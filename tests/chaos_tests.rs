use banditdb::state::DbEvent;
use banditdb::BanditDB;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Test 3.1 — Torn Write Recovery
///
/// Appending a truncated JSON fragment after 100 valid WAL lines simulates a
/// crash mid-write. BanditDB::recover() must skip the corrupt fragment silently
/// via the `if let Ok(event)` guard and restore all 100 valid events. No panic.
#[tokio::test]
async fn test_3_1_torn_write_recovery() {
    let wal = "/tmp/banditdb_test_3_1.jsonl";
    let _ = std::fs::remove_file(wal);

    // Write exactly 100 valid WAL events by hand: 1 CampaignCreated + 99 Predicted.
    {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(wal)
            .unwrap();

        let campaign_event = DbEvent::CampaignCreated {
            campaign_id: "torn_test".to_string(),
            arms: vec!["arm_a".to_string(), "arm_b".to_string()],
            feature_dim: 2,
        };
        writeln!(file, "{}", serde_json::to_string(&campaign_event).unwrap()).unwrap();

        for i in 0..99_usize {
            let event = DbEvent::Predicted {
                interaction_id: format!("iid-{}", i),
                campaign_id: "torn_test".to_string(),
                arm_id: "arm_a".to_string(),
                context: vec![1.0, 0.0],
                timestamp_secs: 0,
            };
            writeln!(file, "{}", serde_json::to_string(&event).unwrap()).unwrap();
        }
    }

    assert_eq!(
        std::fs::read_to_string(wal).unwrap().lines().count(),
        100,
        "Baseline: expected 100 clean WAL lines before injecting corruption"
    );

    // Simulate a crash mid-write: append a partial JSON object with no trailing newline.
    {
        let mut file = std::fs::OpenOptions::new().append(true).open(wal).unwrap();
        write!(file, r#"{{"type":"Rewarded","inter"#).unwrap();
    }

    // Recover: must skip the corrupt fragment and restore all 100 valid events.
    let db = BanditDB::new(wal, "/tmp");

    let campaigns = db.campaigns.read();
    assert!(
        campaigns.contains_key("torn_test"),
        "Campaign not found after recovery — valid events were not replayed"
    );
    drop(campaigns);

    // DB must be fully operational after recovery with a corrupt WAL tail.
    assert!(
        db.predict("torn_test", vec![1.0, 0.0]).is_some(),
        "DB is not functional after torn-write recovery"
    );

    let _ = std::fs::remove_file(wal);
}

/// Test 3.2 — The Orphaned Reward (Silent Failure Mode)
///
/// reward() with an interaction_id absent from the Moka cache is a documented
/// no-op: no model update occurs and no error is returned. This documents the
/// silent failure mode — a caller past TTL gets "OK" but no learning happens.
/// A contrast case verifies that a valid reward DOES update the model.
#[tokio::test]
async fn test_3_2_orphaned_reward_is_noop() {
    let wal = "/tmp/banditdb_test_3_2.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    db.add_campaign("orphan_test", vec!["arm".to_string()], 2);

    let theta_before = {
        let campaigns = db.campaigns.read();
        let campaign = campaigns.get("orphan_test").unwrap();
        let arms = campaign.arms.read();
        arms.get("arm").unwrap().theta.clone()
    };

    // Ghost reward: this interaction_id was never predicted — not in the Moka cache.
    db.reward("ghost-id-that-was-never-predicted", 999.0);

    let theta_after_ghost = {
        let campaigns = db.campaigns.read();
        let campaign = campaigns.get("orphan_test").unwrap();
        let arms = campaign.arms.read();
        arms.get("arm").unwrap().theta.clone()
    };

    assert_eq!(
        theta_before, theta_after_ghost,
        "Orphaned reward mutated theta — reward() must be a no-op for unknown interaction IDs"
    );

    // Contrast: a valid reward DOES update theta, confirming the arm can learn.
    let (_, iid) = db.predict("orphan_test", vec![1.0, 0.0]).unwrap();
    db.reward(&iid, 1.0);

    let theta_after_valid = {
        let campaigns = db.campaigns.read();
        let campaign = campaigns.get("orphan_test").unwrap();
        let arms = campaign.arms.read();
        arms.get("arm").unwrap().theta.clone()
    };

    assert_ne!(
        theta_after_ghost, theta_after_valid,
        "Valid reward should have updated theta — arm model is not learning"
    );

    let _ = std::fs::remove_file(wal);
}

/// Test 3.3 — Idempotent Recovery
///
/// The WAL is a perfect event log. A fresh BanditDB reconstructed from the same
/// WAL file must produce a theta identical to the original trained model.
/// This is the end-to-end proof that crash recovery is lossless.
#[tokio::test]
async fn test_3_3_idempotent_recovery() {
    let wal = "/tmp/banditdb_test_3_3.jsonl";
    let _ = std::fs::remove_file(wal);

    const N: usize = 50;

    // Phase 1: train a model and capture its final theta.
    let theta_original = {
        let db = BanditDB::new(wal, "/tmp");
        db.add_campaign("recovery_campaign", vec!["arm".to_string()], 2);

        for i in 0..N {
            let angle = i as f64 * 0.1;
            let ctx = vec![angle.sin(), angle.cos()];
            if let Some((_, iid)) = db.predict("recovery_campaign", ctx) {
                db.reward(&iid, 1.0);
            }
        }

        // Poll until the async WAL writer has flushed all events to disk.
        let expected = 1 + N * 2; // CampaignCreated + N Predicted + N Rewarded
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let count = std::fs::read_to_string(wal)
                .unwrap_or_default()
                .lines()
                .filter(|l| !l.is_empty())
                .count();
            if count >= expected {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "WAL flush timed out ({} of {} lines written)",
                count,
                expected
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let campaigns = db.campaigns.read();
        let campaign = campaigns.get("recovery_campaign").unwrap();
        let arms = campaign.arms.read();
        arms.get("arm").unwrap().theta.clone()
    };
    // db is dropped here; the WAL file persists on disk.

    // Phase 2: reconstruct from WAL and verify theta is identical.
    let db_recovered = BanditDB::new(wal, "/tmp");

    let theta_recovered = {
        let campaigns = db_recovered.campaigns.read();
        let campaign = campaigns.get("recovery_campaign").unwrap();
        let arms = campaign.arms.read();
        arms.get("arm").unwrap().theta.clone()
    };

    assert_eq!(
        theta_original,
        theta_recovered,
        "Recovered theta doesn't match original — WAL replay is not lossless.\nOriginal:  {:?}\nRecovered: {:?}",
        theta_original,
        theta_recovered
    );

    let _ = std::fs::remove_file(wal);
}

/// Test 3.4 — Concurrent Checkpoint Safety
///
/// checkpoint() reads the WAL, writes Parquet, and rotates the WAL while the
/// async background writer is actively appending to it. The operation must not
/// panic or deadlock. A clean Err is also acceptable. The key assertion is:
/// reaching the end of this test without a panic.
#[tokio::test]
async fn test_3_4_concurrent_export_safety() {
    let wal     = "/tmp/banditdb_test_3_4.jsonl";
    let data_dir = "/tmp/banditdb_test_3_4_data";
    std::fs::create_dir_all(data_dir).unwrap();
    let _ = std::fs::remove_file(wal);
    // Clean up any leftover exports / checkpoint from a previous run
    let _ = std::fs::remove_dir_all(format!("{}/exports", data_dir));
    let _ = std::fs::remove_file(format!("{}/checkpoint.json", data_dir));

    let db = Arc::new(BanditDB::new(wal, data_dir));
    db.add_campaign("export_stress", vec!["a".to_string(), "b".to_string()], 3);

    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // 10 concurrent predict→reward tasks — constantly appending Predicted +
    // Rewarded events to the WAL while checkpoint() runs.
    for i in 0..10_usize {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        handles.push(tokio::spawn(async move {
            let ctx = vec![(i as f64 * 0.1).sin(), (i as f64 * 0.1).cos(), 0.5];
            while !stop.load(Ordering::Relaxed) {
                if let Some((_, iid)) = db.predict("export_stress", ctx.clone()) {
                    db.reward(&iid, 1.0);
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }));
    }

    // Give tasks time to build up non-trivial WAL content before checkpointing.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Fire checkpoint (includes Parquet export + WAL rotation) while tasks are
    // still actively writing to the same file.
    let checkpoint_result = db.checkpoint().await;

    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.await.unwrap();
    }

    // Reaching here proves no panic occurred — that is the primary assertion.
    match checkpoint_result {
        Ok(msg) => {
            println!("Checkpoint succeeded: {}", msg);
            // checkpoint.json must exist
            assert!(
                std::path::Path::new(&format!("{}/checkpoint.json", data_dir)).exists(),
                "Checkpoint succeeded but checkpoint.json was not created"
            );
        }
        Err(e) => {
            println!("Checkpoint returned a clean error (acceptable under concurrent writes): {}", e);
        }
    }

    let _ = std::fs::remove_file(wal);
    let _ = std::fs::remove_dir_all(data_dir);
}
