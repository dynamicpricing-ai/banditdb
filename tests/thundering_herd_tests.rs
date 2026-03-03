use banditdb::state::Algorithm;
use banditdb::BanditDB;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Test 2.1 — Commutative b Assertion
///
/// b += context * reward is commutative: addition order doesn't matter. After
/// N=1000 concurrent reward() calls with identical (context=[1.0, 0.0], reward=1.0),
/// b[0] must equal exactly 1000.0 regardless of scheduling order. Any lost write
/// or race condition on the arm's write lock shows up as b[0] != 1000.0.
#[tokio::test]
async fn test_2_1_commutative_b_assertion() {
    let wal = "/tmp/banditdb_test_2_1.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = Arc::new(BanditDB::new(wal, "/tmp"));
    db.add_campaign("stress", vec!["arm".to_string()], 2, 1.0, Algorithm::Linucb);

    const N: usize = 1000;

    // Step 1: 1000 sequential predicts to mint interaction IDs.
    // All land on the same arm (it's the only one) with context = [1.0, 0.0].
    let mut interaction_ids = Vec::with_capacity(N);
    for _ in 0..N {
        if let Some((_, iid)) = db.predict("stress", vec![1.0, 0.0]) {
            interaction_ids.push(iid);
        }
    }
    assert_eq!(interaction_ids.len(), N, "Not all sequential predicts succeeded");

    // Step 2: reward all 1000 interactions concurrently.
    let ids = Arc::new(interaction_ids);
    let mut handles = Vec::with_capacity(N);
    for i in 0..N {
        let db = Arc::clone(&db);
        let ids = Arc::clone(&ids);
        handles.push(tokio::spawn(async move {
            db.reward(&ids[i], 1.0);
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    // Step 3: b[0] must be exactly N.
    // Each update adds context[0] * reward = 1.0 * 1.0 = 1.0 to b[0].
    // Since addition commutes, b[0] == N regardless of concurrent update order.
    let campaigns = db.campaigns.read();
    let campaign = campaigns.get("stress").unwrap();
    let arms = campaign.arms.read();
    let b = &arms.get("arm").unwrap().b;

    assert_eq!(
        b[0],
        N as f64,
        "b[0]={} but expected {}. Dropped {} concurrent reward updates.",
        b[0],
        N as f64,
        N as f64 - b[0]
    );
    assert_eq!(b[1], 0.0, "b[1] must stay 0 since context[1]=0.0");

    let _ = std::fs::remove_file(wal);
}

/// Test 2.2 — WAL Event Count Integrity
///
/// N=500 concurrent predict→reward cycles must produce exactly 1 + 2N WAL lines
/// (1 CampaignCreated + N Predicted + N Rewarded). The unbounded channel must
/// never drop events under concurrent load. We poll the file until all events
/// are flushed by the async background writer.
#[tokio::test]
async fn test_2_2_wal_event_count_integrity() {
    let wal = "/tmp/banditdb_test_2_2.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = Arc::new(BanditDB::new(wal, "/tmp"));
    db.add_campaign("concurrent", vec!["a".to_string(), "b".to_string()], 3, 1.0, Algorithm::Linucb);

    const N: usize = 500;
    let mut handles = Vec::with_capacity(N);

    for i in 0..N {
        let db = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            let ctx = vec![
                (i as f64 * 0.01).sin(),
                (i as f64 * 0.01).cos(),
                (i as f64 * 0.1) % 1.0,
            ];
            if let Some((_, iid)) = db.predict("concurrent", ctx) {
                db.reward(&iid, 1.0);
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    // The WAL writer is async. Poll until all events are flushed or timeout.
    let expected = 1 + 2 * N; // CampaignCreated + N Predicted + N Rewarded
    let deadline = Instant::now() + Duration::from_secs(5);

    loop {
        let content = std::fs::read_to_string(wal).unwrap_or_default();
        let count = content.lines().filter(|l| !l.is_empty()).count();
        if count >= expected {
            assert_eq!(
                count, expected,
                "WAL has {} lines but expected {} — events were duplicated or lost",
                count, expected
            );
            break;
        }
        assert!(
            Instant::now() < deadline,
            "WAL flush timed out after 5s with only {}/{} events written",
            count,
            expected
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let _ = std::fs::remove_file(wal);
}

/// Test 2.3 — Reader Starvation Check
///
/// Under sustained read pressure (9 predict-only threads), the single write thread
/// (predict + reward) must still make measurable progress. This verifies that
/// parking_lot::RwLock's fairness prevents reader starvation of the arm-update
/// write lock (campaigns.read → arms.write nesting).
#[tokio::test]
async fn test_2_3_reader_starvation_check() {
    let wal = "/tmp/banditdb_test_2_3.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = Arc::new(BanditDB::new(wal, "/tmp"));
    db.add_campaign("stress", vec!["a".to_string(), "b".to_string(), "c".to_string()], 4, 1.0, Algorithm::Linucb);

    let error_count = Arc::new(AtomicUsize::new(0));
    let reward_count = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();

    // 9 reader tasks: predict only (acquire campaigns.read + arms.read)
    for i in 0..9_usize {
        let db = Arc::clone(&db);
        let errors = Arc::clone(&error_count);
        let stop = Arc::clone(&stop);
        handles.push(tokio::spawn(async move {
            let ctx = vec![(i as f64 * 0.1).sin(), (i as f64 * 0.1).cos(), 0.5, 0.5];
            while !stop.load(Ordering::Relaxed) {
                if db.predict("stress", ctx.clone()).is_none() {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }));
    }

    // 1 writer task: predict + reward (acquires arms.write — the contended path)
    {
        let db = Arc::clone(&db);
        let errors = Arc::clone(&error_count);
        let rewards = Arc::clone(&reward_count);
        let stop = Arc::clone(&stop);
        handles.push(tokio::spawn(async move {
            let ctx = vec![0.5_f64, 0.5, 0.3, 0.7];
            while !stop.load(Ordering::Relaxed) {
                match db.predict("stress", ctx.clone()) {
                    Some((_, iid)) => {
                        db.reward(&iid, 1.0);
                        rewards.fetch_add(1, Ordering::Relaxed);
                    }
                    None => {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }));
    }

    tokio::time::sleep(Duration::from_secs(2)).await;
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.await.unwrap();
    }

    let errors = error_count.load(Ordering::Relaxed);
    let rewards = reward_count.load(Ordering::Relaxed);

    assert_eq!(errors, 0, "Encountered {} errors under mixed read/write load", errors);
    assert!(
        rewards > 0,
        "Writer thread completed 0 rewards in 2s — write-lock may be starved by readers"
    );

    let _ = std::fs::remove_file(wal);
}
