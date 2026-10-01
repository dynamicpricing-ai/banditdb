// The limits are read from the environment when a BanditDB instance is
// constructed, and the environment is process-global while cargo runs tests in
// parallel threads. ENV_LOCK serialises the tests that set it, which means holding
// a std Mutex guard across the awaits in the test body. That is exactly what this
// lint warns about, and here it is the point: the guard has to outlive the awaits
// or a concurrent test observes the wrong limit. Nothing in these tests can
// deadlock on it — the guard is released at the end of each test.
#![allow(clippy::await_holding_lock)]

//! Campaign admission control: the instance-wide campaign cap and the
//! per-campaign size ceiling.
//!
//! The load-bearing tests here are not the ones proving a limit rejects a create.
//! They are the two proving a limit does NOT apply to recovery or WAL replay: an
//! operator who lowers a limit below what an instance already holds must still be
//! able to restart it. Since v2 refuses to start rather than come up empty, a limit
//! enforced on the recovery path would turn a config edit into an outage with no
//! way back except editing the config again from memory.

use banditdb::engine::campaign_memory_estimate;
use banditdb::state::{Algorithm, EngineError, NeuralLinUCBConfig};
use banditdb::BanditDB;

fn data_dir_for(wal: &str) -> String {
    let stem = std::path::Path::new(wal)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unnamed".to_string());
    let dir = format!("/tmp/bdb_{stem}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn arms(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("arm_{i}")).collect()
}

/// Env vars are process-global, so the tests that need them run under one mutex
/// and restore the previous value on the way out.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct EnvGuard(&'static str, Option<String>);

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        std::env::set_var(key, value);
        EnvGuard(key, prev)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.1 {
            Some(v) => std::env::set_var(self.0, v),
            None => std::env::remove_var(self.0),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// The cap applies to creates
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_campaign_cap_rejects_creates_past_the_limit() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _env = EnvGuard::set("BANDITDB_MAX_CAMPAIGNS", "3");

    let wal = "/tmp/banditdb_test_campaign_cap.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    for i in 0..3 {
        db.add_campaign(
            &format!("c{i}"),
            arms(2),
            4,
            1.0,
            Algorithm::Linucb,
            None,
            None,
        )
        .await
        .expect("creates below the cap must succeed");
    }

    let err = db
        .add_campaign(
            "one_too_many",
            arms(2),
            4,
            1.0,
            Algorithm::Linucb,
            None,
            None,
        )
        .await
        .expect_err("the fourth create must be refused");
    assert!(
        matches!(err, EngineError::LimitExceeded(_)),
        "expected LimitExceeded, got {err:?}"
    );
    // The message has to carry the numbers, or the operator cannot act on it.
    let msg = err.to_string();
    assert!(
        msg.contains('3') && msg.contains("BANDITDB_MAX_CAMPAIGNS"),
        "message must name the limit and the count: {msg}"
    );

    // Deleting one frees a slot: the cap is a ceiling, not a high-water mark.
    db.delete_campaign("c0").await.unwrap();
    db.add_campaign(
        "replacement",
        arms(2),
        4,
        1.0,
        Algorithm::Linucb,
        None,
        None,
    )
    .await
    .expect("a create must succeed once a slot is freed");

    let _ = std::fs::remove_file(wal);
}

/// An archived campaign still occupies memory, so it must still occupy a slot.
/// Archiving to make room would otherwise let an instance hold unbounded state
/// while reporting itself under the limit.
#[tokio::test]
async fn test_archived_campaigns_still_count() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _env = EnvGuard::set("BANDITDB_MAX_CAMPAIGNS", "2");

    let wal = "/tmp/banditdb_test_cap_archived.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("a", arms(2), 4, 1.0, Algorithm::Linucb, None, None)
        .await
        .unwrap();
    db.add_campaign("b", arms(2), 4, 1.0, Algorithm::Linucb, None, None)
        .await
        .unwrap();
    db.archive_campaign("a").await.unwrap();

    assert!(
        db.add_campaign("c", arms(2), 4, 1.0, Algorithm::Linucb, None, None)
            .await
            .is_err(),
        "archiving must not free a campaign slot — the state is still resident"
    );

    let _ = std::fs::remove_file(wal);
}

// ════════════════════════════════════════════════════════════════════════════
// The cap does NOT apply to recovery or replay
// ════════════════════════════════════════════════════════════════════════════

/// Lowering the cap below the number of campaigns in a checkpoint must not stop
/// the instance from starting.
#[tokio::test]
async fn test_lowered_cap_does_not_break_checkpoint_recovery() {
    let _lock = ENV_LOCK.lock().unwrap();
    let wal = "/tmp/banditdb_test_cap_recovery.jsonl";
    let _ = std::fs::remove_file(wal);
    let data_dir = data_dir_for(wal);

    {
        let _env = EnvGuard::set("BANDITDB_MAX_CAMPAIGNS", "5");
        let db = BanditDB::new(wal, &data_dir);
        for i in 0..5 {
            db.add_campaign(
                &format!("c{i}"),
                arms(2),
                4,
                1.0,
                Algorithm::Linucb,
                None,
                None,
            )
            .await
            .unwrap();
        }
        db.checkpoint().await.expect("checkpoint must succeed");
    }

    // Operator lowers the limit, then restarts.
    let _env = EnvGuard::set("BANDITDB_MAX_CAMPAIGNS", "2");
    let db = BanditDB::new(wal, &data_dir);
    assert_eq!(
        db.campaigns.read().len(),
        5,
        "recovery must restore every campaign in the checkpoint, limit notwithstanding"
    );

    // New creates are still refused while over the limit.
    assert!(
        db.add_campaign("new", arms(2), 4, 1.0, Algorithm::Linucb, None, None)
            .await
            .is_err(),
        "creates must stay refused while the instance is over its lowered limit"
    );

    let _ = std::fs::remove_file(wal);
}

/// Same rule for the WAL: replay must not enforce the cap, or a lowered limit
/// makes an existing log unreplayable.
#[tokio::test]
async fn test_lowered_cap_does_not_break_wal_replay() {
    let _lock = ENV_LOCK.lock().unwrap();
    let wal = "/tmp/banditdb_test_cap_replay.jsonl";
    let _ = std::fs::remove_file(wal);
    let data_dir = data_dir_for(wal);

    {
        let _env = EnvGuard::set("BANDITDB_MAX_CAMPAIGNS", "4");
        let db = BanditDB::new(wal, &data_dir);
        for i in 0..4 {
            db.add_campaign(
                &format!("w{i}"),
                arms(2),
                4,
                1.0,
                Algorithm::Linucb,
                None,
                None,
            )
            .await
            .unwrap();
        }
        // No checkpoint: these survive only as WAL records.
    }

    let _env = EnvGuard::set("BANDITDB_MAX_CAMPAIGNS", "1");
    let db = BanditDB::new(wal, &data_dir);
    assert_eq!(
        db.campaigns.read().len(),
        4,
        "WAL replay must restore every campaign, limit notwithstanding"
    );

    let _ = std::fs::remove_file(wal);
}

// ════════════════════════════════════════════════════════════════════════════
// Per-campaign size ceiling
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_size_ceiling_rejects_an_oversized_campaign() {
    let _lock = ENV_LOCK.lock().unwrap();
    // 1 MB: comfortably over a small LinUCB campaign, far under a wide one.
    let _env = EnvGuard::set("BANDITDB_MAX_CAMPAIGN_BYTES", "1048576");

    let wal = "/tmp/banditdb_test_size_ceiling.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("small", arms(5), 16, 1.0, Algorithm::Linucb, None, None)
        .await
        .expect("5 arms at d=16 is ~10 KB and must be admitted");

    // 20 arms at d=128 is 20 × 8 × 128² ≈ 2.6 MB.
    let err = db
        .add_campaign("wide", arms(20), 128, 1.0, Algorithm::Linucb, None, None)
        .await
        .expect_err("an oversized campaign must be refused");
    assert!(matches!(err, EngineError::LimitExceeded(_)), "got {err:?}");
    let msg = err.to_string();
    assert!(
        msg.contains("MB") && msg.contains("BANDITDB_MAX_CAMPAIGN_BYTES"),
        "message must state the estimate and the limit: {msg}"
    );

    let _ = std::fs::remove_file(wal);
}

/// Default is unlimited, so an existing deployment sees no behaviour change.
#[tokio::test]
async fn test_size_ceiling_is_off_by_default() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _env = EnvGuard::set("BANDITDB_MAX_CAMPAIGN_BYTES", "0");

    let wal = "/tmp/banditdb_test_size_default.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("wide", arms(20), 128, 1.0, Algorithm::Linucb, None, None)
        .await
        .expect("with the ceiling disabled, size must not be checked");

    let _ = std::fs::remove_file(wal);
}

// ════════════════════════════════════════════════════════════════════════════
// The estimator itself
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_memory_estimate_matches_hand_computation() {
    // LinUCB: arms × (8d² + 16d + 64)
    let d = 64u64;
    let expected = 10 * (8 * d * d + 16 * d + 64);
    assert_eq!(
        campaign_memory_estimate(10, 64, &Algorithm::Linucb),
        expected
    );

    // Thompson sampling doubles it: the Cholesky factor is cached per arm.
    assert_eq!(
        campaign_memory_estimate(10, 64, &Algorithm::ThompsonSampling),
        2 * expected,
        "TS must account for the cached Cholesky factor"
    );
}

#[test]
fn test_neural_estimate_is_dominated_by_the_replay_buffer() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _env = EnvGuard::set("BANDITDB_NEURAL_BUFFER_CAP", "50000");

    let cfg = NeuralLinUCBConfig {
        context_dim: 256,
        embed_dim: 32,
        hidden_dim: 128,
        hidden_layers: 2,
        retrain_every: 200,
        retrain_steps: 100,
        learning_rate: 1e-3,
        lambda: 1.0,
    };
    let total = campaign_memory_estimate(6, 32, &Algorithm::NeuralLinUCB(cfg));
    let arms_only = 6 * (8 * 32 * 32 + 16 * 32 + 64);

    // 50,000 × (8 × 256 + 70) ≈ 105 MB against ~50 KB of arm state.
    assert!(total > 100 * 1024 * 1024, "expected ~105 MB, got {total}");
    assert!(
        total > arms_only * 1000,
        "the buffer must dominate: arms {arms_only} vs total {total}"
    );
}
