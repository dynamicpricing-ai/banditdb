//! P0.3 acceptance tests — group-commit fsync on the WAL.
//!
//! What fsync buys and what it does not:
//!
//! `write()` puts a record in the OS page cache. That survives process death — a
//! SIGKILLed process loses nothing already written — but not power loss, kernel
//! panic, or VM preemption. `sync_all()` is what makes it durable against those.
//! Because `scripts/crash_injection.sh` uses SIGKILL, it cannot observe this
//! behaviour at all; these tests cover it instead, by asserting on the fsync
//! counter directly.
//!
//! Two paths must both work:
//!   * interval — under sustained load, one fsync covers many records
//!   * idle — before the writer parks, so a lone reward is not left unsynced
//!     waiting for company. This keeps RPO near zero when traffic is light.

use banditdb::BanditDB;
use banditdb::state::Algorithm;
use std::fs;
use std::sync::atomic::Ordering;
use std::time::Duration;

fn setup(dir: &str) -> BanditDB {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    db.add_campaign("c", vec!["A".into(), "B".into()], 2, 1.0, Algorithm::Linucb, None, None)
        .unwrap();
    db
}

async fn settle() {
    tokio::time::sleep(Duration::from_millis(400)).await;
}

#[tokio::test]
async fn durable_events_are_fsynced() {
    let dir = "/tmp/banditdb_p03_durable";
    let db = setup(dir);
    settle().await;
    let before = db.wal_fsyncs.load(Ordering::Relaxed);

    let (_, iid) = db.predict("c", vec![0.2, 0.4]).expect("predict");
    db.reward(&iid, 1.0).expect("reward");
    settle().await;

    assert!(
        db.wal_fsyncs.load(Ordering::Relaxed) > before,
        "a reward is state-bearing and must reach disk — no fsync was issued, so it \
         would be lost to power loss even though the process stayed alive"
    );
    let _ = fs::remove_dir_all(dir);
}

/// The idle path: a single reward arriving alone must be synced promptly rather
/// than waiting out the full commit window.
#[tokio::test]
async fn lone_reward_is_synced_without_waiting_for_the_window() {
    let dir = "/tmp/banditdb_p03_idle";
    // A long window would hide the idle path if it were missing.
    std::env::set_var("BANDITDB_FSYNC_INTERVAL_MS", "60000");
    let db = setup(dir);
    settle().await;
    let before = db.wal_fsyncs.load(Ordering::Relaxed);

    let (_, iid) = db.predict("c", vec![0.5, 0.1]).expect("predict");
    db.reward(&iid, 1.0).expect("reward");
    settle().await;

    assert!(
        db.wal_fsyncs.load(Ordering::Relaxed) > before,
        "with a 60s commit window the interval path cannot have fired, so this proves \
         the idle-sync path is missing — a lone reward would sit unsynced for a minute"
    );
    std::env::remove_var("BANDITDB_FSYNC_INTERVAL_MS");
    let _ = fs::remove_dir_all(dir);
}

/// Best-effort records alone must not trigger fsync — predictions are not worth a
/// disk round trip, and syncing for them would defeat the split made in P0.4.
#[tokio::test]
async fn predictions_alone_do_not_force_a_sync() {
    let dir = "/tmp/banditdb_p03_besteffort";
    let db = setup(dir);
    settle().await;
    let before = db.wal_fsyncs.load(Ordering::Relaxed);

    for i in 0..50 {
        db.predict("c", vec![(i % 7) as f64 / 7.0, 0.5]).expect("predict");
    }
    settle().await;

    assert_eq!(
        db.wal_fsyncs.load(Ordering::Relaxed), before,
        "predictions are best-effort; issuing an fsync for them would put a disk round \
         trip back on the serving path that P0.4 removed"
    );
    let _ = fs::remove_dir_all(dir);
}

/// Under a burst the commit window should amortise: far fewer fsyncs than rewards.
#[tokio::test]
async fn burst_of_rewards_is_batched_into_fewer_syncs() {
    let dir = "/tmp/banditdb_p03_batching";
    let db = setup(dir);
    settle().await;
    let before = db.wal_fsyncs.load(Ordering::Relaxed);

    // Collect interactions first so the rewards land as a tight burst.
    let mut iids = Vec::new();
    for i in 0..400 {
        let (_, iid) = db.predict("c", vec![(i % 11) as f64 / 11.0, 0.3]).expect("predict");
        iids.push(iid);
    }
    for iid in &iids {
        db.reward(iid, 1.0).expect("reward");
    }
    settle().await;

    let syncs = db.wal_fsyncs.load(Ordering::Relaxed) - before;
    assert!(syncs > 0, "a burst of rewards must still be synced");
    assert!(
        syncs < iids.len() as u64,
        "group commit did not amortise: {syncs} fsyncs for {} rewards means every \
         record paid for its own disk round trip",
        iids.len()
    );
    let _ = fs::remove_dir_all(dir);
}
