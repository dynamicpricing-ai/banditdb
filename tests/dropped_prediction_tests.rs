//! An acknowledged reward must survive restart even when its prediction's WAL
//! record was dropped.
//!
//! Prediction records are best-effort: when the WAL queue is full they are
//! dropped and counted, but the prediction stays in memory so its reward is still
//! accepted. The reward is fsynced before it is acknowledged, yet replay used to
//! skip it — with no `Predicted` record it had nothing to match against — so a
//! restart before the next checkpoint silently lost it.

use banditdb::engine::WalMessage;
use banditdb::state::Algorithm;
use banditdb::BanditDB;
use std::fs;
use std::sync::atomic::Ordering;

fn reward_count(db: &BanditDB) -> u64 {
    db.campaigns.read()["c"].arms.read()["a"]
        .reward_count
        .load(Ordering::Relaxed)
}

// Single-threaded runtime: the WAL writer cannot run until the test awaits, so a
// synchronous burst of predictions fills the queue and the last ones are dropped.
#[tokio::test(flavor = "current_thread")]
async fn reward_for_dropped_prediction_survives_restart() {
    // Keep every prediction in the pending cache; eviction is a different failure.
    std::env::set_var("BANDITDB_MAX_PENDING_INTERACTIONS", "1000000");
    let dir = "/tmp/banditdb_dropped_prediction";
    let wal = format!("{dir}/wal.jsonl");
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();

    let db = BanditDB::new(&wal, dir);
    db.add_campaign("c", vec!["a".into()], 2, 1.0, Algorithm::Linucb, None, None)
        .await
        .unwrap();

    let first = db.predict("c", vec![0.5, 0.5]).unwrap().1;
    let mut last = String::new();
    for _ in 0..100_010 {
        last = db.predict("c", vec![0.5, 0.5]).unwrap().1;
    }
    assert!(
        db.wal_dropped.load(Ordering::Relaxed) > 0,
        "the burst must overflow the WAL queue"
    );

    // Let the writer drain so the rewards themselves are not refused.
    let (tx, rx) = tokio::sync::oneshot::channel();
    db.event_tx
        .send(WalMessage::Checkpoint { reply: tx })
        .await
        .unwrap();
    rx.await.unwrap();

    db.reward(&first, 1.0).await.unwrap();
    db.reward(&last, 1.0).await.unwrap();
    assert_eq!(reward_count(&db), 2);

    // Only the reward whose prediction went unlogged pays for carrying it. Counted
    // on raw bytes so the check holds for both JSON and MessagePack WALs.
    let log = fs::read(&wal).unwrap();
    let carried = log
        .windows(b"unlogged_prediction".len())
        .filter(|w| *w == b"unlogged_prediction")
        .count();
    assert_eq!(
        carried, 1,
        "a reward whose prediction is in the WAL must not repeat it"
    );

    drop(db);
    let db2 = BanditDB::new(&wal, dir);
    assert_eq!(
        reward_count(&db2),
        2,
        "an acknowledged reward was lost on restart because its prediction record had been dropped"
    );
    let _ = fs::remove_dir_all(dir);
}
