//! `checkpoint.prev` must be a lossless fallback.
//!
//! Rotation discards the WAL segment a checkpoint subsumes. When `checkpoint.json`
//! was unreadable, recovery fell back to `checkpoint.prev` and replayed a WAL that
//! began at the *newer* checkpoint's boundary — so every event between the two
//! checkpoints was gone, including acknowledged rewards and whole campaigns, and
//! the server still came up healthy.
//!
//! Rotation now keeps that segment, and the rotated WAL records which checkpoint
//! it starts at, so recovery can rebuild `prev + segment + WAL` exactly.

use banditdb::engine::WalMessage;
use banditdb::state::Algorithm;
use banditdb::BanditDB;
use std::fs;
use std::sync::atomic::Ordering;

fn fresh(dir: &str) -> String {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    format!("{dir}/wal.jsonl")
}

fn rewards(db: &BanditDB, campaign: &str, arm: &str) -> Option<u64> {
    db.campaigns.read().get(campaign)
        .map(|c| c.arms.read()[arm].reward_count.load(Ordering::Relaxed))
}

fn corrupt_current_checkpoint(dir: &str) {
    fs::write(format!("{dir}/checkpoint.json"), b"{not a checkpoint").unwrap();
}

async fn interact_n(db: &BanditDB, campaign: &str, arm: &str, n: usize) {
    for _ in 0..n {
        db.interact(campaign, arm, vec![0.5, 0.5], 1.0).await.unwrap();
    }
}

#[tokio::test]
async fn fallback_to_previous_checkpoint_is_lossless() {
    let dir = "/tmp/banditdb_fallback_lossless";
    let wal = fresh(dir);

    let db = BanditDB::new(&wal, dir);
    db.add_campaign("c", vec!["a".into()], 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    interact_n(&db, "c", "a", 3).await;
    db.checkpoint().await.unwrap();

    // Everything here lives only in the segment the next rotation discards.
    interact_n(&db, "c", "a", 5).await;
    db.add_campaign("d", vec!["x".into()], 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    let pending = db.predict("d", vec![0.5, 0.5]).unwrap().1;
    db.checkpoint().await.unwrap();

    interact_n(&db, "c", "a", 2).await;
    db.reward(&pending, 1.0).await.unwrap();
    drop(db);

    corrupt_current_checkpoint(dir);
    let db = BanditDB::new(&wal, dir);
    assert_eq!(rewards(&db, "c", "a"), Some(10), "events between the two checkpoints were lost");
    assert_eq!(rewards(&db, "d", "x"), Some(1),
        "a campaign created between the checkpoints, and its acknowledged reward, were lost");

    // A process that started from the fallback must itself stay recoverable: its
    // next checkpoint cannot leave the unreadable file as the new fallback.
    interact_n(&db, "c", "a", 4).await;
    db.checkpoint().await.unwrap();
    interact_n(&db, "c", "a", 1).await;
    drop(db);

    corrupt_current_checkpoint(dir);
    let db = BanditDB::new(&wal, dir);
    assert_eq!(rewards(&db, "c", "a"), Some(15), "second fallback lost events");
    assert_eq!(rewards(&db, "d", "x"), Some(1));
    let _ = fs::remove_dir_all(dir);
}

/// A crash after the new checkpoint is written but before the WAL is rotated
/// leaves the old, unrotated WAL in place. Falling back from there must replay it
/// once — not the saved segment and then the same events again from the WAL.
#[tokio::test]
async fn fallback_after_crash_before_rotation_applies_each_event_once() {
    let dir = "/tmp/banditdb_fallback_unrotated";
    let wal = fresh(dir);

    let db = BanditDB::new(&wal, dir);
    db.add_campaign("c", vec!["a".into()], 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    interact_n(&db, "c", "a", 3).await;
    db.checkpoint().await.unwrap();
    interact_n(&db, "c", "a", 5).await;

    // Capture the WAL exactly as the second checkpoint will find it.
    let (tx, rx) = tokio::sync::oneshot::channel();
    db.event_tx.send(WalMessage::Checkpoint { reply: tx }).await.unwrap();
    rx.await.unwrap();
    let unrotated = fs::read(&wal).unwrap();

    db.checkpoint().await.unwrap();
    drop(db);

    // Put back the pre-rotation WAL, and lose the new checkpoint.
    fs::write(&wal, &unrotated).unwrap();
    corrupt_current_checkpoint(dir);

    let db = BanditDB::new(&wal, dir);
    assert_eq!(rewards(&db, "c", "a"), Some(8));
    let _ = fs::remove_dir_all(dir);
}
