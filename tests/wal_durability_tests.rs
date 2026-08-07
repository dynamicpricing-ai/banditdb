//! P0.4 acceptance tests — prediction logging is best-effort, reward logging is not.
//!
//! Background (docs/PRODUCTION_STAGE1.md P0.4): every write path used to share one
//! failure policy. A saturated WAL writer therefore failed `predict` with a 503,
//! making read-shaped traffic fail because *logging* fell behind. Predictions are
//! recoverable — the interaction cache holds them — so they are now dropped and
//! counted instead. Rewards and campaign lifecycle events keep a hard guarantee.

use banditdb::BanditDB;
use banditdb::state::{Algorithm, EngineError};
use std::fs;
use std::sync::atomic::Ordering;

async fn fresh(dir: &str) -> BanditDB {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    db.add_campaign("c", vec!["A".into(), "B".into()], 2, 1.0, Algorithm::Linucb, None, None).await
        .unwrap();
    db
}

/// Saturate the WAL channel so the next enqueue cannot succeed.
/// Returns once the channel reports no free capacity.
fn saturate_wal(db: &BanditDB) -> bool {
    for i in 0..200_000 {
        if db.event_tx.capacity() == 0 {
            return true;
        }
        let ctx = vec![(i % 7) as f64 / 7.0, (i % 5) as f64 / 5.0];
        if db.predict("c", ctx).is_err() {
            // A prediction must never fail on a full channel — that is the bug.
            return db.event_tx.capacity() == 0;
        }
    }
    db.event_tx.capacity() == 0
}

#[tokio::test]
async fn prediction_survives_a_saturated_wal_and_is_counted() {
    let dir = "/tmp/banditdb_p04_best_effort";
    let db = fresh(dir).await;

    if !saturate_wal(&db) {
        eprintln!("SKIPPED prediction_survives_a_saturated_wal_and_is_counted: \
                   writer drained faster than the test could fill the channel");
        let _ = fs::remove_dir_all(dir);
        return;
    }

    let before = db.wal_dropped.load(Ordering::Relaxed);
    let result = db.predict("c", vec![0.5, 0.5]);
    assert!(
        result.is_ok(),
        "predict must not fail when the WAL writer is saturated — prediction logging \
         is best-effort and a logging backlog must not take down the serving path"
    );
    assert!(
        db.wal_dropped.load(Ordering::Relaxed) > before,
        "a dropped prediction record must increment wal_dropped so operators can see \
         that late rewards will stop matching"
    );

    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn reward_still_fails_hard_on_a_saturated_wal() {
    let dir = "/tmp/banditdb_p04_required";
    let db = fresh(dir).await;

    // Take one interaction before saturating so there is something to reward.
    let (_, iid) = db.predict("c", vec![0.1, 0.2]).expect("initial predict");

    if !saturate_wal(&db) {
        eprintln!("SKIPPED reward_still_fails_hard_on_a_saturated_wal: channel never filled");
        let _ = fs::remove_dir_all(dir);
        return;
    }

    match db.reward(&iid, 1.0).await {
        Err(EngineError::WalFull) => {}
        Err(EngineError::WalUnavailable) => {}
        other => panic!(
            "reward carries state that exists nowhere else and must surface enqueue \
             failure to the caller; got {other:?}"
        ),
    }

    let _ = fs::remove_dir_all(dir);
}

/// `interact()` writes a Predicted and a Rewarded describing the same interaction.
/// Both must be Required: dropping the prediction would leave a reward referencing
/// an interaction that WAL replay never saw, silently discarding the observation.
#[tokio::test]
async fn interact_treats_its_paired_events_as_durable() {
    let dir = "/tmp/banditdb_p04_interact";
    let db = fresh(dir).await;

    if !saturate_wal(&db) {
        eprintln!("SKIPPED interact_treats_its_paired_events_as_durable: channel never filled");
        let _ = fs::remove_dir_all(dir);
        return;
    }

    let before = db.wal_dropped.load(Ordering::Relaxed);
    let result = db.interact("c", "A", vec![0.3, 0.4], 1.0).await;
    assert!(
        result.is_err(),
        "interact must fail rather than half-log a paired prediction/reward"
    );
    assert_eq!(
        db.wal_dropped.load(Ordering::Relaxed), before,
        "interact must not silently drop its prediction record as best-effort"
    );

    let _ = fs::remove_dir_all(dir);
}

/// Campaign lifecycle events are state-bearing and must be durable before the
/// caller is told they succeeded.
///
/// They were `Required` — hard failure policy, eventually fsynced — but the caller
/// did not wait, so a create followed immediately by process death could be lost
/// while the client believed the campaign existed. Same class as the reward gap
/// P0.3b closed, deferred at the time to keep that change reviewable.
#[tokio::test]
async fn campaign_lifecycle_is_durable_before_returning() {
    let dir = "/tmp/banditdb_p13_lifecycle";
    let db = fresh(dir).await;

    // Each of these must be on disk by the time it returns. The fsync counter is
    // the observable: a lifecycle op that skipped durability would not advance it.
    let before = db.wal_fsyncs.load(Ordering::Relaxed);

    db.add_campaign("lifecycle", vec!["A".into(), "B".into()], 2, 1.0,
                    Algorithm::Linucb, None, None).await.expect("create");
    let after_create = db.wal_fsyncs.load(Ordering::Relaxed);
    assert!(after_create > before, "create returned without an fsync covering it");

    db.archive_campaign("lifecycle").await.expect("archive");
    let after_archive = db.wal_fsyncs.load(Ordering::Relaxed);
    assert!(after_archive > after_create, "archive returned without an fsync covering it");

    db.restore_campaign("lifecycle").await.expect("restore");
    let after_restore = db.wal_fsyncs.load(Ordering::Relaxed);
    assert!(after_restore > after_archive, "restore returned without an fsync covering it");

    db.delete_campaign("lifecycle").await.expect("delete");
    assert!(
        db.wal_fsyncs.load(Ordering::Relaxed) > after_restore,
        "delete returned without an fsync covering it — a delete that is acknowledged \
         but not durable resurrects the campaign on restart"
    );

    let _ = fs::remove_dir_all(dir);
}

/// A campaign created and acknowledged must survive an immediate restart.
#[tokio::test]
async fn created_campaign_survives_immediate_restart() {
    let dir = "/tmp/banditdb_p13_restart";
    let db = fresh(dir).await;

    db.add_campaign("survivor", vec!["A".into(), "B".into()], 2, 1.0,
                    Algorithm::Linucb, None, None).await.expect("create");
    // No checkpoint: recovery must find it by replaying the WAL alone.
    drop(db);

    let recovered = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    assert!(
        recovered.campaigns.read().contains_key("survivor"),
        "a campaign whose creation was acknowledged did not survive restart"
    );

    let _ = fs::remove_dir_all(dir);
}

/// Baseline: with a healthy writer nothing is dropped and both paths succeed.
#[tokio::test]
async fn healthy_wal_drops_nothing() {
    let dir = "/tmp/banditdb_p04_healthy";
    let db = fresh(dir).await;

    for i in 0..200 {
        let ctx = vec![(i % 9) as f64 / 9.0, (i % 4) as f64 / 4.0];
        let (arm, iid) = db.predict("c", ctx).expect("predict");
        db.reward(&iid, if arm == "A" { 1.0 } else { 0.0 }).await.expect("reward");
    }

    assert_eq!(
        db.wal_dropped.load(Ordering::Relaxed), 0,
        "no records should be dropped under normal load"
    );

    let _ = fs::remove_dir_all(dir);
}
