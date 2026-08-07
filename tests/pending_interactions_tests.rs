//! P0.5 acceptance tests — bounded interaction cache and the persisted pending set.
//!
//! Two defects (docs/PRODUCTION_STAGE1.md P0.5):
//!
//!   1. The cache had a TTL but no capacity limit. At 1,000 predictions/sec against
//!      the default 24h TTL that is 86.4M records, each holding a context vector —
//!      an OOM kill, not a graceful degradation.
//!   2. Unmatched predictions were re-emitted into the WAL on every checkpoint, so a
//!      campaign with a low conversion rate rewrote its whole backlog each time.
//!      They now travel inside the checkpoint instead.

use banditdb::BanditDB;
use banditdb::state::{Algorithm, CheckpointData};
use std::fs;
use std::sync::atomic::Ordering;

async fn setup(dir: &str) -> BanditDB {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    db.add_campaign("c", vec!["A".into(), "B".into()], 2, 1.0, Algorithm::Linucb, None, None).await
        .unwrap();
    db
}

fn ctx(i: usize) -> Vec<f64> {
    vec![(i % 9) as f64 / 9.0, (i % 5) as f64 / 5.0]
}

fn read_checkpoint(dir: &str) -> CheckpointData {
    let raw = fs::read_to_string(format!("{dir}/checkpoint.json")).expect("checkpoint.json");
    serde_json::from_str(&raw).expect("valid checkpoint")
}

// ---------------------------------------------------------------------------
// Capacity bound
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cache_is_bounded_and_reports_evictions() {
    let dir = "/tmp/banditdb_p05_bounded";
    std::env::set_var("BANDITDB_MAX_PENDING_INTERACTIONS", "100");
    let db = setup(dir).await;

    // Predict without rewarding, so nothing is invalidated and the cache can only grow.
    for i in 0..2_000 {
        db.predict("c", ctx(i)).expect("predict");
    }
    db.interactions.run_pending_tasks();

    let held = db.interactions.entry_count();
    assert!(
        held <= 200,
        "cache held {held} entries against a 100-entry limit — unbounded growth is \
         the OOM path this bound exists to prevent"
    );
    assert!(
        db.interactions_evicted.load(Ordering::Relaxed) > 0,
        "evictions must be counted: each one is a prediction whose reward can no \
         longer be matched, which operators need to alert on"
    );

    std::env::remove_var("BANDITDB_MAX_PENDING_INTERACTIONS");
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// Pending set replaces WAL re-emission
// ---------------------------------------------------------------------------

#[tokio::test]
async fn checkpoint_carries_unmatched_predictions() {
    let dir = "/tmp/banditdb_p05_pending";
    let db = setup(dir).await;

    // 10 rewarded (matched), 15 left in flight.
    for i in 0..10 {
        let (arm, iid) = db.predict("c", ctx(i)).expect("predict");
        db.reward(&iid, if arm == "A" { 1.0 } else { 0.0 }).await.expect("reward");
    }
    let mut in_flight = Vec::new();
    for i in 10..25 {
        let (_, iid) = db.predict("c", ctx(i)).expect("predict");
        in_flight.push(iid);
    }

    db.checkpoint().await.expect("checkpoint");
    let cp = read_checkpoint(dir);

    assert_eq!(
        cp.pending_interactions.len(), in_flight.len(),
        "every unmatched prediction must be carried in the checkpoint; a missing one \
         is a reward that can never be matched after rotation"
    );
    for iid in &in_flight {
        assert!(cp.pending_interactions.contains_key(iid), "in-flight {iid} not carried");
    }

    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn checkpoint_does_not_carry_matched_predictions() {
    let dir = "/tmp/banditdb_p05_matched";
    let db = setup(dir).await;

    for i in 0..20 {
        let (arm, iid) = db.predict("c", ctx(i)).expect("predict");
        db.reward(&iid, if arm == "A" { 1.0 } else { 0.0 }).await.expect("reward");
    }
    db.checkpoint().await.expect("checkpoint");

    assert!(
        read_checkpoint(dir).pending_interactions.is_empty(),
        "rewarded interactions are settled and must not be carried forward — doing so \
         would grow every future checkpoint without bound"
    );
    let _ = fs::remove_dir_all(dir);
}

/// The behaviour the pending set exists for: a reward arriving after a restart,
/// for a prediction made before the checkpoint, must still be matched.
#[tokio::test]
async fn late_reward_matches_across_checkpoint_and_restart() {
    let dir = "/tmp/banditdb_p05_late_reward";
    let db = setup(dir).await;

    let (_, iid) = db.predict("c", ctx(1)).expect("predict");
    db.checkpoint().await.expect("checkpoint");
    let rewards_before = db.campaign_report("c").unwrap().total_rewards;
    drop(db);

    let recovered = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    recovered.reward(&iid, 1.0).await.expect(
        "a reward for an interaction predicted before the checkpoint must still match \
         after restart — the pending set is what carries it across rotation",
    );

    assert_eq!(
        recovered.campaign_report("c").unwrap().total_rewards,
        rewards_before + 1,
        "late reward was accepted but not applied"
    );
    let _ = fs::remove_dir_all(dir);
}

/// Repeated checkpoints with a stable backlog must not grow the WAL: that
/// compounding rewrite is what the pending set removes.
#[tokio::test]
async fn repeated_checkpoints_do_not_rewrite_the_backlog() {
    let dir = "/tmp/banditdb_p05_no_amplification";
    let db = setup(dir).await;

    for i in 0..40 {
        db.predict("c", ctx(i)).expect("predict");
    }

    let mut sizes = Vec::new();
    for _ in 0..4 {
        db.checkpoint().await.expect("checkpoint");
        sizes.push(fs::metadata(format!("{dir}/wal.jsonl")).map(|m| m.len()).unwrap_or(0));
    }

    assert!(
        sizes.iter().all(|&s| s < 4_096),
        "WAL grew across checkpoints with no new traffic ({sizes:?}) — the unmatched \
         backlog is being rewritten each time"
    );
    let _ = fs::remove_dir_all(dir);
}
