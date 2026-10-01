//! `interact` and `predict` must reject a request the campaign cannot apply,
//! before anything reaches the WAL.
//!
//! `interact` used to check context values but not the campaign's dimension or
//! whether the arm exists. A wrong-length context was logged and then panicked in
//! the rank-one update; recovery replayed the same records and panicked again, so
//! the process could not restart. An unknown arm was logged and reported success
//! without training anything.

use banditdb::engine::WalMessage;
use banditdb::state::{Algorithm, EngineError, NeuralLinUCBConfig};
use banditdb::BanditDB;
use std::fs;
use std::sync::atomic::Ordering;

async fn setup(dir: &str) -> BanditDB {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    db.add_campaign(
        "c",
        vec!["A".into(), "B".into()],
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
    )
    .await
    .unwrap();
    db
}

/// Flush the WAL and return its contents.
async fn wal_contents(db: &BanditDB, dir: &str) -> String {
    let (tx, rx) = tokio::sync::oneshot::channel();
    db.event_tx
        .send(WalMessage::Checkpoint { reply: tx })
        .await
        .unwrap();
    rx.await.unwrap();
    fs::read_to_string(format!("{dir}/wal.jsonl")).unwrap()
}

fn reward_count(db: &BanditDB, campaign: &str) -> u64 {
    let campaigns = db.campaigns.read();
    let arms = campaigns.get(campaign).unwrap().arms.read();
    arms.values()
        .map(|a| a.reward_count.load(Ordering::Relaxed))
        .sum()
}

#[tokio::test]
async fn interact_rejects_wrong_context_dimension_before_logging() {
    let dir = "/tmp/banditdb_interact_wrong_dim";
    let db = setup(dir).await;

    for ctx in [vec![1.0], vec![1.0, 2.0, 3.0]] {
        let len = ctx.len();
        let res = db.interact("c", "A", ctx, 1.0).await;
        assert!(
            matches!(res, Err(EngineError::BadRequest(_))),
            "context of length {len} on a dim-2 campaign must be rejected, got {res:?}"
        );
    }

    let wal = wal_contents(&db, dir).await;
    assert!(
        !wal.contains("Predicted") && !wal.contains("Rewarded"),
        "a rejected interact must not reach the WAL:\n{wal}"
    );
    assert_eq!(db.rewarded_count.load(Ordering::Relaxed), 0);

    // The valid shape still works.
    db.interact("c", "A", vec![0.5, 0.5], 1.0).await.unwrap();
    assert_eq!(reward_count(&db, "c"), 1);
    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn interact_rejects_unknown_arm_before_logging() {
    let dir = "/tmp/banditdb_interact_unknown_arm";
    let db = setup(dir).await;

    let res = db.interact("c", "ghost", vec![0.5, 0.5], 1.0).await;
    assert!(
        matches!(res, Err(EngineError::NotFound(_))),
        "an arm the campaign does not have must be rejected, got {res:?}"
    );

    let wal = wal_contents(&db, dir).await;
    assert!(
        !wal.contains("ghost"),
        "a rejected interact must not reach the WAL:\n{wal}"
    );
    assert_eq!(db.rewarded_count.load(Ordering::Relaxed), 0);
    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn predict_rejects_wrong_context_dimension() {
    let dir = "/tmp/banditdb_predict_wrong_dim";
    let db = setup(dir).await;

    let res = db.predict("c", vec![1.0, 2.0, 3.0]);
    assert!(
        matches!(res, Err(EngineError::BadRequest(_))),
        "context of length 3 on a dim-2 campaign must be rejected, got {res:?}"
    );
    let _ = fs::remove_dir_all(dir);
}

/// Neural arms live in the embedding space, so the expected context length is the
/// network's input dimension — not the arm dimension.
#[tokio::test]
async fn neural_interact_checks_input_dimension_not_arm_dimension() {
    let dir = "/tmp/banditdb_interact_neural_dim";
    let db = setup(dir).await;
    let cfg = NeuralLinUCBConfig {
        context_dim: 4,
        embed_dim: 8,
        hidden_dim: 16,
        hidden_layers: 2,
        retrain_every: 1000,
        retrain_steps: 5,
        learning_rate: 1e-3,
        lambda: 1.0,
    };
    db.add_campaign(
        "n",
        vec!["A".into()],
        cfg.embed_dim,
        1.0,
        Algorithm::NeuralLinUCB(cfg),
        None,
        None,
    )
    .await
    .unwrap();

    let res = db.interact("n", "A", vec![0.1; 8], 1.0).await;
    assert!(
        matches!(res, Err(EngineError::BadRequest(_))),
        "an embedding-length context must be rejected on a context_dim=4 campaign, got {res:?}"
    );
    let res = db.predict("n", vec![0.1; 8]);
    assert!(
        matches!(res, Err(EngineError::BadRequest(_))),
        "predict must apply the same rule, got {res:?}"
    );
    let _ = fs::remove_dir_all(dir);
}

/// A data dir whose WAL already holds a wrong-dimension reward must still open.
/// The bad record is skipped; everything else is recovered.
#[tokio::test]
async fn recovery_skips_reward_with_wrong_dimension() {
    let dir = "/tmp/banditdb_recover_wrong_dim";
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let wal = format!("{dir}/wal.jsonl");
    fs::write(&wal, concat!(
        r#"{"CampaignCreated":{"campaign_id":"c","arms":["A"],"feature_dim":2,"alpha":1.0,"algorithm":"linucb"}}"#, "\n",
        r#"{"Predicted":{"interaction_id":"bad","campaign_id":"c","arm_id":"A","context":[1.0,2.0,3.0],"timestamp_secs":1,"arm_propensities":null,"is_reemit":false}}"#, "\n",
        r#"{"Rewarded":{"interaction_id":"bad","reward":1.0,"timestamp_secs":1}}"#, "\n",
        r#"{"Predicted":{"interaction_id":"good","campaign_id":"c","arm_id":"A","context":[1.0,2.0],"timestamp_secs":1,"arm_propensities":null,"is_reemit":false}}"#, "\n",
        r#"{"Rewarded":{"interaction_id":"good","reward":1.0,"timestamp_secs":1}}"#, "\n",
    )).unwrap();

    let db = BanditDB::new(&wal, dir);
    assert_eq!(
        reward_count(&db, "c"),
        1,
        "only the well-formed reward should be applied"
    );
    let _ = fs::remove_dir_all(dir);
}
