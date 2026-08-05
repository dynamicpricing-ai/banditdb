#![cfg(feature = "neural")]
//! Covers the retrain path once it is driven by the background worker rather than
//! by `checkpoint()`. These exercise the public entry points the worker calls.

use banditdb::BanditDB;
use banditdb::state::{Algorithm, NeuralLinUCBConfig};
use std::fs;

fn neural_cfg(retrain_every: usize) -> NeuralLinUCBConfig {
    NeuralLinUCBConfig {
        context_dim:   2,
        embed_dim:     8,
        hidden_dim:    32,
        hidden_layers: 2,
        retrain_every,
        retrain_steps: 5,
        learning_rate: 1e-3,
        lambda:        1.0,
    }
}

/// Drive `n` predict/reward round trips through the campaign.
async fn drive(db: &BanditDB, campaign: &str, n: usize) {
    for i in 0..n {
        let ctx = vec![(i % 7) as f64 / 7.0, (i % 3) as f64 / 3.0];
        if let Ok((arm, iid)) = db.predict(campaign, ctx) {
            let reward = if arm == "A" { 1.0 } else { 0.0 };
            let _ = db.reward(&iid, reward).await;
        }
    }
}

fn setup(dir: &str, retrain_every: usize) -> BanditDB {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    let cfg = neural_cfg(retrain_every);
    db.add_campaign(
        "c",
        vec!["A".to_string(), "B".to_string()],
        // Neural arm matrices live in embedding space. The HTTP handler substitutes
        // cfg.embed_dim here (src/main.rs); the engine API takes the value literally.
        cfg.embed_dim,
        1.0,
        Algorithm::NeuralLinUCB(cfg),
        None,
        None,
    )
    .unwrap();
    db
}

#[tokio::test]
async fn campaign_becomes_due_then_clears_after_retrain() {
    let dir = "/tmp/banditdb_test_retrain_due";
    let db = setup(dir, 20);

    assert!(
        db.campaigns_due_for_retrain().is_empty(),
        "a fresh campaign has no accumulated rewards and must not be due"
    );

    drive(&db, "c", 40).await;
    assert_eq!(
        db.campaigns_due_for_retrain(),
        vec!["c".to_string()],
        "40 rewards past a retrain_every of 20 must mark the campaign due"
    );

    assert!(db.retrain_campaign("c"), "retrain should report success");
    assert!(
        db.campaigns_due_for_retrain().is_empty(),
        "retrain resets reward_count, so the campaign must no longer be due"
    );

    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn retrain_persists_weights_and_keeps_serving() {
    let dir = "/tmp/banditdb_test_retrain_weights";
    let db = setup(dir, 20);
    drive(&db, "c", 40).await;

    assert!(db.retrain_campaign("c"));
    assert!(
        std::path::Path::new(&format!("{dir}/neural/c.safetensors")).exists(),
        "retrain must persist MLP weights so recovery can restore them"
    );

    // Arm matrices are rebuilt in the new embedding space; predict must still work.
    let (arm, _) = db.predict("c", vec![0.5, 0.5]).expect("predict after retrain");
    assert!(arm == "A" || arm == "B");

    let _ = fs::remove_dir_all(dir);
}

/// The worker path must publish its retrained weights, not just train in place.
/// `retrain_campaign` and `checkpoint()` share `retrain_campaign_locked`, and the
/// publish lives there — without it the background worker would train forever while
/// predictions kept serving the initial snapshot.
#[tokio::test]
async fn worker_retrain_publishes_weights_to_prediction_path() {
    let dir = "/tmp/banditdb_test_retrain_publishes";
    let db = setup(dir, 20);
    let probe = ndarray::Array1::from_vec(vec![0.3, 0.7]);

    let before = {
        let campaigns = db.campaigns.read();
        let campaign = campaigns.get("c").unwrap();
        let w = std::sync::Arc::clone(&campaign.neural_weights.as_ref().unwrap().read());
        w.embed(&probe)
    };

    drive(&db, "c", 40).await;
    assert!(db.retrain_campaign("c"), "retrain should run");

    let after = {
        let campaigns = db.campaigns.read();
        let campaign = campaigns.get("c").unwrap();
        let w = std::sync::Arc::clone(&campaign.neural_weights.as_ref().unwrap().read());
        w.embed(&probe)
    };

    assert_ne!(
        before.to_vec(), after.to_vec(),
        "worker-driven retrain did not republish weights — predictions would still \
         be served by the pre-training snapshot"
    );

    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn retrain_campaign_is_safe_on_unknown_and_non_neural_campaigns() {
    let dir = "/tmp/banditdb_test_retrain_missing";
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    db.add_campaign("linear", vec!["A".into(), "B".into()], 2, 1.0, Algorithm::Linucb, None, None)
        .unwrap();

    assert!(!db.retrain_campaign("does_not_exist"));
    assert!(!db.retrain_campaign("linear"), "a LinUCB campaign has no MLP to retrain");
    assert!(db.campaigns_due_for_retrain().is_empty());

    let _ = fs::remove_dir_all(dir);
}

/// The buffer must retain more than the old 5,000-entry cap, and retrain must stay
/// usable once the buffer exceeds the minibatch size.
#[tokio::test]
async fn buffer_retains_beyond_legacy_cap() {
    std::env::set_var("BANDITDB_NEURAL_BUFFER_CAP", "8000");
    std::env::set_var("BANDITDB_NEURAL_BATCH_SIZE", "100");

    let dir = "/tmp/banditdb_test_retrain_buffer";
    let db = setup(dir, 100_000); // never auto-due; drive the retrain manually
    drive(&db, "c", 6_000).await;

    {
        let campaigns = db.campaigns.read();
        let campaign = campaigns.get("c").unwrap();
        let neural = campaign.neural.as_ref().unwrap().lock();
        assert!(
            neural.buffer.len() > 5_000,
            "buffer held {} entries; the old hard-coded cap of 5,000 should no longer apply",
            neural.buffer.len()
        );
    }

    // Buffer (6k) far exceeds batch_size (100) — retrain must sample, not choke.
    assert!(db.retrain_campaign("c"));

    std::env::remove_var("BANDITDB_NEURAL_BUFFER_CAP");
    std::env::remove_var("BANDITDB_NEURAL_BATCH_SIZE");
    let _ = fs::remove_dir_all(dir);
}
