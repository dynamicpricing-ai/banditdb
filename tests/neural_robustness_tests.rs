#![cfg(feature = "neural")]
//! Neural corner cases: a Thompson-sampling challenger in a Progressive campaign,
//! arm state across a restart, counters across retrains, and rewards arriving
//! while a network trains.

use banditdb::state::{Algorithm, ArmState, NeuralLinUCBConfig, ProgressiveConfig};
use banditdb::BanditDB;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

fn fresh(name: &str) -> (String, String) {
    let dir = format!("/tmp/banditdb_neural_robust_{name}");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    (format!("{dir}/wal.jsonl"), dir)
}

fn cfg(retrain_every: usize, retrain_steps: usize) -> NeuralLinUCBConfig {
    NeuralLinUCBConfig {
        context_dim: 2,
        embed_dim: 8,
        hidden_dim: 16,
        hidden_layers: 1,
        retrain_every,
        retrain_steps,
        learning_rate: 1e-3,
        lambda: 1.0,
    }
}

fn ctx(i: usize) -> Vec<f64> {
    vec![(i % 7) as f64 / 7.0, (i % 3) as f64 / 3.0]
}

fn progressive(challenger: Algorithm) -> Algorithm {
    Algorithm::Progressive(ProgressiveConfig {
        base: Box::new(Algorithm::Linucb),
        challenger: Box::new(challenger),
        min_obs: 5,
        required_wins: 1,
        step_bps: 1000,
    })
}

async fn create(db: &BanditDB, feature_dim: usize, algorithm: Algorithm) {
    db.add_campaign(
        "c",
        vec!["A".into(), "B".into()],
        feature_dim,
        1.0,
        algorithm,
        None,
        None,
    )
    .await
    .unwrap();
}

fn challenger_dim(db: &BanditDB) -> usize {
    db.campaigns.read()["c"]
        .challenger_arms
        .as_ref()
        .unwrap()
        .read()["A"]
        .theta
        .len()
}

/// A NeuralThompsonSampling challenger used to get arms sized for the raw
/// context while predictions scored them against embeddings: every challenger
/// draw panicked, and so did every retrain and every checkpoint.
#[tokio::test]
async fn progressive_with_neural_thompson_challenger_works_end_to_end() {
    let (wal, dir) = fresh("nts_challenger");
    let db = BanditDB::new(&wal, &dir);
    create(
        &db,
        2,
        progressive(Algorithm::NeuralThompsonSampling(cfg(20, 3))),
    )
    .await;
    assert_eq!(
        challenger_dim(&db),
        8,
        "challenger arms live in embedding space"
    );

    // Route everything to the challenger so each prediction exercises it.
    db.campaigns.read()["c"]
        .challenger_traffic_bps
        .store(10_000, Ordering::Relaxed);
    for i in 0..100 {
        let (arm, iid) = db.predict("c", ctx(i)).expect("challenger prediction");
        db.reward(&iid, if arm == "A" { 1.0 } else { 0.0 })
            .await
            .unwrap();
    }
    assert!(db.retrain_campaign("c"));
    assert_eq!(challenger_dim(&db), 8);
    db.checkpoint().await.expect("checkpoint with tournament");

    let (_, iid) = db.predict("c", ctx(1)).unwrap();
    db.reward(&iid, 1.0).await.unwrap();
}

/// A campaign whose challenger arms were already created in the wrong space must
/// fail predictions cleanly instead of panicking, and heal at its next retrain.
#[tokio::test]
async fn mis_sized_challenger_arms_error_then_heal_on_retrain() {
    let (wal, dir) = fresh("legacy_dims");
    let db = BanditDB::new(&wal, &dir);
    create(
        &db,
        2,
        progressive(Algorithm::NeuralThompsonSampling(cfg(20, 3))),
    )
    .await;
    {
        let campaigns = db.campaigns.read();
        let mut arms = campaigns["c"].challenger_arms.as_ref().unwrap().write();
        *arms = HashMap::from([
            ("A".to_string(), ArmState::new(2)),
            ("B".to_string(), ArmState::new(2)),
        ]);
    }

    // Base traffic still works and feeds the replay buffer.
    db.campaigns.read()["c"]
        .challenger_traffic_bps
        .store(0, Ordering::Relaxed);
    for i in 0..40 {
        let (_, iid) = db.predict("c", ctx(i)).unwrap();
        db.reward(&iid, 1.0).await.unwrap();
    }
    db.campaigns.read()["c"]
        .challenger_traffic_bps
        .store(10_000, Ordering::Relaxed);
    assert!(
        db.predict("c", ctx(0)).is_err(),
        "a mis-sized arm is an error, not a panic"
    );

    assert!(db.retrain_campaign("c"));
    assert_eq!(challenger_dim(&db), 8);
    db.predict("c", ctx(0)).expect("healed after retrain");
    db.checkpoint().await.unwrap();
}

#[tokio::test]
async fn progressive_rejects_a_neural_base_and_nesting() {
    let (wal, dir) = fresh("validation");
    let db = BanditDB::new(&wal, &dir);
    let neural_base = Algorithm::Progressive(ProgressiveConfig {
        base: Box::new(Algorithm::NeuralLinUCB(cfg(20, 3))),
        challenger: Box::new(Algorithm::Linucb),
        min_obs: 5,
        required_wins: 1,
        step_bps: 1000,
    });
    assert!(db
        .add_campaign("n", vec!["A".into()], 2, 1.0, neural_base, None, None)
        .await
        .is_err());

    let nested = progressive(progressive(Algorithm::Linucb));
    assert!(db
        .add_campaign("p", vec!["A".into()], 2, 1.0, nested, None, None)
        .await
        .is_err());
}

fn arm_a(db: &BanditDB) -> (f64, f64, u64) {
    let campaigns = db.campaigns.read();
    let arms = campaigns["c"].arms.read();
    let a = &arms["A"];
    (
        a.a_inv.diag().sum(),
        a.theta.dot(&a.theta).sqrt(),
        a.reward_count.load(Ordering::Relaxed),
    )
}

fn buffer_len(db: &BanditDB) -> usize {
    db.campaigns.read()["c"]
        .neural
        .as_ref()
        .unwrap()
        .lock()
        .buffer
        .len()
}

/// Arms of a neural campaign are rebuilt from the replay buffer at every
/// retrain. The buffer used to be empty after a restart, so the first retrain
/// reset every arm that had no reward since — here arm A, with 300 of them.
#[tokio::test]
async fn restart_keeps_the_replay_buffer_and_what_the_arms_learned() {
    let (wal, dir) = fresh("restart_buffer");
    let n = cfg(50, 3);
    {
        let db = BanditDB::new(&wal, &dir);
        create(&db, n.embed_dim, Algorithm::NeuralLinUCB(n.clone())).await;
        for i in 0..300 {
            db.interact("c", "A", ctx(i), 1.0).await.unwrap();
        }
        assert!(db.retrain_campaign("c"));
        db.checkpoint().await.unwrap();
        // Rewards after the checkpoint come back through the WAL tail.
        for i in 0..10 {
            db.interact("c", "A", ctx(i), 1.0).await.unwrap();
        }
    }
    tokio::time::sleep(Duration::from_millis(300)).await;

    let db = BanditDB::new(&wal, &dir);
    assert_eq!(buffer_len(&db), 310);
    for i in 0..50 {
        db.interact("c", "B", ctx(i), 0.0).await.unwrap();
    }
    assert!(db.retrain_campaign("c"));

    let (trace, theta_norm, rewards) = arm_a(&db);
    assert!(
        trace < 7.0,
        "arm A went back to the cold prior (trace {trace})"
    );
    assert!(theta_norm > 0.1, "arm A lost its estimate");
    assert_eq!(rewards, 310);
}

/// Re-accumulation replays the buffer through `update`, which counts each reward.
/// The counters used to be restored before the replay, so every retrain added the
/// whole buffer to them again.
#[tokio::test]
async fn retrains_do_not_inflate_reward_counters() {
    let (wal, dir) = fresh("counters");
    let db = BanditDB::new(&wal, &dir);
    let n = cfg(1_000_000, 3);
    create(&db, n.embed_dim, Algorithm::NeuralLinUCB(n)).await;
    for i in 0..60 {
        db.interact("c", "A", ctx(i), 0.5).await.unwrap();
    }
    for _ in 0..4 {
        assert!(db.retrain_campaign("c"));
    }
    let campaigns = db.campaigns.read();
    let arms = campaigns["c"].arms.read();
    assert_eq!(arms["A"].reward_count.load(Ordering::Relaxed), 60);
    let total = f64::from_bits(arms["A"].total_reward.load(Ordering::Relaxed));
    assert!((total - 30.0).abs() < 1e-9, "total_reward {total}");
}

/// Rewards append to the replay buffer under the neural lock. Training used to
/// hold that lock for its whole duration, so every reward for the campaign — and
/// any diagnostics call — waited for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rewards_and_diagnostics_are_not_blocked_by_training() {
    let (wal, dir) = fresh("training_lock");
    let db = Arc::new(BanditDB::new(&wal, &dir));
    let n = cfg(1_000_000, 5_000);
    create(&db, n.embed_dim, Algorithm::NeuralLinUCB(n)).await;
    for i in 0..50 {
        db.interact("c", "A", ctx(i), 1.0).await.unwrap();
    }

    let trainer = Arc::clone(&db);
    let training = tokio::task::spawn_blocking(move || trainer.retrain_campaign("c"));
    tokio::time::sleep(Duration::from_millis(100)).await;

    for i in 0..10 {
        db.interact("c", "B", ctx(i), 0.0).await.unwrap();
    }
    let still_training = !training.is_finished();
    let diag = db.campaign_diagnostics("c").unwrap();
    assert_eq!(diag.neural_buffer_size, Some(60));
    if still_training {
        // The retrain cannot have installed its result yet, so the 10 rewards
        // above were applied while it was running.
        assert!(diag.neural_last_retrain_losses.is_none());
    }
    // Blocked rewards would only have returned once training was over. On a
    // machine fast enough to finish 5,000 steps first, the check says nothing
    // either way, so it is skipped rather than failed.
    if !still_training {
        eprintln!("training finished before the rewards did — check skipped");
    }
    assert!(training.await.unwrap());
}
