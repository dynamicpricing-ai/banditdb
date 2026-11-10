//! Checkpoints must keep working, and stay readable, in the corner cases that
//! used to break them: mixed context lengths in one export, non-finite values in
//! the snapshot, and unbounded decay on arms that receive no rewards.

use banditdb::engine::CheckpointLoad;
use banditdb::state::{Algorithm, ArmState};
use banditdb::BanditDB;
use ndarray::Array1;
use std::sync::atomic::Ordering;

fn fresh(name: &str) -> (String, String) {
    let dir = format!("/tmp/banditdb_ckpt_robust_{name}");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    (format!("{dir}/wal.jsonl"), dir)
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

async fn create(db: &BanditDB, id: &str, dim: usize, half_life: Option<f64>) {
    db.add_campaign(
        id,
        vec!["a".into(), "b".into()],
        dim,
        1.0,
        Algorithm::Linucb,
        None,
        half_life,
    )
    .await
    .unwrap();
}

fn shards(dir: &str, campaign: &str) -> usize {
    std::fs::read_dir(format!("{dir}/exports"))
        .map(|d| {
            d.flatten()
                .filter(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    n.starts_with(&format!("{campaign}_")) && n.ends_with(".parquet")
                })
                .count()
        })
        .unwrap_or(0)
}

fn a_inv_max_diag(db: &BanditDB, campaign: &str, arm: &str) -> f64 {
    db.campaigns.read().get(campaign).unwrap().arms.read()[arm]
        .a_inv
        .diag()
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
}

/// A campaign id deleted and recreated with another feature_dim leaves two context
/// lengths in one WAL segment. The export used to index every row by the first
/// row's length and panic, failing this and every later checkpoint.
#[tokio::test]
async fn checkpoint_survives_mixed_context_lengths_in_one_segment() {
    let (wal, dir) = fresh("mixed_dims");
    let db = BanditDB::new(&wal, &dir);

    create(&db, "x", 5, None).await;
    for _ in 0..30 {
        let (_, iid) = db.predict("x", vec![0.1; 5]).unwrap();
        db.reward(&iid, 1.0).await.unwrap();
    }
    db.delete_campaign("x").await.unwrap();
    create(&db, "x", 2, None).await;
    for _ in 0..3 {
        let (_, iid) = db.predict("x", vec![0.1; 2]).unwrap();
        db.reward(&iid, 1.0).await.unwrap();
    }

    let msg = db.checkpoint().await.expect("checkpoint must succeed");
    assert!(msg.contains("33 interactions exported"), "{msg}");
    assert_eq!(shards(&dir, "x"), 2, "one shard per context length");
    assert!(std::path::Path::new(&format!("{dir}/checkpoint.json")).exists());
    db.checkpoint().await.expect("the next checkpoint must succeed too");
}

/// An arm that gets no rewards is widened by decay at every checkpoint. Uncapped,
/// A⁻¹ passed the cold prior after one checkpoint and overflowed to infinity after
/// ~155, leaving θ = NaN and a checkpoint.json that could not be read back.
#[tokio::test]
async fn decay_never_widens_an_arm_past_the_cold_prior() {
    let (wal, dir) = fresh("decay_cap");
    let db = BanditDB::new(&wal, &dir);
    create(&db, "d", 2, Some(1.0)).await;

    for _ in 0..200 {
        db.last_checkpoint_secs
            .store(now() - 36_000, Ordering::Relaxed);
        db.checkpoint().await.unwrap();
    }
    assert!((a_inv_max_diag(&db, "d", "a") - 1.0).abs() < 1e-9);

    drop(db);
    assert!(matches!(
        BanditDB::load_checkpoint(&dir),
        CheckpointLoad::Loaded(_)
    ));
    let cp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/checkpoint.json")).unwrap())
            .unwrap();
    let stored = cp["campaigns"]["d"]["arms"]["a"]["a_inv"]["data"][0]
        .as_f64()
        .expect("a_inv must be stored as a number, not null");
    assert!((stored - 1.0).abs() < 1e-9);
}

/// D-LinUCB forgetting: a learned direction widens by ~1/γ, a never-seen
/// direction stays at the prior and does not hold the learned one back, θ moves
/// only slightly, and many half-lives end at the cold prior — not beyond it.
#[tokio::test]
async fn decay_forgets_toward_the_prior_per_direction() {
    let (wal, dir) = fresh("decay_trained");
    let db = BanditDB::new(&wal, &dir);
    create(&db, "d", 2, Some(1.0)).await;
    // Only feature 0 is ever observed; feature 1 stays at the prior.
    for _ in 0..50 {
        db.interact("d", "a", vec![1.0, 0.0], 1.0).await.unwrap();
    }
    let arm = |db: &BanditDB| db.campaigns.read().get("d").unwrap().arms.read()["a"].clone();
    let before = arm(&db);
    assert!(before.a_inv[[0, 0]] < 0.05);
    assert!((before.a_inv[[1, 1]] - 1.0).abs() < 1e-9);

    // One half-life (γ = 0.5).
    db.last_checkpoint_secs
        .store(now() - 3_600, Ordering::Relaxed);
    db.checkpoint().await.unwrap();
    let after = arm(&db);
    let ratio = after.a_inv[[0, 0]] / before.a_inv[[0, 0]];
    assert!((1.9..=2.0).contains(&ratio), "learned direction: x{ratio}");
    assert!((after.a_inv[[1, 1]] - 1.0).abs() < 1e-9, "unseen direction");
    let shrink = after.theta[0] / before.theta[0];
    assert!((0.95..=1.0).contains(&shrink), "theta moved by x{shrink}");

    // Many half-lives: everything is forgotten, and no further.
    for _ in 0..20 {
        db.last_checkpoint_secs
            .store(now() - 36_000, Ordering::Relaxed);
        db.checkpoint().await.unwrap();
    }
    let end = arm(&db);
    assert!((end.a_inv[[0, 0]] - 1.0).abs() < 1e-6);
    assert!(end.a_inv.iter().all(|v| *v <= 1.0 + 1e-9));
    assert!(end.theta[0].abs() < 1e-6);
}

/// Arms inflated past the prior by the old unbounded decay are pulled back to it.
#[test]
fn decay_pulls_an_over_inflated_arm_back_to_the_prior() {
    let mut arm = ArmState::new(3);
    arm.a_inv *= 1e12;
    for _ in 0..40 {
        arm.decay(0.5);
    }
    for i in 0..3 {
        assert!((arm.a_inv[[i, i]] - 1.0).abs() < 1e-6, "{}", arm.a_inv[[i, i]]);
    }
}

/// The inverse-space form must equal the definition A' = γA + (1−γ)I, b' = γb.
#[test]
fn decay_matches_the_d_linucb_definition() {
    let mut arm = ArmState::new(3);
    for (x, r) in [
        ([1.0, 0.2, -0.5], 1.0),
        ([0.3, 0.9, 0.1], 0.0),
        ([-0.4, 0.5, 0.8], 1.0),
        ([0.7, -0.6, 0.2], 0.5),
    ] {
        arm.update(&Array1::from_vec(x.to_vec()), r);
    }
    let gamma = 0.7;
    let a_inv_before = arm.a_inv.clone();
    let b_before = arm.b.clone();
    arm.decay(gamma);

    // A' · A'⁻¹ = I  with  A' = γA + (1−γ)I  ⇔  (γI + (1−γ)A⁻¹) · A'⁻¹ = A⁻¹.
    let m = &a_inv_before * (1.0 - gamma) + ndarray::Array2::<f64>::eye(3) * gamma;
    let lhs = m.dot(&arm.a_inv);
    for (x, y) in lhs.iter().zip(a_inv_before.iter()) {
        assert!((x - y).abs() < 1e-12, "{x} vs {y}");
    }
    for (x, y) in arm.b.iter().zip(b_before.iter()) {
        assert!((x - gamma * y).abs() < 1e-12);
    }
    for (x, y) in arm.theta.iter().zip(arm.a_inv.dot(&arm.b).iter()) {
        assert!((x - y).abs() < 1e-12);
    }
}

/// A checkpoint that fails after applying decay must not apply the same interval
/// again when it is retried.
#[tokio::test]
async fn failed_checkpoint_does_not_reapply_decay() {
    let (wal, dir) = fresh("decay_retry");
    let db = BanditDB::new(&wal, &dir);
    create(&db, "d", 2, Some(1.0)).await;
    for _ in 0..200 {
        db.interact("d", "a", vec![1.0, 0.0], 1.0).await.unwrap();
        db.interact("d", "a", vec![0.0, 1.0], 1.0).await.unwrap();
    }
    let before = a_inv_max_diag(&db, "d", "a");

    // Make the checkpoint fail after the decay step: `exports` cannot be created.
    std::fs::write(format!("{dir}/exports"), b"not a directory").unwrap();
    db.last_checkpoint_secs
        .store(now() - 3_600, Ordering::Relaxed);
    assert!(db.checkpoint().await.is_err());
    let after_first = a_inv_max_diag(&db, "d", "a");
    assert!((after_first / before - 2.0).abs() < 0.02);

    for _ in 0..3 {
        assert!(db.checkpoint().await.is_err());
    }
    let after_retries = a_inv_max_diag(&db, "d", "a");
    assert!(
        (after_retries / after_first - 1.0).abs() < 0.01,
        "retries re-applied decay: {after_first} -> {after_retries}"
    );
}

/// alpha large enough to make scores infinite gave NaN propensities, which went
/// into the pending-interaction set of the checkpoint as `null`.
#[tokio::test]
async fn oversized_alpha_is_rejected() {
    let (wal, dir) = fresh("alpha");
    let db = BanditDB::new(&wal, &dir);
    let r = db
        .add_campaign(
            "h",
            vec!["a".into(), "b".into()],
            2,
            1e308,
            Algorithm::Linucb,
            None,
            None,
        )
        .await;
    assert!(r.is_err());
    assert!(db.campaigns.read().get("h").is_none());
}

/// A campaign that already carries an extreme alpha (created before the bound, or
/// replayed from an old WAL) must still produce finite propensities and a
/// checkpoint that reads back.
#[tokio::test]
async fn infinite_scores_do_not_reach_the_checkpoint() {
    let (wal, dir) = fresh("inf_scores");
    {
        // Seed the WAL directly: replay bypasses creation-time validation.
        let event = serde_json::json!({"CampaignCreated": {
            "campaign_id": "h", "arms": ["a", "b"], "feature_dim": 2, "alpha": 1e308
        }});
        std::fs::write(&wal, format!("{event}\n")).unwrap();
    }
    let db = BanditDB::new(&wal, &dir);
    let (_, iid) = db.predict("h", vec![10.0, 10.0]).unwrap();
    let props = db.interactions.get(&iid).unwrap().arm_propensities.unwrap();
    assert!(props.values().all(|p| p.is_finite()), "{props:?}");
    assert!((props.values().sum::<f64>() - 1.0).abs() < 1e-9);

    db.checkpoint().await.unwrap();
    drop(db);
    match BanditDB::load_checkpoint(&dir) {
        CheckpointLoad::Loaded(cp) => assert_eq!(cp.pending_interactions.len(), 1),
        other => panic!("checkpoint must read back, got {other:?}"),
    }
}

/// If NaN does reach an arm, the checkpoint refuses to write rather than replace a
/// readable checkpoint.json with one the next start cannot parse.
#[tokio::test]
async fn checkpoint_refuses_to_write_non_finite_arm_state() {
    let (wal, dir) = fresh("nan_arm");
    let db = BanditDB::new(&wal, &dir);
    create(&db, "good", 2, None).await;
    create(&db, "bad", 2, None).await;
    db.checkpoint().await.unwrap();

    db.campaigns.read().get("bad").unwrap().arms.write().get_mut("a").unwrap().theta[0] = f64::NAN;
    let err = db.checkpoint().await.unwrap_err();
    assert!(err.contains("'bad'") && err.contains("'a'"), "{err}");

    // The checkpoint on disk is still the readable one.
    assert!(matches!(
        BanditDB::load_checkpoint(&dir),
        CheckpointLoad::Loaded(_)
    ));

    // Removing the poisoned campaign unblocks checkpointing.
    db.delete_campaign("bad").await.unwrap();
    db.checkpoint().await.unwrap();
}

/// A rank-one update that would produce NaN is skipped, leaving the arm as it was.
#[test]
fn update_never_commits_non_finite_state() {
    let mut arm = ArmState::new(2);
    arm.a_inv *= 1e200;
    let before = arm.a_inv.clone();
    arm.update(&Array1::from_vec(vec![1000.0, 1000.0]), 1.0);
    assert!(arm.a_inv.iter().all(|v| v.is_finite()));
    assert!(arm.theta.iter().all(|v| v.is_finite()));
    assert_eq!(arm.a_inv, before);
}
