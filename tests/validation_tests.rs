//! P0.6 acceptance tests — every write path validates its inputs.
//!
//! Two defects (docs/PRODUCTION_STAGE1.md P0.6):
//!
//!   1. Only `context.len()` was checked. Values were not, and finiteness alone is
//!      insufficient: `1e200` is finite, but `update()` squares it while forming
//!      `x·A⁻¹·x`, which overflows to infinity, and `inf / inf` yields NaN. That NaN
//!      lands in `a_inv` and `theta`, makes every later score NaN, and is written to
//!      the checkpoint — so it survives restart and the campaign is permanently dead.
//!   2. `/campaign/:id/interact` validated only the IDs before calling the engine,
//!      so it bypassed the length and reward-range checks the other routes applied.
//!
//! Validation now lives in the engine, which is the boundary every caller shares —
//! HTTP handlers, the SDK, and embedded users alike.

use banditdb::BanditDB;
use banditdb::state::{Algorithm, NeuralLinUCBConfig};
use std::fs;

async fn setup(dir: &str) -> BanditDB {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    db.add_campaign("c", vec!["A".into(), "B".into()], 2, 1.0, Algorithm::Linucb, None, None).await
        .unwrap();
    db
}

/// Values that must never reach the matrix math, with the reason each is dangerous.
fn hostile_values() -> Vec<(&'static str, f64)> {
    vec![
        ("NaN",            f64::NAN),
        ("+inf",           f64::INFINITY),
        ("-inf",           f64::NEG_INFINITY),
        ("1e200",          1e200),          // finite, but squares to inf
        ("-1e200",         -1e200),
        ("f64::MAX",       f64::MAX),
        ("1e300",          1e300),
    ]
}

fn theta_is_finite(db: &BanditDB, campaign: &str) -> bool {
    let campaigns = db.campaigns.read();
    let c = campaigns.get(campaign).unwrap();
    let arms = c.arms.read();
    arms.values().all(|a| {
        a.theta.iter().all(|v| v.is_finite())
            && a.a_inv.iter().all(|v| v.is_finite())
            && a.b.iter().all(|v| v.is_finite())
    })
}

#[tokio::test]
async fn predict_rejects_hostile_context_values() {
    let dir = "/tmp/banditdb_p06_predict";
    let db = setup(dir).await;

    for (label, v) in hostile_values() {
        for pos in 0..2 {
            let mut ctx = vec![0.5, 0.5];
            ctx[pos] = v;
            assert!(
                db.predict("c", ctx).is_err(),
                "predict accepted {label} at index {pos}"
            );
        }
    }

    assert!(
        theta_is_finite(&db, "c"),
        "a rejected context still reached the arm matrices — once NaN is in a_inv it \
         is persisted to the checkpoint and survives restart"
    );
    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn interact_rejects_hostile_context_values() {
    let dir = "/tmp/banditdb_p06_interact_ctx";
    let db = setup(dir).await;

    for (label, v) in hostile_values() {
        assert!(
            db.interact("c", "A", vec![v, 0.5], 1.0).await.is_err(),
            "interact accepted {label} — this route bypassed validation entirely"
        );
    }

    assert!(theta_is_finite(&db, "c"), "interact corrupted the arm matrices");
    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn interact_enforces_reward_range_and_campaign_existence() {
    let dir = "/tmp/banditdb_p06_interact_reward";
    let db = setup(dir).await;

    for bad in [f64::NAN, f64::INFINITY, 5.0, -0.5, 1.000_001] {
        assert!(
            db.interact("c", "A", vec![0.5, 0.5], bad).await.is_err(),
            "interact accepted out-of-contract reward {bad}"
        );
    }
    assert!(
        db.interact("missing", "A", vec![0.5, 0.5], 1.0).await.is_err(),
        "interact accepted a reward for a campaign that does not exist"
    );

    // The valid case still works.
    assert!(db.interact("c", "A", vec![0.5, 0.5], 1.0).await.is_ok());
    assert!(theta_is_finite(&db, "c"));
    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn context_dimension_limits_are_enforced() {
    let dir = "/tmp/banditdb_p06_dims";
    let db = setup(dir).await;

    assert!(db.predict("c", vec![]).is_err(), "empty context must be rejected");
    assert!(
        db.predict("c", vec![0.1; db.max_feature_dim + 1]).is_err(),
        "context longer than max_feature_dim must be rejected before allocation"
    );
    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn campaign_creation_rejects_degenerate_parameters() {
    let dir = "/tmp/banditdb_p06_campaign";
    let db = setup(dir).await;

    for bad_alpha in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(
            db.add_campaign("x", vec!["A".into()], 2, bad_alpha, Algorithm::Linucb, None, None).await.is_err(),
            "alpha {bad_alpha} must be rejected: it makes every score NaN or inverts exploration"
        );
    }

    for bad_hl in [Some(0.0), Some(-5.0), Some(f64::NAN)] {
        assert!(
            db.add_campaign("y", vec!["A".into()], 2, 1.0, Algorithm::Linucb, None, bad_hl).await.is_err(),
            "decay half-life {bad_hl:?} must be rejected"
        );
    }

    assert!(
        db.add_campaign("ok", vec!["A".into()], 2, 0.0, Algorithm::Linucb, None, None).await.is_ok(),
        "alpha = 0 is valid (pure exploitation) and must still be accepted"
    );
    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn neural_config_rejects_degenerate_dimensions() {
    let dir = "/tmp/banditdb_p06_neural_cfg";
    let db = setup(dir).await;

    let base = NeuralLinUCBConfig {
        context_dim: 4, embed_dim: 8, hidden_dim: 16, hidden_layers: 2,
        retrain_every: 10, retrain_steps: 5, learning_rate: 1e-3, lambda: 1.0,
    };

    let mut zero_ctx = base.clone();          zero_ctx.context_dim = 0;
    let mut zero_embed = base.clone();        zero_embed.embed_dim = 0;
    let mut zero_hidden = base.clone();       zero_hidden.hidden_dim = 0;
    let mut zero_layers = base.clone();       zero_layers.hidden_layers = 0;
    let mut bad_lr = base.clone();            bad_lr.learning_rate = f64::NAN;
    let mut neg_lr = base.clone();            neg_lr.learning_rate = -1.0;
    let mut bad_lambda = base.clone();        bad_lambda.lambda = f64::INFINITY;

    for (label, cfg) in [
        ("context_dim=0", zero_ctx), ("embed_dim=0", zero_embed),
        ("hidden_dim=0", zero_hidden), ("hidden_layers=0", zero_layers),
        ("learning_rate=NaN", bad_lr), ("learning_rate<0", neg_lr),
        ("lambda=inf", bad_lambda),
    ] {
        assert!(
            db.add_campaign("n", vec!["A".into()], 8, 1.0,
                            Algorithm::NeuralLinUCB(cfg), None, None).await.is_err(),
            "neural config with {label} must be rejected — zero dimensions build \
             degenerate matrices whose dot products panic on a length mismatch"
        );
    }

    assert!(
        db.add_campaign("n_ok", vec!["A".into()], 8, 1.0,
                        Algorithm::NeuralLinUCB(base), None, None).await.is_ok(),
        "a valid neural config must still be accepted"
    );
    let _ = fs::remove_dir_all(dir);
}

/// Sweep a wide range of magnitudes: everything accepted must leave the matrices
/// finite, and everything rejected must leave them untouched.
#[tokio::test]
async fn magnitude_sweep_never_corrupts_arm_state() {
    let dir = "/tmp/banditdb_p06_sweep";
    let db = setup(dir).await;

    // Range deliberately spans the overflow threshold: f64 squaring only overflows
    // above ~1e154, so a sweep stopping at 1e30 never reaches the dangerous region.
    //
    // Each magnitude gets its own campaign so it meets a clean identity A⁻¹. Reusing
    // one campaign hides the bug — accumulated updates shrink A⁻¹, so by the time the
    // sweep reaches large values the products stay finite by accident.
    let mut accepted = 0;
    for exp in -160i32..=160 {
        let v = 10f64.powi(exp);
        for (sign, candidate) in [("p", v), ("n", -v)] {
            let name = format!("s{sign}{exp}");
            db.add_campaign(&name, vec!["A".into(), "B".into()], 2, 1.0,
                            Algorithm::Linucb, None, None).await.unwrap();

            if let Ok((arm, iid)) = db.predict(&name, vec![candidate, 0.5]) {
                accepted += 1;
                db.reward(&iid, if arm == "A" { 1.0 } else { 0.0 }).await.expect("reward");
            }
            assert!(
                theta_is_finite(&db, &name),
                "arm state went non-finite after an accepted context value of {candidate:e} — \
                 the validator let through a magnitude that overflows the rank-one update"
            );
        }
    }

    assert!(accepted > 0, "the sweep rejected everything; it is not exercising the accept path");
    let _ = fs::remove_dir_all(dir);
}
