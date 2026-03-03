use banditdb::state::Algorithm;
use banditdb::BanditDB;

/// Test 1.2 — Asymptotic Convergence to Known Theta
///
/// We define a ground-truth weight vector (true_theta) and train the engine with
/// noiseless synthetic rewards computed directly from it. After 500 steps the
/// learned theta must converge within epsilon=0.1. Uses deterministic sin/cos
/// contexts to span the unit circle without requiring a rand dependency.
#[tokio::test]
async fn test_1_2_asymptotic_convergence() {
    let wal = "/tmp/banditdb_test_convergence.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    db.add_campaign("convergence", vec!["arm".to_string()], 2, 1.0, Algorithm::Linucb);

    let true_theta = [3.0_f64, -2.0_f64];

    // Deterministic contexts tracing the unit circle (sin/cos at 0.1-radian steps).
    // Dense, balanced coverage of all directions with no rand crate needed.
    for i in 0..500_usize {
        let angle = i as f64 * 0.1;
        let ctx = vec![angle.sin(), angle.cos()];
        let reward = true_theta[0] * ctx[0] + true_theta[1] * ctx[1];

        if let Some((_, iid)) = db.predict("convergence", ctx) {
            db.reward(&iid, reward);
        }
    }

    let campaigns = db.campaigns.read();
    let campaign = campaigns.get("convergence").unwrap();
    let arms = campaign.arms.read();
    let theta = &arms.get("arm").unwrap().theta;

    let error = ((theta[0] - true_theta[0]).powi(2) + (theta[1] - true_theta[1]).powi(2)).sqrt();

    assert!(
        error < 0.1,
        "Theta failed to converge after 500 steps. error={:.4}, learned=[{:.3}, {:.3}], true={:?}",
        error,
        theta[0],
        theta[1],
        true_theta
    );

    let _ = std::fs::remove_file(wal);
}

/// Test 1.4 — Wrong Feature Dim Returns None, Not a Panic
///
/// Without dimension validation, ndarray panics inside dot() when context.len()
/// doesn't match theta.len(). The engine must reject the request and return None
/// before any math is attempted. Three sub-cases: over-sized, under-sized, empty.
#[tokio::test]
async fn test_1_4_wrong_feature_dim_no_panic() {
    let wal = "/tmp/banditdb_test_dim_fuzz.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    db.add_campaign("dim_test", vec!["a".to_string(), "b".to_string()], 2, 1.0, Algorithm::Linucb);

    // Baseline: correct dim must succeed.
    assert!(db.predict("dim_test", vec![1.0, 0.0]).is_some());

    // Over-sized context: 3 features instead of 2.
    assert!(
        db.predict("dim_test", vec![1.0, 0.0, 0.0]).is_none(),
        "Over-sized context should return None"
    );

    // Under-sized context: 1 feature instead of 2.
    assert!(
        db.predict("dim_test", vec![1.0]).is_none(),
        "Under-sized context should return None"
    );

    // Edge case: empty context vector.
    assert!(
        db.predict("dim_test", vec![]).is_none(),
        "Empty context should return None"
    );

    let _ = std::fs::remove_file(wal);
}

/// Test V.1 — Duplicate Campaign Creation Is Rejected
///
/// Calling add_campaign twice with the same campaign_id must return false on the
/// second call and leave the existing matrices untouched. Without this guard a
/// duplicate create silently resets all learned weights to zero.
#[tokio::test]
async fn test_v1_duplicate_campaign_rejected() {
    let wal = "/tmp/banditdb_test_v1.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");

    // First create must succeed
    assert!(db.add_campaign("dup_test", vec!["arm_a".to_string()], 2, 1.0, Algorithm::Linucb));

    // Train it so theta is non-zero
    let (_, iid) = db.predict("dup_test", vec![1.0, 0.0]).unwrap();
    db.reward(&iid, 1.0);

    let theta_before = {
        let c = db.campaigns.read();
        let campaign = c.get("dup_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm_a").unwrap().theta.clone();
        x
    };

    // Second create with same id must be rejected
    assert!(
        !db.add_campaign("dup_test", vec!["arm_a".to_string()], 2, 1.0, Algorithm::Linucb),
        "Duplicate campaign creation must return false"
    );

    // Matrices must be completely unchanged
    let theta_after = {
        let c = db.campaigns.read();
        let campaign = c.get("dup_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm_a").unwrap().theta.clone();
        x
    };

    assert_eq!(
        theta_before, theta_after,
        "Duplicate campaign creation must not reset learned weights"
    );

    let _ = std::fs::remove_file(wal);
}

/// Test V.2 — Double-Reward Is Rejected
///
/// The same interaction_id must only update the model once. After the first
/// reward the interaction is removed from the TTL cache. A second call with
/// the same id must return false and leave theta unchanged.
#[tokio::test]
async fn test_v2_double_reward_rejected() {
    let wal = "/tmp/banditdb_test_v2.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    db.add_campaign("double_reward_test", vec!["arm".to_string()], 2, 1.0, Algorithm::Linucb);

    let (_, iid) = db.predict("double_reward_test", vec![1.0, 0.0]).unwrap();

    // First reward must succeed and update the model
    assert!(db.reward(&iid, 1.0), "First reward must return true");

    let theta_after_first = {
        let c = db.campaigns.read();
        let campaign = c.get("double_reward_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm").unwrap().theta.clone();
        x
    };

    // Second reward with the same id must be rejected
    assert!(
        !db.reward(&iid, 1.0),
        "Second reward with same interaction_id must return false"
    );

    let theta_after_second = {
        let c = db.campaigns.read();
        let campaign = c.get("double_reward_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm").unwrap().theta.clone();
        x
    };

    assert_eq!(
        theta_after_first, theta_after_second,
        "Double reward must not update theta a second time"
    );

    let _ = std::fs::remove_file(wal);
}

/// Test V.3 — Reward for Unknown Interaction Returns False
///
/// Rewarding an interaction_id that was never predicted (or whose TTL expired)
/// must return false and must not mutate any campaign's theta.
#[tokio::test]
async fn test_v3_unknown_interaction_reward_rejected() {
    let wal = "/tmp/banditdb_test_v3.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    db.add_campaign("unknown_iid_test", vec!["arm".to_string()], 2, 1.0, Algorithm::Linucb);

    let theta_before = {
        let c = db.campaigns.read();
        let campaign = c.get("unknown_iid_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm").unwrap().theta.clone();
        x
    };

    assert!(
        !db.reward("interaction-id-that-never-existed", 1.0),
        "Reward for unknown interaction_id must return false"
    );

    let theta_after = {
        let c = db.campaigns.read();
        let campaign = c.get("unknown_iid_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm").unwrap().theta.clone();
        x
    };

    assert_eq!(
        theta_before, theta_after,
        "Unknown interaction reward must not mutate theta"
    );

    let _ = std::fs::remove_file(wal);
}

/// Test V.4 — Non-Finite Reward Is Rejected, Out-of-Range Is Warned But Applied
///
/// The engine rejects Inf and NaN rewards entirely (existing guard in update()).
/// A reward outside [0, 1] (e.g. 5.0) is a caller mistake that the HTTP handler
/// warns about but the engine still applies — this test verifies both behaviours.
#[tokio::test]
async fn test_v4_reward_range_behaviour() {
    let wal = "/tmp/banditdb_test_v4.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    db.add_campaign("range_test", vec!["arm".to_string()], 2, 1.0, Algorithm::Linucb);

    // Non-finite reward: engine must reject it, theta stays at zero
    let (_, iid_inf) = db.predict("range_test", vec![1.0, 0.0]).unwrap();
    db.reward(&iid_inf, f64::INFINITY);

    let theta_after_inf = {
        let c = db.campaigns.read();
        let campaign = c.get("range_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm").unwrap().theta.clone();
        x
    };
    assert!(
        theta_after_inf.iter().all(|&v| v == 0.0),
        "Inf reward must not update theta"
    );

    // Out-of-range but finite reward: engine applies it (handler warns, but does not block)
    let (_, iid_oob) = db.predict("range_test", vec![1.0, 0.0]).unwrap();
    assert!(db.reward(&iid_oob, 5.0), "Out-of-range finite reward must still return true");

    let theta_after_oob = {
        let c = db.campaigns.read();
        let campaign = c.get("range_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm").unwrap().theta.clone();
        x
    };
    assert!(
        theta_after_oob.iter().any(|&v| v != 0.0),
        "Out-of-range finite reward must update theta"
    );

    let _ = std::fs::remove_file(wal);
}

/// Original learning test: the engine must converge to context-specific arm selection
/// after 50 training rounds of positive reinforcement.
#[tokio::test]
async fn test_bandit_learns_context() {
    let wal = "/tmp/banditdb_test_learns_context.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    db.add_campaign("homepage", vec!["layout_a".to_string(), "layout_b".to_string()], 2, 1.0, Algorithm::Linucb);

    let mobile_context = vec![1.0, 0.0];
    let desktop_context = vec![0.0, 1.0];

    for _ in 0..50 {
        let (arm, iid) = db.predict("homepage", mobile_context.clone()).unwrap();
        db.reward(&iid, if arm == "layout_a" { 1.0 } else { 0.0 });

        let (arm, iid) = db.predict("homepage", desktop_context.clone()).unwrap();
        db.reward(&iid, if arm == "layout_b" { 1.0 } else { 0.0 });
    }

    let (mobile_pred, _) = db.predict("homepage", mobile_context).unwrap();
    assert_eq!(mobile_pred, "layout_a");

    let _ = std::fs::remove_file(wal);
}

/// TS Test 1 — Thompson Sampling learns context-dependent preferences.
///
/// Same structure as test_bandit_learns_context but uses ThompsonSampling.
/// After 100 training rounds the model must reliably prefer the rewarded arm.
/// Retried up to 3× since TS is stochastic.
#[tokio::test]
async fn test_ts_learns_context() {
    let wal = "/tmp/banditdb_test_ts_learns.jsonl";

    for attempt in 0..3 {
        let _ = std::fs::remove_file(wal);
        let db = BanditDB::new(wal, "/tmp");
        db.add_campaign("ts_homepage", vec!["layout_a".to_string(), "layout_b".to_string()], 2, 1.0, Algorithm::ThompsonSampling);

        let mobile_context  = vec![1.0, 0.0];
        let desktop_context = vec![0.0, 1.0];

        for _ in 0..100 {
            let (arm, iid) = db.predict("ts_homepage", mobile_context.clone()).unwrap();
            db.reward(&iid, if arm == "layout_a" { 1.0 } else { 0.0 });

            let (arm, iid) = db.predict("ts_homepage", desktop_context.clone()).unwrap();
            db.reward(&iid, if arm == "layout_b" { 1.0 } else { 0.0 });
        }

        let (mobile_pred, _) = db.predict("ts_homepage", mobile_context.clone()).unwrap();
        if mobile_pred == "layout_a" {
            let _ = std::fs::remove_file(wal);
            return; // passed
        }
        if attempt == 2 {
            assert_eq!(mobile_pred, "layout_a", "TS failed to learn context after 3 attempts");
        }
    }

    let _ = std::fs::remove_file(wal);
}

/// TS Test 2 — Thompson Sampling explores all arms with neutral context.
///
/// With 3 arms and no rewards, TS must visit all 3 arms across 50 predictions
/// (stochastic exploration ensures diversity).
#[tokio::test]
async fn test_ts_explores() {
    let wal = "/tmp/banditdb_test_ts_explores.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    db.add_campaign("ts_explore", vec!["a".to_string(), "b".to_string(), "c".to_string()], 2, 1.0, Algorithm::ThompsonSampling);

    let mut seen = std::collections::HashSet::new();
    for _ in 0..50 {
        if let Some((arm, _)) = db.predict("ts_explore", vec![1.0, 1.0]) {
            seen.insert(arm);
        }
    }

    assert_eq!(
        seen.len(), 3,
        "TS must explore all 3 arms across 50 predictions, but only saw: {:?}", seen
    );

    let _ = std::fs::remove_file(wal);
}

/// TS Test 3 — Algorithm field survives checkpoint/recovery round-trip.
///
/// Creates a TS campaign, trains it, checkpoints, then recovers. The recovered
/// campaign must still have algorithm == ThompsonSampling and predict successfully.
#[tokio::test]
async fn test_ts_checkpoint_recovery() {
    let data_dir = "/tmp/banditdb_ts_ckpt_test";
    let wal_path = format!("{}/bandit_wal.jsonl", data_dir);
    let _ = std::fs::remove_dir_all(data_dir);
    std::fs::create_dir_all(data_dir).unwrap();

    let db = BanditDB::new(&wal_path, data_dir);
    db.add_campaign("ts_camp", vec!["x".to_string(), "y".to_string()], 2, 1.0, Algorithm::ThompsonSampling);

    for i in 0..20_usize {
        let ctx = vec![(i as f64 * 0.3).sin(), (i as f64 * 0.3).cos()];
        if let Some((arm, iid)) = db.predict("ts_camp", ctx) {
            db.reward(&iid, if arm == "x" { 1.0 } else { 0.0 });
        }
    }

    db.checkpoint().await.unwrap();
    drop(db);

    let db2 = BanditDB::new(&wal_path, data_dir);

    {
        let campaigns = db2.campaigns.read();
        let camp = campaigns.get("ts_camp").expect("ts_camp must survive recovery");
        assert_eq!(
            camp.algorithm, Algorithm::ThompsonSampling,
            "algorithm field must be ThompsonSampling after checkpoint recovery"
        );
    }

    assert!(
        db2.predict("ts_camp", vec![1.0, 0.0]).is_some(),
        "predict must work on recovered TS campaign"
    );

    let _ = std::fs::remove_dir_all(data_dir);
}

/// TS Test 4 — LinUCB and Thompson Sampling campaigns coexist without interference.
#[tokio::test]
async fn test_linucb_ts_coexist() {
    let wal = "/tmp/banditdb_test_coexist.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    db.add_campaign("ucb_camp", vec!["a".to_string(), "b".to_string()], 2, 1.0, Algorithm::Linucb);
    db.add_campaign("ts_camp",  vec!["a".to_string(), "b".to_string()], 2, 1.0, Algorithm::ThompsonSampling);

    for _ in 0..20 {
        if let Some((_, iid)) = db.predict("ucb_camp", vec![1.0, 0.0]) {
            db.reward(&iid, 1.0);
        }
        if let Some((_, iid)) = db.predict("ts_camp", vec![1.0, 0.0]) {
            db.reward(&iid, 1.0);
        }
    }

    assert!(db.predict("ucb_camp", vec![1.0, 0.0]).is_some(), "LinUCB campaign must still predict");
    assert!(db.predict("ts_camp",  vec![1.0, 0.0]).is_some(), "TS campaign must still predict");

    {
        let campaigns = db.campaigns.read();
        assert_eq!(campaigns.get("ucb_camp").unwrap().algorithm, Algorithm::Linucb);
        assert_eq!(campaigns.get("ts_camp").unwrap().algorithm,  Algorithm::ThompsonSampling);
    }

    let _ = std::fs::remove_file(wal);
}
