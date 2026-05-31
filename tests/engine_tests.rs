use banditdb::engine::WalMessage;
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
    let _ = db.add_campaign("convergence", vec!["arm".to_string()], 2, 1.0, Algorithm::Linucb, None, None);

    let true_theta = [3.0_f64, -2.0_f64];

    // Deterministic contexts tracing the unit circle (sin/cos at 0.1-radian steps).
    // Dense, balanced coverage of all directions with no rand crate needed.
    for i in 0..500_usize {
        let angle = i as f64 * 0.1;
        let ctx = vec![angle.sin(), angle.cos()];
        let reward = true_theta[0] * ctx[0] + true_theta[1] * ctx[1];

        if let Ok((_, iid)) = db.predict("convergence", ctx) {
            let _ = db.reward(&iid, reward);
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
    let _ = db.add_campaign("dim_test", vec!["a".to_string(), "b".to_string()], 2, 1.0, Algorithm::Linucb, None, None);

    // Baseline: correct dim must succeed.
    assert!(db.predict("dim_test", vec![1.0, 0.0]).is_ok());

    // Over-sized context: 3 features instead of 2.
    assert!(
        db.predict("dim_test", vec![1.0, 0.0, 0.0]).is_err(),
        "Over-sized context should return None"
    );

    // Under-sized context: 1 feature instead of 2.
    assert!(
        db.predict("dim_test", vec![1.0]).is_err(),
        "Under-sized context should return None"
    );

    // Edge case: empty context vector.
    assert!(
        db.predict("dim_test", vec![]).is_err(),
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
    assert!(db.add_campaign("dup_test", vec!["arm_a".to_string()], 2, 1.0, Algorithm::Linucb, None, None).is_ok());

    // Train it so theta is non-zero
    let (_, iid) = db.predict("dup_test", vec![1.0, 0.0]).unwrap();
    let _ = db.reward(&iid, 1.0);

    let theta_before = {
        let c = db.campaigns.read();
        let campaign = c.get("dup_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm_a").unwrap().theta.clone();
        x
    };

    // Second create with same id must be rejected
    assert!(
        db.add_campaign("dup_test", vec!["arm_a".to_string()], 2, 1.0, Algorithm::Linucb, None, None).is_err(),
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
    let _ = db.add_campaign("double_reward_test", vec!["arm".to_string()], 2, 1.0, Algorithm::Linucb, None, None);

    let (_, iid) = db.predict("double_reward_test", vec![1.0, 0.0]).unwrap();

    // First reward must succeed and update the model
    assert!(db.reward(&iid, 1.0).is_ok(), "First reward must return true");

    let theta_after_first = {
        let c = db.campaigns.read();
        let campaign = c.get("double_reward_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm").unwrap().theta.clone();
        x
    };

    // Second reward with the same id must be rejected
    assert!(
        db.reward(&iid, 1.0).is_err(),
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
    let _ = db.add_campaign("unknown_iid_test", vec!["arm".to_string()], 2, 1.0, Algorithm::Linucb, None, None);

    let theta_before = {
        let c = db.campaigns.read();
        let campaign = c.get("unknown_iid_test").unwrap();
        let arms = campaign.arms.read();
        let x = arms.get("arm").unwrap().theta.clone();
        x
    };

    assert!(
        db.reward("interaction-id-that-never-existed", 1.0).is_err(),
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
    let _ = db.add_campaign("range_test", vec!["arm".to_string()], 2, 1.0, Algorithm::Linucb, None, None);

    // Non-finite reward: engine must reject it, theta stays at zero
    let (_, iid_inf) = db.predict("range_test", vec![1.0, 0.0]).unwrap();
    let _ = db.reward(&iid_inf, f64::INFINITY);

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
    assert!(db.reward(&iid_oob, 5.0).is_ok(), "Out-of-range finite reward must still return true");

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
    let _ = db.add_campaign("homepage", vec!["layout_a".to_string(), "layout_b".to_string()], 2, 1.0, Algorithm::Linucb, None, None);

    let mobile_context = vec![1.0, 0.0];
    let desktop_context = vec![0.0, 1.0];

    for _ in 0..50 {
        let (arm, iid) = db.predict("homepage", mobile_context.clone()).unwrap();
        let _ = db.reward(&iid, if arm == "layout_a" { 1.0 } else { 0.0 });

        let (arm, iid) = db.predict("homepage", desktop_context.clone()).unwrap();
        let _ = db.reward(&iid, if arm == "layout_b" { 1.0 } else { 0.0 });
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
        let _ = db.add_campaign("ts_homepage", vec!["layout_a".to_string(), "layout_b".to_string()], 2, 1.0, Algorithm::ThompsonSampling, None, None);

        let mobile_context  = vec![1.0, 0.0];
        let desktop_context = vec![0.0, 1.0];

        for _ in 0..100 {
            let (arm, iid) = db.predict("ts_homepage", mobile_context.clone()).unwrap();
            let _ = db.reward(&iid, if arm == "layout_a" { 1.0 } else { 0.0 });

            let (arm, iid) = db.predict("ts_homepage", desktop_context.clone()).unwrap();
            let _ = db.reward(&iid, if arm == "layout_b" { 1.0 } else { 0.0 });
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
    let _ = db.add_campaign("ts_explore", vec!["a".to_string(), "b".to_string(), "c".to_string()], 2, 1.0, Algorithm::ThompsonSampling, None, None);

    let mut seen = std::collections::HashSet::new();
    for _ in 0..50 {
        if let Ok((arm, _)) = db.predict("ts_explore", vec![1.0, 1.0]) {
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
    let _ = db.add_campaign("ts_camp", vec!["x".to_string(), "y".to_string()], 2, 1.0, Algorithm::ThompsonSampling, None, None);

    for i in 0..20_usize {
        let ctx = vec![(i as f64 * 0.3).sin(), (i as f64 * 0.3).cos()];
        if let Ok((arm, iid)) = db.predict("ts_camp", ctx) {
            let _ = db.reward(&iid, if arm == "x" { 1.0 } else { 0.0 });
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
        db2.predict("ts_camp", vec![1.0, 0.0]).is_ok(),
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
    let _ = db.add_campaign("ucb_camp", vec!["a".to_string(), "b".to_string()], 2, 1.0, Algorithm::Linucb, None, None);
    let _ = db.add_campaign("ts_camp",  vec!["a".to_string(), "b".to_string()], 2, 1.0, Algorithm::ThompsonSampling, None, None);

    for _ in 0..20 {
        if let Ok((_, iid)) = db.predict("ucb_camp", vec![1.0, 0.0]) {
            let _ = db.reward(&iid, 1.0);
        }
        if let Ok((_, iid)) = db.predict("ts_camp", vec![1.0, 0.0]) {
            let _ = db.reward(&iid, 1.0);
        }
    }

    assert!(db.predict("ucb_camp", vec![1.0, 0.0]).is_ok(), "LinUCB campaign must still predict");
    assert!(db.predict("ts_camp",  vec![1.0, 0.0]).is_ok(), "TS campaign must still predict");

    {
        let campaigns = db.campaigns.read();
        assert_eq!(campaigns.get("ucb_camp").unwrap().algorithm, Algorithm::Linucb);
        assert_eq!(campaigns.get("ts_camp").unwrap().algorithm,  Algorithm::ThompsonSampling);
    }

    let _ = std::fs::remove_file(wal);
}

/// Campaign metadata is stored, survives checkpoint+recovery, and is absent when not set.
#[tokio::test]
async fn test_campaign_metadata_roundtrip() {
    let data_dir = "/tmp/banditdb_metadata_test_data";
    let wal = format!("{}/bandit_wal.jsonl", data_dir);
    let _ = std::fs::remove_dir_all(data_dir);
    std::fs::create_dir_all(data_dir).unwrap();

    let meta = serde_json::json!({
        "owner": "recommendations-team",
        "features": ["user_age", "session_length"],
        "version": 1
    });

    {
        let db = BanditDB::new(&wal, data_dir);
        let _ = db.add_campaign("meta_camp", vec!["a".to_string()], 2, 1.0, Algorithm::Linucb, Some(meta.clone()), None);
        let _ = db.add_campaign("bare_camp", vec!["a".to_string()], 2, 1.0, Algorithm::Linucb, None, None);

        // Verify metadata is in memory immediately
        let campaigns = db.campaigns.read();
        let stored = campaigns.get("meta_camp").unwrap().metadata.as_ref().unwrap();
        assert_eq!(stored["owner"], "recommendations-team");
        assert_eq!(stored["features"][0], "user_age");
        assert!(campaigns.get("bare_camp").unwrap().metadata.is_none());
        drop(campaigns);

        db.checkpoint().await.unwrap();
    }

    // Recover from checkpoint and verify metadata survives
    {
        let db2 = BanditDB::new(&wal, data_dir);
        let campaigns = db2.campaigns.read();
        let stored = campaigns.get("meta_camp").unwrap().metadata.as_ref().unwrap();
        assert_eq!(stored["owner"], "recommendations-team");
        assert_eq!(stored["version"], 1);
        assert!(campaigns.get("bare_camp").unwrap().metadata.is_none());
    }

    let _ = std::fs::remove_dir_all(data_dir);
}

/// Metadata must survive WAL-only recovery (pre-checkpoint crash path).
///
/// If the process dies before checkpoint() is ever called, the CampaignCreated
/// event in the WAL is the only persistence record. This test simulates that by
/// using the WAL flush barrier (WalMessage::Checkpoint) to guarantee the event
/// is on disk, then dropping the db without ever calling db.checkpoint(). The
/// recovered instance must see identical metadata via pure WAL replay.
#[tokio::test]
async fn test_campaign_metadata_wal_only_recovery() {
    let data_dir = "/tmp/banditdb_metadata_wal_test_data";
    let wal = format!("{}/bandit_wal.jsonl", data_dir);
    let _ = std::fs::remove_dir_all(data_dir);
    std::fs::create_dir_all(data_dir).unwrap();

    let meta = serde_json::json!({
        "owner": "recommendations-team",
        "features": ["user_age", "session_length"],
        "version": 2,
        "active": true
    });

    {
        let db = BanditDB::new(&wal, data_dir);
        let _ = db.add_campaign(
            "meta_wal_camp",
            vec!["a".to_string(), "b".to_string()],
            2,
            1.0,
            Algorithm::Linucb,
            Some(meta.clone()),
            None,
        );
        let _ = db.add_campaign(
            "bare_wal_camp",
            vec!["a".to_string()],
            2,
            1.0,
            Algorithm::Linucb,
            None,
            None,
        );

        // Flush barrier: guarantees both CampaignCreated events are on disk.
        // This is NOT db.checkpoint() — no checkpoint.json is written.
        let (ftx, frx) = tokio::sync::oneshot::channel::<u64>();
        db.event_tx.send(WalMessage::Checkpoint { reply: ftx }).await.unwrap();
        let wal_size = frx.await.unwrap();
        assert!(wal_size > 0, "WAL must be non-empty after add_campaign");

        // Drop without checkpointing — simulates a crash before any checkpoint
    }

    // Confirm no checkpoint.json was created — we are testing pure WAL recovery
    assert!(
        !std::path::Path::new(&format!("{}/checkpoint.json", data_dir)).exists(),
        "checkpoint.json must not exist — this test verifies WAL-only recovery"
    );

    // Recover from WAL alone
    {
        let db2 = BanditDB::new(&wal, data_dir);
        let campaigns = db2.campaigns.read();

        assert!(campaigns.contains_key("meta_wal_camp"), "meta_wal_camp must survive WAL replay");
        assert!(campaigns.contains_key("bare_wal_camp"), "bare_wal_camp must survive WAL replay");

        let stored = campaigns
            .get("meta_wal_camp")
            .unwrap()
            .metadata
            .as_ref()
            .expect("metadata must survive WAL replay");

        assert_eq!(stored["owner"], "recommendations-team");
        assert_eq!(stored["features"][0], "user_age");
        assert_eq!(stored["features"][1], "session_length");
        assert_eq!(stored["version"], 2);
        assert_eq!(stored["active"], true);

        assert!(
            campaigns.get("bare_wal_camp").unwrap().metadata.is_none(),
            "bare_wal_camp metadata must remain None after WAL replay"
        );
    }

    let _ = std::fs::remove_dir_all(data_dir);
}

// ════════════════════════════════════════════════════════════════════════════
// TS Propensity Tests
//
// These tests verify the adaptive Monte Carlo propensity implementation.
// Before this feature, TS predictions logged None for arm_propensities.
// The tests confirm propensities are now:
//   1. Present (Some) after every TS prediction
//   2. A valid probability distribution (sum = 1, all values in [0, 1])
//   3. Concentrated toward the winning arm after training
//
// The adaptive sample count (N = 8–64 driven by A_inv diagonal) is an internal
// detail exercised indirectly through the distribution properties above.
// ════════════════════════════════════════════════════════════════════════════

/// TS Propensity 1 — every TS prediction must produce Some arm_propensities.
///
/// Before adaptive Monte Carlo, TS returned None. This is the simplest
/// regression guard: a single fresh prediction must populate the propensity map.
#[tokio::test]
async fn test_ts_propensity_is_some() {
    let wal = "/tmp/banditdb_test_ts_prop_some.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    let _ = db.add_campaign(
        "ts_prop_some",
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        3, 1.0, Algorithm::ThompsonSampling, None, None,
    );

    let (_, iid) = db.predict("ts_prop_some", vec![1.0, 0.0, 0.0]).unwrap();
    let record = db.interactions.get(iid.as_str())
        .expect("interaction must be in pending cache after predict");

    assert!(
        record.arm_propensities.is_some(),
        "TS must log propensities via adaptive Monte Carlo — got None"
    );

    let _ = std::fs::remove_file(wal);
}

/// TS Propensity 2 — propensities form a valid probability distribution.
///
/// For every prediction: one entry per arm, each value in [0, 1], sum = 1.0.
/// Checked over 20 consecutive predictions to cover different posterior samples.
#[tokio::test]
async fn test_ts_propensity_valid_distribution() {
    let wal = "/tmp/banditdb_test_ts_prop_dist.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    let arm_names = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let _ = db.add_campaign(
        "ts_prop_dist",
        arm_names.clone(),
        2, 1.0, Algorithm::ThompsonSampling, None, None,
    );

    for round in 0..20 {
        let (_, iid) = db.predict("ts_prop_dist", vec![1.0, 0.0]).unwrap();
        let record = db.interactions.get(iid.as_str())
            .expect("interaction must be in cache");
        let props = record.arm_propensities.as_ref()
            .expect("propensities must be Some on every TS prediction");

        assert_eq!(
            props.len(), 3,
            "round {round}: propensity map must have one entry per arm, got {}", props.len()
        );

        for name in &arm_names {
            let p = *props.get(name).unwrap_or_else(||
                panic!("round {round}: arm '{name}' missing from propensities")
            );
            assert!(
                (0.0..=1.0).contains(&p),
                "round {round}: propensity for '{name}' = {p:.6} outside [0, 1]"
            );
        }

        let sum: f64 = props.values().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "round {round}: propensities must sum to 1.0, got {sum:.12}"
        );
    }

    let _ = std::fs::remove_file(wal);
}

/// TS Propensity 3 — after training, the winning arm's propensity exceeds
/// the uniform baseline (1 / n_arms).
///
/// Trains a 2-arm campaign for 150 rounds: "win" always rewarded on [1.0, 0.0],
/// "lose" never rewarded. The winning arm's propensity must then exceed 0.5.
/// Retried up to 3× because TS is stochastic.
#[tokio::test]
async fn test_ts_propensity_concentrates_after_learning() {
    let wal = "/tmp/banditdb_test_ts_prop_conc.jsonl";

    for attempt in 0..3 {
        let _ = std::fs::remove_file(wal);
        let db = BanditDB::new(wal, "/tmp");
        let _ = db.add_campaign(
            "ts_prop_conc",
            vec!["win".to_string(), "lose".to_string()],
            2, 1.0, Algorithm::ThompsonSampling, None, None,
        );

        for _ in 0..150 {
            if let Ok((arm, iid)) = db.predict("ts_prop_conc", vec![1.0, 0.0]) {
                let _ = db.reward(&iid, if arm == "win" { 1.0 } else { 0.0 });
            }
        }

        let (_, iid) = db.predict("ts_prop_conc", vec![1.0, 0.0]).unwrap();
        let record = db.interactions.get(iid.as_str()).expect("in cache");
        let props  = record.arm_propensities.as_ref().expect("Some");
        let p_win  = *props.get("win").expect("win arm in propensity map");

        if p_win > 0.5 {
            let _ = std::fs::remove_file(wal);
            return;
        }
        if attempt == 2 {
            panic!(
                "winning arm propensity should exceed 0.5 after 150 training rounds, got {p_win:.3}"
            );
        }
    }

    let _ = std::fs::remove_file(wal);
}

/// TS Propensity 4 — LinUCB propensities are unaffected by the TS changes.
///
/// LinUCB uses softmax over UCB scores, not Monte Carlo sampling.
/// This guards against regressions: LinUCB propensities must still be Some,
/// valid, and non-degenerate after the TS implementation was added.
#[tokio::test]
async fn test_linucb_propensity_unaffected_by_ts_changes() {
    let wal = "/tmp/banditdb_test_linucb_prop.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    let _ = db.add_campaign(
        "ucb_prop",
        vec!["a".to_string(), "b".to_string()],
        2, 1.0, Algorithm::Linucb, None, None,
    );

    let (_, iid) = db.predict("ucb_prop", vec![1.0, 0.0]).unwrap();
    let record = db.interactions.get(iid.as_str()).expect("in cache");
    let props  = record.arm_propensities.as_ref()
        .expect("LinUCB must still produce softmax propensities");

    assert_eq!(props.len(), 2, "one entry per arm");

    let sum: f64 = props.values().sum();
    assert!((sum - 1.0).abs() < 1e-9, "LinUCB propensities must sum to 1.0, got {sum}");

    for (arm, &p) in props {
        assert!(
            p > 0.0 && p < 1.0,
            "LinUCB propensity for '{arm}' should be in (0, 1) on a fresh campaign, got {p:.4}"
        );
    }

    let _ = std::fs::remove_file(wal);
}
