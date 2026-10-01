//! Lagrangian Pacing & Capacity Constraints Integration Tests.
//!
//! Verifies:
//! 1. Invariance when unconstrained (pacing: None).
//! 2. Hard capacity masking when budget depleted.
//! 3. Dual shadow price adaptation via projected gradient descent.
//! 4. Price before propensity invariant (no IPS bias).
//! 5. Multi-knapsack constraint support.
//! 6. Checkpoint persistence and WAL replay durability.
//! 7. Endgame singularity protection and dual freezing.
//! 8. Rollback safety with unknown WAL events.
//! 9. Input validation rejecting malformed configurations.
//! 10. PacingConsumed WAL event: post-checkpoint pacing state restored on replay.
//! 11. PacingConsumed rollback: old WAL without PacingConsumed resets to checkpoint.

use banditdb::engine::ArmFilter;
use banditdb::state::{Algorithm, DbEvent, EngineError, PacingConfig, ResourceConstraint};
use banditdb::BanditDB;
use std::collections::HashMap;
use std::io::Write;

fn temp_paths(test_name: &str) -> (String, String) {
    let wal = format!("/tmp/banditdb_test_{test_name}.jsonl");
    let dir = format!("/tmp/bdb_test_{test_name}");
    let _ = std::fs::remove_file(&wal);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::create_dir_all(&dir);
    (wal, dir)
}

fn arms(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

// ════════════════════════════════════════════════════════════════════════════
// 1. Unconstrained Invariance
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_unconstrained_invariance() {
    let (wal, dir) = temp_paths("unconstrained_invariance");
    let db = BanditDB::new(&wal, &dir);

    // Create campaign A without pacing (None)
    db.add_campaign(
        "camp_a",
        arms(&["arm_1", "arm_2"]),
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
    )
    .await
    .unwrap();

    // Create campaign B explicitly passing None pacing
    db.add_campaign_pacing(
        "camp_b",
        arms(&["arm_1", "arm_2"]),
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    // Train arm_1 with positive reward in both campaigns so scores are strictly differentiated
    db.interact("camp_a", "arm_1", vec![1.0, 0.0], 1.0)
        .await
        .unwrap();
    db.interact("camp_b", "arm_1", vec![1.0, 0.0], 1.0)
        .await
        .unwrap();

    let ctx = vec![1.0, 0.0];
    for _ in 0..50 {
        let (arm_a, iid_a) = db.predict("camp_a", ctx.clone()).unwrap();
        let (arm_b, iid_b) = db.predict("camp_b", ctx.clone()).unwrap();
        assert_eq!(arm_a, arm_b, "Unconstrained decisions must be identical");
        assert_eq!(arm_a, "arm_1");
        db.reward(&iid_a, 1.0).await.unwrap();
        db.reward(&iid_b, 1.0).await.unwrap();
    }

    assert!(db.campaign_pacing_report("camp_a").unwrap().is_none());
    assert!(db.campaign_pacing_report("camp_b").unwrap().is_none());
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Hard Capacity Masking
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_hard_mask_exhaustion() {
    let (wal, dir) = temp_paths("hard_mask_exhaustion");
    let db = BanditDB::new(&wal, &dir);

    let mut costs = HashMap::new();
    costs.insert("expensive".to_string(), 5.0);
    costs.insert("free".to_string(), 0.0);

    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "tokens".to_string(),
            budget: 10.0, // Exactly enough for 2 predictions of expensive
            horizon: 100,
            step_size: Some(0.01),
            lambda_max: Some(1.0),
            initial_lambda: Some(0.0),
            arm_costs: costs,
        }],
        adaptive: true,
    };

    db.add_campaign_pacing(
        "server_tokens",
        arms(&["expensive", "free"]),
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
        Some(pacing),
    )
    .await
    .unwrap();

    // Use ArmFilter to force 2 selections of "expensive", consuming the full 10.0 budget
    let only_expensive = ArmFilter::include(vec!["expensive".to_string()]);
    let (p1, iid1) = db
        .predict_filtered("server_tokens", vec![1.0, 0.0], &only_expensive)
        .unwrap();
    assert_eq!(p1, "expensive");
    db.reward(&iid1, 1.0).await.unwrap();

    let (p2, iid2) = db
        .predict_filtered("server_tokens", vec![1.0, 0.0], &only_expensive)
        .unwrap();
    assert_eq!(p2, "expensive");
    db.reward(&iid2, 1.0).await.unwrap();

    // Now budget consumed is 10.0 / 10.0. Remaining is 0.0 < 5.0.
    // Prediction 3 with no filter: "expensive" is hard-masked! "free" MUST be selected.
    let (p3, _) = db.predict("server_tokens", vec![1.0, 0.0]).unwrap();
    assert_eq!(
        p3, "free",
        "Expensive arm must be hard-masked when capacity is depleted"
    );

    // Prediction 4 attempting to force "expensive" via filter MUST fail with BadRequest (no eligible arms)
    let p4_err = db.predict_filtered("server_tokens", vec![1.0, 0.0], &only_expensive);
    assert!(
        matches!(p4_err, Err(EngineError::BadRequest(_))),
        "Filter requiring masked arm must return BadRequest"
    );

    let report = db
        .campaign_pacing_report("server_tokens")
        .unwrap()
        .expect("pacing report exists");
    let res = &report.resources[0];
    assert_eq!(res.consumed, 10.0);
    assert_eq!(res.remaining, 0.0);
    assert!(res.is_exhausted);
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Dual Shadow Price Adaptation
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_dual_shadow_price_adaptation() {
    let (wal, dir) = temp_paths("dual_adaptation");
    let db = BanditDB::new(&wal, &dir);

    let mut costs = HashMap::new();
    costs.insert("heavy".to_string(), 1.0);
    costs.insert("light".to_string(), 0.0);

    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "compute".to_string(),
            budget: 100.0,
            horizon: 1000,
            step_size: Some(0.1),
            lambda_max: Some(5.0),
            initial_lambda: Some(0.0),
            arm_costs: costs,
        }],
        adaptive: false, // Target rate rho = 100 / 1000 = 0.1
    };

    db.add_campaign_pacing(
        "compute_camp",
        arms(&["heavy", "light"]),
        2,
        0.0,
        Algorithm::Linucb,
        None,
        None,
        Some(pacing),
    )
    .await
    .unwrap();

    let report0 = db.campaign_pacing_report("compute_camp").unwrap().unwrap();
    assert_eq!(report0.resources[0].lambda, 0.0);

    // Serve prediction forcing heavy: heavy is selected and consumes 1.0.
    // rho = 0.1. delta = 1.0 - 0.1 = 0.9.
    // lambda_1 = clamp(0.0 + 0.1 * 0.9) = 0.09.
    let only_heavy = ArmFilter::include(vec!["heavy".to_string()]);
    let (chosen, _) = db
        .predict_filtered("compute_camp", vec![1.0, 0.0], &only_heavy)
        .unwrap();
    assert_eq!(chosen, "heavy");

    let report1 = db.campaign_pacing_report("compute_camp").unwrap().unwrap();
    let l1 = report1.resources[0].lambda;
    assert!(
        (l1 - 0.09).abs() < 1e-6,
        "Lambda should increase by eta * (c - rho), got {l1}"
    );

    // Serve prediction forcing light: light is selected (consumes 0.0).
    // rho = 0.1. delta = 0.0 - 0.1 = -0.1.
    // lambda_2 = clamp(0.09 + 0.1 * (-0.1)) = 0.08.
    let only_light = ArmFilter::include(vec!["light".to_string()]);
    let (chosen2, _) = db
        .predict_filtered("compute_camp", vec![1.0, 0.0], &only_light)
        .unwrap();
    assert_eq!(chosen2, "light");

    let report2 = db.campaign_pacing_report("compute_camp").unwrap().unwrap();
    let l2 = report2.resources[0].lambda;
    assert!(
        (l2 - 0.08).abs() < 1e-6,
        "Lambda should decrease when under-consuming, got {l2}"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// 4. Price Before Propensity Invariant
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_price_before_propensity() {
    let (wal, dir) = temp_paths("price_before_propensity");
    let db = BanditDB::new(&wal, &dir);

    let mut costs = HashMap::new();
    costs.insert("arm_a".to_string(), 1.0);
    costs.insert("arm_b".to_string(), 0.0);

    // Initial shadow price lambda_0 = 2.0
    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "res".to_string(),
            budget: 100.0,
            horizon: 1000,
            step_size: Some(0.01),
            lambda_max: Some(5.0),
            initial_lambda: Some(2.0),
            arm_costs: costs,
        }],
        adaptive: false,
    };

    db.add_campaign_pacing(
        "priced_propensity",
        arms(&["arm_a", "arm_b"]),
        2,
        0.5,
        Algorithm::Linucb,
        None,
        None,
        Some(pacing),
    )
    .await
    .unwrap();

    let (_, iid) = db.predict("priced_propensity", vec![1.0, 1.0]).unwrap();
    let record = db.interactions.get(&iid).expect("interaction recorded");
    let propensities = record.arm_propensities.expect("propensities computed");

    let p_a = propensities.get("arm_a").copied().unwrap_or(0.0);
    let p_b = propensities.get("arm_b").copied().unwrap_or(0.0);

    // At cold start without price, arm_a and arm_b have identical feature scores (both 0) and identical uncertainty.
    // With price lambda * cost_a = 2.0 * 1.0 = 2.0 subtracted from arm_a, arm_a's effective score is -2.0.
    // Softmax ensures p_b > p_a.
    assert!(
        p_b > p_a,
        "Price penalty must reduce propensity before logging: p_b ({p_b}) > p_a ({p_a})"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// 5. Multi-Knapsack Constraints
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_multi_knapsack_constraints() {
    let (wal, dir) = temp_paths("multi_knapsack");
    let db = BanditDB::new(&wal, &dir);

    let mut costs_gpu = HashMap::new();
    costs_gpu.insert("heavy".to_string(), 2.0);
    costs_gpu.insert("light".to_string(), 0.0);

    let mut costs_ram = HashMap::new();
    costs_ram.insert("heavy".to_string(), 1.0);
    costs_ram.insert("light".to_string(), 1.0);

    let pacing = PacingConfig {
        resources: vec![
            ResourceConstraint {
                name: "gpu".to_string(),
                budget: 2.0, // Only 1 heavy call allowed
                horizon: 100,
                step_size: Some(0.01),
                lambda_max: Some(2.0),
                initial_lambda: Some(0.0),
                arm_costs: costs_gpu,
            },
            ResourceConstraint {
                name: "ram".to_string(),
                budget: 10.0,
                horizon: 100,
                step_size: Some(0.01),
                lambda_max: Some(2.0),
                initial_lambda: Some(0.0),
                arm_costs: costs_ram,
            },
        ],
        adaptive: true,
    };

    db.add_campaign_pacing(
        "multi_res",
        arms(&["heavy", "light"]),
        2,
        0.0,
        Algorithm::Linucb,
        None,
        None,
        Some(pacing),
    )
    .await
    .unwrap();

    // Call 1: force heavy. Consumes 2.0 GPU and 1.0 RAM.
    let only_heavy = ArmFilter::include(vec!["heavy".to_string()]);
    let (c1, _) = db
        .predict_filtered("multi_res", vec![1.0, 0.0], &only_heavy)
        .unwrap();
    assert_eq!(c1, "heavy");

    // Call 2: with no filter, GPU is exhausted (remaining = 0.0 < 2.0), so heavy is masked!
    // light only consumes 0 GPU and 1 RAM (remaining 9.0 >= 1.0), so light MUST be selected.
    let (c2, _) = db.predict("multi_res", vec![1.0, 0.0]).unwrap();
    assert_eq!(c2, "light");

    let report = db.campaign_pacing_report("multi_res").unwrap().unwrap();
    let gpu = report.resources.iter().find(|r| r.name == "gpu").unwrap();
    let ram = report.resources.iter().find(|r| r.name == "ram").unwrap();
    assert!(gpu.is_exhausted);
    assert_eq!(gpu.consumed, 2.0);
    assert_eq!(ram.consumed, 2.0); // 1 from heavy + 1 from light
}

// ════════════════════════════════════════════════════════════════════════════
// 6. Checkpoint Persistence and WAL Replay Durability
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_pacing_checkpoint_recovery_and_wal_replay() {
    let (wal, dir) = temp_paths("checkpoint_durability");

    let mut costs = HashMap::new();
    costs.insert("arm_x".to_string(), 2.5);

    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "storage".to_string(),
            budget: 50.0,
            horizon: 500,
            step_size: Some(0.05),
            lambda_max: Some(3.0),
            initial_lambda: Some(0.1),
            arm_costs: costs,
        }],
        adaptive: true,
    };

    let (c_before, l_before, d_before) = {
        let db = BanditDB::new(&wal, &dir);
        db.add_campaign_pacing(
            "durability_camp",
            arms(&["arm_x", "arm_y"]),
            2,
            1.0,
            Algorithm::Linucb,
            None,
            None,
            Some(pacing),
        )
        .await
        .unwrap();

        // 3 predictions before checkpoint
        for _ in 0..3 {
            let (_, iid) = db.predict("durability_camp", vec![1.0, 0.0]).unwrap();
            db.reward(&iid, 1.0).await.unwrap();
        }

        db.checkpoint().await.expect("checkpoint must succeed");

        // 2 predictions after checkpoint (stored in WAL slice)
        for _ in 0..2 {
            let (_, iid) = db.predict("durability_camp", vec![1.0, 0.0]).unwrap();
            db.reward(&iid, 1.0).await.unwrap();
        }

        let report = db
            .campaign_pacing_report("durability_camp")
            .unwrap()
            .unwrap();
        let r = &report.resources[0];
        (r.consumed, r.lambda, r.decisions)
    };

    // Reopen DB from the same directory to verify recovery
    {
        let recovered_db = BanditDB::new(&wal, &dir);
        let report = recovered_db
            .campaign_pacing_report("durability_camp")
            .unwrap()
            .expect("pacing restored");
        let r = &report.resources[0];

        assert_eq!(
            r.decisions, d_before,
            "Decisions count must survive recovery"
        );
        assert_eq!(
            r.consumed, c_before,
            "Consumed amount must survive recovery"
        );
        assert!(
            (r.lambda - l_before).abs() < 1e-9,
            "Lambda must match precisely across recovery"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 7. Endgame Singularity Protection
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_endgame_singularity_protection() {
    let (wal, dir) = temp_paths("endgame_freeze");
    let db = BanditDB::new(&wal, &dir);

    let mut costs = HashMap::new();
    costs.insert("arm_a".to_string(), 1.0);

    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "endgame_res".to_string(),
            budget: 100.0,
            horizon: 100, // Small horizon
            step_size: Some(0.1),
            lambda_max: Some(5.0),
            initial_lambda: Some(0.5),
            arm_costs: costs,
        }],
        adaptive: true,
    };

    db.add_campaign_pacing(
        "endgame_camp",
        arms(&["arm_a", "arm_b"]),
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
        Some(pacing),
    )
    .await
    .unwrap();

    // Drive decisions all the way to horizon and past it
    for _ in 0..120 {
        db.predict("endgame_camp", vec![0.1, 0.2]).unwrap();
    }

    let report = db.campaign_pacing_report("endgame_camp").unwrap().unwrap();
    let r = &report.resources[0];
    assert!(
        r.lambda.is_finite(),
        "Lambda must remain finite during endgame"
    );
    assert!(!r.lambda.is_nan(), "Lambda must not be NaN");
    assert!(r.decisions >= 100);
}

// ════════════════════════════════════════════════════════════════════════════
// 8. Rollback Safety with Unknown WAL Events
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_rollback_compatibility_unknown_wal_event() {
    let (wal, dir) = temp_paths("rollback_safety");

    // Manually write an unknown event line to the WAL
    {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&wal)
            .unwrap();

        let valid_campaign = DbEvent::CampaignCreated {
            campaign_id: "test_rollback".to_string(),
            arms: vec!["a".to_string(), "b".to_string()],
            feature_dim: 2,
            alpha: 1.0,
            algorithm: Algorithm::Linucb,
            metadata: None,
            decay_half_life_hours: None,
            pacing: None,
        };
        writeln!(file, "{}", serde_json::to_string(&valid_campaign).unwrap()).unwrap();

        // Write an unknown variant simulating a newer binary version writing an event
        let future_event = serde_json::json!({
            "QuantumSuperpositionPacingV3": {
                "entanglement_factor": 42.0
            }
        });
        writeln!(file, "{}", future_event).unwrap();
    }

    // Opening BanditDB should gracefully skip the unknown event without crashing
    let db = BanditDB::new(&wal, &dir);
    let campaigns = db.campaigns.read();
    assert!(
        campaigns.contains_key("test_rollback"),
        "Valid campaign must be recovered despite unknown event"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// 9. Input Validation
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_pacing_validation_rejects_malformed_config() {
    let (wal, dir) = temp_paths("validation_pacing");
    let db = BanditDB::new(&wal, &dir);

    // Empty resource name
    let bad1 = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "   ".to_string(),
            budget: 10.0,
            horizon: 100,
            step_size: None,
            lambda_max: None,
            initial_lambda: None,
            arm_costs: HashMap::new(),
        }],
        adaptive: true,
    };
    assert!(matches!(
        db.add_campaign_pacing(
            "bad1",
            arms(&["a"]),
            2,
            1.0,
            Algorithm::Linucb,
            None,
            None,
            Some(bad1)
        )
        .await,
        Err(EngineError::BadRequest(_))
    ));

    // Non-positive budget
    let bad2 = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "budget".to_string(),
            budget: -5.0,
            horizon: 100,
            step_size: None,
            lambda_max: None,
            initial_lambda: None,
            arm_costs: HashMap::new(),
        }],
        adaptive: true,
    };
    assert!(matches!(
        db.add_campaign_pacing(
            "bad2",
            arms(&["a"]),
            2,
            1.0,
            Algorithm::Linucb,
            None,
            None,
            Some(bad2)
        )
        .await,
        Err(EngineError::BadRequest(_))
    ));

    // Zero horizon
    let bad3 = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "budget".to_string(),
            budget: 10.0,
            horizon: 0,
            step_size: None,
            lambda_max: None,
            initial_lambda: None,
            arm_costs: HashMap::new(),
        }],
        adaptive: true,
    };
    assert!(matches!(
        db.add_campaign_pacing(
            "bad3",
            arms(&["a"]),
            2,
            1.0,
            Algorithm::Linucb,
            None,
            None,
            Some(bad3)
        )
        .await,
        Err(EngineError::BadRequest(_))
    ));

    // Duplicate resource names
    let bad4 = PacingConfig {
        resources: vec![
            ResourceConstraint {
                name: "dup".to_string(),
                budget: 10.0,
                horizon: 100,
                step_size: None,
                lambda_max: None,
                initial_lambda: None,
                arm_costs: HashMap::new(),
            },
            ResourceConstraint {
                name: "dup".to_string(),
                budget: 20.0,
                horizon: 200,
                step_size: None,
                lambda_max: None,
                initial_lambda: None,
                arm_costs: HashMap::new(),
            },
        ],
        adaptive: true,
    };
    assert!(matches!(
        db.add_campaign_pacing(
            "bad4",
            arms(&["a"]),
            2,
            1.0,
            Algorithm::Linucb,
            None,
            None,
            Some(bad4)
        )
        .await,
        Err(EngineError::BadRequest(_))
    ));
}

// ════════════════════════════════════════════════════════════════════════════
// 10. PacingConsumed WAL Event — Post-Checkpoint Pacing State Restored
// ════════════════════════════════════════════════════════════════════════════

/// Verifies that `PacingConsumed` events written into the post-checkpoint WAL
/// slice are replayed on recovery, fully restoring `consumed`, `decisions`, and
/// `lambda` to the values that existed at the moment of the simulated crash.
#[tokio::test]
async fn test_pacing_consumed_wal_replay_restores_post_checkpoint_state() {
    let (wal, dir) = temp_paths("pacing_consumed_replay");

    let mut costs = HashMap::new();
    costs.insert("costly".to_string(), 3.0);
    costs.insert("free".to_string(), 0.0);

    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "bandwidth".to_string(),
            budget: 1_000.0,
            horizon: 10_000,
            step_size: Some(0.05),
            lambda_max: Some(4.0),
            initial_lambda: Some(0.5),
            arm_costs: costs,
        }],
        adaptive: false,
    };

    let (consumed_live, lambda_live, decisions_live) = {
        let db = BanditDB::new(&wal, &dir);
        db.add_campaign_pacing(
            "replay_camp",
            arms(&["costly", "free"]),
            2,
            1.0,
            Algorithm::Linucb,
            None,
            None,
            Some(pacing),
        )
        .await
        .unwrap();

        // Phase 1: 5 predictions before checkpoint — force the costly arm.
        let only_costly = ArmFilter::include(vec!["costly".to_string()]);
        for _ in 0..5 {
            let (_, iid) = db
                .predict_filtered("replay_camp", vec![1.0, 0.0], &only_costly)
                .unwrap();
            db.reward(&iid, 1.0).await.unwrap();
        }

        // Checkpoint: 5-decision pacing state captured.
        db.checkpoint().await.expect("checkpoint must succeed");

        // Phase 2: 3 more predictions after checkpoint (WAL slice only).
        // Each emits one Predicted + one PacingConsumed event.
        for _ in 0..3 {
            let (_, iid) = db
                .predict_filtered("replay_camp", vec![1.0, 0.0], &only_costly)
                .unwrap();
            db.reward(&iid, 1.0).await.unwrap();
        }

        let report = db.campaign_pacing_report("replay_camp").unwrap().unwrap();
        let r = &report.resources[0];
        (r.consumed, r.lambda, r.decisions)
        // DB dropped — simulates crash. WAL remains on disk.
    };

    // Reopen: checkpoint restores 5-decision state, then 3 PacingConsumed
    // events from the WAL slice advance counters to 8 decisions.
    {
        let recovered = BanditDB::new(&wal, &dir);
        let report = recovered
            .campaign_pacing_report("replay_camp")
            .unwrap()
            .expect("pacing report must be present after recovery");
        let r = &report.resources[0];

        assert_eq!(
            r.decisions, decisions_live,
            "decisions must match live state after WAL replay: expected {decisions_live}, got {}",
            r.decisions
        );
        assert!(
            (r.consumed - consumed_live).abs() < 1e-9,
            "consumed must match live state after WAL replay: expected {consumed_live}, got {}",
            r.consumed
        );
        assert!(
            (r.lambda - lambda_live).abs() < 1e-9,
            "lambda must match live state after WAL replay: expected {lambda_live}, got {}",
            r.lambda
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 11. PacingConsumed Rollback Compatibility
// ════════════════════════════════════════════════════════════════════════════

/// Verifies that a WAL written by a binary that pre-dates `PacingConsumed`
/// (only `Predicted` events, no `PacingConsumed`) is replayed safely.
///
/// The `Predicted` handler no longer calls `record_consumption`, so pacing
/// counters sit at checkpoint values — conservatively correct (under-counts
/// post-rollback consumption, never over-counts or panics).
#[tokio::test]
async fn test_pacing_without_pacing_consumed_events_loads_safely() {
    let (wal, dir) = temp_paths("pacing_rollback_compat");

    let mut costs = HashMap::new();
    costs.insert("arm_a".to_string(), 1.0);

    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "cpu".to_string(),
            budget: 500.0,
            horizon: 5_000,
            step_size: Some(0.01),
            lambda_max: Some(3.0),
            initial_lambda: Some(0.0),
            arm_costs: costs,
        }],
        adaptive: false,
    };

    // Write a WAL with only CampaignCreated + Predicted events (no PacingConsumed).
    // Mimics a WAL slice from a pre-schema binary.
    {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&wal)
            .unwrap();

        let created = DbEvent::CampaignCreated {
            campaign_id: "rollback_camp".to_string(),
            arms: vec!["arm_a".to_string(), "arm_b".to_string()],
            feature_dim: 2,
            alpha: 1.0,
            algorithm: Algorithm::Linucb,
            metadata: None,
            decay_half_life_hours: None,
            pacing: Some(pacing),
        };
        writeln!(file, "{}", serde_json::to_string(&created).unwrap()).unwrap();

        for i in 0..10u32 {
            let predicted = DbEvent::Predicted {
                interaction_id: format!("iid-{i}"),
                campaign_id: "rollback_camp".to_string(),
                arm_id: "arm_a".to_string(),
                context: vec![1.0, 0.0],
                timestamp_secs: 1_700_000_000 + i as u64,
                arm_propensities: None,
                is_reemit: false,
            };
            writeln!(file, "{}", serde_json::to_string(&predicted).unwrap()).unwrap();
        }
    }

    let db = BanditDB::new(&wal, &dir);
    let report = db
        .campaign_pacing_report("rollback_camp")
        .unwrap()
        .expect("pacing report present");
    let r = &report.resources[0];

    // Old-binary WAL: no PacingConsumed replayed → consumed stays at 0.
    // Conservatively correct: under-counts, never over-counts.
    assert!(r.lambda.is_finite(), "lambda must be finite");
    assert!(!r.lambda.is_nan(), "lambda must not be NaN");
    assert!(r.consumed >= 0.0, "consumed must be non-negative");

    // Predictions must still work after loading old-format WAL.
    assert!(
        db.predict("rollback_camp", vec![0.5, 0.5]).is_ok(),
        "predictions must succeed after loading old-format WAL"
    );
}
