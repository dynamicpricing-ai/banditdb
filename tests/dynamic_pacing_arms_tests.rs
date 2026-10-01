use banditdb::state::{Algorithm, ArmStatus, PacingConfig, ResourceConstraint, WarmStart};
use banditdb::BanditDB;
use std::collections::HashMap;

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

#[tokio::test]
async fn test_pacing_with_dynamic_arms() {
    let (wal, dir) = temp_paths("pacing_with_dynamic_arms");
    let db = BanditDB::new(&wal, &dir);

    // 1. Create a paced campaign with 2 arms.
    let mut costs = HashMap::new();
    costs.insert("paid_1".to_string(), 1.0);
    costs.insert("paid_2".to_string(), 1.0);

    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "budget".to_string(),
            budget: 10.0,
            horizon: 100,
            step_size: None,
            lambda_max: None,
            initial_lambda: None,
            arm_costs: costs,
        }],
        adaptive: false,
    };

    db.add_campaign_pacing(
        "dyn_camp",
        arms(&["paid_1", "paid_2"]),
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
        Some(pacing),
    )
    .await
    .unwrap();

    // 2. Consume some budget.
    for _ in 0..5 {
        let (arm, iid) = db.predict("dyn_camp", vec![1.0, 0.0]).unwrap();
        db.reward(&iid, 1.0).await.unwrap();
        assert!(arm == "paid_1" || arm == "paid_2");
    }

    let report = db.campaign_pacing_report("dyn_camp").unwrap().unwrap();
    let consumed_initial = report.resources[0].consumed;
    assert!(consumed_initial > 0.0, "Budget should be consumed");

    // 3. Pause an arm.
    db.set_arm_status("dyn_camp", "paid_2", ArmStatus::Paused)
        .await
        .unwrap();

    // 4. Add a new arm dynamically.
    // It is not in the original pacing config arm_costs, so its cost defaults to 0.0.
    db.add_arm("dyn_camp", "free_new", None, &WarmStart::None)
        .await
        .unwrap();

    // 5. Predict again.
    // We should be able to select both paid_1 and free_new.
    let mut selected_new = false;
    for _ in 0..20 {
        let (arm, _) = db.predict("dyn_camp", vec![0.0, 1.0]).unwrap();
        if arm == "free_new" {
            selected_new = true;
        }
        assert_ne!(arm, "paid_2", "Paused arm must not be selected");
    }
    assert!(selected_new, "Dynamically added arm should be selectable");

    // 6. Exhaust the budget.
    // Pause free_new temporarily so that only paid_1 is eligible, forcing budget consumption.
    db.set_arm_status("dyn_camp", "free_new", ArmStatus::Paused)
        .await
        .unwrap();

    // Predict until paid_1 exhausts the remaining budget.
    // Since free_new is paused, once budget is exhausted, predict will return BadRequest.
    let mut exhausted = false;
    for _ in 0..10 {
        match db.predict("dyn_camp", vec![1.0, 0.0]) {
            Ok((arm, _)) => assert_eq!(arm, "paid_1", "Only paid_1 should be selectable right now"),
            Err(e) => {
                assert!(
                    e.to_string().contains("exhausted capacity"),
                    "Expected capacity exhaustion error"
                );
                exhausted = true;
                break;
            }
        }
    }
    assert!(exhausted, "paid_1 should have exhausted the budget");

    // Reactivate free_new.
    db.set_arm_status("dyn_camp", "free_new", ArmStatus::Active)
        .await
        .unwrap();

    // Once budget is exhausted, paid_1 will be masked out by the Lagrangian engine.
    // Predict should then exclusively return the free_new arm.
    let report2 = db.campaign_pacing_report("dyn_camp").unwrap().unwrap();
    assert!(
        report2.resources[0].is_exhausted,
        "Budget should be exhausted"
    );

    for _ in 0..10 {
        let (arm, _) = db.predict("dyn_camp", vec![1.0, 0.0]).unwrap();
        assert_eq!(
            arm, "free_new",
            "Only unconstrained dynamic arm is left after budget exhaustion"
        );
    }
}

#[tokio::test]
async fn test_dynamic_arm_with_custom_costs() {
    let (wal, dir) = temp_paths("dynamic_arm_custom_costs");
    let db = BanditDB::new(&wal, &dir);

    let mut costs = HashMap::new();
    costs.insert("seed".to_string(), 1.0);

    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "tokens".to_string(),
            budget: 5.0,
            horizon: 50,
            step_size: Some(0.01),
            lambda_max: Some(5.0),
            initial_lambda: Some(0.0),
            arm_costs: costs,
        }],
        adaptive: false,
    };

    db.add_campaign_pacing(
        "pacing_dyn_costs",
        arms(&["seed"]),
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
        Some(pacing),
    )
    .await
    .unwrap();

    // Dynamically add a paid arm with cost 2.0
    let mut arm2_costs = HashMap::new();
    arm2_costs.insert("tokens".to_string(), 2.0);

    db.add_arm_with_costs(
        "pacing_dyn_costs",
        "paid_dynamic",
        None,
        &WarmStart::None,
        Some(arm2_costs),
    )
    .await
    .unwrap();

    // Force selection of paid_dynamic
    let filter = banditdb::engine::ArmFilter::include(vec!["paid_dynamic".to_string()]);
    let (chosen, iid) = db
        .predict_filtered("pacing_dyn_costs", vec![1.0, 0.0], &filter)
        .unwrap();
    assert_eq!(chosen, "paid_dynamic");
    db.reward(&iid, 1.0).await.unwrap();

    let report = db
        .campaign_pacing_report("pacing_dyn_costs")
        .unwrap()
        .unwrap();
    assert_eq!(
        report.resources[0].consumed, 2.0,
        "Dynamic arm cost must be charged to budget"
    );
    assert_eq!(report.resources[0].remaining, 3.0);
}

#[tokio::test]
async fn test_dynamic_arm_unknown_resource_rejected() {
    let (wal, dir) = temp_paths("dyn_unknown_res");
    let db = BanditDB::new(&wal, &dir);

    let mut costs = HashMap::new();
    costs.insert("tokens".to_string(), 1.0);

    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "tokens".to_string(),
            budget: 5.0,
            horizon: 50,
            step_size: Some(0.01),
            lambda_max: Some(5.0),
            initial_lambda: Some(0.0),
            arm_costs: costs,
        }],
        adaptive: false,
    };

    db.add_campaign_pacing(
        "camp_res_check",
        arms(&["seed"]),
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
        Some(pacing),
    )
    .await
    .unwrap();

    let mut bad_costs = HashMap::new();
    bad_costs.insert("non_existent".to_string(), 1.0);

    let res = db
        .add_arm_with_costs(
            "camp_res_check",
            "bad_arm",
            None,
            &WarmStart::None,
            Some(bad_costs),
        )
        .await;
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("unknown resource 'non_existent'"));
}

#[tokio::test]
async fn test_dynamic_arm_unpaced_campaign_with_costs_rejected() {
    let (wal, dir) = temp_paths("dyn_unpaced_costs");
    let db = BanditDB::new(&wal, &dir);

    db.add_campaign(
        "camp_unpaced",
        arms(&["seed"]),
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
    )
    .await
    .unwrap();

    let mut costs = HashMap::new();
    costs.insert("tokens".to_string(), 1.0);

    let res = db
        .add_arm_with_costs(
            "camp_unpaced",
            "arm_with_cost",
            None,
            &WarmStart::None,
            Some(costs),
        )
        .await;
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("has no pacing configured"));
}

