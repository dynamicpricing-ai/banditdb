use banditdb::engine::ArmFilter;
use banditdb::state::{Algorithm, PacingConfig, ResourceConstraint};
use banditdb::BanditDB;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinHandle;

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

/// Validates that under massive concurrent load (100 threads), the lock-free CAS
/// loops for `consumed` and `lambda` do not drop updates, deadlock, or corrupt state.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_pacing_cas_contention_and_exact_consumption() {
    let (wal, dir) = temp_paths("pacing_concurrency");
    let db = Arc::new(BanditDB::new(&wal, &dir));

    let mut costs = HashMap::new();
    costs.insert("paid".to_string(), 1.0);
    costs.insert("free".to_string(), 0.0);

    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "compute".to_string(),
            budget: 100_000.0,
            horizon: 100_000,
            step_size: Some(0.01),
            lambda_max: Some(10.0),
            initial_lambda: Some(0.0),
            arm_costs: costs,
        }],
        adaptive: false,
    };

    db.add_campaign_pacing(
        "concurrent_camp",
        arms(&["paid", "free"]),
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
        Some(pacing),
    )
    .await
    .unwrap();

    let num_tasks = 100;
    let requests_per_task = 100;
    let mut handles: Vec<JoinHandle<()>> = Vec::new();

    // Spawn 100 concurrent tasks, each firing 100 predictions as fast as possible.
    // Force the "paid" arm so we hit the CAS loops maximally.
    for _ in 0..num_tasks {
        let db_clone = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            let filter = ArmFilter::include(vec!["paid".to_string()]);
            for _ in 0..requests_per_task {
                let (_, iid) = db_clone
                    .predict_filtered("concurrent_camp", vec![1.0, 0.0], &filter)
                    .unwrap();
                db_clone.reward(&iid, 1.0).await.unwrap();
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let report = db
        .campaign_pacing_report("concurrent_camp")
        .unwrap()
        .unwrap();
    let r = &report.resources[0];

    // Exactly 100 * 100 = 10,000 decisions. Cost is 1.0 each.
    let expected_decisions = (num_tasks * requests_per_task) as u64;
    let expected_consumed = expected_decisions as f64;

    assert_eq!(
        r.decisions, expected_decisions,
        "Concurrent decisions count must match exactly"
    );
    assert_eq!(
        r.consumed, expected_consumed,
        "Concurrent consumed budget must match exactly (CAS loop correctness)"
    );
    assert!(
        r.lambda.is_finite() && !r.lambda.is_nan(),
        "Concurrent lambda updates must remain numerically stable"
    );
}

/// Validates that under heavy concurrent contention at budget exhaustion,
/// the atomic reservation guarantees zero budget oversell: consumed never exceeds budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_concurrent_exhaustion_zero_overshoot() {
    let (wal, dir) = temp_paths("pacing_zero_overshoot");
    let db = Arc::new(BanditDB::new(&wal, &dir));

    let mut costs = HashMap::new();
    costs.insert("arm_a".to_string(), 1.0);
    costs.insert("arm_b".to_string(), 1.0);

    let budget_limit = 10.0;
    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "limited_credits".to_string(),
            budget: budget_limit,
            horizon: 1_000,
            step_size: Some(0.01),
            lambda_max: Some(5.0),
            initial_lambda: Some(0.0),
            arm_costs: costs,
        }],
        adaptive: false,
    };

    db.add_campaign_pacing(
        "overshoot_camp",
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

    let num_tasks = 50;
    let requests_per_task = 5;
    let mut handles: Vec<JoinHandle<()>> = Vec::new();

    // 50 threads * 5 requests = 250 requests competing for 10.0 budget.
    for _ in 0..num_tasks {
        let db_clone = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            for _ in 0..requests_per_task {
                let _ = db_clone.predict("overshoot_camp", vec![1.0, 0.0]);
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let report = db
        .campaign_pacing_report("overshoot_camp")
        .unwrap()
        .unwrap();
    let r = &report.resources[0];

    assert!(
        r.consumed <= budget_limit,
        "Total consumed ({}) must NEVER exceed budget ({}) even under concurrent stampede",
        r.consumed,
        budget_limit
    );
    assert_eq!(
        r.consumed, budget_limit,
        "All available budget must be utilized exactly"
    );
    assert_eq!(r.remaining, 0.0, "Remaining budget must be exactly 0");
    assert!(r.is_exhausted, "Resource must be marked as exhausted");
}

/// Validates that under concurrent contention, once the preferred paid arm's budget is exhausted,
/// predictions seamlessly fall back to the unconstrained free arm with zero failures and zero oversell.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_concurrent_exhaustion_fallback_to_free_arm() {
    let (wal, dir) = temp_paths("pacing_concurrency_fallback");
    let db = Arc::new(BanditDB::new(&wal, &dir));

    let mut costs = HashMap::new();
    costs.insert("paid".to_string(), 1.0);
    costs.insert("free".to_string(), 0.0);

    let budget_limit = 10.0;
    let pacing = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "limited_credits".to_string(),
            budget: budget_limit,
            horizon: 1_000,
            step_size: Some(0.001),
            lambda_max: Some(0.001),
            initial_lambda: Some(0.0),
            arm_costs: costs,
        }],
        adaptive: false,
    };

    db.add_campaign_pacing(
        "fallback_camp",
        arms(&["paid", "free"]),
        2,
        0.01, // Low alpha so mean reward estimate strictly drives arm ranking
        Algorithm::Linucb,
        None,
        None,
        Some(pacing),
    )
    .await
    .unwrap();

    // Reward "paid" with 1.0 and "free" with 0.0 so "paid" is strictly preferred.
    let filter_paid = banditdb::engine::ArmFilter::include(vec!["paid".to_string()]);
    let (_, iid_p) = db
        .predict_filtered("fallback_camp", vec![1.0, 0.0], &filter_paid)
        .unwrap();
    db.reward(&iid_p, 1.0).await.unwrap();

    let filter_free = banditdb::engine::ArmFilter::include(vec!["free".to_string()]);
    let (_, iid_f) = db
        .predict_filtered("fallback_camp", vec![1.0, 0.0], &filter_free)
        .unwrap();
    db.reward(&iid_f, 0.0).await.unwrap();

    let num_tasks = 40;
    let requests_per_task = 5;
    let mut handles: Vec<JoinHandle<Vec<String>>> = Vec::new();

    // 40 threads * 5 requests = 200 requests. 10.0 budget total (1 already consumed in setup).
    for _ in 0..num_tasks {
        let db_clone = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            let mut chosen_arms = Vec::new();
            for _ in 0..requests_per_task {
                let (chosen, _) = db_clone.predict("fallback_camp", vec![1.0, 0.0]).unwrap();
                chosen_arms.push(chosen);
            }
            chosen_arms
        }));
    }

    let mut all_chosen = Vec::new();
    for h in handles {
        let arms = h.await.unwrap();
        all_chosen.extend(arms);
    }

    let paid_count = all_chosen.iter().filter(|&a| a == "paid").count();
    let free_count = all_chosen.iter().filter(|&a| a == "free").count();

    // Setup consumed 1.0, so exactly 9 more requests got "paid", and 191 got "free"
    assert_eq!(
        paid_count, 9,
        "Exactly 9 remaining paid requests should succeed"
    );
    assert_eq!(
        free_count, 191,
        "All remaining requests should smoothly fallback to free"
    );

    let report = db.campaign_pacing_report("fallback_camp").unwrap().unwrap();
    let r = &report.resources[0];

    assert_eq!(
        r.consumed, budget_limit,
        "Total consumed must exactly equal budget limit"
    );
    assert_eq!(r.remaining, 0.0);
    assert!(r.is_exhausted);
}
