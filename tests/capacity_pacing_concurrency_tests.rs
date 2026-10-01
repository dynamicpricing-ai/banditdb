use banditdb::BanditDB;
use banditdb::engine::ArmFilter;
use banditdb::state::{Algorithm, PacingConfig, ResourceConstraint};
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
            name:           "compute".to_string(),
            budget:         100_000.0,
            horizon:        100_000,
            step_size:      Some(0.01),
            lambda_max:     Some(10.0),
            initial_lambda: Some(0.0),
            arm_costs:      costs,
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
    ).await.unwrap();

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
                let (_, iid) = db_clone.predict_filtered("concurrent_camp", vec![1.0, 0.0], &filter).unwrap();
                db_clone.reward(&iid, 1.0).await.unwrap();
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let report = db.campaign_pacing_report("concurrent_camp").unwrap().unwrap();
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
