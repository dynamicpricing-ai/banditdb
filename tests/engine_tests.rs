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
    db.add_campaign("convergence", vec!["arm".to_string()], 2);

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
    db.add_campaign("dim_test", vec!["a".to_string(), "b".to_string()], 2);

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

/// Original learning test: the engine must converge to context-specific arm selection
/// after 50 training rounds of positive reinforcement.
#[tokio::test]
async fn test_bandit_learns_context() {
    let wal = "/tmp/banditdb_test_learns_context.jsonl";
    let _ = std::fs::remove_file(wal);

    let db = BanditDB::new(wal, "/tmp");
    db.add_campaign("homepage", vec!["layout_a".to_string(), "layout_b".to_string()], 2);

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
