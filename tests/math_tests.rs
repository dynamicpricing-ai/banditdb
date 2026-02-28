use banditdb::state::ArmState;
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Frobenius norm of (matrix − Identity). Measures how far a matrix is from I.
fn frobenius_error_from_identity(m: &Array2<f64>) -> f64 {
    let identity = Array2::<f64>::eye(m.nrows());
    let diff = m - &identity;
    diff.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Build the ground-truth A matrix independently from scratch.
/// A = Identity + Σ(x_i · x_i^T) for each context x_i.
fn build_a_matrix(dim: usize, contexts: &[Array1<f64>]) -> Array2<f64> {
    let mut a = Array2::<f64>::eye(dim);
    for ctx in contexts {
        for i in 0..dim {
            for j in 0..dim {
                a[[i, j]] += ctx[i] * ctx[j];
            }
        }
    }
    a
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test 1.1 — Sherman-Morrison Invariant
///
/// The master correctness proof: after any sequence of rank-1 updates, A_inv * A must
/// equal the Identity matrix. We rebuild A from scratch independently and verify the
/// product via Frobenius norm. If this holds, the entire update formula is correct —
/// no other matrix test is needed.
#[test]
fn test_1_1_sherman_morrison_invariant() {
    let dim = 3;
    let mut arm = ArmState::new(dim);

    let contexts = vec![
        Array1::from_vec(vec![1.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0, 0.5]),
        Array1::from_vec(vec![0.7, 0.3, 1.0]),
        Array1::from_vec(vec![-0.5, 0.8, -0.2]),
        Array1::from_vec(vec![1.0, 1.0, 1.0]),
    ];

    for ctx in &contexts {
        arm.update(ctx, 1.0);
    }

    let a_true = build_a_matrix(dim, &contexts);
    let product = arm.a_inv.dot(&a_true);
    let error = frobenius_error_from_identity(&product);

    assert!(
        error < 1e-10,
        "A_inv * A deviated from Identity (Frobenius error: {:.2e}). Sherman-Morrison invariant broken.",
        error
    );
}

/// Test 1.3 — Exploration Monotonicity
///
/// A fresh arm has maximum uncertainty. After N updates the UCB exploration bonus
/// must strictly decrease. We use reward=0.0 to hold expected_reward at zero,
/// isolating the variance term as the only moving part of the score.
#[test]
fn test_1_3_exploration_monotonicity() {
    let dim = 2;
    let mut arm = ArmState::new(dim);
    let context = Array1::from_vec(vec![1.0, 0.0]);
    let alpha = 1.0;

    let score_before = arm.score(&context, alpha);

    for _ in 0..100 {
        arm.update(&context, 0.0);
    }

    let score_after = arm.score(&context, alpha);

    assert!(
        score_after < score_before,
        "Exploration bonus failed to shrink after 100 updates: before={:.4}, after={:.4}",
        score_before,
        score_after
    );
}

/// Test 1.5 — Collinearity Does Not Produce NaN
///
/// Flooding with one identical context vector drives A_inv toward a singular matrix.
/// Due to f64 rounding, x^T * A_inv * x can drift slightly negative after ~100k
/// iterations. sqrt(negative) = NaN, poisoning all future scores.
/// Fix: variance.max(0.0) before sqrt in score().
#[test]
fn test_1_5_collinearity_no_nan() {
    let dim = 2;
    let mut arm = ArmState::new(dim);
    let context = Array1::from_vec(vec![1.0, 1.0]);

    for _ in 0..100_000 {
        arm.update(&context, 1.0);
    }

    let score = arm.score(&context, 1.0);

    assert!(
        score.is_finite(),
        "Score became NaN/Inf after 100k collinear updates: {}. variance.max(0.0) fix missing.",
        score
    );
    assert!(
        arm.theta.iter().all(|x| x.is_finite()),
        "Theta was corrupted after collinear flooding: {:?}",
        arm.theta
    );
}

/// Test 1.6 — Infinite Reward Does Not Corrupt Theta
///
/// reward=f64::INFINITY causes b = context * Inf = [Inf, Inf].
/// With off-diagonal terms in A_inv, theta = A_inv * b produces Inf − Inf = NaN.
/// This is the f64 overflow path to the same NaN bug as Test 1.5, reached via
/// the reward argument rather than collinear context flooding.
/// Fix: early return in update() if !reward.is_finite().
#[test]
fn test_1_6_infinite_reward_no_nan() {
    let dim = 2;
    let mut arm = ArmState::new(dim);
    let context = Array1::from_vec(vec![1.0, 1.0]);

    // One normal update creates off-diagonal terms in A_inv that trigger
    // Inf − Inf = NaN if a subsequent Inf reward is not rejected.
    arm.update(&context, 1.0);
    let theta_before = arm.theta.clone();

    arm.update(&context, f64::INFINITY);

    assert!(
        arm.score(&context, 1.0).is_finite(),
        "Score became NaN after f64::INFINITY reward — is_finite guard missing in update()."
    );
    assert_eq!(
        arm.theta, theta_before,
        "Theta was mutated by a non-finite reward. update() must reject it before touching state."
    );
}
