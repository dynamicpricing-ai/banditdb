use crate::state::ArmState;
use ndarray::{Array1, Array2, Axis};

/// Cholesky decomposition. Manual O(d³); used only by checkpoint decay.
pub(crate) fn cholesky(m: &Array2<f64>) -> Array2<f64> {
    let n = m.shape()[0];
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = s.max(0.0).sqrt();
            } else if l[[j, j]] > 1e-10 {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    l
}

/// Solve `M·X = B` for symmetric positive-definite `M` via Cholesky.
/// Returns `None` if `M` is not positive definite.
fn spd_solve(m: &Array2<f64>, b: &Array2<f64>) -> Option<Array2<f64>> {
    let n = m.shape()[0];
    let l = cholesky(m);
    if !(0..n).all(|i| l[[i, i]].is_finite() && l[[i, i]] > 1e-10) {
        return None;
    }
    // Forward substitution: L·Y = B, one row of Y at a time.
    let mut y = b.clone();
    for i in 0..n {
        let mut row = y.row(i).to_owned();
        for k in 0..i {
            row.scaled_add(-l[[i, k]], &y.row(k));
        }
        row /= l[[i, i]];
        y.row_mut(i).assign(&row);
    }
    // Back substitution: Lᵀ·X = Y.
    let mut x = y;
    for i in (0..n).rev() {
        let mut row = x.row(i).to_owned();
        for k in (i + 1)..n {
            row.scaled_add(-l[[k, i]], &x.row(k));
        }
        row /= l[[i, i]];
        x.row_mut(i).assign(&row);
    }
    Some(x)
}

impl ArmState {
    pub fn score(&self, context: &Array1<f64>, alpha: f64) -> f64 {
        let expected_reward = self.theta.dot(context);
        // Clamp to 0.0: collinear updates push f64 variance slightly negative → NaN in sqrt.
        let variance = context.dot(&self.a_inv.dot(context)).max(0.0);
        expected_reward + alpha * variance.sqrt()
    }

    /// Mean and standard deviation of this arm's reward estimate at `context`:
    /// `θ·x` and `sqrt(xᵀA⁻¹x)`.
    pub fn posterior(&self, context: &Array1<f64>) -> (f64, f64) {
        let mean = self.theta.dot(context);
        // Clamp to 0.0: collinear updates push f64 variance slightly negative → NaN in sqrt.
        let variance = context.dot(&self.a_inv.dot(context)).max(0.0);
        (mean, variance.sqrt())
    }

    /// Thompson Sampling score: sample θ̃ ~ N(θ, v²·A⁻¹) and return θ̃·x.
    ///
    /// θ̃·x is a linear function of a Gaussian vector, so it is itself Gaussian:
    /// `N(θ·x, v²·xᵀA⁻¹x)`. One scalar normal draw therefore has exactly the
    /// distribution of drawing the whole d-dimensional θ̃ and projecting it, with
    /// no Cholesky factor of A⁻¹ — which cost O(d³) after every reward and O(d²)
    /// per draw.
    pub fn score_ts(&self, context: &Array1<f64>, v: f64) -> f64 {
        use rand_distr::{Distribution, StandardNormal};
        let (mean, sd) = self.posterior(context);
        let z: f64 = StandardNormal.sample(&mut rand::thread_rng());
        mean + v * sd * z
    }

    /// Exponential forgetting with discount `gamma` ∈ (0, 1), as in D-LinUCB
    /// (Russac et al. 2019): observations are discounted, the ridge prior is not.
    ///
    /// ```text
    /// A' = γ·A + (1 − γ)·I        b' = γ·b
    /// ```
    ///
    /// Every direction forgets toward the cold prior (`A⁻¹ = I`, θ = 0) and stops
    /// there: an arm can lose everything it observed, but never becomes less
    /// certain than an arm that has observed nothing. Discounting the prior as well
    /// (`A⁻¹/γ`) has no such limit — an arm that receives no rewards grew by `1/γ`
    /// at every checkpoint until `A⁻¹` overflowed to infinity.
    ///
    /// Only `A⁻¹` is stored, so the update is applied as
    /// `A'⁻¹ = (γ·I + (1 − γ)·A⁻¹)⁻¹ · A⁻¹`. The matrix being inverted has
    /// eigenvalues in `[γ, 1]`, so it is well conditioned. Cost is O(d³) per arm,
    /// paid once per checkpoint.
    ///
    /// `gamma` is the discount factor. It is unrelated to the pacing shadow price,
    /// the neural L2 weight and the warm-start strength, all of which are λ.
    pub fn decay(&mut self, gamma: f64) {
        if !gamma.is_finite() || gamma <= 0.0 || gamma >= 1.0 {
            return;
        }
        let n = self.theta.len();
        let mut m = &self.a_inv * (1.0 - gamma);
        for i in 0..n {
            m[[i, i]] += gamma;
        }
        let Some(a_inv) = spd_solve(&m, &self.a_inv) else {
            // A⁻¹ is already damaged (not positive semi-definite, or non-finite).
            // Leave the arm as it is rather than make it worse.
            return;
        };
        let a_inv = (&a_inv + &a_inv.t()) * 0.5;
        let b = &self.b * gamma;
        let theta = a_inv.dot(&b);
        if !a_inv.iter().all(|v| v.is_finite()) || !theta.iter().all(|v| v.is_finite()) {
            return;
        }
        self.a_inv = a_inv;
        self.b = b;
        self.theta = theta;
    }

    pub fn update(&mut self, context: &Array1<f64>, reward: f64) {
        // Non-finite reward causes Inf - Inf = NaN in subsequent dot products. Reject early.
        if !reward.is_finite() {
            return;
        }

        let a_inv_x = self.a_inv.dot(context);
        let x_a_inv_x = context.dot(&a_inv_x);
        let denom = 1.0 + x_a_inv_x;
        // A healthy A⁻¹ is positive semi-definite, so denom ≥ 1. Anything else means
        // the matrix is already damaged or the product overflowed; applying the
        // update would write NaN into the arm, which no later update can clear.
        if !denom.is_finite() || denom <= 0.0 {
            return;
        }
        let col = a_inv_x.clone().insert_axis(Axis(1));
        let row = a_inv_x.clone().insert_axis(Axis(0));
        let numerator = col.dot(&row);

        let a_inv = &self.a_inv - &(numerator / denom);

        // Enforce symmetry to prevent numeric drift from accumulated rank-1 updates.
        // A_inv is theoretically symmetric; this corrects f64 rounding that would
        // otherwise cause score() variance and cholesky() to diverge from the true posterior.
        let a_inv = (&a_inv + &a_inv.t()) * 0.5;

        let b = &self.b + &(context * reward);
        let theta = a_inv.dot(&b);
        // Commit only a finite result: a skipped reward costs one observation, a
        // NaN costs the arm.
        if !theta.iter().all(|v| v.is_finite()) || !a_inv.diag().iter().all(|v| v.is_finite()) {
            return;
        }
        self.a_inv = a_inv;
        self.b = b;
        self.theta = theta;

        use std::sync::atomic::Ordering;
        self.reward_count.fetch_add(1, Ordering::Relaxed);

        // CAS loop: f64 total_reward stored as bit-cast AtomicU64.
        let mut current_bits = self.total_reward.load(Ordering::Relaxed);
        loop {
            let current_val = f64::from_bits(current_bits);
            let new_val = current_val + reward;
            let new_bits = new_val.to_bits();
            match self.total_reward.compare_exchange_weak(
                current_bits,
                new_bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_bits = actual,
            }
        }
    }
}
