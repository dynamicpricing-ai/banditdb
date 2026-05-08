use crate::state::ArmState;
use ndarray::{Array1, Array2, Axis};

/// Cholesky decomposition.
/// Currently a manual O(d^3) implementation. 
/// TODO: Integrate ndarray-linalg with a BLAS backend for d > 100.
pub(crate) fn cholesky(m: &Array2<f64>) -> Array2<f64> {
    let n = m.shape()[0];
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[[i, j]];
            for k in 0..j { s -= l[[i, k]] * l[[j, k]]; }
            if i == j {
                l[[i, j]] = s.max(0.0).sqrt();
            } else if l[[j, j]] > 1e-10 {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    l
}

impl ArmState {
    pub fn score(&self, context: &Array1<f64>, alpha: f64) -> f64 {
        let expected_reward = self.theta.dot(context);
        // Clamp to 0.0: collinear updates push f64 variance slightly negative → NaN in sqrt.
        let variance = context.dot(&self.a_inv.dot(context)).max(0.0);
        expected_reward + alpha * variance.sqrt()
    }

    /// Thompson Sampling score: sample θ̃ ~ N(θ, v²·A⁻¹) and return θ̃·x.
    /// The Cholesky factor L is computed once after each `update()` and cached;
    /// subsequent calls in the same scoring pass reuse it (O(d²) instead of O(d³)).
    pub fn score_ts(&self, context: &Array1<f64>, v: f64) -> f64 {
        use rand::thread_rng;
        use rand_distr::{Distribution, StandardNormal};
        let d = self.theta.len();

        // Acquire the per-arm cache lock; compute L only if invalidated by a prior update.
        let l = {
            let mut guard = self.chol_cache.lock();
            if guard.is_none() {
                *guard = Some(cholesky(&self.a_inv));
            }
            guard.as_ref().unwrap().clone()
        };

        let mut rng = thread_rng();
        let z = Array1::from_iter((0..d).map(|_| StandardNormal.sample(&mut rng)));
        (&self.theta + &(v * l.dot(&z))).dot(context)
    }

    pub fn update(&mut self, context: &Array1<f64>, reward: f64) {
        // Non-finite reward causes Inf - Inf = NaN in subsequent dot products. Reject early.
        if !reward.is_finite() {
            return;
        }

        let a_inv_x   = self.a_inv.dot(context);
        let x_a_inv_x = context.dot(&a_inv_x);
        let col       = a_inv_x.clone().insert_axis(Axis(1));
        let row       = a_inv_x.clone().insert_axis(Axis(0));
        let numerator = col.dot(&row);

        self.a_inv = &self.a_inv - &(numerator / (1.0 + x_a_inv_x));

        // Enforce symmetry to prevent numeric drift from accumulated rank-1 updates.
        // A_inv is theoretically symmetric; this corrects f64 rounding that would
        // otherwise cause score() variance and cholesky() to diverge from the true posterior.
        self.a_inv = (&self.a_inv + &self.a_inv.t()) * 0.5;

        self.b     = &self.b + &(context * reward);
        self.theta = self.a_inv.dot(&self.b);

        // Invalidate Cholesky cache — recomputed lazily on next score_ts call.
        *self.chol_cache.lock() = None;

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
