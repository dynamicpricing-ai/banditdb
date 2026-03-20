use crate::state::ArmState;
use ndarray::{Array1, Array2, Axis};

fn cholesky(m: &Array2<f64>) -> Array2<f64> {
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
    /// Predict the UCB score for a given context
    pub fn score(&self, context: &Array1<f64>, alpha: f64) -> f64 {
        // 1. Exploit: Expected reward (Theta dot Context)
        let expected_reward = self.theta.dot(context);
        
        // 2. Explore: Variance/Uncertainty bound
        // Clamp to 0.0: after many collinear updates, f64 rounding can push variance
        // slightly negative, causing sqrt(negative) = NaN.
        let variance = context.dot(&self.a_inv.dot(context)).max(0.0);
        let exploration_bound = alpha * variance.sqrt();
        
        expected_reward + exploration_bound
    }

    /// Thompson Sampling score: sample θ̃ ~ N(θ, v²·A⁻¹) and return θ̃·x
    pub fn score_ts(&self, context: &Array1<f64>, v: f64) -> f64 {
        use rand::thread_rng;
        use rand_distr::{Distribution, StandardNormal};
        let d = self.theta.len();
        let l = cholesky(&self.a_inv);
        let mut rng = thread_rng();
        let z = Array1::from_iter((0..d).map(|_| StandardNormal.sample(&mut rng)));
        let theta_tilde = &self.theta + &(v * l.dot(&z));
        theta_tilde.dot(context)
    }

    /// Sherman-Morrison Rank-1 Update (Instant Learning)
    pub fn update(&mut self, context: &Array1<f64>, reward: f64) {
        // Guard: a non-finite reward (Inf, NaN) causes b = [Inf,...], then theta = A_inv * b.
        // With off-diagonal terms in A_inv, this produces Inf - Inf = NaN. Reject early.
        if !reward.is_finite() {
            return;
        }

        let a_inv_x = self.a_inv.dot(context);
        let x_a_inv_x = context.dot(&a_inv_x);
        
        // Outer product to get the numerator matrix
        let col = a_inv_x.clone().insert_axis(Axis(1));
        let row = a_inv_x.clone().insert_axis(Axis(0));
        let numerator = col.dot(&row);
        
        // Update A^-1 natively without full inversion
        self.a_inv = &self.a_inv - &(numerator / (1.0 + x_a_inv_x));
        
        // Update b vector
        self.b = &self.b + &(context * reward);
        
        // Update cached weights
        self.theta = self.a_inv.dot(&self.b);
        self.reward_count += 1;
        self.total_reward += reward;
    }
}
