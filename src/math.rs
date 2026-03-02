use crate::state::ArmState;
use ndarray::{Array1, Axis};

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
    }
}
