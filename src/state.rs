use ndarray::{Array1, Array2};
use parking_lot::RwLock;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

fn default_none_map() -> Option<HashMap<String, f64>> {
    None
}

pub const DEFAULT_ALPHA: f64 = 1.0;

fn default_alpha() -> f64 {
    DEFAULT_ALPHA
}

// ---------------------------------------------------------------------------
// Typed engine error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum EngineError {
    NotFound(String),
    AlreadyExists(String),
    Archived(String),
    WalFull,
    WalUnavailable,
    BadRequest(String),
    /// A configured quota refused the request. Distinct from `BadRequest`: the
    /// request is well formed and would succeed against an instance with more
    /// headroom, so the caller needs the limit and the current usage, not a
    /// validation message.
    LimitExceeded(String),
    Internal(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::NotFound(m) => write!(f, "{m}"),
            EngineError::AlreadyExists(m) => write!(f, "{m}"),
            EngineError::Archived(m) => write!(f, "{m}"),
            EngineError::WalFull => write!(f, "wal:full — server is busy, retry momentarily"),
            EngineError::WalUnavailable => {
                write!(f, "wal:unavailable — storage error, check server logs")
            }
            EngineError::BadRequest(m) => write!(f, "{m}"),
            EngineError::LimitExceeded(m) => write!(f, "{m}"),
            EngineError::Internal(m) => write!(f, "{m}"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NeuralLinUCBConfig {
    pub context_dim: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub hidden_layers: usize,
    pub retrain_every: usize,
    pub retrain_steps: usize,
    pub learning_rate: f64,
    pub lambda: f64,
}

impl Default for NeuralLinUCBConfig {
    fn default() -> Self {
        Self {
            context_dim: 64,
            embed_dim: 32,
            hidden_dim: 128,
            hidden_layers: 2,
            retrain_every: 200,
            retrain_steps: 100,
            learning_rate: 1e-3,
            lambda: 1.0,
        }
    }
}

/// Configuration for the Progressive self-tuning tournament.
///
/// Progressive runs a base model and a challenger in parallel ("shadow learning").
/// Every reward updates both models. At each checkpoint the engine evaluates both
/// with SNIPS (Self-Normalised Importance-Weighted Policy Evaluation). If the
/// challenger wins `required_wins` consecutive checkpoints by more than 10%, one
/// traffic step (`step_bps`) shifts toward the challenger — and vice-versa for
/// the base. Traffic ramps gradually; it never jumps more than `step_bps` per
/// checkpoint, and it never drops below 10% (exploration) or above 90%.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProgressiveConfig {
    pub base: Box<Algorithm>,
    pub challenger: Box<Algorithm>,
    /// Minimum buffer entries per arm required before any traffic shift fires.
    #[serde(default = "default_progressive_min_obs")]
    pub min_obs: usize,
    /// Consecutive checkpoint wins required to earn one traffic step.
    #[serde(default = "default_progressive_required_wins")]
    pub required_wins: usize,
    /// Traffic change per confirmed win run, in basis points (1000 = 10%).
    #[serde(default = "default_progressive_step_bps")]
    pub step_bps: u32,
}

fn default_progressive_min_obs() -> usize {
    100
}
fn default_progressive_required_wins() -> usize {
    3
}
fn default_progressive_step_bps() -> u32 {
    1000
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum Algorithm {
    #[default]
    Linucb,
    ThompsonSampling,
    #[serde(rename = "neural_lin_ucb")]
    NeuralLinUCB(NeuralLinUCBConfig),
    /// Neural Thompson Sampling (Zhang et al. 2021).
    /// Same MLP embedding and retrain procedure as NeuralLinUCB, but exploration
    /// uses Thompson Sampling draws (w ~ N(θ, σ²A⁻¹)) instead of UCB bounds.
    /// Better long-run convergence; higher early regret than NeuralLinUCB.
    #[serde(rename = "neural_thompson_sampling")]
    NeuralThompsonSampling(NeuralLinUCBConfig),
    Progressive(ProgressiveConfig),
}

/// Outcome of a single tournament SNIPS evaluation round, returned by the extracted
/// evaluate_tournament helper so the checkpoint loop can act on it without 7-level nesting.
#[cfg(feature = "neural")]
pub enum TournamentOutcome {
    /// Not enough data yet — hold current traffic split.
    Hold,
    /// Challenger won `required_wins` consecutive rounds — promote by one step_bps.
    ChallengerStep(u32),
    /// Base won `required_wins` consecutive rounds — demote by one step_bps.
    BaseStep(u32),
    /// Neither side exceeded the margin — streak decayed, no traffic change.
    Inconclusive,
}

/// Lifecycle state of a single arm.
///
/// Exclusion is always soft: a paused or retired arm keeps its matrices and keeps
/// learning from rewards. Predictions made before the pause are still in flight and
/// their rewards still arrive; dropping them would throw away data the arm paid for.
/// It also matters for the neural path, where `retrain` looks every buffered
/// interaction's arm up by id — hard-deleting an arm would strand those entries.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ArmStatus {
    /// Eligible for selection.
    #[default]
    Active,
    /// Not selectable, still learns. Reversible; the intended state for
    /// "out of stock", "creative paused", "seasonal".
    Paused,
    /// Not selectable, still absorbs in-flight rewards. Semantically permanent,
    /// but restorable — nothing is deleted.
    Retired,
}

impl ArmStatus {
    pub fn as_u8(self) -> u8 {
        match self {
            ArmStatus::Active => 0,
            ArmStatus::Paused => 1,
            ArmStatus::Retired => 2,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => ArmStatus::Paused,
            2 => ArmStatus::Retired,
            _ => ArmStatus::Active,
        }
    }
}

/// A materialised warm-start prior for a new arm: ridge regression centred on
/// `mean` instead of on zero, with `strength` pseudo-observations of pull.
///
/// The mean is resolved **when the arm is added** and written into the WAL, never
/// recomputed at replay. Replay starts from a checkpoint, so the source arms' θ at
/// that point in the log are not the θ the original call saw — re-resolving would
/// make recovery diverge from the live state.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ArmPrior {
    pub mean: Vec<f64>,
    pub strength: f64,
}

/// How a newly added arm should borrow from the arms that already exist.
///
/// Request-level input, resolved to an [`ArmPrior`] by the engine. `strength` is in
/// pseudo-observations: 1.0 leaves the new arm exactly as uncertain as a cold one
/// (so it still gets explored) while starting its estimate at the borrowed mean.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(tag = "from", rename_all = "snake_case")]
pub enum WarmStart {
    /// Cold start: θ = 0, A⁻¹ = I. The historical behaviour.
    #[default]
    None,
    /// Mean θ over every active arm in the campaign.
    Population {
        #[serde(default = "default_warm_start_strength")]
        strength: f64,
    },
    /// Mean θ over the active arms sharing the new arm's `group`.
    Group {
        #[serde(default = "default_warm_start_strength")]
        strength: f64,
    },
    /// Mean θ over an explicit list of arms, whatever their status.
    Arms {
        arms: Vec<String>,
        #[serde(default = "default_warm_start_strength")]
        strength: f64,
    },
}

pub fn default_warm_start_strength() -> f64 {
    1.0
}

/// Upper bound on prior strength. Past this the prior is numerically
/// indistinguishable from a frozen arm: A⁻¹ = I/λ leaves no exploration bonus.
pub const MAX_WARM_START_STRENGTH: f64 = 1e6;

#[derive(Debug)]
pub struct ArmState {
    pub a_inv: Array2<f64>,
    pub b: Array1<f64>,
    pub theta: Array1<f64>,
    /// Selection eligibility. Atomic so pausing an arm needs only a read lock on
    /// the arms map, never a write lock that would block the prediction path.
    pub status: AtomicU8,
    /// Optional hierarchy label. Arms in a group lend their θ to new arms in the
    /// same group via `WarmStart::Group`.
    pub group: Option<String>,
    /// Cached Cholesky factor L (A_inv = L·Lᵀ) for Thompson Sampling.
    /// Computed lazily on the first `score_ts` call after each `update`, then
    /// reused until the next update invalidates it. LinUCB never touches this.
    pub chol_cache: parking_lot::Mutex<Option<Array2<f64>>>,
    pub prediction_count: AtomicU64,
    pub reward_count: AtomicU64,
    pub total_reward: AtomicU64,
}

impl Clone for ArmState {
    fn clone(&self) -> Self {
        Self {
            a_inv: self.a_inv.clone(),
            b: self.b.clone(),
            theta: self.theta.clone(),
            status: AtomicU8::new(self.status.load(Ordering::Relaxed)),
            group: self.group.clone(),
            chol_cache: parking_lot::Mutex::new(None), // don't copy stale cache
            prediction_count: AtomicU64::new(self.prediction_count.load(Ordering::Relaxed)),
            reward_count: AtomicU64::new(self.reward_count.load(Ordering::Relaxed)),
            total_reward: AtomicU64::new(self.total_reward.load(Ordering::Relaxed)),
        }
    }
}

impl Serialize for ArmState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Shadow {
            a_inv: Array2<f64>,
            b: Array1<f64>,
            theta: Array1<f64>,
            status: ArmStatus,
            #[serde(skip_serializing_if = "Option::is_none")]
            group: Option<String>,
            prediction_count: u64,
            reward_count: u64,
            total_reward: f64,
        }
        let shadow = Shadow {
            a_inv: self.a_inv.clone(),
            b: self.b.clone(),
            theta: self.theta.clone(),
            status: ArmStatus::from_u8(self.status.load(Ordering::Relaxed)),
            group: self.group.clone(),
            prediction_count: self.prediction_count.load(Ordering::Relaxed),
            reward_count: self.reward_count.load(Ordering::Relaxed),
            total_reward: f64::from_bits(self.total_reward.load(Ordering::Relaxed)),
        };
        shadow.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ArmState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Shadow {
            a_inv: Array2<f64>,
            b: Array1<f64>,
            #[allow(dead_code)]
            // stored value is intentionally ignored; theta is recomputed from a_inv·b
            theta: Array1<f64>,
            // Checkpoints written before arm lifecycle existed have neither field;
            // those arms load as active and ungrouped, which is what they were.
            #[serde(default)]
            status: ArmStatus,
            #[serde(default)]
            group: Option<String>,
            #[serde(default)]
            prediction_count: u64,
            #[serde(default)]
            reward_count: u64,
            #[serde(default)]
            total_reward: f64,
        }
        let shadow = Shadow::deserialize(deserializer)?;
        let theta = shadow.a_inv.dot(&shadow.b);
        Ok(Self {
            a_inv: shadow.a_inv,
            b: shadow.b,
            theta,
            status: AtomicU8::new(shadow.status.as_u8()),
            group: shadow.group,
            chol_cache: parking_lot::Mutex::new(None), // recomputed lazily on first score_ts
            prediction_count: AtomicU64::new(shadow.prediction_count),
            reward_count: AtomicU64::new(shadow.reward_count),
            total_reward: AtomicU64::new(shadow.total_reward.to_bits()),
        })
    }
}

impl ArmState {
    pub fn new(dim: usize) -> Self {
        Self {
            a_inv: Array2::eye(dim),
            b: Array1::zeros(dim),
            theta: Array1::zeros(dim),
            status: AtomicU8::new(ArmStatus::Active.as_u8()),
            group: None,
            chol_cache: parking_lot::Mutex::new(None),
            prediction_count: AtomicU64::new(0),
            reward_count: AtomicU64::new(0),
            total_reward: AtomicU64::new(0.0f64.to_bits()),
        }
    }

    /// Cold start with the ridge prior centred on `mean` instead of on zero.
    ///
    /// The default state is already a prior — ridge regression with precision
    /// A = I and mean 0. Shifting the centre needs no new math, only a different
    /// starting point:
    ///
    /// ```text
    /// A⁻¹ = I / λ     b = λ·μ₀     ⇒  θ = A⁻¹b = μ₀
    /// ```
    ///
    /// Sherman-Morrison, scoring, Thompson sampling and checkpoint decay all keep
    /// working unchanged — the prior is just λ pseudo-observations that real data
    /// outweighs as it arrives. At λ = 1 the new arm carries a cold arm's
    /// uncertainty, so it is still explored; it merely starts from a sensible guess
    /// rather than from zero.
    pub fn with_prior(dim: usize, mean: &Array1<f64>, strength: f64) -> Self {
        let mut state = Self::new(dim);
        state.a_inv = Array2::eye(dim) / strength;
        state.b = mean * strength;
        state.theta = mean.clone();
        state
    }

    pub fn status(&self) -> ArmStatus {
        ArmStatus::from_u8(self.status.load(Ordering::Relaxed))
    }

    pub fn set_status(&self, status: ArmStatus) {
        self.status.store(status.as_u8(), Ordering::Relaxed);
    }

    /// Eligible for selection. Paused and retired arms keep learning either way.
    pub fn is_active(&self) -> bool {
        self.status() == ArmStatus::Active
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InteractionRecord {
    pub campaign_id: String,
    pub arm_id: String,
    pub context: Array1<f64>,
    #[serde(default)]
    pub arm_propensities: Option<HashMap<String, f64>>,
    #[serde(default)]
    pub timestamp_secs: u64,
    /// False when this prediction's WAL record was dropped under load, so replay
    /// cannot find it and its reward must carry it instead. Not persisted: a
    /// record restored from a checkpoint is recoverable from that checkpoint.
    #[serde(skip, default = "logged_default")]
    pub logged: bool,
}

fn logged_default() -> bool {
    true
}

/// One completed prediction→reward pair, ready to write as a flat Parquet row.
#[derive(Debug)]
pub struct CompletedInteraction {
    pub interaction_id: String,
    pub arm_id: String,
    pub context: Vec<f64>,
    pub reward: f64,
    pub predicted_at: u64,
    pub rewarded_at: u64,
    /// Propensity of the chosen arm under the logging policy.
    /// LinUCB: softmax-normalised UCB score.
    /// Thompson Sampling: adaptive Monte Carlo frequency (N=8–64, driven by A_inv diagonal).
    pub propensity: Option<f64>,
}

// ---------------------------------------------------------------------------
// Capacity Constraints & Lagrangian Pacing
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ResourceConstraint {
    pub name: String,
    /// Total resource budget over the horizon window (B_j > 0).
    pub budget: f64,
    /// Number of decisions in this window (T > 0). Default: 10,000.
    #[serde(default = "default_pacing_horizon")]
    pub horizon: u64,
    /// Dual gradient descent step size (eta). Default: auto-tuned from horizon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_size: Option<f64>,
    /// Maximum shadow price multiplier (lambda_max). Default: auto-tuned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lambda_max: Option<f64>,
    /// Initial shadow price multiplier (lambda_0). Default: 0.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_lambda: Option<f64>,
    /// Per-arm resource consumption (c_{j, a} >= 0). Default 0.0 for unlisted arms.
    #[serde(default)]
    pub arm_costs: HashMap<String, f64>,
}

pub fn default_pacing_horizon() -> u64 {
    10_000
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PacingConfig {
    pub resources: Vec<ResourceConstraint>,
    /// Whether to use adaptive target pacing rate (B_rem / T_rem) vs uniform (B / T).
    /// Default: true (recommended baseline).
    #[serde(default = "default_pacing_adaptive")]
    pub adaptive: bool,
}

pub fn default_pacing_adaptive() -> bool {
    true
}

/// Live runtime state for an individual constrained resource knapsack.
#[derive(Debug)]
pub struct ResourceState {
    pub name: String,
    pub budget: f64,
    pub horizon: u64,
    pub step_size: f64,
    pub lambda_max: f64,
    pub arm_costs: RwLock<HashMap<String, f64>>,
    pub consumed: AtomicU64,  // f64::to_bits
    pub lambda: AtomicU64,    // f64::to_bits
    pub decisions: AtomicU64, // total decisions evaluated in this window
}

impl ResourceState {
    pub fn new(cfg: &ResourceConstraint) -> Self {
        let budget = cfg.budget.max(1e-6);
        let horizon = cfg.horizon.max(1);

        // If step_size is not specified, derive variance-normalized step size:
        // eta = 2.0 / sqrt(horizon) (Section 5.2 / Section 4.6 of lagrangian_pacing_mathematics.md)
        let step_size = cfg
            .step_size
            .unwrap_or_else(|| (2.0 / (horizon as f64).sqrt()).max(1e-5));

        let min_cost = cfg
            .arm_costs
            .values()
            .copied()
            .filter(|&c| c > 0.0)
            .fold(f64::INFINITY, f64::min);
        let lambda_max = cfg.lambda_max.unwrap_or_else(|| {
            if min_cost.is_finite() && min_cost > 0.0 {
                (1.0 / min_cost).max(1.0)
            } else {
                1.0
            }
        });

        let init_lambda = cfg.initial_lambda.unwrap_or(0.0).clamp(0.0, lambda_max);

        Self {
            name: cfg.name.clone(),
            budget,
            horizon,
            step_size,
            lambda_max,
            arm_costs: RwLock::new(cfg.arm_costs.clone()),
            consumed: AtomicU64::new(0.0f64.to_bits()),
            lambda: AtomicU64::new(init_lambda.to_bits()),
            decisions: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn cost(&self, arm_id: &str) -> f64 {
        self.arm_costs.read().get(arm_id).copied().unwrap_or(0.0)
    }

    pub fn set_arm_cost(&self, arm_id: &str, cost: f64) {
        self.arm_costs.write().insert(arm_id.to_string(), cost);
    }

    #[inline(always)]
    pub fn consumed(&self) -> f64 {
        f64::from_bits(self.consumed.load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub fn lambda(&self) -> f64 {
        f64::from_bits(self.lambda.load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub fn remaining_budget(&self) -> f64 {
        (self.budget - self.consumed()).max(0.0)
    }

    /// Hard feasibility mask M_t:
    /// Returns true if arm consumes this resource and remaining capacity is insufficient.
    #[inline(always)]
    pub fn is_masked(&self, arm_id: &str) -> bool {
        let c = self.cost(arm_id);
        c > 0.0 && self.remaining_budget() < c
    }

    /// Target consumption rate per decision epoch rho_{j, t}.
    pub fn target_rate(&self, adaptive: bool) -> f64 {
        self.target_rate_at(
            self.decisions.load(Ordering::Relaxed),
            self.remaining_budget(),
            adaptive,
        )
    }

    /// Compute target rate at a specific epoch `t` (number of decisions already taken).
    fn target_rate_at(&self, t: u64, remaining_budget: f64, adaptive: bool) -> f64 {
        if !adaptive {
            self.budget / (self.horizon.max(1) as f64)
        } else {
            let rem_h = self.horizon.saturating_sub(t);
            // Endgame freeze boundary:
            // When (T - t) < max(50, 0.05 * T), clamp remaining horizon to avoid singularity.
            let min_rem = (50u64).max((self.horizon as f64 * 0.05) as u64);
            let clamped_rem = rem_h.max(min_rem) as f64;
            remaining_budget / clamped_rem
        }
    }

    /// Dual projected gradient descent step on consumption:
    /// lambda_{j, t+1} = \Pi_{[0, lambda_max]} [ lambda_j + eta * (c_{j, a} - rho_{j, t}) ]
    pub fn record_and_update(&self, consumed_amount: f64, adaptive: bool) {
        // t = epoch BEFORE this decision (fetch_add returns the old value).
        let t = self.decisions.fetch_add(1, Ordering::Relaxed);

        // Pre-decision remaining budget for accurate target pacing rate calculation.
        let cur_consumed_bits = self.consumed.load(Ordering::Relaxed);
        let cur_consumed = f64::from_bits(cur_consumed_bits);
        let pre_remaining = (self.budget - cur_consumed).max(0.0);

        if consumed_amount > 0.0 {
            let mut cur = cur_consumed_bits;
            loop {
                let next = (f64::from_bits(cur) + consumed_amount).to_bits();
                match self.consumed.compare_exchange_weak(
                    cur,
                    next,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(b) => cur = b,
                }
            }
        }

        // Freeze dual multiplier in the endgame window to avoid singularity.
        let min_rem = (50u64).max((self.horizon as f64 * 0.05) as u64);
        if self.horizon > min_rem && t >= self.horizon - min_rem {
            return;
        }

        let rho = self.target_rate_at(t, pre_remaining, adaptive);
        let delta = consumed_amount - rho;
        let mut cur_lambda = self.lambda.load(Ordering::Relaxed);
        loop {
            let updated =
                (f64::from_bits(cur_lambda) + self.step_size * delta).clamp(0.0, self.lambda_max);
            match self.lambda.compare_exchange_weak(
                cur_lambda,
                updated.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(b) => cur_lambda = b,
            }
        }
    }

    pub fn rollback_consumption(&self, amount: f64) {
        if amount > 0.0 {
            let mut cur = self.consumed.load(Ordering::Relaxed);
            loop {
                let cur_f = f64::from_bits(cur);
                let next = (cur_f - amount).max(0.0).to_bits();
                match self.consumed.compare_exchange_weak(
                    cur,
                    next,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(b) => cur = b,
                }
            }
        }
    }
}

/// Live runtime state for all constraints on a campaign.
#[derive(Debug)]
pub struct PacingState {
    pub resources: Vec<ResourceState>,
    pub adaptive: bool,
}

impl PacingState {
    pub fn new(cfg: PacingConfig) -> Self {
        Self {
            resources: cfg.resources.iter().map(ResourceState::new).collect(),
            adaptive: cfg.adaptive,
        }
    }

    #[inline(always)]
    pub fn is_arm_masked(&self, arm_id: &str) -> bool {
        self.resources.iter().any(|r| r.is_masked(arm_id))
    }

    #[inline(always)]
    pub fn total_price(&self, arm_id: &str) -> f64 {
        self.resources
            .iter()
            .map(|r| r.lambda() * r.cost(arm_id))
            .sum()
    }

    pub fn record_consumption(&self, arm_id: &str) {
        for r in &self.resources {
            let cost = r.cost(arm_id);
            r.record_and_update(cost, self.adaptive);
        }
    }

    pub fn try_record_consumption(&self, arm_id: &str) -> bool {
        // Fast pre-check: if any resource is already masked, return false immediately.
        for r in &self.resources {
            let cost = r.cost(arm_id);
            if cost > 0.0 && r.remaining_budget() < cost {
                return false;
            }
        }

        let mut reserved: Vec<(&ResourceState, f64)> = Vec::with_capacity(self.resources.len());
        for r in &self.resources {
            let cost = r.cost(arm_id);
            if cost > 0.0 {
                let mut cur = r.consumed.load(Ordering::Relaxed);
                let mut ok = false;
                loop {
                    let cur_f = f64::from_bits(cur);
                    if cur_f + cost > r.budget + 1e-9 {
                        break;
                    }
                    let next = (cur_f + cost).to_bits();
                    match r.consumed.compare_exchange_weak(
                        cur,
                        next,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            ok = true;
                            break;
                        }
                        Err(b) => cur = b,
                    }
                }
                if !ok {
                    // Rollback any earlier resources that were already reserved
                    for (prev_r, prev_cost) in reserved {
                        prev_r.rollback_consumption(prev_cost);
                    }
                    return false;
                }
                reserved.push((r, cost));
            }
        }

        // All resources successfully reserved consumption! Now perform dual step updates.
        for r in &self.resources {
            let cost = r.cost(arm_id);
            let t = r.decisions.fetch_add(1, Ordering::Relaxed);
            let cur_consumed = f64::from_bits(r.consumed.load(Ordering::Relaxed));
            let pre_remaining = (r.budget - (cur_consumed - cost)).max(0.0);

            let min_rem = (50u64).max((r.horizon as f64 * 0.05) as u64);
            if r.horizon > min_rem && t >= r.horizon - min_rem {
                continue;
            }

            let rho = r.target_rate_at(t, pre_remaining, self.adaptive);
            let delta = cost - rho;
            let mut cur_lambda = r.lambda.load(Ordering::Relaxed);
            loop {
                let updated =
                    (f64::from_bits(cur_lambda) + r.step_size * delta).clamp(0.0, r.lambda_max);
                match r.lambda.compare_exchange_weak(
                    cur_lambda,
                    updated.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(b) => cur_lambda = b,
                }
            }
        }

        true
    }

    pub fn rollback_consumption(&self, arm_id: &str) {
        for r in &self.resources {
            let cost = r.cost(arm_id);
            if cost > 0.0 {
                r.rollback_consumption(cost);
            }
        }
    }

    pub fn set_arm_costs(&self, arm_id: &str, costs: &HashMap<String, f64>) {
        for r in &self.resources {
            if let Some(&cost) = costs.get(&r.name) {
                r.set_arm_cost(arm_id, cost);
            }
        }
    }

    pub fn report(&self) -> PacingReport {
        PacingReport {
            adaptive: self.adaptive,
            resources: self
                .resources
                .iter()
                .map(|r| {
                    let consumed = r.consumed();
                    let budget = r.budget;
                    let remaining = r.remaining_budget();
                    let utilization = if budget > 0.0 {
                        (consumed / budget).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    ResourceReport {
                        name: r.name.clone(),
                        budget,
                        consumed,
                        remaining,
                        utilization,
                        lambda: r.lambda(),
                        horizon: r.horizon,
                        decisions: r.decisions.load(Ordering::Relaxed),
                        is_exhausted: remaining <= 0.0,
                    }
                })
                .collect(),
        }
    }

    pub fn checkpoint(&self) -> PacingCheckpoint {
        PacingCheckpoint {
            adaptive: self.adaptive,
            resources: self
                .resources
                .iter()
                .map(|r| ResourceCheckpoint {
                    name: r.name.clone(),
                    budget: r.budget,
                    horizon: r.horizon,
                    step_size: r.step_size,
                    lambda_max: r.lambda_max,
                    arm_costs: r.arm_costs.read().clone(),
                    consumed: r.consumed(),
                    lambda: r.lambda(),
                    decisions: r.decisions.load(Ordering::Relaxed),
                })
                .collect(),
        }
    }

    pub fn from_checkpoint(chk: PacingCheckpoint) -> Self {
        Self {
            adaptive: chk.adaptive,
            resources: chk
                .resources
                .into_iter()
                .map(|r| ResourceState {
                    name: r.name,
                    budget: r.budget,
                    horizon: r.horizon,
                    step_size: r.step_size,
                    lambda_max: r.lambda_max,
                    arm_costs: RwLock::new(r.arm_costs),
                    consumed: AtomicU64::new(r.consumed.to_bits()),
                    lambda: AtomicU64::new(r.lambda.to_bits()),
                    decisions: AtomicU64::new(r.decisions),
                })
                .collect(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ResourceCheckpoint {
    pub name: String,
    pub budget: f64,
    pub horizon: u64,
    pub step_size: f64,
    pub lambda_max: f64,
    pub arm_costs: HashMap<String, f64>,
    pub consumed: f64,
    pub lambda: f64,
    pub decisions: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PacingCheckpoint {
    pub resources: Vec<ResourceCheckpoint>,
    pub adaptive: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ResourceReport {
    pub name: String,
    pub budget: f64,
    pub consumed: f64,
    pub remaining: f64,
    pub utilization: f64,
    pub lambda: f64,
    pub horizon: u64,
    pub decisions: u64,
    pub is_exhausted: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PacingReport {
    pub adaptive: bool,
    pub resources: Vec<ResourceReport>,
}

// --- Checkpoint structs ---

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CampaignCheckpoint {
    pub alpha: f64,
    #[serde(default)]
    pub algorithm: Algorithm,
    pub arms: HashMap<String, ArmState>,
    #[serde(default)]
    pub challenger_arms: Option<HashMap<String, ArmState>>,
    /// Challenger traffic in basis points (0–10000; 1000 = 10% initial exploration).
    /// Persisted so promotion progress survives a restart.
    #[serde(default)]
    pub challenger_traffic_bps: u32,
    /// Tournament win streak: +N = N consecutive challenger wins, −N = base wins.
    /// Resets to 0 after each traffic adjustment.
    #[serde(default)]
    pub tournament_wins: i32,
    /// Soft-deleted campaigns are preserved in the checkpoint but excluded from
    /// predictions and reward updates. Survives restart via this flag.
    #[serde(default)]
    pub archived: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Entropy snapshot written at checkpoint time; used to compute EntropyTrend on next diagnostics call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entropy_snapshot: Option<f64>,
    /// Half-life for time-aware checkpoint decay. None = no forgetting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decay_half_life_hours: Option<f64>,
    /// Lagrangian pacing / capacity constraints checkpoint state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pacing: Option<PacingCheckpoint>,
}

// --- Campaign report (business-level convergence signal) ---

/// Per-arm statistics returned by `GET /campaign/:id/report`.
#[derive(Serialize, Debug)]
pub struct ArmReportStats {
    /// Fraction of total predictions routed to this arm (0.0–1.0).
    pub traffic_share: f64,
    pub predictions: u64,
    pub rewards: u64,
    /// Observed mean reward. None when fewer than 10 rewards received.
    pub mean_reward: Option<f64>,
    /// Lower bound of the 95% confidence interval on mean reward.
    pub reward_lower_ci: Option<f64>,
    /// Upper bound of the 95% confidence interval on mean reward.
    pub reward_upper_ci: Option<f64>,
    /// Selection eligibility. Non-active arms keep their history in this report —
    /// traffic shares are shares of everything the campaign ever served.
    pub status: ArmStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
}

/// Business-level campaign report returned by `GET /campaign/:id/report`.
///
/// The `converged` field answers "is this campaign done?":
/// - `true`  → leading arm has a statistically significant advantage (95% CI).
/// - `false` → the leading arm leads but CIs still overlap.
/// - `null`  → not enough data to assess (< 30 rewards per arm).
///
/// Validate convergence with the causal forest analysis in the Python SDK:
/// if `arm_traffic_share` matches `causal_analysis()` arm assignment percentages,
/// the bandit has found the causally correct structure.
#[derive(Serialize, Debug)]
pub struct CampaignReport {
    pub campaign_id: String,
    pub archived: bool,
    pub algorithm: Algorithm,
    pub alpha: f64,
    pub total_predictions: u64,
    pub total_rewards: u64,
    pub overall_reward_rate: Option<f64>,
    pub arms: HashMap<String, ArmReportStats>,
    /// Arm with the highest mean reward (requires at least 10 rewards).
    pub leading_arm: Option<String>,
    /// Statistical convergence at 95% confidence level.
    pub converged: Option<bool>,
    /// Percentage of traffic currently routed to the challenger (Progressive only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenger_traffic_pct: Option<f64>,
    /// Current tournament win streak (Progressive only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tournament_win_streak: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pacing: Option<PacingReport>,
}

// --- Per-arm and campaign diagnostics ---

#[derive(Serialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "snake_case")]
pub enum EntropyStatus {
    Ok,
    Warning,
    Critical,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum EntropyTrend {
    Stable,
    Falling,
    Recovering,
    Unknown,
}

/// Per-arm diagnostics: reward stats and A_inv condition proxy.
#[derive(Serialize, Debug)]
pub struct ArmDiagnostics {
    pub predictions: u64,
    pub rewards: u64,
    pub avg_reward: Option<f64>,
    /// L2 norm of θ — grows as the arm accumulates correlated positive rewards.
    pub theta_norm: f64,
    /// Smallest diagonal entry of A_inv (high = high remaining uncertainty for this dim).
    pub a_inv_diag_min: f64,
    /// Largest diagonal entry of A_inv.
    pub a_inv_diag_max: f64,
    pub status: ArmStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
}

/// Full campaign diagnostics snapshot returned by `GET /campaign/:id/diagnostics`.
#[derive(Serialize, Debug)]
pub struct CampaignDiagnosticsData {
    pub campaign_id: String,
    pub archived: bool,
    pub algorithm: Algorithm,
    pub alpha: f64,
    pub arm_count: usize,
    /// Arms eligible for selection right now. Entropy and the convergence signal
    /// below are computed over these only — a paused arm's historical traffic would
    /// otherwise make a collapsed campaign look healthy.
    pub active_arm_count: usize,
    pub total_predictions: u64,
    pub total_rewards: u64,
    pub overall_avg_reward: Option<f64>,
    pub arm_stats: HashMap<String, ArmDiagnostics>,
    /// Progressive campaigns only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenger_traffic_pct: Option<f64>,
    /// Progressive campaigns only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tournament_win_streak: Option<i32>,
    /// Neural / Progressive-with-neural-challenger only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neural_buffer_size: Option<usize>,
    /// Loss at each gradient step of the most recent neural retrain. Absent until first retrain fires.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neural_last_retrain_losses: Option<Vec<f32>>,
    /// Normalised selection entropy (0 = fully collapsed, 1 = uniform across arms).
    pub selection_entropy: f64,
    pub entropy_status: EntropyStatus,
    pub entropy_trend: EntropyTrend,
    /// Statistical convergence signal (Guard 1): suppresses false-positive alerts
    /// when one arm has genuinely won. None = insufficient reward data.
    pub converged: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub likely_cause: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_action: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pacing: Option<PacingReport>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckpointData {
    pub wal_offset: u64,     // byte position in WAL; recovery replays from here
    pub timestamp_secs: u64, // unix epoch, for diagnostics
    pub campaigns: HashMap<String, CampaignCheckpoint>,
    /// Predictions still awaiting a reward when this checkpoint was taken.
    ///
    /// These used to be re-emitted into the WAL as `is_reemit` Predicted records so
    /// a late reward could still match after rotation, which rewrote the whole
    /// unmatched set on every checkpoint. Carrying them here instead costs one copy
    /// per checkpoint rather than one WAL record each, and recovery restores the
    /// cache directly instead of replaying them.
    ///
    /// Empty on checkpoints written before this field existed; such files simply
    /// recover the old way from any re-emitted records still in the WAL.
    #[serde(default)]
    pub pending_interactions: HashMap<String, InteractionRecord>,
    /// Increments with every checkpoint. The WAL rotated by checkpoint N begins
    /// with `DbEvent::WalStart { generation: N }`, and the segment it discarded is
    /// kept as `wal_segment.N`, so recovery can tell which WAL history belongs to
    /// which checkpoint. 0 on checkpoints written before generations existed.
    #[serde(default)]
    pub generation: u64,
}

// --- The Write-Ahead Log Events ---
// Make sure this has `pub enum DbEvent` so other files can see it!
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DbEvent {
    CampaignCreated {
        campaign_id: String,
        arms: Vec<String>,
        feature_dim: usize,
        #[serde(default = "default_alpha")]
        alpha: f64,
        #[serde(default)]
        algorithm: Algorithm,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decay_half_life_hours: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pacing: Option<PacingConfig>,
    },
    Predicted {
        interaction_id: String,
        campaign_id: String,
        arm_id: String,
        context: Vec<f64>,
        #[serde(default)]
        timestamp_secs: u64,
        /// Per-arm propensity under the logging policy.
        /// LinUCB/NeuralLinUCB: softmax-normalised UCB scores.
        /// Thompson Sampling: adaptive Monte Carlo frequency estimate (N=8–64).
        /// Absent in WAL records written before propensity logging — deserialises to None.
        #[serde(default = "default_none_map")]
        arm_propensities: Option<HashMap<String, f64>>,
        /// True when this record was re-emitted at checkpoint to keep an
        /// in-flight prediction matchable after WAL rotation. Such predictions
        /// were already counted live and captured in the checkpoint snapshot, so
        /// WAL replay must NOT re-increment prediction_count for them. Old WAL
        /// records (written before this field) deserialise to false.
        #[serde(default)]
        is_reemit: bool,
    },
    Rewarded {
        interaction_id: String,
        reward: f64,
        #[serde(default)]
        timestamp_secs: u64,
        /// The prediction being rewarded, present only when its `Predicted`
        /// record was dropped from the WAL. Replay restores the prediction from
        /// here, so an acknowledged reward is never orphaned. Absent in older WALs.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        unlogged_prediction: Option<InteractionRecord>,
    },
    /// A new arm joined a live campaign.
    ///
    /// Both priors are already resolved to concrete vectors — see [`ArmPrior`] for
    /// why they cannot be recomputed at replay. `challenger_*` is present only for
    /// Progressive campaigns, whose challenger arms may live in a different
    /// (embedding) space than the base arms.
    ArmAdded {
        campaign_id: String,
        arm_id: String,
        base_dim: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        group: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prior: Option<ArmPrior>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        challenger_dim: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        challenger_prior: Option<ArmPrior>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        costs: Option<HashMap<String, f64>>,
        #[serde(default)]
        timestamp_secs: u64,
    },
    /// An arm was paused, retired, or brought back. Matrices are untouched.
    ArmStatusChanged {
        campaign_id: String,
        arm_id: String,
        status: ArmStatus,
        #[serde(default)]
        timestamp_secs: u64,
    },
    CampaignDeleted {
        campaign_id: String,
    },
    CampaignArchived {
        campaign_id: String,
        #[serde(default)]
        timestamp_secs: u64,
    },
    CampaignRestored {
        campaign_id: String,
        #[serde(default)]
        timestamp_secs: u64,
    },
    /// First record of a WAL rotated by checkpoint `generation`: the file holds
    /// exactly the events after that checkpoint. Carries no state; replay skips it.
    WalStart {
        generation: u64,
    },
    /// Records that a prediction selected `arm_id` and the campaign's pacing
    /// budget was charged accordingly.
    ///
    /// Emitted with `BestEffort` durability alongside every `Predicted` event for
    /// campaigns that have `pacing` configured. Separating it from `Predicted` means:
    ///
    /// 1. WAL replay can restore pacing counters (`consumed`, `lambda`, `decisions`)
    ///    independently of whether the `Predicted` record survived — WAL rotation and
    ///    budget saturation can drop `Predicted` while this still lands.
    /// 2. The replay handler for `Predicted` does NOT call `record_consumption`;
    ///    only this variant does. There is therefore no double-count risk.
    /// 3. Rollback safety: binaries that pre-date this variant deserialise it as
    ///    `Unknown` and skip it with a warning — the existing `#[serde(other)]`
    ///    guard covers this automatically.
    ///
    /// Arm costs are intentionally NOT stored here; they are derived from the
    /// campaign's live `PacingState::arm_costs` at replay time. This means the
    /// event is a compact "what happened" record that does not embed a snapshot of
    /// the cost table (which can be large for multi-arm campaigns).
    PacingConsumed {
        campaign_id: String,
        arm_id: String,
        #[serde(default)]
        timestamp_secs: u64,
    },
    /// Unknown / future variant for forward-compatibility during rollback
    #[serde(other)]
    Unknown,
}
