use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize, Serializer, Deserializer};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

fn default_none_map() -> Option<HashMap<String, f64>> { None }

pub const DEFAULT_ALPHA: f64 = 1.0;

fn default_alpha() -> f64 { DEFAULT_ALPHA }

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
    Internal(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::NotFound(m)      => write!(f, "{m}"),
            EngineError::AlreadyExists(m) => write!(f, "{m}"),
            EngineError::Archived(m)      => write!(f, "{m}"),
            EngineError::WalFull          => write!(f, "wal:full — server is busy, retry momentarily"),
            EngineError::WalUnavailable   => write!(f, "wal:unavailable — storage error, check server logs"),
            EngineError::BadRequest(m)    => write!(f, "{m}"),
            EngineError::Internal(m)      => write!(f, "{m}"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NeuralLinUCBConfig {
    pub context_dim:    usize,
    pub embed_dim:      usize,
    pub hidden_dim:     usize,
    pub hidden_layers:  usize,
    pub retrain_every:  usize,
    pub retrain_steps:  usize,
    pub learning_rate:  f64,
    pub lambda:         f64,
}

impl Default for NeuralLinUCBConfig {
    fn default() -> Self {
        Self {
            context_dim:   64,
            embed_dim:     32,
            hidden_dim:    128,
            hidden_layers: 2,
            retrain_every: 200,
            retrain_steps: 100,
            learning_rate: 1e-3,
            lambda:        1.0,
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
    pub base:          Box<Algorithm>,
    pub challenger:    Box<Algorithm>,
    /// Minimum buffer entries per arm required before any traffic shift fires.
    #[serde(default = "default_progressive_min_obs")]
    pub min_obs:       usize,
    /// Consecutive checkpoint wins required to earn one traffic step.
    #[serde(default = "default_progressive_required_wins")]
    pub required_wins: usize,
    /// Traffic change per confirmed win run, in basis points (1000 = 10%).
    #[serde(default = "default_progressive_step_bps")]
    pub step_bps:      u32,
}

fn default_progressive_min_obs()       -> usize { 100  }
fn default_progressive_required_wins() -> usize { 3    }
fn default_progressive_step_bps()      -> u32   { 1000 }

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    Linucb,
    ThompsonSampling,
    #[serde(rename = "neural_lin_ucb")]
    NeuralLinUCB(NeuralLinUCBConfig),
    Progressive(ProgressiveConfig),
}

impl Default for Algorithm {
    fn default() -> Self { Algorithm::Linucb }
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

#[derive(Debug)]
pub struct ArmState {
    pub a_inv: Array2<f64>,
    pub b: Array1<f64>,
    pub theta: Array1<f64>,
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
            a_inv:             self.a_inv.clone(),
            b:                 self.b.clone(),
            theta:             self.theta.clone(),
            chol_cache:        parking_lot::Mutex::new(None), // don't copy stale cache
            prediction_count:  AtomicU64::new(self.prediction_count.load(Ordering::Relaxed)),
            reward_count:      AtomicU64::new(self.reward_count.load(Ordering::Relaxed)),
            total_reward:      AtomicU64::new(self.total_reward.load(Ordering::Relaxed)),
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
            prediction_count: u64,
            reward_count: u64,
            total_reward: f64,
        }
        let shadow = Shadow {
            a_inv: self.a_inv.clone(),
            b: self.b.clone(),
            theta: self.theta.clone(),
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
            #[allow(dead_code)] // stored value is intentionally ignored; theta is recomputed from a_inv·b
            theta: Array1<f64>,
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
            a_inv:            shadow.a_inv,
            b:                shadow.b,
            theta,
            chol_cache:       parking_lot::Mutex::new(None), // recomputed lazily on first score_ts
            prediction_count: AtomicU64::new(shadow.prediction_count),
            reward_count:     AtomicU64::new(shadow.reward_count),
            total_reward:     AtomicU64::new(shadow.total_reward.to_bits()),
        })
    }
}

impl ArmState {
    pub fn new(dim: usize) -> Self {
        Self {
            a_inv:            Array2::eye(dim),
            b:                Array1::zeros(dim),
            theta:            Array1::zeros(dim),
            chol_cache:       parking_lot::Mutex::new(None),
            prediction_count: AtomicU64::new(0),
            reward_count:     AtomicU64::new(0),
            total_reward:     AtomicU64::new(0.0f64.to_bits()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct InteractionRecord {
    pub campaign_id:      String,
    pub arm_id:           String,
    pub context:          Array1<f64>,
    pub arm_propensities: Option<HashMap<String, f64>>,
    pub timestamp_secs:   u64,
}

/// One completed prediction→reward pair, ready to write as a flat Parquet row.
#[derive(Debug)]
pub struct CompletedInteraction {
    pub interaction_id: String,
    pub arm_id:         String,
    pub context:        Vec<f64>,
    pub reward:         f64,
    pub predicted_at:   u64,
    pub rewarded_at:    u64,
    /// Softmax-normalised UCB propensity of the chosen arm (LinUCB only).
    /// None for Thompson Sampling campaigns (propensity logging added in a future iteration).
    pub propensity:     Option<f64>,
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
}

// --- Campaign report (business-level convergence signal) ---

/// Per-arm statistics returned by `GET /campaign/:id/report`.
#[derive(Serialize, Debug)]
pub struct ArmReportStats {
    /// Fraction of total predictions routed to this arm (0.0–1.0).
    pub traffic_share:   f64,
    pub predictions:     u64,
    pub rewards:         u64,
    /// Observed mean reward. None when fewer than 10 rewards received.
    pub mean_reward:     Option<f64>,
    /// Lower bound of the 95% confidence interval on mean reward.
    pub reward_lower_ci: Option<f64>,
    /// Upper bound of the 95% confidence interval on mean reward.
    pub reward_upper_ci: Option<f64>,
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
    pub campaign_id:         String,
    pub archived:            bool,
    pub algorithm:           Algorithm,
    pub total_predictions:   u64,
    pub total_rewards:       u64,
    pub overall_reward_rate: Option<f64>,
    pub arms:                HashMap<String, ArmReportStats>,
    /// Arm with the highest mean reward (requires at least 10 rewards).
    pub leading_arm:         Option<String>,
    /// Statistical convergence at 95% confidence level.
    pub converged:           Option<bool>,
    /// Percentage of traffic currently routed to the challenger (Progressive only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenger_traffic_pct: Option<f64>,
    /// Current tournament win streak (Progressive only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tournament_win_streak:  Option<i32>,
}

// --- Per-arm and campaign diagnostics ---

/// Per-arm diagnostics: reward stats and A_inv condition proxy.
#[derive(Serialize, Debug)]
pub struct ArmDiagnostics {
    pub predictions:    u64,
    pub rewards:        u64,
    pub avg_reward:     Option<f64>,
    /// L2 norm of θ — grows as the arm accumulates correlated positive rewards.
    pub theta_norm:     f64,
    /// Smallest diagonal entry of A_inv (high = high remaining uncertainty for this dim).
    pub a_inv_diag_min: f64,
    /// Largest diagonal entry of A_inv.
    pub a_inv_diag_max: f64,
}

/// Full campaign diagnostics snapshot returned by `GET /campaign/:id/diagnostics`.
#[derive(Serialize, Debug)]
pub struct CampaignDiagnosticsData {
    pub campaign_id:          String,
    pub archived:             bool,
    pub algorithm:            Algorithm,
    pub arm_count:            usize,
    pub total_predictions:    u64,
    pub total_rewards:        u64,
    pub overall_avg_reward:   Option<f64>,
    pub arm_stats:            HashMap<String, ArmDiagnostics>,
    /// Progressive campaigns only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenger_traffic_pct: Option<f64>,
    /// Progressive campaigns only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tournament_win_streak:  Option<i32>,
    /// Neural / Progressive-with-neural-challenger only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neural_buffer_size:   Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckpointData {
    pub wal_offset: u64,      // byte position in WAL; recovery replays from here
    pub timestamp_secs: u64,  // unix epoch, for diagnostics
    pub campaigns: HashMap<String, CampaignCheckpoint>,
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
    },
    Predicted {
        interaction_id: String,
        campaign_id:    String,
        arm_id:         String,
        context:        Vec<f64>,
        #[serde(default)]
        timestamp_secs: u64,
        /// Softmax-normalised UCB propensity for every arm (LinUCB only).
        /// Absent in WAL records written before propensity logging was added — deserialises to None.
        /// None for Thompson Sampling campaigns.
        #[serde(default = "default_none_map")]
        arm_propensities: Option<HashMap<String, f64>>,
    },
    Rewarded {
        interaction_id: String,
        reward:         f64,
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
}
