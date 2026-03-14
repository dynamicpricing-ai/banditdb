use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn default_none_map() -> Option<HashMap<String, f64>> { None }

fn default_alpha() -> f64 { 1.0 }

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    Linucb,
    ThompsonSampling,
}

impl Default for Algorithm {
    fn default() -> Self { Algorithm::Linucb }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ArmState {
    pub a_inv: Array2<f64>,
    pub b: Array1<f64>,
    pub theta: Array1<f64>,
    #[serde(default)]
    pub prediction_count: u64,
    #[serde(default)]
    pub reward_count: u64,
}

impl ArmState {
    pub fn new(dim: usize) -> Self {
        Self {
            a_inv: Array2::eye(dim),
            b: Array1::zeros(dim),
            theta: Array1::zeros(dim),
            prediction_count: 0,
            reward_count: 0,
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
}