use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ArmState {
    pub a_inv: Array2<f64>,
    pub b: Array1<f64>,
    pub theta: Array1<f64>,
}

impl ArmState {
    pub fn new(dim: usize) -> Self {
        Self {
            a_inv: Array2::eye(dim),
            b: Array1::zeros(dim),
            theta: Array1::zeros(dim),
        }
    }
}

#[derive(Clone, Debug)]
pub struct InteractionRecord {
    pub campaign_id: String,
    pub arm_id: String,
    pub context: Array1<f64>,
    pub probability: f64,
}

// --- Checkpoint structs ---

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CampaignCheckpoint {
    pub alpha: f64,
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
    },
    Predicted {
        interaction_id: String,
        campaign_id: String,
        arm_id: String,
        context: Vec<f64>,
    },
    Rewarded {
        interaction_id: String,
        reward: f64,
    },
    CampaignDeleted {
        campaign_id: String,
    },
}