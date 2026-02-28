use crate::state::{ArmState, DbEvent, InteractionRecord};
use moka::sync::Cache;
use ndarray::Array1;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use uuid::Uuid;
use polars::prelude::*;

pub struct Campaign {
    pub alpha: f64,
    pub arms: RwLock<HashMap<String, ArmState>>,
}

pub struct BanditDB {
    pub campaigns: RwLock<HashMap<String, Campaign>>,
    pub interactions: Cache<String, InteractionRecord>,
    pub event_tx: UnboundedSender<DbEvent>, // The fast channel to the disk writer
}

impl BanditDB {
    pub fn new(wal_path: &str) -> Self {
        let (tx, mut rx) = unbounded_channel::<DbEvent>();

        // 1. Spawn the Background Disk Writer Thread
        let path = wal_path.to_string();
        tokio::spawn(async move {
            let mut file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
            while let Some(event) = rx.recv().await {
                let json = serde_json::to_string(&event).unwrap();
                writeln!(file, "{}", json).unwrap(); // Appends to wal.jsonl
            }
        });

        let db = Self {
            campaigns: RwLock::new(HashMap::new()),
            interactions: Cache::builder().time_to_live(Duration::from_secs(86400)).build(),
            event_tx: tx,
        };

        // 2. Crash Recovery: Read the WAL on startup
        db.recover(wal_path);
        db
    }

    /// Reads the WAL line by line and perfectly reconstructs the RAM state
    fn recover(&self, wal_path: &str) {
        if let Ok(file) = File::open(wal_path) {
            let reader = BufReader::new(file);
            let mut count = 0;
            for line in reader.lines().flatten() {
                if let Ok(event) = serde_json::from_str::<DbEvent>(&line) {
                    self.apply_event_to_memory(event);
                    count += 1;
                }
            }
            println!("🔄 Recovered {} events from WAL.", count);
        }
    }

    /// The unified math & memory updater
    fn apply_event_to_memory(&self, event: DbEvent) {
        match event {
            DbEvent::CampaignCreated { campaign_id, arms, feature_dim } => {
                let mut arms_map = HashMap::new();
                for arm in arms {
                    arms_map.insert(arm, ArmState::new(feature_dim));
                }
                self.campaigns.write().insert(
                    campaign_id,
                    Campaign { alpha: 1.0, arms: RwLock::new(arms_map) },
                );
            }
            DbEvent::Predicted { interaction_id, campaign_id, arm_id, context } => {
                self.interactions.insert(
                    interaction_id,
                    InteractionRecord {
                        campaign_id, arm_id, context: Array1::from_vec(context), probability: 0.0,
                    },
                );
            }
            DbEvent::Rewarded { interaction_id, reward } => {
                if let Some(record) = self.interactions.get(&interaction_id) {
                    if let Some(campaign) = self.campaigns.read().get(&record.campaign_id) {
                        if let Some(arm_state) = campaign.arms.write().get_mut(&record.arm_id) {
                            arm_state.update(&record.context, reward);
                        }
                    }
                }
            }
        }
    }

    // --- The Public API ---

    pub fn add_campaign(&self, campaign_id: &str, arms: Vec<String>, feature_dim: usize) {
        let event = DbEvent::CampaignCreated { campaign_id: campaign_id.to_string(), arms, feature_dim };
        self.apply_event_to_memory(event.clone()); // Update Math
        let _ = self.event_tx.send(event);         // Save to Disk
    }

    pub fn predict(&self, campaign_id: &str, context: Vec<f64>) -> Option<(String, String)> {
        let campaigns = self.campaigns.read();
        let campaign = campaigns.get(campaign_id)?;
        let context_arr = Array1::from_vec(context.clone());

        let arms = campaign.arms.read();

        // Guard: reject context whose dimension doesn't match the campaign's feature space.
        // Without this, ndarray panics inside dot() with a shape mismatch error.
        if let Some((_, first_arm)) = arms.iter().next() {
            if context_arr.len() != first_arm.theta.len() {
                return None;
            }
        }

        let mut best_arm = String::new();
        let mut max_score = f64::NEG_INFINITY;

        for (arm_id, state) in arms.iter() {
            let score = state.score(&context_arr, campaign.alpha);
            if score > max_score {
                max_score = score;
                best_arm = arm_id.clone();
            }
        }

        let interaction_id = Uuid::new_v4().to_string();
        let event = DbEvent::Predicted {
            interaction_id: interaction_id.clone(), campaign_id: campaign_id.to_string(),
            arm_id: best_arm.clone(), context,
        };

        self.apply_event_to_memory(event.clone()); // Remember context
        let _ = self.event_tx.send(event);         // Save to Disk

        Some((best_arm, interaction_id))
    }

    pub fn reward(&self, interaction_id: &str, reward: f64) {
        let event = DbEvent::Rewarded { interaction_id: interaction_id.to_string(), reward };
        self.apply_event_to_memory(event.clone()); // Update Math
        let _ = self.event_tx.send(event);         // Save to Disk
    }

    /// Converts the current JSONL Write-Ahead Log into a highly compressed Parquet file.
    pub fn export_to_parquet(&self, wal_path: &str, output_path: &str) -> Result<String, String> {
        // 1. Read the JSON lines file using Polars' insanely fast multi-threaded reader
        let df_result = JsonLineReader::from_path(wal_path)
            .map_err(|e| format!("Failed to read WAL: {}", e))?
            .finish();

        match df_result {
            Ok(mut df) => {
                // 2. Create the Parquet file
                let mut file = std::fs::File::create(output_path)
                    .map_err(|e| format!("Failed to create Parquet file: {}", e))?;

                // 3. Compress and write the DataFrame to Parquet
                ParquetWriter::new(&mut file)
                    .with_compression(ParquetCompression::Snappy)
                    .finish(&mut df)
                    .map_err(|e| format!("Failed to write Parquet: {}", e))?;

                Ok(format!("Successfully exported {} rows to {}", df.height(), output_path))
            }
            Err(e) => Err(format!("Failed to parse WAL into DataFrame: {}", e)),
        }
    }
}
