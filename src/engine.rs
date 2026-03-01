use crate::state::{ArmState, CampaignCheckpoint, CheckpointData, DbEvent, InteractionRecord};
use moka::sync::Cache;
use ndarray::Array1;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::oneshot;
use uuid::Uuid;
use polars::prelude::*;

pub enum WalMessage {
    Event(DbEvent),
    Checkpoint { reply: oneshot::Sender<u64> },
}

pub struct Campaign {
    pub alpha: f64,
    pub arms: RwLock<HashMap<String, ArmState>>,
}

pub struct BanditDB {
    pub campaigns:      RwLock<HashMap<String, Campaign>>,
    pub interactions:   Cache<String, InteractionRecord>,
    pub event_tx:       UnboundedSender<WalMessage>,
    pub rewarded_count: AtomicU64,
    pub wal_path:       String,
    pub data_dir:       String,
}

impl BanditDB {
    pub fn new(wal_path: &str, data_dir: &str) -> Self {
        let (tx, mut rx) = unbounded_channel::<WalMessage>();

        // 1. Spawn the Background Disk Writer Thread
        let path = wal_path.to_string();
        tokio::spawn(async move {
            let mut file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
            while let Some(msg) = rx.recv().await {
                match msg {
                    WalMessage::Event(event) => {
                        let json = serde_json::to_string(&event).unwrap();
                        writeln!(file, "{}", json).unwrap();
                        file.flush().unwrap();
                    }
                    WalMessage::Checkpoint { reply } => {
                        file.flush().unwrap();
                        file.sync_all().unwrap();
                        // seek(End(0)) returns the true file size regardless of whether
                        // any writes have occurred in the current session (stream_position()
                        // returns 0 on O_APPEND files until the first write).
                        let offset = file.seek(SeekFrom::End(0)).unwrap();
                        let _ = reply.send(offset);
                    }
                }
            }
        });

        let ttl_secs: u64 = std::env::var("BANDITDB_REWARD_TTL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(86400);

        let db = Self {
            campaigns:      RwLock::new(HashMap::new()),
            interactions:   Cache::builder().time_to_live(Duration::from_secs(ttl_secs)).build(),
            event_tx:       tx,
            rewarded_count: AtomicU64::new(0),
            wal_path:       wal_path.to_string(),
            data_dir:       data_dir.to_string(),
        };

        // 2. Crash Recovery: Load checkpoint then replay WAL tail
        db.recover(wal_path, data_dir);
        db
    }

    fn recover(&self, wal_path: &str, data_dir: &str) {
        // Phase 1: Load checkpoint if one exists
        let mut wal_start_offset: u64 = 0;
        let checkpoint_path = format!("{}/checkpoint.json", data_dir);

        match fs::read_to_string(&checkpoint_path).ok()
            .and_then(|s| serde_json::from_str::<CheckpointData>(&s).ok())
        {
            Some(checkpoint) => {
                let n = checkpoint.campaigns.len();
                for (campaign_id, camp) in checkpoint.campaigns {
                    self.campaigns.write().insert(
                        campaign_id,
                        Campaign { alpha: camp.alpha, arms: RwLock::new(camp.arms) },
                    );
                }
                wal_start_offset = checkpoint.wal_offset;
                println!("[recovery] Loaded checkpoint: {} campaigns, WAL offset {} bytes", n, wal_start_offset);
            }
            None => {
                println!("[recovery] No checkpoint found — replaying WAL from beginning");
            }
        }

        // Phase 2: Replay WAL events after the checkpoint offset
        match File::open(wal_path) {
            Err(_) => {
                println!("[recovery] No WAL found — starting fresh");
            }
            Ok(mut file) => {
                if wal_start_offset > 0 {
                    file.seek(SeekFrom::Start(wal_start_offset)).unwrap();
                }
                let reader = BufReader::new(file);
                let mut count = 0;
                for line in reader.lines().flatten() {
                    if let Ok(event) = serde_json::from_str::<DbEvent>(&line) {
                        self.apply_event_to_memory(event);
                        count += 1;
                    }
                }
                println!("[recovery] Replayed {} WAL events from tail", count);
            }
        }

        println!("[recovery] Ready: {} campaigns loaded", self.campaigns.read().len());
    }

    pub async fn checkpoint(&self) -> Result<String, String> {
        // 1. Send flush barrier through the WAL channel — writer drains all prior
        //    events to disk before replying with the confirmed byte offset.
        let (reply_tx, reply_rx) = oneshot::channel::<u64>();
        self.event_tx
            .send(WalMessage::Checkpoint { reply: reply_tx })
            .map_err(|_| "WAL channel closed".to_string())?;

        let wal_offset = reply_rx.await.map_err(|_| "WAL writer closed".to_string())?;

        // 2. Snapshot all campaign matrices under read lock.
        //    At this point every event applied to memory is also on disk at or before wal_offset.
        let campaigns_snapshot: HashMap<String, CampaignCheckpoint> = {
            let campaigns = self.campaigns.read();
            campaigns.iter().map(|(id, campaign)| {
                let arms_snapshot: HashMap<String, ArmState> = campaign.arms.read()
                    .iter()
                    .map(|(arm_id, state)| (arm_id.clone(), state.clone()))
                    .collect();
                (id.clone(), CampaignCheckpoint { alpha: campaign.alpha, arms: arms_snapshot })
            }).collect()
        };

        // 3. Serialise checkpoint envelope
        let data = CheckpointData {
            wal_offset,
            timestamp_secs: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            campaigns: campaigns_snapshot,
        };
        let json = serde_json::to_string(&data).map_err(|e| e.to_string())?;

        // 4. Atomic write: write to .tmp then rename — crash-safe
        let tmp_path  = format!("{}/checkpoint.tmp",  self.data_dir);
        let dest_path = format!("{}/checkpoint.json", self.data_dir);
        fs::write(&tmp_path, &json).map_err(|e| e.to_string())?;
        fs::rename(&tmp_path, &dest_path).map_err(|e| e.to_string())?;

        let msg = format!(
            "Checkpoint written: {} campaigns, WAL offset {} bytes",
            data.campaigns.len(), wal_offset
        );
        println!("[checkpoint] {}", msg);
        Ok(msg)
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
            DbEvent::CampaignDeleted { campaign_id } => {
                self.campaigns.write().remove(&campaign_id);
            }
        }
    }

    // --- The Public API ---

    pub fn add_campaign(&self, campaign_id: &str, arms: Vec<String>, feature_dim: usize) {
        let event = DbEvent::CampaignCreated { campaign_id: campaign_id.to_string(), arms, feature_dim };
        self.apply_event_to_memory(event.clone()); // Update Math
        let _ = self.event_tx.send(WalMessage::Event(event));         // Save to Disk
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
        let _ = self.event_tx.send(WalMessage::Event(event));         // Save to Disk

        Some((best_arm, interaction_id))
    }

    pub fn delete_campaign(&self, campaign_id: &str) -> bool {
        if !self.campaigns.read().contains_key(campaign_id) {
            return false;
        }
        let event = DbEvent::CampaignDeleted { campaign_id: campaign_id.to_string() };
        self.apply_event_to_memory(event.clone());
        let _ = self.event_tx.send(WalMessage::Event(event));
        true
    }

    pub fn reward(&self, interaction_id: &str, reward: f64) {
        let event = DbEvent::Rewarded { interaction_id: interaction_id.to_string(), reward };
        self.apply_event_to_memory(event.clone());
        let _ = self.event_tx.send(WalMessage::Event(event));
        self.rewarded_count.fetch_add(1, Ordering::Relaxed);
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
