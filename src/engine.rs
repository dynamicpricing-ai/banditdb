use crate::state::{ArmState, CampaignCheckpoint, CheckpointData, CompletedInteraction, DbEvent, InteractionRecord};
use moka::sync::Cache;
use ndarray::Array1;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::oneshot;
use uuid::Uuid;
use polars::prelude::*;

pub enum WalMessage {
    Event(DbEvent),
    Checkpoint { reply: oneshot::Sender<u64> },
    Rotate { checkpoint_offset: u64, reply: oneshot::Sender<()> },
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
        let writer_data_dir = data_dir.to_string();
        tokio::spawn(async move {
            let mut file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
            // One-slot buffer for non-Event messages drained during batch collection.
            // try_recv() removes the message it reads, so a Checkpoint or Rotate pulled
            // out while batching must be held here and processed on the next iteration.
            let mut peeked: Option<WalMessage> = None;
            loop {
                let msg = match peeked.take() {
                    Some(m) => m,
                    None => match rx.recv().await {
                        Some(m) => m,
                        None => break,
                    },
                };
                match msg {
                    WalMessage::Event(event) => {
                        // Drain every Event already queued in the channel into a batch.
                        // Any non-Event message (Checkpoint, Rotate) stops the drain and
                        // is saved in `peeked` for the next outer loop iteration.
                        let mut batch = vec![event];
                        loop {
                            match rx.try_recv() {
                                Ok(WalMessage::Event(e)) => batch.push(e),
                                Ok(other) => { peeked = Some(other); break; }
                                Err(_) => break,
                            }
                        }
                        for e in &batch {
                            let json = serde_json::to_string(e).unwrap();
                            writeln!(file, "{}", json).unwrap();
                        }
                        // One flush per burst instead of one per event.
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
                    WalMessage::Rotate { checkpoint_offset, reply } => {
                        // Flush and sync everything currently in the old file
                        file.flush().unwrap();
                        file.sync_all().unwrap();

                        // Read the tail: events written after the checkpoint offset (including
                        // any post-Checkpoint-barrier events that arrived before this Rotate)
                        let mut old = File::open(&path).unwrap();
                        old.seek(SeekFrom::Start(checkpoint_offset)).unwrap();
                        let mut tail = Vec::new();
                        old.read_to_end(&mut tail).unwrap();
                        drop(old);

                        // Atomic replace: write tail to tmp then rename over the WAL
                        let tmp = format!("{}/wal_rotation.tmp", writer_data_dir);
                        fs::write(&tmp, &tail).unwrap();
                        fs::rename(&tmp, &path).unwrap();

                        // Reopen — old fd is now detached from the directory; new fd points
                        // to the rotated file. Future Event writes go to the correct file.
                        file = OpenOptions::new().create(true).append(true).open(&path).unwrap();

                        println!(
                            "[rotation] WAL rotated: freed {} bytes, tail {} bytes",
                            checkpoint_offset,
                            tail.len()
                        );
                        let _ = reply.send(());
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
                wal_start_offset = checkpoint.wal_offset;

                println!("[recovery] Checkpoint snapshot: {} campaigns, WAL offset {} bytes, taken at epoch {}",
                    checkpoint.campaigns.len(), checkpoint.wal_offset, checkpoint.timestamp_secs);
                for (campaign_id, camp) in &checkpoint.campaigns {
                    let mut arms: Vec<&String> = camp.arms.keys().collect();
                    arms.sort();
                    let feature_dim = camp.arms.values().next().map(|a| a.theta.len()).unwrap_or(0);
                    println!("[recovery]   campaign={} arms={:?} feature_dim={}", campaign_id, arms, feature_dim);
                }

                for (campaign_id, camp) in checkpoint.campaigns {
                    self.campaigns.write().insert(
                        campaign_id,
                        Campaign { alpha: camp.alpha, arms: RwLock::new(camp.arms) },
                    );
                }
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
                    // After WAL rotation the file contains only the tail and starts at byte 0.
                    // If the stored offset exceeds the file size the WAL was rotated; replay from 0.
                    let file_len = file.seek(SeekFrom::End(0)).unwrap();
                    let seek_to = if wal_start_offset <= file_len { wal_start_offset } else { 0 };
                    file.seek(SeekFrom::Start(seek_to)).unwrap();
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

        // 5. Write completed prediction→reward pairs from this WAL segment to Parquet.
        //    Read WAL 0..wal_offset, join Predicted+Rewarded on interaction_id, append to
        //    per-campaign Parquet files. Only matched pairs are written.
        let export_dir = format!("{}/exports", self.data_dir);
        fs::create_dir_all(&export_dir).map_err(|e| e.to_string())?;

        let mut predicted: HashMap<String, (String, String, Vec<f64>, u64)> = HashMap::new();
        let mut rewarded:  HashMap<String, (f64, u64)>                      = HashMap::new();

        if let Ok(wal_file) = File::open(&self.wal_path) {
            let limited = wal_file.take(wal_offset);
            for line in BufReader::new(limited).lines().flatten() {
                match serde_json::from_str::<DbEvent>(&line) {
                    Ok(DbEvent::Predicted { interaction_id, campaign_id, arm_id, context, timestamp_secs }) => {
                        predicted.insert(interaction_id, (campaign_id, arm_id, context, timestamp_secs));
                    }
                    Ok(DbEvent::Rewarded { interaction_id, reward, timestamp_secs }) => {
                        rewarded.insert(interaction_id, (reward, timestamp_secs));
                    }
                    _ => {}
                }
            }
        }

        // Match pairs and group by campaign
        let mut by_campaign: HashMap<String, Vec<CompletedInteraction>> = HashMap::new();
        let mut matched: HashSet<String> = HashSet::new();

        for (iid, (reward, rewarded_at)) in &rewarded {
            if let Some((campaign_id, arm_id, context, predicted_at)) = predicted.get(iid) {
                matched.insert(iid.clone());
                by_campaign.entry(campaign_id.clone()).or_default().push(CompletedInteraction {
                    interaction_id: iid.clone(),
                    arm_id:         arm_id.clone(),
                    context:        context.clone(),
                    reward:         *reward,
                    predicted_at:   *predicted_at,
                    rewarded_at:    *rewarded_at,
                });
            }
        }

        let mut parquet_rows = 0usize;
        for (campaign_id, interactions) in &by_campaign {
            let feature_dim = interactions[0].context.len();
            if let Err(e) = write_campaign_parquet(&export_dir, campaign_id, interactions, feature_dim) {
                println!("[checkpoint] Parquet write failed for {}: {}", campaign_id, e);
            } else {
                parquet_rows += interactions.len();
            }
        }

        // 6. Re-emit in-flight (unmatched) Predicted events into the WAL tail so that
        //    their reward — however delayed — lands in the same future WAL segment and
        //    can be matched at the next checkpoint.
        let mut reemit_count = 0usize;
        for (iid, record) in self.interactions.iter() {
            if !matched.contains(iid.as_ref()) {
                let event = DbEvent::Predicted {
                    interaction_id: iid.as_ref().clone(),
                    campaign_id:    record.campaign_id.clone(),
                    arm_id:         record.arm_id.clone(),
                    context:        record.context.to_vec(),
                    timestamp_secs: record.timestamp_secs,
                };
                let _ = self.event_tx.send(WalMessage::Event(event));
                reemit_count += 1;
            }
        }

        // 7. Rotate WAL — discard the prefix already embedded in the checkpoint
        let (rot_tx, rot_rx) = oneshot::channel::<()>();
        self.event_tx
            .send(WalMessage::Rotate { checkpoint_offset: wal_offset, reply: rot_tx })
            .map_err(|_| "WAL channel closed during rotation".to_string())?;
        rot_rx.await.map_err(|_| "WAL writer closed during rotation".to_string())?;

        let msg = format!(
            "Checkpoint written and WAL rotated: {} campaigns, offset {} bytes, {} interactions exported, {} in-flight re-emitted",
            data.campaigns.len(), wal_offset, parquet_rows, reemit_count
        );
        println!("[checkpoint] {}", msg);
        Ok(msg)
    }

    /// The unified math & memory updater
    fn apply_event_to_memory(&self, event: DbEvent) {
        match event {
            DbEvent::CampaignCreated { campaign_id, arms, feature_dim, alpha } => {
                let mut arms_map = HashMap::new();
                for arm in arms {
                    arms_map.insert(arm, ArmState::new(feature_dim));
                }
                self.campaigns.write().insert(
                    campaign_id,
                    Campaign { alpha, arms: RwLock::new(arms_map) },
                );
            }
            DbEvent::Predicted { interaction_id, campaign_id, arm_id, context, timestamp_secs } => {
                if let Some(campaign) = self.campaigns.read().get(&campaign_id) {
                    if let Some(arm) = campaign.arms.write().get_mut(&arm_id) {
                        arm.prediction_count += 1;
                    }
                }
                self.interactions.insert(
                    interaction_id,
                    InteractionRecord {
                        campaign_id, arm_id, context: Array1::from_vec(context),
                        probability: 0.0, timestamp_secs,
                    },
                );
            }
            DbEvent::Rewarded { interaction_id, reward, .. } => {
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

    pub fn add_campaign(&self, campaign_id: &str, arms: Vec<String>, feature_dim: usize, alpha: f64) -> bool {
        if self.campaigns.read().contains_key(campaign_id) {
            return false;
        }
        let event = DbEvent::CampaignCreated { campaign_id: campaign_id.to_string(), arms, feature_dim, alpha };
        self.apply_event_to_memory(event.clone());
        let _ = self.event_tx.send(WalMessage::Event(event));
        true
    }

    pub fn predict(&self, campaign_id: &str, context: Vec<f64>) -> Option<(String, String)> {
        // Score all arms under read locks, then drop every guard before calling
        // apply_event_to_memory. apply_event_to_memory acquires arms.write() to
        // increment prediction_count — a same-thread read→write on the same RwLock
        // is always a deadlock with parking_lot, so no guard may be live at that point.
        let best_arm = {
            let campaigns = self.campaigns.read();
            let campaign = campaigns.get(campaign_id)?;
            let context_arr = Array1::from_vec(context.clone());
            let arms = campaign.arms.read();

            // Guard: reject context whose dimension doesn't match the campaign's feature space.
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
            best_arm
            // campaigns and arms guards dropped here
        };

        let interaction_id = Uuid::new_v4().to_string();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let event = DbEvent::Predicted {
            interaction_id: interaction_id.clone(), campaign_id: campaign_id.to_string(),
            arm_id: best_arm.clone(), context, timestamp_secs: now,
        };

        self.apply_event_to_memory(event.clone());
        let _ = self.event_tx.send(WalMessage::Event(event));

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
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let event = DbEvent::Rewarded { interaction_id: interaction_id.to_string(), reward, timestamp_secs: now };
        self.apply_event_to_memory(event.clone());
        let _ = self.event_tx.send(WalMessage::Event(event));
        self.rewarded_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the directory where per-campaign Parquet files are written.
    pub fn export_dir(&self) -> String {
        format!("{}/exports", self.data_dir)
    }
}

/// Append completed interactions to a per-campaign Parquet file.
///
/// Schema: interaction_id | arm_id | reward | predicted_at | rewarded_at | feature_0 … feature_N
///
/// Uses Option A (read-existing + concat + full rewrite) for simplicity.
/// The file is written atomically via a .tmp rename.
pub fn write_campaign_parquet(
    export_dir: &str,
    campaign_id: &str,
    interactions: &[CompletedInteraction],
    feature_dim: usize,
) -> Result<(), String> {
    if interactions.is_empty() {
        return Ok(());
    }

    // Build new-rows DataFrame
    let new_df = interactions_to_df(interactions, feature_dim)?;

    // If an existing file is present, read it and concatenate
    let path = format!("{}/{}.parquet", export_dir, campaign_id);
    let mut combined = if std::path::Path::new(&path).exists() {
        let existing = LazyFrame::scan_parquet(&path, ScanArgsParquet::default())
            .map_err(|e| e.to_string())?
            .collect()
            .map_err(|e| e.to_string())?;
        concat([existing.lazy(), new_df.lazy()], UnionArgs::default())
            .map_err(|e| e.to_string())?
            .collect()
            .map_err(|e| e.to_string())?
    } else {
        new_df
    };

    // Atomic write via tmp rename
    let tmp = format!("{}/{}.parquet.tmp", export_dir, campaign_id);
    let mut file = File::create(&tmp).map_err(|e| e.to_string())?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut combined)
        .map_err(|e| e.to_string())?;
    fs::rename(&tmp, &path).map_err(|e| e.to_string())?;

    Ok(())
}

fn interactions_to_df(interactions: &[CompletedInteraction], feature_dim: usize) -> Result<DataFrame, String> {
    let interaction_ids: Vec<&str> = interactions.iter().map(|r| r.interaction_id.as_str()).collect();
    let arm_ids:         Vec<&str> = interactions.iter().map(|r| r.arm_id.as_str()).collect();
    let rewards:         Vec<f64>  = interactions.iter().map(|r| r.reward).collect();
    let predicted_ats:   Vec<i64>  = interactions.iter().map(|r| r.predicted_at as i64).collect();
    let rewarded_ats:    Vec<i64>  = interactions.iter().map(|r| r.rewarded_at  as i64).collect();

    let mut series: Vec<Series> = vec![
        Series::new("interaction_id".into(), interaction_ids),
        Series::new("arm_id".into(),         arm_ids),
        Series::new("reward".into(),         rewards),
        Series::new("predicted_at".into(),   predicted_ats),
        Series::new("rewarded_at".into(),    rewarded_ats),
    ];

    for f in 0..feature_dim {
        let col: Vec<f64> = interactions.iter().map(|r| r.context[f]).collect();
        let name = format!("feature_{}", f);
        series.push(Series::new(name.as_str().into(), col));
    }

    DataFrame::new(series).map_err(|e| e.to_string())
}
