use crate::state::{Algorithm, ArmDiagnostics, ArmReportStats, ArmState, CampaignCheckpoint, CampaignDiagnosticsData, CampaignReport, CheckpointData, CompletedInteraction, DbEvent, EngineError, EntropyStatus, EntropyTrend, InteractionRecord};
#[cfg(feature = "neural")]
use crate::state::{ProgressiveConfig, TournamentOutcome};
#[cfg(feature = "neural")]
use crate::neural::NeuralLinUCBState;
use moka::sync::Cache;
use ndarray::Array1;
use parking_lot::RwLock;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write as _};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{channel, error::TrySendError, Sender};
use tokio::sync::oneshot;
use uuid::Uuid;
use polars::prelude::*;

/// Basis-point bounds for the Progressive tournament traffic ramp.
const BPS_FLOOR: u32 = 1_000;   // 10% — minimum challenger exploration
#[cfg(feature = "neural")]
const BPS_CEIL:  u32 = 9_000;   // 90% — maximum before full promotion
const BPS_SCALE: u32 = 10_000;  // denominator for the U[0, BPS_SCALE) draw

pub enum WalMessage {
    Event(Arc<DbEvent>),
    Checkpoint { reply: oneshot::Sender<u64> },
    Rotate { checkpoint_offset: u64, reply: oneshot::Sender<()> },
}

/// WAL serialisation format.
/// Controlled by `BANDITDB_WAL_FORMAT` (default: json, opt-in: msgpack).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum WalFormat { Json, Msgpack }

impl WalFormat {
    pub fn from_env() -> Self {
        match std::env::var("BANDITDB_WAL_FORMAT").as_deref() {
            Ok("msgpack") => WalFormat::Msgpack,
            _             => WalFormat::Json,
        }
    }
}

/// Magic bytes written at the start of every new binary WAL file.
const WAL_MAGIC: &[u8; 4] = b"BDMP";

/// Detect the WAL format of an existing file by reading its first 4 bytes.
fn detect_wal_format(path: &str) -> WalFormat {
    if let Ok(mut f) = File::open(path) {
        let mut magic = [0u8; 4];
        if f.read_exact(&mut magic).is_ok() && &magic == WAL_MAGIC {
            return WalFormat::Msgpack;
        }
    }
    WalFormat::Json
}

/// Read `DbEvent`s from a WAL file slice, auto-detecting the format.
/// `start_offset` is the byte to start reading from (0 for the full file).
/// `end_offset`   is the exclusive byte limit (0 means read to end-of-file).
fn read_wal_slice(
    path:         &str,
    start_offset: u64,
    end_offset:   u64,
    format:       WalFormat,
) -> Vec<DbEvent> {
    let Ok(mut file) = File::open(path) else { return vec![] };

    // For binary files the first 4 bytes are the magic header.
    // Adjust start_offset so we never seek into the middle of the magic.
    let data_start = match format {
        WalFormat::Msgpack => start_offset.max(WAL_MAGIC.len() as u64),
        WalFormat::Json    => start_offset,
    };

    if file.seek(SeekFrom::Start(data_start)).is_err() { return vec![] }

    let mut events = Vec::new();

    match format {
        WalFormat::Json => {
            let reader: Box<dyn Read> = if end_offset > 0 {
                Box::new(file.take(end_offset.saturating_sub(data_start)))
            } else {
                Box::new(file)
            };
            for line in BufReader::new(reader).lines().map_while(Result::ok) {
                if let Ok(e) = serde_json::from_str::<DbEvent>(&line) {
                    events.push(e);
                }
            }
        }
        WalFormat::Msgpack => {
            let limit = if end_offset > 0 { end_offset.saturating_sub(data_start) } else { u64::MAX };
            let mut reader = file.take(limit);
            let mut len_buf = [0u8; 4];
            while reader.read_exact(&mut len_buf).is_ok() {
                let len = u32::from_le_bytes(len_buf) as usize;
                let mut bytes = vec![0u8; len];
                if reader.read_exact(&mut bytes).is_err() { break; }
                if let Ok(e) = rmp_serde::from_slice::<DbEvent>(&bytes) {
                    events.push(e);
                }
            }
        }
    }
    events
}

/// Monotonic Unix timestamp in whole seconds. Returns 0 on the (impossible) pre-epoch case.
fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

pub struct Campaign {
    pub alpha:                  f64,
    pub algorithm:              Algorithm,
    pub arms:                   RwLock<HashMap<String, ArmState>>,
    pub challenger_arms:        Option<RwLock<HashMap<String, ArmState>>>,
    pub metadata:               Option<serde_json::Value>,
    #[cfg(feature = "neural")]
    pub neural:                 Option<parking_lot::Mutex<NeuralLinUCBState>>,
    /// Challenger traffic in basis points (0–10000). Progressive campaigns start
    /// at 1000 (10% exploration) and ramp up/down based on SNIPS tournament results.
    pub challenger_traffic_bps: AtomicU32,
    /// Tournament win streak persisted through checkpoints. See ProgressiveConfig.
    pub tournament_wins:        std::sync::atomic::AtomicI32,
    /// Soft-deleted: archived campaigns are frozen — no new predictions or rewards.
    pub archived:               AtomicBool,
    /// f64 bits of the selection entropy computed at the last checkpoint.
    /// f64::NAN (the initial value) means no checkpoint has been written yet.
    pub last_checkpoint_entropy: AtomicU64,
    /// If set, A_inv and b are rescaled at each checkpoint to implement exponential
    /// forgetting over wall-clock time. None = no forgetting.
    pub decay_half_life_hours: Option<f64>,
}

impl Campaign {
    pub fn new(
        alpha:     f64,
        algorithm: Algorithm,
        arms:      RwLock<HashMap<String, ArmState>>,
        // None = derive challenger_arms from algorithm (new campaign).
        // Some(loaded) = restore checkpointed matrices (recovery path).
        challenger_arms:       Option<RwLock<HashMap<String, ArmState>>>,
        metadata:              Option<serde_json::Value>,
        decay_half_life_hours: Option<f64>,
    ) -> Self {
        // When challenger_arms is not explicitly provided (new campaign path), derive them
        // from the algorithm config so the caller doesn't have to duplicate the logic.
        let challenger_arms = challenger_arms.or_else(|| {
            if let Algorithm::Progressive(cfg) = &algorithm {
                let (arm_names, base_dim): (Vec<String>, usize) = {
                    let r = arms.read();
                    let dim = r.values().next().map(|a| a.theta.len()).unwrap_or(0);
                    (r.keys().cloned().collect(), dim)
                };
                let challenger_dim = match cfg.challenger.as_ref() {
                    Algorithm::NeuralLinUCB(ncfg) => ncfg.embed_dim,
                    _ => base_dim,
                };
                let c_map: HashMap<String, ArmState> = arm_names.into_iter()
                    .map(|name| (name, ArmState::new(challenger_dim)))
                    .collect();
                Some(RwLock::new(c_map))
            } else {
                None
            }
        });
        #[cfg(feature = "neural")]
        let neural = match &algorithm {
            Algorithm::NeuralLinUCB(cfg) | Algorithm::NeuralThompsonSampling(cfg) => {
                match NeuralLinUCBState::new(cfg) {
                    Ok(state) => Some(parking_lot::Mutex::new(state)),
                    Err(e) => { tracing::error!(error = %e, "neural: failed to init network"); None }
                }
            }
            Algorithm::Progressive(cfg) => {
                let neural_cfg = match cfg.challenger.as_ref() {
                    Algorithm::NeuralLinUCB(c) | Algorithm::NeuralThompsonSampling(c) => Some(c),
                    _ => None,
                };
                if let Some(neural_cfg) = neural_cfg {
                    match NeuralLinUCBState::new(neural_cfg) {
                        Ok(state) => Some(parking_lot::Mutex::new(state)),
                        Err(e) => { tracing::error!(error = %e, "neural: failed to init challenger network"); None }
                    }
                } else { None }
            }
            _ => None,
        };

        let initial_traffic = if let Algorithm::Progressive(_) = &algorithm { BPS_FLOOR } else { 0 };

        Self {
            alpha,
            algorithm,
            arms,
            challenger_arms,
            metadata,
            #[cfg(feature = "neural")]
            neural,
            challenger_traffic_bps:  AtomicU32::new(initial_traffic),
            tournament_wins:         std::sync::atomic::AtomicI32::new(0),
            archived:                AtomicBool::new(false),
            last_checkpoint_entropy: AtomicU64::new(f64::NAN.to_bits()),
            decay_half_life_hours,
        }
    }

    /// Returns the embedding of `context` for Algorithm 1 scoring.
    /// Identity for LinUCB / ThompsonSampling; MLP forward pass for NeuralLinUCB.
    #[cfg(feature = "neural")]
    pub fn embed(&self, context: &Array1<f64>) -> Array1<f64> {
        if let Some(neural) = &self.neural {
            return neural.lock().embed(context);
        }
        context.clone()
    }

    #[cfg(not(feature = "neural"))]
    pub fn embed(&self, context: &Array1<f64>) -> Array1<f64> {
        context.clone()
    }

}

/// ## Consistency model
///
/// `predict()`, `reward()`, and `add_campaign()` each perform two steps in order:
/// (1) enqueue an event to the WAL channel via `try_send`, then
/// (2) apply it to in-memory matrices via `apply_event_to_memory`.
///
/// These two steps are not atomic. Under concurrent load, two threads may enqueue
/// their events in order A → B (establishing WAL order) but have the scheduler
/// run their memory-apply steps in order B → A. After a crash, WAL replay restores
/// A → B order, producing a theta that differs from the pre-crash in-memory state.
///
/// This divergence is intentional and acceptable. Sherman-Morrison updates are
/// non-commutative in theory, but bandit convergence is insensitive to the ordering
/// of events within a millisecond scheduling window — both linearizations are valid
/// approximations of the true posterior. The checkpoint always captures the actual
/// in-memory state; post-crash WAL replay produces a valid (though not bit-identical)
/// approximation of that state.
///
/// In short: the system is AP at the per-event level. If you need strict
/// WAL-order == memory-order consistency, the WAL writer would need to become the
/// sole applier of events to memory, at the cost of one round-trip latency per request.
pub struct BanditDB {
    pub campaigns:        RwLock<HashMap<String, Campaign>>,
    pub interactions:     Cache<String, InteractionRecord>,
    pub event_tx:         Sender<WalMessage>,
    pub rewarded_count:   AtomicU64,
    pub wal_path:         String,
    pub data_dir:         String,
    /// Upper bound on feature_dim at campaign creation and context length at predict time.
    /// Prevents arm-matrix OOM from untrusted callers. Env: BANDITDB_MAX_FEATURE_DIM (default 4096).
    pub max_feature_dim:  usize,
    /// Upper bound on the number of arms per campaign.
    /// Env: BANDITDB_MAX_ARMS (default 1000).
    pub max_arms:         usize,
    /// False when the WAL writer task has encountered an unrecoverable I/O error.
    /// Exposed to the /health endpoint and checked by all write paths.
    pub wal_healthy:      Arc<AtomicBool>,
    /// WAL serialisation format determined at startup from `BANDITDB_WAL_FORMAT`.
    pub wal_format:       WalFormat,
    /// Optional audit log channel. When set, write-path operations emit a JSON
    /// summary line to a dedicated audit file (separate from application logs).
    pub audit_tx:             Option<tokio::sync::mpsc::Sender<String>>,
    /// Unix timestamp (seconds) of the last successful checkpoint. Used to compute
    /// elapsed time for time-aware campaign decay. Initialised to startup time so
    /// a restart followed immediately by checkpoint does not over-decay.
    pub last_checkpoint_secs: AtomicU64,
}

impl BanditDB {
    pub fn new(wal_path: &str, data_dir: &str) -> Self {
        let (tx, mut rx) = channel::<WalMessage>(100_000);

        let wal_healthy        = Arc::new(AtomicBool::new(true));
        let wal_healthy_writer = Arc::clone(&wal_healthy);

        // 1. Spawn the WAL writer task.
        //
        // Supports JSON (default) and MessagePack (opt-in via BANDITDB_WAL_FORMAT=msgpack).
        // Binary files are identified by a 4-byte magic header `BDMP`; JSON files have none.
        // Transient I/O errors are retried up to MAX_WAL_RETRIES times with exponential back-off.
        const MAX_WAL_RETRIES: u32 = 5;

        let write_format    = WalFormat::from_env();
        let path            = wal_path.to_string();
        let writer_data_dir = data_dir.to_string();

        tokio::spawn(async move {
            let mut file = match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(f)  => f,
                Err(e) => {
                    tracing::error!(error = %e, path = %path, "WAL writer: cannot open file");
                    wal_healthy_writer.store(false, Ordering::SeqCst);
                    return;
                }
            };

            // Write magic header when creating a new binary WAL file.
            if write_format == WalFormat::Msgpack {
                if let Ok(0) = file.seek(SeekFrom::End(0)) {
                    let _ = file.write_all(WAL_MAGIC);
                    let _ = file.flush();
                }
            }

            // One-slot peek buffer for non-Event messages surfaced during batch drain.
            let mut peeked: Option<WalMessage> = None;

            loop {
                let msg = match peeked.take() {
                    Some(m) => m,
                    None    => match rx.recv().await {
                        Some(m) => m,
                        None    => break,
                    },
                };

                let mut fatal_err: Option<std::io::Error> = None;

                match msg {
                    WalMessage::Event(event_arc) => {
                        let mut batch = vec![event_arc];
                        loop {
                            match rx.try_recv() {
                                Ok(WalMessage::Event(e)) => batch.push(e),
                                Ok(other) => { peeked = Some(other); break; }
                                Err(_)    => break,
                            }
                        }

                        let mut attempt = 0u32;
                        loop {
                            let mut write_err: Option<std::io::Error> = None;
                            'write: {
                                for e in &batch {
                                    match write_format {
                                        WalFormat::Json => {
                                            let json = serde_json::to_string(e.as_ref()).unwrap_or_default();
                                            if let Err(err) = writeln!(file, "{json}") {
                                                write_err = Some(err); break 'write;
                                            }
                                        }
                                        WalFormat::Msgpack => {
                                            match rmp_serde::to_vec_named(e.as_ref()) {
                                                Err(err) => {
                                                    write_err = Some(std::io::Error::new(
                                                        std::io::ErrorKind::InvalidData, err));
                                                    break 'write;
                                                }
                                                Ok(bytes) => {
                                                    let len = (bytes.len() as u32).to_le_bytes();
                                                    if let Err(err) = file.write_all(&len) { write_err = Some(err); break 'write; }
                                                    if let Err(err) = file.write_all(&bytes) { write_err = Some(err); break 'write; }
                                                }
                                            }
                                        }
                                    }
                                }
                                if let Err(err) = file.flush() { write_err = Some(err); }
                            }
                            match write_err {
                                None => break,
                                Some(e) if attempt < MAX_WAL_RETRIES => {
                                    attempt += 1;
                                    let delay_ms = 50u64 * (1 << attempt);
                                    tracing::warn!(error = %e, attempt, delay_ms,
                                        "WAL writer: transient error — retrying");
                                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                                    if let Ok(f) = OpenOptions::new().create(true).append(true).open(&path) {
                                        file = f;
                                    }
                                }
                                Some(e) => { fatal_err = Some(e); break; }
                            }
                        }
                    }

                    WalMessage::Checkpoint { reply } => {
                        match file.flush()
                            .and_then(|_| file.sync_all())
                            .and_then(|_| file.seek(SeekFrom::End(0)))
                        {
                            Ok(offset) => { let _ = reply.send(offset); }
                            Err(e)     => fatal_err = Some(e),
                        }
                    }

                    WalMessage::Rotate { checkpoint_offset, reply } => {
                        if let Err(e) = file.flush().and_then(|_| file.sync_all()) {
                            fatal_err = Some(e);
                        } else {
                            let rotate = (|| -> std::io::Result<(usize, std::fs::File)> {
                                let mut old = File::open(&path)?;
                                old.seek(SeekFrom::Start(checkpoint_offset))?;
                                let mut tail = Vec::new();
                                old.read_to_end(&mut tail)?;
                                drop(old);

                                // Prepend binary magic to the new file if in msgpack mode.
                                let new_content: Vec<u8> = if write_format == WalFormat::Msgpack {
                                    let mut v = Vec::with_capacity(WAL_MAGIC.len() + tail.len());
                                    v.extend_from_slice(WAL_MAGIC);
                                    v.extend_from_slice(&tail);
                                    v
                                } else {
                                    tail
                                };

                                let tmp = format!("{writer_data_dir}/wal_rotation.tmp");
                                fs::write(&tmp, &new_content)?;
                                fs::rename(&tmp, &path)?;
                                let new_file = OpenOptions::new().append(true).open(&path)?;
                                Ok((new_content.len(), new_file))
                            })();
                            match rotate {
                                Ok((total_len, new_file)) => {
                                    file = new_file;
                                    tracing::info!(
                                        freed_bytes  = checkpoint_offset,
                                        new_file_len = total_len,
                                        "WAL rotated"
                                    );
                                    let _ = reply.send(());
                                }
                                Err(e) => fatal_err = Some(e),
                            }
                        }
                    }
                }

                if let Some(e) = fatal_err {
                    tracing::error!(error = %e, "WAL writer: unrecoverable I/O error — shutting down");
                    wal_healthy_writer.store(false, Ordering::SeqCst);
                    break;
                }
            }
        });

        let wal_format = write_format; // already determined above for the writer task

        let ttl_secs: u64 = std::env::var("BANDITDB_REWARD_TTL_SECS")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(86400);
        let max_feature_dim: usize = std::env::var("BANDITDB_MAX_FEATURE_DIM")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
        let max_arms: usize = std::env::var("BANDITDB_MAX_ARMS")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(1000);

        if wal_format == WalFormat::Msgpack {
            tracing::info!("WAL format: MessagePack (binary, length-framed)");
        }

        // 2. Optional audit log writer task.
        //    Activate by setting BANDITDB_AUDIT_LOG=/path/to/audit.jsonl.
        //    Each line is a JSON object: { "ts": <unix_secs>, "action": "...", "campaign_id": "..." }
        //    Uses a bounded channel (10_000) — audit lines are dropped with a warning under disk pressure
        //    rather than buffering unboundedly.
        let audit_tx: Option<tokio::sync::mpsc::Sender<String>> =
            match std::env::var("BANDITDB_AUDIT_LOG") {
                Err(_) => None,
                Ok(audit_path) => {
                    let (atx, mut arx) = tokio::sync::mpsc::channel::<String>(10_000);
                    tracing::info!(path = %audit_path, "audit logging enabled");
                    tokio::spawn(async move {
                        use std::io::Write as _;
                        let mut f = match OpenOptions::new().create(true).append(true).open(&audit_path) {
                            Ok(f)  => f,
                            Err(e) => {
                                tracing::error!(error = %e, path = %audit_path, "audit: cannot open log file");
                                return;
                            }
                        };
                        while let Some(line) = arx.recv().await {
                            let _ = writeln!(f, "{line}");
                            let _ = f.flush();
                        }
                    });
                    Some(atx)
                }
            };

        let db = Self {
            campaigns:            RwLock::new(HashMap::new()),
            interactions:         Cache::builder().time_to_live(Duration::from_secs(ttl_secs)).build(),
            event_tx:             tx,
            rewarded_count:       AtomicU64::new(0),
            wal_path:             wal_path.to_string(),
            data_dir:             data_dir.to_string(),
            max_feature_dim,
            max_arms,
            wal_healthy,
            wal_format,
            audit_tx,
            last_checkpoint_secs: AtomicU64::new(now_secs()),
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
                self.last_checkpoint_secs.store(checkpoint.timestamp_secs, Ordering::Relaxed);

                tracing::info!(
                    campaigns  = checkpoint.campaigns.len(),
                    wal_offset = checkpoint.wal_offset,
                    epoch      = checkpoint.timestamp_secs,
                    "recovery: checkpoint snapshot"
                );
                for (campaign_id, camp) in &checkpoint.campaigns {
                    let mut arms: Vec<&String> = camp.arms.keys().collect();
                    arms.sort();
                    let feature_dim = camp.arms.values().next().map(|a| a.theta.len()).unwrap_or(0);
                    tracing::info!(campaign = %campaign_id, ?arms, feature_dim, "recovery: campaign");
                }

                for (campaign_id, camp) in checkpoint.campaigns {
                    let challenger_arms = camp.challenger_arms.map(RwLock::new);
                    let campaign = Campaign::new(camp.alpha, camp.algorithm, RwLock::new(camp.arms), challenger_arms, camp.metadata, camp.decay_half_life_hours);
                    campaign.challenger_traffic_bps.store(camp.challenger_traffic_bps, Ordering::Relaxed);
                    campaign.tournament_wins.store(camp.tournament_wins, Ordering::Relaxed);
                    campaign.archived.store(camp.archived, Ordering::Relaxed);
                    if let Some(e) = camp.entropy_snapshot {
                        campaign.last_checkpoint_entropy.store(e.to_bits(), Ordering::Relaxed);
                    }
                    self.campaigns.write().insert(campaign_id, campaign);
                }

                // Load saved neural weights if present
                #[cfg(feature = "neural")]
                {
                    let neural_dir = self.neural_dir();
                    let campaigns = self.campaigns.read();
                    for (campaign_id, campaign) in campaigns.iter() {
                        if let Some(neural) = &campaign.neural {
                            let path = format!("{neural_dir}/{campaign_id}.safetensors");
                            if std::path::Path::new(&path).exists() {
                                match neural.lock().load(&path) {
                                    Ok(_)  => tracing::info!(campaign = %campaign_id, "recovery: loaded neural weights"),
                                    Err(e) => tracing::warn!(campaign = %campaign_id, error = %e, "recovery: failed to load neural weights"),
                                }
                            }
                        }
                    }
                }
            }
            None => {
                tracing::info!("recovery: no checkpoint found — replaying WAL from beginning");
            }
        }

        // Phase 2: Replay WAL events after the checkpoint offset.
        // Auto-detect format (JSON vs MessagePack) from the file's magic bytes.
        if !std::path::Path::new(wal_path).exists() {
            tracing::info!("recovery: no WAL found — starting fresh");
        } else {
            let fmt = detect_wal_format(wal_path);

            // After WAL rotation the file contains only the tail starting at byte 0.
            // If the stored offset exceeds the current file size, the WAL was rotated — replay from 0.
            let file_len = File::open(wal_path)
                .and_then(|mut f| f.seek(SeekFrom::End(0)))
                .unwrap_or(0);
            let start = if wal_start_offset <= file_len { wal_start_offset } else { 0 };

            let events = read_wal_slice(wal_path, start, 0, fmt);
            let count  = events.len();
            for event in events {
                self.apply_event_to_memory(&event);
            }
            tracing::info!(events = count, ?fmt, "recovery: WAL tail replayed");
        }

        tracing::info!(campaigns = self.campaigns.read().len(), "recovery: ready");
    }

    pub async fn checkpoint(&self) -> Result<String, String> {
        // Fast-fail: if the WAL writer is dead the blocking send below would hang.
        if !self.wal_healthy.load(Ordering::SeqCst) {
            return Err("WAL writer is unavailable — server needs restart".to_string());
        }

        // 1. Send flush barrier through the WAL channel — writer drains all prior
        //    events to disk before replying with the confirmed byte offset.
        let (reply_tx, reply_rx) = oneshot::channel::<u64>();
        self.event_tx
            .send(WalMessage::Checkpoint { reply: reply_tx })
            .await
            .map_err(|_| "WAL channel closed".to_string())?;

        let wal_offset = reply_rx.await.map_err(|_| "WAL writer closed".to_string())?;

        // 2. Parquet export: read WAL [0, wal_offset), join Predicted+Rewarded pairs,
        //    write Parquet shards. Runs inside spawn_blocking so the synchronous WAL
        //    scan and Parquet I/O do not stall the tokio async runtime.
        //
        //    Done BEFORE the neural/tournament block so WAL rotation (step 9) doesn't
        //    discard events we still need to read.
        let export_dir = format!("{}/exports", self.data_dir);
        fs::create_dir_all(&export_dir).map_err(|e| e.to_string())?;

        let wal_path_clone  = self.wal_path.clone();
        let export_dir_clone = export_dir.clone();
        let wal_fmt         = self.wal_format;

        let (matched, parquet_rows) = tokio::task::spawn_blocking(move || {
            // Scan the WAL for matched Predicted+Rewarded pairs.
            #[allow(clippy::type_complexity)] // matched-pair accumulator; a named type alias would not aid clarity here
            let mut predicted: HashMap<String, (String, String, Vec<f64>, Option<HashMap<String, f64>>, u64)> = HashMap::new();
            let mut rewarded:  HashMap<String, (f64, u64)> = HashMap::new();

            for event in read_wal_slice(&wal_path_clone, 0, wal_offset, wal_fmt) {
                match event {
                    DbEvent::Predicted { interaction_id, campaign_id, arm_id, context, timestamp_secs, arm_propensities, .. } => {
                        predicted.insert(interaction_id, (campaign_id, arm_id, context, arm_propensities, timestamp_secs));
                    }
                    DbEvent::Rewarded { interaction_id, reward, timestamp_secs } => {
                        rewarded.insert(interaction_id, (reward, timestamp_secs));
                    }
                    _ => {}
                }
            }

            let mut by_campaign: HashMap<String, Vec<CompletedInteraction>> = HashMap::new();
            let mut matched:     HashSet<String>                             = HashSet::new();

            for (iid, (reward, rewarded_at)) in &rewarded {
                if let Some((campaign_id, arm_id, context, arm_propensities, predicted_at)) = predicted.get(iid) {
                    matched.insert(iid.clone());
                    let propensity = arm_propensities.as_ref().and_then(|m| m.get(arm_id.as_str())).copied();
                    by_campaign.entry(campaign_id.clone()).or_default().push(CompletedInteraction {
                        interaction_id: iid.clone(),
                        arm_id:         arm_id.clone(),
                        context:        context.clone(),
                        reward:         *reward,
                        predicted_at:   *predicted_at,
                        rewarded_at:    *rewarded_at,
                        propensity,
                    });
                }
            }

            let mut rows = 0usize;
            for (campaign_id, interactions) in &by_campaign {
                let feature_dim = interactions[0].context.len();
                match write_campaign_parquet(&export_dir_clone, campaign_id, interactions, feature_dim) {
                    Err(e) => tracing::error!(campaign = %campaign_id, error = %e, "checkpoint: Parquet write failed"),
                    Ok(())  => rows += interactions.len(),
                }
            }

            (matched, rows)
        }).await.map_err(|e| format!("checkpoint export task panicked: {e}"))?;

        // 6. Re-emit in-flight (unmatched) Predicted events into the WAL tail so that
        //    their reward — however delayed — lands in the same future WAL segment and
        //    can be matched at the next checkpoint.
        let mut reemit_count = 0usize;
        for (iid, record) in self.interactions.iter() {
            if !matched.contains(iid.as_ref()) {
                let event = DbEvent::Predicted {
                    interaction_id:   iid.as_ref().clone(),
                    campaign_id:      record.campaign_id.clone(),
                    arm_id:           record.arm_id.clone(),
                    context:          record.context.to_vec(),
                    timestamp_secs:   record.timestamp_secs,
                    arm_propensities: record.arm_propensities.clone(),
                    // Re-emitted for reward matching; already counted + in checkpoint.
                    is_reemit:        true,
                };
                let _ = self.event_tx.send(WalMessage::Event(Arc::new(event))).await;
                reemit_count += 1;
            }
        }

        // 7. (Neural) Run Algorithm 2 on campaigns that have accumulated enough rewards,
        //    then re-accumulate arm matrices in embedding space (warm start).
        //    Runs before WAL rotation so the updated weights are included in the checkpoint.
        #[cfg(feature = "neural")]
        {
            let neural_dir = self.neural_dir();
            fs::create_dir_all(&neural_dir).ok();

            let campaigns = self.campaigns.read();
            for (campaign_id, campaign) in campaigns.iter() {
                let Some(neural_mutex) = &campaign.neural else { continue };
                let mut neural = neural_mutex.lock();

                let needs_retrain  = neural.should_retrain();
                let is_progressive = matches!(&campaign.algorithm, Algorithm::Progressive(_));

                // Skip the whole block if neither a retrain nor a tournament evaluation is due.
                if !needs_retrain && !is_progressive { continue }

                // 7a. Algorithm 2: retrain the MLP and warm-start arm matrices.
                //
                // Lock order fix: predict() holds arms.read() then acquires neural.lock()
                // via embed(). Holding neural.lock() while calling arms.write() inverts
                // that order → deadlock. Fix: retrain + save under neural lock, drop it,
                // then apply new arm states, then re-acquire neural lock for tournament.
                if needs_retrain {
                    let arms_snapshot: HashMap<String, ArmState> = {
                        let target_arms = campaign.challenger_arms.as_ref().unwrap_or(&campaign.arms);
                        target_arms.read().iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                    };

                    let new_arm_states = match neural.retrain(&arms_snapshot) {
                        Err(e) => { tracing::error!(campaign = %campaign_id, error = %e, "checkpoint: neural retrain failed"); None }
                        Ok(_)  => Some(neural.reaccumulate(&arms_snapshot)),
                    };

                    let weights_path = format!("{neural_dir}/{campaign_id}.safetensors");
                    if let Err(e) = neural.save(&weights_path) {
                        tracing::warn!(campaign = %campaign_id, error = %e, "checkpoint: failed to save neural weights");
                    }
                    {
                        let losses = &neural.last_retrain_losses;
                        let initial = losses.first().copied().unwrap_or(0.0);
                        let final_l = losses.last().copied().unwrap_or(0.0);
                        let improv  = if initial > 0.0 { (initial - final_l) / initial * 100.0 } else { 0.0 };
                        tracing::info!(
                            campaign        = %campaign_id,
                            steps           = losses.len(),
                            initial_loss    = format!("{initial:.4}"),
                            final_loss      = format!("{final_l:.4}"),
                            improvement_pct = format!("{improv:.1}"),
                            "checkpoint: neural retrain complete"
                        );
                    }

                    // Drop neural lock before arms.write() to preserve lock order.
                    drop(neural);

                    if let Some(new_arm_states) = new_arm_states {
                        let target_arms = campaign.challenger_arms.as_ref().unwrap_or(&campaign.arms);
                        let mut arms_write = target_arms.write();
                        for (arm_id, new_state) in new_arm_states {
                            arms_write.insert(arm_id, new_state);
                        }
                    }

                    // 7b. Tournament — re-acquire neural lock (no arms lock held).
                    if is_progressive {
                        let neural = neural_mutex.lock();
                        if let Algorithm::Progressive(cfg) = &campaign.algorithm {
                            run_tournament(campaign_id, campaign, &neural, cfg);
                        }
                    }
                } else {
                    // 7b. No retrain due — tournament evaluation only (neural lock still held).
                    if let Algorithm::Progressive(cfg) = &campaign.algorithm {
                        run_tournament(campaign_id, campaign, &neural, cfg);
                    }
                }
            }
        }

        // 4b. Time-aware decay: apply exponential forgetting to campaigns that have
        //     decay_half_life_hours set. Done after neural retrain so fresh matrices
        //     are decayed, and before snapshot so checkpoint.json reflects the decayed state.
        {
            let now  = now_secs();
            let last = self.last_checkpoint_secs.load(Ordering::Relaxed);
            let elapsed_hours = now.saturating_sub(last) as f64 / 3600.0;

            for (campaign_id, campaign) in self.campaigns.read().iter() {
                let Some(half_life) = campaign.decay_half_life_hours else { continue };
                if half_life <= 0.0 || elapsed_hours <= 0.0 { continue }

                let lambda     = 0.5_f64.powf(elapsed_hours / half_life).max(0.01);
                let inv_lambda = 1.0 / lambda;

                let apply_decay = |arms: &RwLock<HashMap<String, ArmState>>| {
                    for arm in arms.write().values_mut() {
                        arm.a_inv *= inv_lambda;
                        arm.b     *= lambda;
                        arm.theta  = arm.a_inv.dot(&arm.b);
                        *arm.chol_cache.lock() = None;
                    }
                };
                apply_decay(&campaign.arms);
                if let Some(c_arms) = &campaign.challenger_arms {
                    apply_decay(c_arms);
                }
                tracing::debug!(campaign = %campaign_id, lambda, elapsed_hours, "checkpoint: decay applied");
            }
        }

        // 5. Snapshot all campaign matrices under read lock — done AFTER neural retrain
        //    and tournament evaluation so checkpoint.json captures post-retrain arm matrices
        //    and post-tournament challenger_traffic_bps / tournament_wins.
        let campaigns_snapshot: HashMap<String, CampaignCheckpoint> = {
            let campaigns = self.campaigns.read();
            campaigns.iter().map(|(id, campaign)| {
                let arms_snapshot: HashMap<String, ArmState> = campaign.arms.read()
                    .iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                let challenger_snapshot = campaign.challenger_arms.as_ref().map(|c| {
                    c.read().iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                });
                let entropy_at_checkpoint = selection_entropy(
                    &arms_snapshot.values()
                        .map(|s| s.prediction_count.load(Ordering::Relaxed))
                        .collect::<Vec<_>>(),
                );
                campaign.last_checkpoint_entropy.store(entropy_at_checkpoint.to_bits(), Ordering::Relaxed);

                (id.clone(), CampaignCheckpoint {
                    alpha:                  campaign.alpha,
                    algorithm:              campaign.algorithm.clone(),
                    arms:                   arms_snapshot,
                    challenger_arms:        challenger_snapshot,
                    challenger_traffic_bps: campaign.challenger_traffic_bps.load(Ordering::SeqCst),
                    tournament_wins:        campaign.tournament_wins.load(Ordering::SeqCst),
                    archived:               campaign.archived.load(Ordering::SeqCst),
                    metadata:               campaign.metadata.clone(),
                    entropy_snapshot:       Some(entropy_at_checkpoint),
                    decay_half_life_hours:  campaign.decay_half_life_hours,
                })
            }).collect()
        };

        let timestamp_secs = now_secs();
        self.last_checkpoint_secs.store(timestamp_secs, Ordering::Relaxed);
        let data = CheckpointData { wal_offset, timestamp_secs, campaigns: campaigns_snapshot };
        let json = serde_json::to_string(&data).map_err(|e| e.to_string())?;

        let tmp_path  = format!("{}/checkpoint.tmp",  self.data_dir);
        let dest_path = format!("{}/checkpoint.json", self.data_dir);
        fs::write(&tmp_path, &json).map_err(|e| e.to_string())?;
        fs::rename(&tmp_path, &dest_path).map_err(|e| e.to_string())?;

        // 9. Rotate WAL — discard the prefix already embedded in the checkpoint
        let (rot_tx, rot_rx) = oneshot::channel::<()>();
        self.event_tx
            .send(WalMessage::Rotate { checkpoint_offset: wal_offset, reply: rot_tx })
            .await
            .map_err(|_| "WAL channel closed during rotation".to_string())?;
        rot_rx.await.map_err(|_| "WAL writer closed during rotation".to_string())?;

        let msg = format!(
            "Checkpoint written and WAL rotated: {} campaigns, offset {} bytes, {} interactions exported, {} in-flight re-emitted",
            data.campaigns.len(), wal_offset, parquet_rows, reemit_count
        );
        tracing::info!(
            campaigns  = data.campaigns.len(),
            wal_offset,
            parquet_rows,
            reemit_count,
            "checkpoint complete"
        );
        Ok(msg)
    }

    /// Apply a WAL event to in-memory state. Called from both the hot path (online updates)
    /// and from WAL replay during recovery. Takes a reference — callers own the Arc.
    fn apply_event_to_memory(&self, event: &DbEvent) {
        match event {
            DbEvent::CampaignCreated { campaign_id, arms, feature_dim, alpha, algorithm, metadata, decay_half_life_hours } => {
                let arms_map: HashMap<String, ArmState> = arms.iter()
                    .map(|arm| (arm.clone(), ArmState::new(*feature_dim)))
                    .collect();
                // challenger_arms derived by Campaign::new from algorithm (None = derive).
                self.campaigns.write().insert(
                    campaign_id.clone(),
                    Campaign::new(*alpha, algorithm.clone(), RwLock::new(arms_map), None, metadata.clone(), *decay_half_life_hours),
                );
            }
            DbEvent::Predicted { interaction_id, campaign_id, arm_id, context, timestamp_secs, arm_propensities, is_reemit } => {
                // WAL replay path: insert into the interactions cache so delayed rewards
                // can match.
                self.interactions.insert(
                    interaction_id.clone(),
                    InteractionRecord {
                        campaign_id: campaign_id.clone(),
                        arm_id:      arm_id.clone(),
                        context:     Array1::from_vec(context.clone()),
                        arm_propensities: arm_propensities.clone(),
                        timestamp_secs: *timestamp_secs,
                    },
                );
                // Restore the prediction counter on replay so it survives restart
                // instead of resetting to the checkpoint value. fetch_add works under a
                // read lock since prediction_count is atomic. Base arms only — the WAL
                // Predicted event does not record whether a Progressive challenger draw
                // was used, so the base/challenger split is not reconstructable here.
                //
                // SKIP re-emitted predictions: they were already counted live and are
                // captured in the checkpoint snapshot, so counting them here would
                // double-count after a checkpoint+rotation cycle.
                if !is_reemit {
                    if let Some(campaign) = self.campaigns.read().get(campaign_id) {
                        if let Some(arm_state) = campaign.arms.read().get(arm_id.as_str()) {
                            arm_state.prediction_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
            DbEvent::Rewarded { interaction_id, reward, .. } => {
                if let Some(record) = self.interactions.get(interaction_id.as_str()) {
                    if let Some(campaign) = self.campaigns.read().get(&record.campaign_id) {
                        // Helper: pick embedding based on which sub-algorithm a branch uses.
                        let embed_for = |algo: &Algorithm| -> Array1<f64> {
                            match algo {
                                Algorithm::NeuralLinUCB(_) => campaign.embed(&record.context),
                                _ => record.context.clone(),
                            }
                        };

                        // Shadow learning: every reward updates both base and challenger.
                        let base_features = match &campaign.algorithm {
                            Algorithm::Progressive(cfg) => embed_for(cfg.base.as_ref()),
                            Algorithm::NeuralLinUCB(_) | Algorithm::NeuralThompsonSampling(_) => campaign.embed(&record.context),
                            _ => record.context.clone(),
                        };
                        if let Some(arm_state) = campaign.arms.write().get_mut(record.arm_id.as_str()) {
                            arm_state.update(&base_features, *reward);
                        }

                        if let Some(c_arms) = &campaign.challenger_arms {
                            let challenger_features = match &campaign.algorithm {
                                Algorithm::Progressive(cfg) => embed_for(cfg.challenger.as_ref()),
                                _ => campaign.embed(&record.context),
                            };
                            if let Some(arm_state) = c_arms.write().get_mut(record.arm_id.as_str()) {
                                arm_state.update(&challenger_features, *reward);
                            }
                        }

                        #[cfg(feature = "neural")]
                        if let Some(neural) = &campaign.neural {
                            let propensity = record.arm_propensities.as_ref()
                                .and_then(|m| m.get(&record.arm_id))
                                .cloned()
                                .unwrap_or(1.0);
                            neural.lock().push(
                                record.context.to_vec(),
                                record.arm_id.clone(),
                                *reward,
                                propensity,
                            );
                        }
                    }
                    self.interactions.invalidate(interaction_id.as_str());
                }
            }
            DbEvent::CampaignDeleted { campaign_id } => {
                self.campaigns.write().remove(campaign_id.as_str());
            }
            DbEvent::CampaignArchived { campaign_id, .. } => {
                if let Some(c) = self.campaigns.read().get(campaign_id.as_str()) {
                    c.archived.store(true, Ordering::Relaxed);
                }
            }
            DbEvent::CampaignRestored { campaign_id, .. } => {
                if let Some(c) = self.campaigns.read().get(campaign_id.as_str()) {
                    c.archived.store(false, Ordering::Relaxed);
                }
            }
        }
    }

    // --- The Public API ---

    fn wal_try_send(&self, event: Arc<DbEvent>) -> Result<(), EngineError> {
        self.event_tx.try_send(WalMessage::Event(event))
            .map_err(|e| match e {
                TrySendError::Full(_)   => EngineError::WalFull,
                TrySendError::Closed(_) => EngineError::WalUnavailable,
            })
    }

    fn campaign_not_found(id: &str) -> EngineError {
        EngineError::NotFound(format!("Campaign '{id}' not found"))
    }

    #[allow(clippy::too_many_arguments)] // campaign construction params; grouping into a struct would only move the noise
    pub fn add_campaign(
        &self,
        campaign_id:          &str,
        arms:                 Vec<String>,
        feature_dim:          usize,
        alpha:                f64,
        algorithm:            Algorithm,
        metadata:             Option<serde_json::Value>,
        decay_half_life_hours: Option<f64>,
    ) -> Result<(), EngineError> {
        if self.campaigns.read().contains_key(campaign_id) {
            return Err(EngineError::AlreadyExists(format!("Campaign '{campaign_id}' already exists")));
        }
        let event = Arc::new(DbEvent::CampaignCreated {
            campaign_id: campaign_id.to_string(), arms, feature_dim, alpha, algorithm, metadata, decay_half_life_hours,
        });
        // WAL before memory. See BanditDB consistency-model doc comment.
        self.wal_try_send(Arc::clone(&event))?;
        self.apply_event_to_memory(&event);
        self.audit("create", campaign_id, None);
        Ok(())
    }

    pub fn predict(&self, campaign_id: &str, context: Vec<f64>) -> Result<(String, String), EngineError> {
        // All scoring happens under read locks. prediction_count is incremented here
        // (inside the lock, before guards drop) to avoid a second lock acquisition in
        // apply_event_to_memory. Guards are dropped before WAL + cache insert.
        let (best_arm, arm_propensities) = {
            let campaigns = self.campaigns.read();
            let campaign  = campaigns.get(campaign_id)
                .ok_or_else(|| Self::campaign_not_found(campaign_id))?;
            if campaign.archived.load(Ordering::Relaxed) {
                return Err(EngineError::NotFound(format!("Campaign '{campaign_id}' is archived")));
            }
            let context_arr = Array1::from_vec(context.clone());

            let (active_algo, arms_guard) = match &campaign.algorithm {
                Algorithm::Progressive(cfg) => {
                    let traffic_bps = campaign.challenger_traffic_bps.load(Ordering::Relaxed);
                    let use_challenger = rand::thread_rng().gen_range(0u32..BPS_SCALE) < traffic_bps;
                    if use_challenger {
                        let c_arms = campaign.challenger_arms.as_ref()
                            .ok_or_else(|| EngineError::Internal("challenger_arms missing".into()))?;
                        (cfg.challenger.as_ref(), c_arms.read())
                    } else {
                        (cfg.base.as_ref(), campaign.arms.read())
                    }
                }
                _ => (&campaign.algorithm, campaign.arms.read())
            };

            let expected_context_dim = match active_algo {
                Algorithm::NeuralLinUCB(cfg) | Algorithm::NeuralThompsonSampling(cfg) => cfg.context_dim,
                _ => arms_guard.iter().next().map(|(_, a)| a.theta.len()).unwrap_or(0),
            };
            if context_arr.len() != expected_context_dim {
                return Err(EngineError::BadRequest(format!(
                    "Context dimension mismatch: expected {expected_context_dim}, got {}", context_arr.len()
                )));
            }

            let features = match active_algo {
                Algorithm::NeuralLinUCB(_) | Algorithm::NeuralThompsonSampling(_) => campaign.embed(&context_arr),
                _ => context_arr,
            };

            let scores: Vec<(String, f64)> = arms_guard.iter().map(|(arm_id, state)| {
                let score = match active_algo {
                    Algorithm::ThompsonSampling | Algorithm::NeuralThompsonSampling(_) => state.score_ts(&features, campaign.alpha),
                    _ => state.score(&features, campaign.alpha),
                };
                (arm_id.clone(), score)
            }).collect();

            let best_arm = scores.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(id, _)| id.clone())
                .unwrap_or_default();

            // Increment prediction counter here — avoids a second lock acquisition below.
            if let Some(arm_state) = arms_guard.get(best_arm.as_str()) {
                arm_state.prediction_count.fetch_add(1, Ordering::Relaxed);
            }

            let arm_propensities = match active_algo {
                Algorithm::ThompsonSampling | Algorithm::NeuralThompsonSampling(_) => {
                    // Adaptive Monte Carlo: draw N trials and count wins per arm.
                    // N scales with posterior spread (A_inv diagonal) — large near cold-start
                    // where many samples are needed; small once the posterior concentrates.
                    // The first trial's winner is already known (best_arm from initial scores draw).
                    let n = ts_propensity_samples(&arms_guard);
                    let mut counts: HashMap<String, u32> = arms_guard.keys()
                        .map(|id| (id.clone(), 0u32))
                        .collect();
                    *counts.entry(best_arm.clone()).or_insert(0) += 1;
                    for _ in 1..n {
                        if let Some((winner, _)) = arms_guard.iter()
                            .map(|(id, state)| (id.clone(), state.score_ts(&features, campaign.alpha)))
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        {
                            *counts.entry(winner).or_insert(0) += 1;
                        }
                    }
                    let n_f = n as f64;
                    Some(counts.into_iter().map(|(id, c)| (id, c as f64 / n_f)).collect())
                }
                _ => Some(softmax_propensities(&scores)),
            };

            (best_arm, arm_propensities)
            // All guards dropped here.
        };

        let interaction_id = Uuid::new_v4().to_string();
        let now = now_secs();
        let event = Arc::new(DbEvent::Predicted {
            interaction_id:   interaction_id.clone(),
            campaign_id:      campaign_id.to_string(),
            arm_id:           best_arm.clone(),
            context:          context.clone(),
            timestamp_secs:   now,
            arm_propensities: arm_propensities.clone(),
            is_reemit:        false,
        });

        // WAL before memory. See BanditDB consistency-model doc comment.
        self.wal_try_send(Arc::clone(&event))?;

        // Direct cache insert — no lock needed. We skip apply_event_to_memory here
        // because the prediction_count was already incremented above.
        self.interactions.insert(
            interaction_id.clone(),
            InteractionRecord {
                campaign_id:      campaign_id.to_string(),
                arm_id:           best_arm.clone(),
                context:          Array1::from_vec(context),
                arm_propensities,
                timestamp_secs:   now,
            },
        );

        Ok((best_arm, interaction_id))
    }

    pub fn interact(
        &self,
        campaign_id: &str,
        arm_id:      &str,
        context:     Vec<f64>,
        reward:      f64,
    ) -> Result<String, EngineError> {
        let interaction_id = Uuid::new_v4().to_string();
        let now = now_secs();

        // 1. Emit Predicted event
        let pred_event = Arc::new(DbEvent::Predicted {
            interaction_id:   interaction_id.clone(),
            campaign_id:      campaign_id.to_string(),
            arm_id:           arm_id.to_string(),
            context:          context.clone(),
            timestamp_secs:   now,
            arm_propensities: None, // Historical data doesn't usually have propensities
            is_reemit:        false,
        });
        self.wal_try_send(Arc::clone(&pred_event))?;
        self.apply_event_to_memory(&pred_event);

        // 2. Emit Rewarded event
        let reward_event = Arc::new(DbEvent::Rewarded {
            interaction_id: interaction_id.clone(),
            reward,
            timestamp_secs: now,
        });
        self.wal_try_send(Arc::clone(&reward_event))?;
        self.apply_event_to_memory(&reward_event);

        self.rewarded_count.fetch_add(1, Ordering::Relaxed);
        Ok(interaction_id)
    }

    pub fn delete_campaign(&self, campaign_id: &str) -> Result<(), EngineError> {
        if !self.campaigns.read().contains_key(campaign_id) {
            return Err(Self::campaign_not_found(campaign_id));
        }
        let event = Arc::new(DbEvent::CampaignDeleted { campaign_id: campaign_id.to_string() });
        // WAL before memory. See BanditDB consistency-model doc comment.
        self.wal_try_send(Arc::clone(&event))?;
        self.apply_event_to_memory(&event);
        self.audit("delete", campaign_id, None);
        Ok(())
    }

    pub fn reward(&self, interaction_id: &str, reward: f64) -> Result<(), EngineError> {
        if self.interactions.get(interaction_id).is_none() {
            return Err(EngineError::NotFound(
                format!("Interaction '{interaction_id}' not found or already rewarded")
            ));
        }
        let event = Arc::new(DbEvent::Rewarded {
            interaction_id: interaction_id.to_string(), reward, timestamp_secs: now_secs(),
        });
        // WAL before memory. See BanditDB consistency-model doc comment.
        self.wal_try_send(Arc::clone(&event))?;
        self.apply_event_to_memory(&event);
        self.rewarded_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn archive_campaign(&self, campaign_id: &str) -> Result<(), EngineError> {
        if !self.campaigns.read().contains_key(campaign_id) {
            return Err(Self::campaign_not_found(campaign_id));
        }
        let event = Arc::new(DbEvent::CampaignArchived {
            campaign_id: campaign_id.to_string(), timestamp_secs: now_secs(),
        });
        self.wal_try_send(Arc::clone(&event))?;
        self.apply_event_to_memory(&event);
        self.audit("archive", campaign_id, None);
        Ok(())
    }

    pub fn restore_campaign(&self, campaign_id: &str) -> Result<(), EngineError> {
        if !self.campaigns.read().contains_key(campaign_id) {
            return Err(Self::campaign_not_found(campaign_id));
        }
        let event = Arc::new(DbEvent::CampaignRestored {
            campaign_id: campaign_id.to_string(), timestamp_secs: now_secs(),
        });
        self.wal_try_send(Arc::clone(&event))?;
        self.apply_event_to_memory(&event);
        self.audit("restore", campaign_id, None);
        Ok(())
    }

    pub fn campaign_diagnostics(&self, campaign_id: &str) -> Result<CampaignDiagnosticsData, EngineError> {
        let campaigns = self.campaigns.read();
        let campaign  = campaigns.get(campaign_id)
            .ok_or_else(|| Self::campaign_not_found(campaign_id))?;

        let arms_guard = campaign.arms.read();
        let mut total_predictions = 0u64;
        let mut total_rewards     = 0u64;
        let mut total_reward_sum  = 0.0f64;
        let mut arm_stats         = HashMap::new();

        // Single pass over arms — no duplicate HashMap lookup, use ndarray diag() view.
        for (arm_id, state) in arms_guard.iter() {
            let p  = state.prediction_count.load(Ordering::Relaxed);
            let r  = state.reward_count.load(Ordering::Relaxed);
            let tr = f64::from_bits(state.total_reward.load(Ordering::Relaxed));
            total_predictions += p;
            total_rewards     += r;
            total_reward_sum  += tr;

            let (a_inv_diag_min, a_inv_diag_max) = state.a_inv.diag().iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)));

            arm_stats.insert(arm_id.clone(), ArmDiagnostics {
                predictions:    p,
                rewards:        r,
                avg_reward:     if r > 0 { Some(tr / r as f64) } else { None },
                theta_norm:     state.theta.dot(&state.theta).sqrt(),
                a_inv_diag_min,
                a_inv_diag_max,
            });
        }

        let (challenger_traffic_pct, tournament_win_streak) =
            if matches!(&campaign.algorithm, Algorithm::Progressive(_)) {
                (
                    Some(campaign.challenger_traffic_bps.load(Ordering::Relaxed) as f64 / 100.0),
                    Some(campaign.tournament_wins.load(Ordering::Relaxed)),
                )
            } else {
                (None, None)
            };

        #[cfg(feature = "neural")]
        let neural_buffer_size = campaign.neural.as_ref()
            .map(|n| n.lock().buffer.len());
        #[cfg(not(feature = "neural"))]
        let neural_buffer_size: Option<usize> = None;

        #[cfg(feature = "neural")]
        let neural_last_retrain_losses = campaign.neural.as_ref().and_then(|n| {
            let v = n.lock().last_retrain_losses.clone();
            if v.is_empty() { None } else { Some(v) }
        });
        #[cfg(not(feature = "neural"))]
        let neural_last_retrain_losses: Option<Vec<f32>> = None;

        // --- Entropy alerting ---

        // Guard 2: compare current entropy against the snapshot written at last checkpoint.
        let pred_counts: Vec<u64> = arm_stats.values().map(|s| s.predictions).collect();
        let entropy = selection_entropy(&pred_counts);

        let prior_raw = campaign.last_checkpoint_entropy.load(Ordering::Relaxed);
        let prior     = f64::from_bits(prior_raw);
        let entropy_trend = if prior.is_nan() {
            EntropyTrend::Unknown
        } else if entropy < prior - 0.1 {
            EntropyTrend::Falling
        } else if entropy > prior + 0.1 {
            EntropyTrend::Recovering
        } else {
            EntropyTrend::Stable
        };

        // Guard 1: suppress alert when the campaign has statistically converged.
        let converged = convergence_signal(&arms_guard);

        let entropy_status = classify_entropy_status(entropy, total_predictions, converged);

        let (likely_cause, suggested_action) = if matches!(entropy_status, EntropyStatus::Ok) {
            (None, None)
        } else {
            let min_arm_preds = arm_stats.values().map(|s| s.predictions).min().unwrap_or(0);
            let (cause, action): (&str, &str) = if matches!(entropy_trend, EntropyTrend::Falling) {
                ("recent_collapse",
                 "Entropy dropped since last checkpoint. Check reward pipeline for bugs or recent config changes.")
            } else if min_arm_preds < 50 {
                ("early_lock_in",
                 "One or more arms have very few observations. Consider increasing alpha or resetting the campaign.")
            } else {
                ("sustained_collapse",
                 "Sustained low entropy without a convergence signal. Investigate context distribution shift or add new arms.")
            };
            (Some(cause.to_string()), Some(action.to_string()))
        };

        if !matches!(entropy_status, EntropyStatus::Ok) {
            tracing::warn!(
                campaign      = %campaign_id,
                entropy       = %format!("{entropy:.3}"),
                trend         = ?entropy_trend,
                status        = ?entropy_status,
                likely_cause  = likely_cause.as_deref().unwrap_or("unknown"),
                "entropy: low selection entropy detected"
            );
        }

        Ok(CampaignDiagnosticsData {
            campaign_id:          campaign_id.to_string(),
            archived:             campaign.archived.load(Ordering::Relaxed),
            algorithm:            campaign.algorithm.clone(),
            arm_count:            arms_guard.len(),
            total_predictions,
            total_rewards,
            overall_avg_reward:   if total_rewards > 0 { Some(total_reward_sum / total_rewards as f64) } else { None },
            arm_stats,
            challenger_traffic_pct,
            tournament_win_streak,
            neural_buffer_size,
            neural_last_retrain_losses,
            selection_entropy:    entropy,
            entropy_status,
            entropy_trend,
            converged,
            likely_cause,
            suggested_action,
        })
    }

    /// Emit a JSON line to the audit log using serde_json to prevent injection.
    fn audit(&self, action: &str, campaign_id: &str, detail: Option<&str>) {
        let Some(tx) = &self.audit_tx else { return };
        let mut obj = serde_json::json!({
            "ts":          now_secs(),
            "action":      action,
            "campaign_id": campaign_id,
        });
        if let Some(d) = detail {
            obj["detail"] = serde_json::Value::String(d.to_string());
        }
        match serde_json::to_string(&obj) {
            Ok(line) => { let _ = tx.try_send(line); }
            Err(e)   => tracing::warn!(error = %e, "audit: failed to serialise event"),
        }
    }

    /// Business-level campaign report with convergence signal.
    ///
    /// Uses Wilson-score normal approximation for 95% CIs on mean reward.
    /// `converged = true` means the leading arm's CI lower bound exceeds the
    /// second arm's upper bound — a statistically significant lead.
    pub fn campaign_report(&self, campaign_id: &str) -> Result<CampaignReport, EngineError> {
        let campaigns  = self.campaigns.read();
        let campaign   = campaigns.get(campaign_id)
            .ok_or_else(|| Self::campaign_not_found(campaign_id))?;

        let arms_guard       = campaign.arms.read();
        let total_preds: u64 = arms_guard.values().map(|s| s.prediction_count.load(Ordering::Relaxed)).sum();
        let total_rwds:  u64 = arms_guard.values().map(|s| s.reward_count.load(Ordering::Relaxed)).sum();

        let mut arm_stats: HashMap<String, ArmReportStats> = HashMap::new();

        for (arm_id, state) in arms_guard.iter() {
            let p  = state.prediction_count.load(Ordering::Relaxed);
            let r  = state.reward_count.load(Ordering::Relaxed);
            let tr = f64::from_bits(state.total_reward.load(Ordering::Relaxed));

            let traffic_share = if total_preds > 0 { p as f64 / total_preds as f64 } else { 0.0 };

            let (mean_reward, lower_ci, upper_ci) = if r >= 10 {
                let mean = (tr / r as f64).clamp(0.0, 1.0);
                let z    = 1.96_f64;
                let se   = ((mean * (1.0 - mean)) / r as f64).max(0.0).sqrt();
                (Some(mean), Some((mean - z * se).max(0.0)), Some((mean + z * se).min(1.0)))
            } else {
                (None, None, None)
            };

            arm_stats.insert(arm_id.clone(), ArmReportStats {
                traffic_share,
                predictions:     p,
                rewards:         r,
                mean_reward,
                reward_lower_ci: lower_ci,
                reward_upper_ci: upper_ci,
            });
        }

        // Rank arms by mean_reward descending.
        let mut ranked: Vec<(&String, f64, Option<f64>, Option<f64>)> = arm_stats.iter()
            .filter_map(|(id, s)| s.mean_reward.map(|m| (id, m, s.reward_lower_ci, s.reward_upper_ci)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (leading_arm, converged) = match ranked.as_slice() {
            [] | [_] => (ranked.first().map(|(id, _, _, _)| (*id).clone()), None),
            [top, second, ..] => {
                let top_rewards    = arms_guard.get(top.0.as_str()).map_or(0, |s| s.reward_count.load(Ordering::Relaxed));
                let second_rewards = arms_guard.get(second.0.as_str()).map_or(0, |s| s.reward_count.load(Ordering::Relaxed));
                let converged = if top_rewards >= 30 && second_rewards >= 30 {
                    // Leading arm's lower CI > second arm's upper CI → significant lead
                    Some(top.2.unwrap_or(0.0) > second.3.unwrap_or(1.0))
                } else {
                    None
                };
                (Some(top.0.clone()), converged)
            }
        };

        let overall_reward_rate = if total_rwds > 0 {
            let sum: f64 = arms_guard.values()
                .map(|s| f64::from_bits(s.total_reward.load(Ordering::Relaxed)))
                .sum();
            Some(sum / total_rwds as f64)
        } else {
            None
        };

        let (challenger_traffic_pct, tournament_win_streak) =
            if matches!(&campaign.algorithm, Algorithm::Progressive(_)) {
                (Some(campaign.challenger_traffic_bps.load(Ordering::Relaxed) as f64 / 100.0),
                 Some(campaign.tournament_wins.load(Ordering::Relaxed)))
            } else {
                (None, None)
            };

        Ok(CampaignReport {
            campaign_id:         campaign_id.to_string(),
            archived:            campaign.archived.load(Ordering::Relaxed),
            algorithm:           campaign.algorithm.clone(),
            total_predictions:   total_preds,
            total_rewards:       total_rwds,
            overall_reward_rate,
            arms:                arm_stats,
            leading_arm,
            converged,
            challenger_traffic_pct,
            tournament_win_streak,
        })
    }

    pub fn neural_dir(&self) -> String { format!("{}/neural", self.data_dir) }
    pub fn export_dir(&self) -> String { format!("{}/exports", self.data_dir) }

    /// Lightweight per-campaign entropy status for the /health endpoint.
    /// Computes live entropy from arm prediction counts (one read lock per campaign).
    pub fn entropy_status_all(&self) -> Vec<(String, f64, EntropyStatus)> {
        let campaigns = self.campaigns.read();
        let mut result = Vec::with_capacity(campaigns.len());
        for (id, campaign) in campaigns.iter() {
            if campaign.archived.load(Ordering::Relaxed) { continue; }
            let arms_guard = campaign.arms.read();
            let total_preds: u64 = arms_guard.values()
                .map(|s| s.prediction_count.load(Ordering::Relaxed))
                .sum();
            let pred_counts: Vec<u64> = arms_guard.values()
                .map(|s| s.prediction_count.load(Ordering::Relaxed))
                .collect();
            let converged = convergence_signal(&arms_guard);
            drop(arms_guard);
            let entropy = selection_entropy(&pred_counts);
            let status  = classify_entropy_status(entropy, total_preds, converged);
            result.push((id.clone(), entropy, status));
        }
        result
    }
}

/// Self-Normalised Importance-Weighted Policy Evaluation (SNIPS).
///
/// Estimates the expected reward of a deterministic greedy policy (defined by
/// `arms` + `feature_fn`) using logged data from the stochastic logging policy.
///
/// For each buffer entry (context, logged_arm, reward, propensity):
///   If the greedy policy would have chosen `logged_arm`, contribute:
///     weight = 1 / clamp(propensity, 0.01, 1.0),  capped at 100
///     numerator   += reward × weight
///     denominator += weight
///
/// Returns (snips_estimate, coverage_count). Callers should reject results where
/// coverage_count is too low (too few matched samples for a reliable estimate).
#[cfg(feature = "neural")]
fn snips_score(
    buffer:     &std::collections::VecDeque<(Vec<f64>, String, f64, f64)>,
    arms:       &HashMap<String, ArmState>,
    feature_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
) -> (f64, usize) {
    let mut numerator   = 0.0f64;
    let mut denominator = 0.0f64;
    let mut coverage    = 0usize;

    for (context, logged_arm, reward, propensity) in buffer {
        let features = feature_fn(&Array1::from_vec(context.clone()));

        let best_arm = arms.iter()
            .max_by(|(_, a), (_, b)| {
                a.theta.dot(&features)
                    .partial_cmp(&b.theta.dot(&features))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id.as_str());

        if best_arm == Some(logged_arm.as_str()) {
            let w = (1.0 / propensity.clamp(0.01, 1.0)).min(100.0);
            numerator   += reward * w;
            denominator += w;
            coverage    += 1;
        }
    }

    let estimate = if denominator > 1e-10 { numerator / denominator } else { 0.0 };
    (estimate, coverage)
}

/// Apply a tournament outcome to a campaign's traffic split and log the result.
#[cfg(feature = "neural")]
fn run_tournament(
    campaign_id: &str,
    campaign:    &Campaign,
    neural:      &crate::neural::NeuralLinUCBState,
    cfg:         &ProgressiveConfig,
) {
    match evaluate_tournament(campaign_id, campaign, neural, cfg) {
        TournamentOutcome::Hold => {}
        TournamentOutcome::ChallengerStep(new_bps) => {
            let old = campaign.challenger_traffic_bps.swap(new_bps, Ordering::SeqCst);
            campaign.tournament_wins.store(0, Ordering::SeqCst);
            tracing::info!(campaign = %campaign_id,
                from_pct = old / 100, to_pct = new_bps / 100,
                "tournament: challenger traffic increased");
        }
        TournamentOutcome::BaseStep(new_bps) => {
            let old = campaign.challenger_traffic_bps.swap(new_bps, Ordering::SeqCst);
            campaign.tournament_wins.store(0, Ordering::SeqCst);
            tracing::info!(campaign = %campaign_id,
                from_pct = old / 100, to_pct = new_bps / 100,
                "tournament: base traffic restored");
        }
        TournamentOutcome::Inconclusive => {
            tracing::debug!(campaign = %campaign_id,
                streak = campaign.tournament_wins.load(Ordering::SeqCst),
                "tournament: inconclusive — streak decayed");
        }
    }
}

/// Evaluate one tournament round for a Progressive campaign, returning a `TournamentOutcome`
/// the checkpoint loop can act on without nested conditionals.
///
/// Extracts SNIPS evaluation, coverage guard, streak accumulation, and traffic clamping
/// from the 7-level nested block they previously lived in inside `checkpoint()`.
#[cfg(feature = "neural")]
fn evaluate_tournament(
    campaign_id: &str,
    campaign:    &Campaign,
    neural:      &crate::neural::NeuralLinUCBState,
    cfg:         &ProgressiveConfig,
) -> TournamentOutcome {
    if neural.buffer.is_empty() { return TournamentOutcome::Hold; }

    // Minimum observations per arm before we trust the SNIPS estimate.
    let per_arm_counts = neural.buffer.iter().fold(
        HashMap::<&str, usize>::new(),
        |mut m, (_, arm, _, _)| { *m.entry(arm.as_str()).or_insert(0) += 1; m },
    );
    let min_per_arm = per_arm_counts.values().copied().min().unwrap_or(0);
    if min_per_arm < cfg.min_obs {
        tracing::debug!(campaign = %campaign_id, min_per_arm, required = cfg.min_obs,
            "tournament: insufficient obs/arm — holding");
        return TournamentOutcome::Hold;
    }

    let base_arms      = campaign.arms.read();
    let chal_arm_guard = campaign.challenger_arms.as_ref().map(|c| c.read());

    let (base_snips, base_cov) = snips_score(
        &neural.buffer, &base_arms,
        |ctx| match cfg.base.as_ref() {
            Algorithm::NeuralLinUCB(_) => neural.embed(ctx),
            _ => ctx.clone(),
        },
    );
    let (chal_snips, chal_cov) = match &chal_arm_guard {
        Some(c) => snips_score(
            &neural.buffer, c,
            |ctx| match cfg.challenger.as_ref() {
                Algorithm::NeuralLinUCB(_) => neural.embed(ctx),
                _ => ctx.clone(),
            },
        ),
        None => (0.0, 0),
    };

    let min_cov = (cfg.min_obs / 4).max(5);
    if base_cov < min_cov {
        tracing::debug!(campaign = %campaign_id, base_cov, chal_cov,
            "tournament: base SNIPS coverage too low — holding");
        return TournamentOutcome::Hold;
    }
    // When chal_cov == 0 the challenger's current policy would never choose any
    // historically-logged arm. snips_score already returns (0.0, 0) for this
    // case, so we let it proceed — base clearly wins over a chal_snips of 0.0.

    tracing::info!(
        campaign   = %campaign_id,
        base_snips = format!("{base_snips:.4}"),
        base_cov,
        chal_snips = format!("{chal_snips:.4}"),
        chal_cov,
        "tournament: SNIPS evaluation"
    );

    const MARGIN: f64 = 0.10;
    let current_bps = campaign.challenger_traffic_bps.load(Ordering::SeqCst);
    let chal_wins   = chal_snips > base_snips * (1.0 + MARGIN);
    let base_wins   = base_snips > chal_snips * (1.0 + MARGIN);

    if chal_wins {
        let streak = campaign.tournament_wins.fetch_add(1, Ordering::SeqCst) + 1;
        if streak >= cfg.required_wins as i32 {
            return TournamentOutcome::ChallengerStep(
                (current_bps + cfg.step_bps).min(BPS_CEIL)
            );
        }
        tracing::debug!(campaign = %campaign_id, streak, required = cfg.required_wins,
            "tournament: challenger win");
    } else if base_wins {
        let streak = campaign.tournament_wins.fetch_sub(1, Ordering::SeqCst) - 1;
        if streak <= -(cfg.required_wins as i32) {
            return TournamentOutcome::BaseStep(
                current_bps.saturating_sub(cfg.step_bps).max(BPS_FLOOR)
            );
        }
        tracing::debug!(campaign = %campaign_id, streak = -streak, required = cfg.required_wins,
            "tournament: base win");
    } else {
        // Inconclusive: decay streak toward 0 to demand fresh evidence.
        let streak = campaign.tournament_wins.load(Ordering::SeqCst);
        if streak > 0      { campaign.tournament_wins.fetch_sub(1, Ordering::SeqCst); }
        else if streak < 0 { campaign.tournament_wins.fetch_add(1, Ordering::SeqCst); }
        return TournamentOutcome::Inconclusive;
    }

    TournamentOutcome::Hold
}

/// Append completed interactions to a per-campaign Parquet file shard.
///
/// Schema: interaction_id | arm_id | reward | predicted_at | rewarded_at | feature_0 … feature_N
///
/// Each checkpoint writes a new shard: `{campaign_id}_{timestamp_us}.parquet`.
/// The timestamp is in microseconds to prevent collisions when two checkpoints
/// happen within the same second. This ensures O(1) write performance regardless
/// of campaign history size.
pub fn write_campaign_parquet(
    export_dir:   &str,
    campaign_id:  &str,
    interactions: &[CompletedInteraction],
    feature_dim:  usize,
) -> Result<(), String> {
    if interactions.is_empty() {
        return Ok(());
    }

    let timestamp_us = SystemTime::now()
        .duration_since(UNIX_EPOCH).unwrap_or_default()
        .as_micros() as u64;

    let mut df = interactions_to_df(interactions, feature_dim)?;

    // Guard against path traversal from WAL-sourced campaign IDs that bypass HTTP validation.
    let filename  = format!("{campaign_id}_{timestamp_us}.parquet");
    let tmp_name  = format!("{campaign_id}_{timestamp_us}.parquet.tmp");
    let safe_name = Path::new(&filename).file_name()
        .ok_or_else(|| format!("invalid parquet filename for campaign '{campaign_id}'"))?;
    let safe_tmp  = Path::new(&tmp_name).file_name()
        .ok_or_else(|| format!("invalid parquet tmp filename for campaign '{campaign_id}'"))?;

    let path = Path::new(export_dir).join(safe_name);
    let tmp  = Path::new(export_dir).join(safe_tmp);

    let mut file = File::create(&tmp).map_err(|e| e.to_string())?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df)
        .map_err(|e| e.to_string())?;
    fs::rename(tmp, path).map_err(|e| e.to_string())?;

    Ok(())
}

fn interactions_to_df(interactions: &[CompletedInteraction], feature_dim: usize) -> Result<DataFrame, String> {
    let interaction_ids: Vec<&str>      = interactions.iter().map(|r| r.interaction_id.as_str()).collect();
    let arm_ids:         Vec<&str>      = interactions.iter().map(|r| r.arm_id.as_str()).collect();
    let rewards:         Vec<f64>       = interactions.iter().map(|r| r.reward).collect();
    let predicted_ats:   Vec<i64>       = interactions.iter().map(|r| r.predicted_at as i64).collect();
    let rewarded_ats:    Vec<i64>       = interactions.iter().map(|r| r.rewarded_at  as i64).collect();
    let propensities:    Vec<Option<f64>> = interactions.iter().map(|r| r.propensity).collect();

    let mut series: Vec<Series> = vec![
        Series::new("interaction_id", interaction_ids),
        Series::new("arm_id",         arm_ids),
        Series::new("reward",         rewards),
        Series::new("predicted_at",   predicted_ats),
        Series::new("rewarded_at",    rewarded_ats),
        Series::new("propensity",     propensities),
    ];

    for f in 0..feature_dim {
        let col: Vec<f64> = interactions.iter().map(|r| r.context[f]).collect();
        let name = format!("feature_{}", f);
        series.push(Series::new(name.as_str(), col));
    }

    DataFrame::new(series).map_err(|e| e.to_string())
}

/// Normalised Shannon entropy of an arm selection distribution (0 = collapsed, 1 = uniform).
/// Returns 1.0 when there are fewer than two arms or no predictions yet.
fn selection_entropy(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    if total == 0 || counts.len() < 2 { return 1.0; }
    let log_n = (counts.len() as f64).ln();
    counts.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f64 / total as f64; -p * p.ln() })
        .sum::<f64>() / log_n
}

/// Wilson-score convergence signal: true if the leading arm's 95% CI lower bound
/// exceeds the second arm's upper bound (requires ≥ 30 rewards on both arms).
fn convergence_signal(arms: &HashMap<String, ArmState>) -> Option<bool> {
    let mut ranked: Vec<(f64, f64, f64, u64)> = arms.values()
        .filter_map(|s| {
            let r = s.reward_count.load(Ordering::Relaxed);
            if r < 10 { return None; }
            let tr   = f64::from_bits(s.total_reward.load(Ordering::Relaxed));
            let mean = (tr / r as f64).clamp(0.0, 1.0);
            let se   = ((mean * (1.0 - mean)) / r as f64).max(0.0).sqrt();
            Some((mean - 1.96 * se, mean + 1.96 * se, mean, r))
        })
        .collect();
    ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    match ranked.as_slice() {
        [top, second, ..] if top.3 >= 30 && second.3 >= 30 => Some(top.0 > second.1),
        _ => None,
    }
}

/// Map entropy + guards to a status level.
/// Guard 1: suppress if statistically converged.
/// Guard 2: suppress if fewer than 500 total predictions (insufficient data).
fn classify_entropy_status(entropy: f64, total_preds: u64, converged: Option<bool>) -> EntropyStatus {
    if converged == Some(true) || total_preds < 500 { return EntropyStatus::Ok; }
    if entropy >= 0.4 { EntropyStatus::Ok }
    else if entropy >= 0.2 { EntropyStatus::Warning }
    else { EntropyStatus::Critical }
}

/// Adaptive Monte Carlo sample count for Thompson Sampling propensity estimation.
///
/// Uses the maximum A_inv diagonal across all arms as a proxy for posterior spread.
/// A_inv starts as the identity matrix (diagonal = 1.0) and shrinks as rewards accumulate.
///
/// | max_diag  | Stage               | N  |
/// |-----------|---------------------|----|
/// | > 0.7     | Cold start          | 64 |
/// | 0.3–0.7   | Active learning     | 32 |
/// | 0.1–0.3   | Converging          | 16 |
/// | ≤ 0.1     | Concentrated        |  8 |
///
/// This gives the ideal cost profile for production: high N when traffic is low (cold start),
/// low N when traffic is high (converged) — the sample budget scales inversely with load.
fn ts_propensity_samples(arms: &HashMap<String, ArmState>) -> usize {
    let max_diag = arms.values()
        .map(|s| s.a_inv.diag().iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        .fold(0.0f64, f64::max);
    if max_diag > 0.7 { 64 }
    else if max_diag > 0.3 { 32 }
    else if max_diag > 0.1 { 16 }
    else { 8 }
}

/// Compute softmax-normalised propensities over arm UCB scores.
/// Subtracts the max for numerical stability before exponentiating.
fn softmax_propensities(scores: &[(String, f64)]) -> HashMap<String, f64> {
    let max_score = scores.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|(_, s)| (s - max_score).exp()).collect();
    let sum: f64 = exps.iter().sum();
    scores.iter().zip(exps.iter())
        .map(|((arm_id, _), exp)| (arm_id.clone(), exp / sum))
        .collect()
}
