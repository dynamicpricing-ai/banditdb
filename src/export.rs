//! Parquet export of completed interactions.
//!
//! Every checkpoint scans the WAL segment it is about to rotate away, joins each
//! prediction to its reward, and writes one shard per campaign (and per context
//! length) under `exports/`. Recovery never reads these files; they are the
//! analytics and training record of what was decided and what came of it.

use crate::engine::{for_each_wal_event, WalFormat};
use crate::state::{CompletedInteraction, DbEvent};
use polars::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Export the completed interactions in WAL bytes `[0, wal_offset)` and prune old
/// shards. Returns the number of rows written. Blocking: run it off the async
/// runtime.
pub(crate) fn export_wal_segment(
    wal_path: &str,
    wal_fmt: WalFormat,
    wal_offset: u64,
    export_dir: &str,
    export_retain: usize,
) -> usize {
    // Scan the WAL for matched Predicted+Rewarded pairs.
    #[allow(clippy::type_complexity)]
    // matched-pair accumulator; a named type alias would not aid clarity here
    let mut predicted: HashMap<
        String,
        (String, String, Vec<f64>, Option<HashMap<String, f64>>, u64),
    > = HashMap::new();
    let mut rewarded: HashMap<String, (f64, u64)> = HashMap::new();

    for_each_wal_event(
        wal_path,
        0,
        wal_offset,
        wal_fmt,
        |event| match event {
            DbEvent::Predicted {
                interaction_id,
                campaign_id,
                arm_id,
                context,
                timestamp_secs,
                arm_propensities,
                ..
            } => {
                predicted.insert(
                    interaction_id,
                    (
                        campaign_id,
                        arm_id,
                        context,
                        arm_propensities,
                        timestamp_secs,
                    ),
                );
            }
            DbEvent::Rewarded {
                interaction_id,
                reward,
                timestamp_secs,
                unlogged_prediction,
            } => {
                if let Some(p) = unlogged_prediction {
                    predicted.entry(interaction_id.clone()).or_insert((
                        p.campaign_id,
                        p.arm_id,
                        p.context.to_vec(),
                        p.arm_propensities,
                        p.timestamp_secs,
                    ));
                }
                rewarded.insert(interaction_id, (reward, timestamp_secs));
            }
            _ => {}
        },
    );

    let mut by_campaign: HashMap<String, Vec<CompletedInteraction>> = HashMap::new();

    for (iid, (reward, rewarded_at)) in &rewarded {
        if let Some((campaign_id, arm_id, context, arm_propensities, predicted_at)) =
            predicted.get(iid)
        {
            let propensity = arm_propensities
                .as_ref()
                .and_then(|m| m.get(arm_id.as_str()))
                .copied();
            by_campaign
                .entry(campaign_id.clone())
                .or_default()
                .push(CompletedInteraction {
                    interaction_id: iid.clone(),
                    arm_id: arm_id.clone(),
                    context: context.clone(),
                    reward: *reward,
                    predicted_at: *predicted_at,
                    rewarded_at: *rewarded_at,
                    propensity,
                });
        }
    }

    let mut rows = 0usize;
    for (campaign_id, interactions) in by_campaign {
        prune_export_shards(export_dir, &campaign_id, export_retain);
        // One shard per context length. A campaign id that was deleted and
        // recreated with another feature_dim leaves both lengths in the same
        // WAL segment, and a Parquet shard has one fixed set of feature columns.
        let mut by_dim: std::collections::BTreeMap<usize, Vec<CompletedInteraction>> =
            std::collections::BTreeMap::new();
        for interaction in interactions {
            by_dim
                .entry(interaction.context.len())
                .or_default()
                .push(interaction);
        }
        for (feature_dim, group) in &by_dim {
            match write_campaign_parquet(export_dir, &campaign_id, group, *feature_dim) {
                Err(e) => {
                    tracing::error!(campaign = %campaign_id, feature_dim, error = %e, "checkpoint: Parquet write failed")
                }
                Ok(()) => rows += group.len(),
            }
        }
    }

    rows
}

/// Where a campaign's Parquet shards live: a directory and the file-name stem.
///
/// Tenant campaigns are stored as `{tenant}/{campaign}`, and their shards go in a
/// directory per tenant: `exports/{tenant}/{campaign}_{micros}.parquet`. Before
/// this, the writer kept only the last path component, so every tenant's shards
/// landed flat as `{campaign}_{micros}.parquet`: two tenants' `checkout` shards
/// could not be told apart, and pruning (which looked for `{tenant}/{campaign}_`)
/// never matched them, so they were never deleted. Campaigns without a tenant
/// keep the flat layout.
///
/// Each part must be a plain id (`[A-Za-z0-9_-]{1,128}`, the HTTP and tenant
/// rule). WAL-sourced ids bypass HTTP validation, so anything else — `..`, an
/// empty part, a nested path — is refused rather than written somewhere else.
pub fn export_location(export_dir: &str, campaign_id: &str) -> Result<(PathBuf, String), String> {
    let part = |s: &str| -> Result<String, String> {
        let ok = !s.is_empty()
            && s.len() <= 128
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if ok {
            Ok(s.to_string())
        } else {
            Err(format!(
                "campaign id '{campaign_id}' cannot name an export path"
            ))
        }
    };
    match campaign_id.split_once('/') {
        Some((tenant, name)) => Ok((Path::new(export_dir).join(part(tenant)?), part(name)?)),
        None => Ok((PathBuf::from(export_dir), part(campaign_id)?)),
    }
}

/// The campaign a shard file belongs to: `checkout` for `checkout_1700….parquet`.
///
/// The timestamp is always the last `_` part, so `checkout_v2_1700….parquet` is
/// `checkout_v2`, never `checkout`. Matching by prefix alone would confuse them.
pub fn shard_stem(file_name: &str) -> Option<&str> {
    let stem = file_name.strip_suffix(".parquet")?;
    let (name, ts) = stem.rsplit_once('_')?;
    let ok = !name.is_empty() && !ts.is_empty() && ts.bytes().all(|b| b.is_ascii_digit());
    ok.then_some(name)
}

/// Delete the oldest Parquet shards once a campaign exceeds the retention limit.
///
/// Every checkpoint writes a new shard and nothing ever removed them, so `exports/`
/// grew without bound until the volume filled — at which point checkpointing starts
/// failing and the WAL stops rotating. Recovery never reads these files (it uses
/// checkpoint.json plus the WAL), so pruning is safe: the cost is only that
/// offline analysis loses the oldest history.
///
/// Retention is per campaign, newest first, matched exactly by `shard_stem` in
/// the campaign's own directory (`export_location`): pruning `checkout` never
/// touches `checkout_v2`, nor another tenant's `checkout`.
/// `BANDITDB_EXPORT_RETAIN_SHARDS=0` disables pruning entirely.
pub fn prune_export_shards(export_dir: &str, campaign_id: &str, retain: usize) {
    if retain == 0 {
        return;
    }

    let Ok((dir, stem)) = export_location(export_dir, campaign_id) else {
        return;
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        return;
    };

    // Shard names embed a microsecond timestamp, so lexical order is chronological.
    let mut shards: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| shard_stem(n) == Some(stem.as_str()))
        .collect();
    if shards.len() <= retain {
        return;
    }
    shards.sort();

    let doomed = shards.len() - retain;
    for name in shards.into_iter().take(doomed) {
        match fs::remove_file(dir.join(&name)) {
            Ok(()) => tracing::debug!(shard = %name, "export: pruned old shard"),
            Err(e) => tracing::warn!(shard = %name, error = %e, "export: could not prune shard"),
        }
    }
    tracing::info!(campaign = %campaign_id, pruned = doomed, retained = retain,
        "export: pruned old Parquet shards");
}

/// Append completed interactions to a per-campaign Parquet file shard.
///
/// Schema: interaction_id | arm_id | reward | predicted_at | rewarded_at | feature_0 … feature_N
///
/// Each checkpoint writes a new shard: `{campaign}_{timestamp_us}.parquet`, in
/// `exports/{tenant}/` for a tenant campaign (see `export_location`).
/// The timestamp is in microseconds to prevent collisions when two checkpoints
/// happen within the same second. This ensures O(1) write performance regardless
/// of campaign history size.
pub fn write_campaign_parquet(
    export_dir: &str,
    campaign_id: &str,
    interactions: &[CompletedInteraction],
    feature_dim: usize,
) -> Result<(), String> {
    if interactions.is_empty() {
        return Ok(());
    }

    let mut timestamp_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    let mut df = interactions_to_df(interactions, feature_dim)?;

    // `export_location` refuses ids that are not plain names, so a WAL-sourced id
    // that bypassed HTTP validation cannot write outside its directory.
    let (dir, stem) = export_location(export_dir, campaign_id)?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    // Two shards for one campaign can be written within the same microsecond
    // (one per context length); never overwrite an existing shard.
    while dir.join(format!("{stem}_{timestamp_us}.parquet")).exists() {
        timestamp_us += 1;
    }
    let path = dir.join(format!("{stem}_{timestamp_us}.parquet"));
    let tmp = dir.join(format!("{stem}_{timestamp_us}.parquet.tmp"));

    let mut file = File::create(&tmp).map_err(|e| e.to_string())?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df)
        .map_err(|e| e.to_string())?;
    fs::rename(tmp, path).map_err(|e| e.to_string())?;

    Ok(())
}

fn interactions_to_df(
    interactions: &[CompletedInteraction],
    feature_dim: usize,
) -> Result<DataFrame, String> {
    let interaction_ids: Vec<&str> = interactions
        .iter()
        .map(|r| r.interaction_id.as_str())
        .collect();
    let arm_ids: Vec<&str> = interactions.iter().map(|r| r.arm_id.as_str()).collect();
    let rewards: Vec<f64> = interactions.iter().map(|r| r.reward).collect();
    let predicted_ats: Vec<i64> = interactions.iter().map(|r| r.predicted_at as i64).collect();
    let rewarded_ats: Vec<i64> = interactions.iter().map(|r| r.rewarded_at as i64).collect();
    let propensities: Vec<Option<f64>> = interactions.iter().map(|r| r.propensity).collect();

    // Indexing below assumes every row has `feature_dim` features.
    if let Some(bad) = interactions.iter().find(|r| r.context.len() != feature_dim) {
        return Err(format!(
            "interaction '{}' has {} features, expected {feature_dim}",
            bad.interaction_id,
            bad.context.len()
        ));
    }

    let mut series: Vec<Series> = vec![
        Series::new("interaction_id", interaction_ids),
        Series::new("arm_id", arm_ids),
        Series::new("reward", rewards),
        Series::new("predicted_at", predicted_ats),
        Series::new("rewarded_at", rewarded_ats),
        Series::new("propensity", propensities),
    ];

    for f in 0..feature_dim {
        let col: Vec<f64> = interactions.iter().map(|r| r.context[f]).collect();
        let name = format!("feature_{}", f);
        series.push(Series::new(name.as_str(), col));
    }

    DataFrame::new(series).map_err(|e| e.to_string())
}
