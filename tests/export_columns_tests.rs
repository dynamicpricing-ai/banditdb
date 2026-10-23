//! Parquet feature columns: headers from the campaign's metadata, values as the
//! app sent them.
//!
//! The SDKs and the console encode a named context the same way: values in the
//! order of `metadata.features`, a constant 1 appended (`metadata.context.bias`),
//! scaled to unit length (`normalize`). A shard that kept `feature_0 … feature_N`
//! of that vector was unreadable without the metadata beside it. Now the columns
//! carry the feature names and the encoding is undone, so `requested_gb` holds
//! the value the app sent. Each rule is pinned below, along with every fallback
//! to the numbered columns.

use banditdb::engine::{export_columns, write_campaign_parquet, ExportColumns};
use banditdb::state::{Algorithm, CompletedInteraction};
use banditdb::BanditDB;
use polars::prelude::*;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "banditdb_export_columns_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The SDK encoding: values, then a constant 1, scaled to unit length.
fn encode(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.push(1.0);
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn row(id: &str, context: Vec<f64>) -> CompletedInteraction {
    CompletedInteraction {
        interaction_id: id.into(),
        arm_id: "a".into(),
        context,
        reward: 1.0,
        predicted_at: 1_700_000_000,
        rewarded_at: 1_700_000_001,
        propensity: Some(0.5),
    }
}

fn names(cols: &ExportColumns) -> Vec<&str> {
    cols.names.iter().map(String::as_str).collect()
}

/// The single shard written under `dir`, read back.
fn read_only_shard(dir: &std::path::Path) -> DataFrame {
    let files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    ParquetReader::new(fs::File::open(&files[0]).unwrap())
        .finish()
        .unwrap()
}

fn header(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn column(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
    df.column(name)
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .collect()
}

fn console_meta() -> Value {
    json!({
        "features": ["past_spending", "requested_gb", "thread_count", "site_visits"],
        "context": { "bias": true, "normalize": true },
        "console": { "version": 1, "title": "Proxy pricing" }
    })
}

const FIXED: [&str; 6] = [
    "interaction_id",
    "arm_id",
    "reward",
    "predicted_at",
    "rewarded_at",
    "propensity",
];

// ---------------------------------------------------------------------------
// Which headers
// ---------------------------------------------------------------------------

#[test]
fn no_metadata_keeps_numbered_columns() {
    let cols = export_columns(None, 3);
    assert_eq!(names(&cols), ["feature_0", "feature_1", "feature_2"]);
    assert!(!cols.strip_bias);
}

#[test]
fn console_layout_names_every_feature_and_drops_the_constant() {
    let cols = export_columns(Some(&console_meta()), 5);
    assert_eq!(
        names(&cols),
        [
            "past_spending",
            "requested_gb",
            "thread_count",
            "site_visits"
        ]
    );
    assert!(cols.strip_bias);
}

#[test]
fn layout_defaults_to_bias_and_normalize() {
    let meta = json!({ "features": ["a", "b"], "context": {} });
    let cols = export_columns(Some(&meta), 3);
    assert_eq!(names(&cols), ["a", "b"]);
    assert!(cols.strip_bias);
}

#[test]
fn without_a_layout_bias_is_told_from_the_width() {
    let meta = json!({ "features": ["a", "b"] });
    // One wider than the names: the constant is there.
    let with = export_columns(Some(&meta), 3);
    assert_eq!((names(&with), with.strip_bias), (vec!["a", "b"], true));
    // Same width: written before the constant, scaled, so not recoverable.
    let without = export_columns(Some(&meta), 2);
    assert_eq!(
        (names(&without), without.strip_bias),
        (vec!["a_normalized", "b_normalized"], false)
    );
}

#[test]
fn scaled_without_a_constant_says_so_in_the_header() {
    let meta = json!({ "features": ["a", "b"], "context": { "bias": false, "normalize": true } });
    let cols = export_columns(Some(&meta), 2);
    assert_eq!(names(&cols), ["a_normalized", "b_normalized"]);
    assert!(!cols.strip_bias);
}

#[test]
fn raw_vector_keeps_plain_names() {
    let meta = json!({ "features": ["a", "b"], "context": { "bias": false, "normalize": false } });
    let cols = export_columns(Some(&meta), 2);
    assert_eq!(names(&cols), ["a", "b"]);
    assert!(!cols.strip_bias);
}

#[test]
fn constant_without_scaling_is_still_dropped() {
    let meta = json!({ "features": ["a"], "context": { "bias": true, "normalize": false } });
    let cols = export_columns(Some(&meta), 2);
    assert_eq!((names(&cols), cols.strip_bias), (vec!["a"], true));
}

#[test]
fn anything_that_does_not_line_up_falls_back_to_numbered() {
    let cases = [
        (
            "width too wide",
            json!({ "features": ["a", "b"], "context": { "bias": true } }),
            4,
        ),
        (
            "width too narrow",
            json!({ "features": ["a", "b"], "context": { "bias": true } }),
            2,
        ),
        (
            "no constant, wrong width",
            json!({ "features": ["a", "b"], "context": { "bias": false } }),
            3,
        ),
        ("duplicate names", json!({ "features": ["a", "a"] }), 3),
        (
            "clashes with a fixed column",
            json!({ "features": ["reward", "b"] }),
            3,
        ),
        (
            "clashes with propensity",
            json!({ "features": ["propensity"] }),
            2,
        ),
        ("empty name", json!({ "features": ["", "b"] }), 3),
        ("name too long", json!({ "features": ["x".repeat(129)] }), 2),
        ("not a string", json!({ "features": ["a", 2] }), 3),
        ("not a list", json!({ "features": "a,b" }), 3),
        ("empty list", json!({ "features": [] }), 1),
        (
            "no features key",
            json!({ "console": { "features": [{ "name": "a" }] } }),
            2,
        ),
        ("metadata not an object", json!(["a", "b"]), 3),
    ];
    for (why, meta, width) in cases {
        let cols = export_columns(Some(&meta), width);
        assert_eq!(cols, ExportColumns::numbered(width), "{why}");
    }
}

// ---------------------------------------------------------------------------
// What is written
// ---------------------------------------------------------------------------

#[test]
fn shard_has_named_columns_and_the_values_the_app_sent() {
    let tmp = TempDir::new("named");
    let exports = tmp.0.to_string_lossy().to_string();
    let sent = [
        [0.3366, 0.854, 0.382, 0.77],
        [0.0, 1.0, 0.25, 0.5],
        [0.9, 0.1, 0.0, 1.0],
    ];
    let rows: Vec<_> = sent
        .iter()
        .enumerate()
        .map(|(i, v)| row(&format!("r{i}"), encode(v)))
        .collect();

    write_campaign_parquet(&exports, "proxy", &rows, Some(&console_meta())).unwrap();

    let df = read_only_shard(&tmp.0);
    let mut expected: Vec<String> = FIXED.iter().map(|s| s.to_string()).collect();
    expected.extend(
        [
            "past_spending",
            "requested_gb",
            "thread_count",
            "site_visits",
        ]
        .map(String::from),
    );
    assert_eq!(
        header(&df),
        expected,
        "fixed columns, then one per named feature, no constant"
    );

    for (j, name) in [
        "past_spending",
        "requested_gb",
        "thread_count",
        "site_visits",
    ]
    .iter()
    .enumerate()
    {
        let got = column(&df, name);
        for (i, v) in sent.iter().enumerate() {
            let g = got[i].unwrap();
            assert!(
                (g - v[j]).abs() < 1e-12,
                "{name} row {i}: wrote {g}, app sent {}",
                v[j]
            );
        }
    }
}

#[test]
fn shard_without_metadata_is_unchanged() {
    let tmp = TempDir::new("numbered");
    let exports = tmp.0.to_string_lossy().to_string();
    let ctx = encode(&[0.2, 0.4]);
    write_campaign_parquet(&exports, "plain", &[row("r0", ctx.clone())], None).unwrap();

    let df = read_only_shard(&tmp.0);
    let mut expected: Vec<String> = FIXED.iter().map(|s| s.to_string()).collect();
    expected.extend(["feature_0", "feature_1", "feature_2"].map(String::from));
    assert_eq!(header(&df), expected);
    for (j, v) in ctx.iter().enumerate() {
        assert_eq!(
            column(&df, &format!("feature_{j}")),
            vec![Some(*v)],
            "as the model saw it"
        );
    }
}

#[test]
fn normalized_columns_keep_the_values_the_model_saw() {
    let tmp = TempDir::new("normalized");
    let exports = tmp.0.to_string_lossy().to_string();
    let meta = json!({ "features": ["a", "b"], "context": { "bias": false, "normalize": true } });
    write_campaign_parquet(&exports, "old", &[row("r0", vec![0.6, 0.8])], Some(&meta)).unwrap();

    let df = read_only_shard(&tmp.0);
    assert_eq!(column(&df, "a_normalized"), vec![Some(0.6)]);
    assert_eq!(column(&df, "b_normalized"), vec![Some(0.8)]);
}

#[test]
fn a_bad_constant_writes_null_not_a_guess() {
    let tmp = TempDir::new("bad_constant");
    let exports = tmp.0.to_string_lossy().to_string();
    let meta = json!({ "features": ["a"], "context": { "bias": true } });
    let rows = [
        row("ok", vec![0.5, 0.5]),
        row("zero", vec![0.5, 0.0]),
        row("short", vec![0.5]),
    ];
    write_campaign_parquet(&exports, "c", &rows, Some(&meta)).unwrap();

    let df = read_only_shard(&tmp.0);
    assert_eq!(column(&df, "a"), vec![Some(1.0), None, None]);
}

// ---------------------------------------------------------------------------
// End to end: the campaign's own metadata at checkpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn checkpoint_names_columns_from_each_campaigns_metadata() {
    let tmp = TempDir::new("checkpoint");
    let data_dir = tmp.0.to_string_lossy().to_string();
    let db = BanditDB::new(&format!("{data_dir}/bandit_wal.jsonl"), &data_dir);

    db.add_campaign(
        "t/priced", vec!["x".into(), "y".into()], 3, 1.0, Algorithm::Linucb,
        Some(json!({ "features": ["spend", "visits"], "context": { "bias": true, "normalize": true } })),
        None,
    ).await.unwrap();
    db.add_campaign(
        "t/bare",
        vec!["x".into(), "y".into()],
        3,
        1.0,
        Algorithm::Linucb,
        None,
        None,
    )
    .await
    .unwrap();

    let sent = [0.25, 0.75];
    for id in ["t/priced", "t/bare"] {
        let (_, iid) = db.predict(id, encode(&sent)).unwrap();
        db.reward(&iid, 1.0).await.unwrap();
    }
    db.checkpoint().await.unwrap();

    let dir = tmp.0.join("exports/t");
    let by_stem = |stem: &str| {
        let f = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(&format!("{stem}_"))
            })
            .unwrap_or_else(|| panic!("no shard for {stem}"));
        ParquetReader::new(fs::File::open(f).unwrap())
            .finish()
            .unwrap()
    };

    let priced = by_stem("priced");
    assert_eq!(&header(&priced)[6..], ["spend", "visits"]);
    assert!((column(&priced, "spend")[0].unwrap() - 0.25).abs() < 1e-12);
    assert!((column(&priced, "visits")[0].unwrap() - 0.75).abs() < 1e-12);

    let bare = by_stem("bare");
    assert_eq!(&header(&bare)[6..], ["feature_0", "feature_1", "feature_2"]);
}
