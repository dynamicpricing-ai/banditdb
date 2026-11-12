//! Every exported decision carries its provenance: the full propensity map, the
//! input it was made from, and the budget state behind it — including decisions
//! whose `Predicted` record was rotated away and is carried by the reward.

use banditdb::engine::ArmFilter;
use banditdb::state::{Algorithm, DbEvent, InputRef, PacingConfig, ResourceConstraint};
use banditdb::BanditDB;
use polars::prelude::*;
use std::collections::HashMap;

const SHA: &str = "9f2c000000000000000000000000000000000000000000000000000000000001";

fn fresh(name: &str) -> (String, String) {
    let dir = format!("/tmp/banditdb_export_provenance_{name}");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    (format!("{dir}/wal.jsonl"), dir)
}

fn input(n: usize) -> InputRef {
    InputRef {
        uri: format!("s3://acme-inputs/c-{n}/manifest.json"),
        sha256: Some(SHA.into()),
        media_type: Some("application/json".into()),
    }
}

/// Every shard of `campaign`, stacked into one frame.
fn export(dir: &str, campaign: &str) -> DataFrame {
    let mut frames: Vec<DataFrame> = std::fs::read_dir(format!("{dir}/exports"))
        .unwrap()
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.starts_with(&format!("{campaign}_")) && n.ends_with(".parquet")
        })
        .map(|e| {
            ParquetReader::new(std::fs::File::open(e.path()).unwrap())
                .finish()
                .unwrap()
        })
        .collect();
    let mut df = frames.remove(0);
    for f in frames {
        df.vstack_mut(&f).unwrap();
    }
    df
}

fn str_col(df: &DataFrame, name: &str) -> Vec<Option<String>> {
    df.column(name)
        .unwrap_or_else(|_| panic!("missing column {name}"))
        .str()
        .unwrap()
        .into_iter()
        .map(|v| v.map(str::to_string))
        .collect()
}

fn f64_col(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
    df.column(name)
        .unwrap_or_else(|_| panic!("missing column {name}"))
        .f64()
        .unwrap()
        .into_iter()
        .collect()
}

async fn paced_campaign(db: &BanditDB) {
    db.add_campaign_pacing(
        "c",
        vec!["a".into(), "b".into()],
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
        Some(PacingConfig {
            resources: vec![ResourceConstraint {
                name: "promo".into(),
                budget: 100.0,
                horizon: 1_000,
                step_size: None,
                lambda_max: None,
                initial_lambda: Some(0.25),
                arm_costs: HashMap::from([("a".into(), 1.0), ("b".into(), 2.0)]),
            }],
            adaptive: true,
        }),
    )
    .await
    .unwrap();
}

/// Predict with an input reference on a budgeted campaign, reward, checkpoint:
/// the row holds the propensity map, the reference and the budget state as the
/// decision saw it.
#[tokio::test]
async fn exported_row_carries_propensities_input_and_budget() {
    let (wal, dir) = fresh("round_trip");
    let db = BanditDB::new(&wal, &dir);
    paced_campaign(&db).await;

    let (arm, iid) = db
        .predict_with_ref("c", vec![0.5, 0.5], &ArmFilter::default(), Some(input(1)))
        .unwrap();
    db.reward(&iid, 1.0).await.unwrap();
    db.checkpoint().await.unwrap();

    let df = export(&dir, "c");
    assert_eq!(df.height(), 1);
    assert_eq!(str_col(&df, "reward_status"), vec![Some("rewarded".into())]);
    assert_eq!(str_col(&df, "input_uri"), vec![Some(input(1).uri)]);
    assert_eq!(str_col(&df, "input_sha256"), vec![Some(SHA.into())]);
    assert_eq!(
        str_col(&df, "input_media_type"),
        vec![Some("application/json".into())]
    );

    let props: HashMap<String, f64> =
        serde_json::from_str(str_col(&df, "arm_propensities")[0].as_deref().unwrap()).unwrap();
    assert_eq!(props.len(), 2);
    assert!((props.values().sum::<f64>() - 1.0).abs() < 1e-9);
    assert_eq!(f64_col(&df, "propensity"), vec![Some(props[&arm])]);

    // First decision: the full budget was left, and λ was the initial value —
    // the price that shaped the decision, not the one after its own update.
    let cost = if arm == "a" { 1.0 } else { 2.0 };
    assert_eq!(f64_col(&df, "cost__promo"), vec![Some(cost)]);
    assert_eq!(f64_col(&df, "lambda__promo"), vec![Some(0.25)]);
    assert_eq!(f64_col(&df, "remaining__promo"), vec![Some(100.0)]);
}

/// A reward that arrives after a checkpoint carries its prediction, since the
/// `Predicted` record was rotated away. The provenance must travel with it.
#[tokio::test]
async fn late_reward_keeps_input_and_budget() {
    let (wal, dir) = fresh("late_reward");
    let db = BanditDB::new(&wal, &dir);
    paced_campaign(&db).await;

    let (_, first) = db
        .predict_with_ref("c", vec![0.5, 0.5], &ArmFilter::default(), Some(input(1)))
        .unwrap();
    let (_, second) = db
        .predict_with_ref("c", vec![0.1, 0.9], &ArmFilter::default(), Some(input(2)))
        .unwrap();
    db.checkpoint().await.unwrap();
    db.reward(&first, 1.0).await.unwrap();
    db.reward(&second, 0.0).await.unwrap();
    db.checkpoint().await.unwrap();

    let df = export(&dir, "c")
        .sort(["predicted_at", "input_uri"], Default::default())
        .unwrap();
    assert_eq!(df.height(), 2);
    let mut uris = str_col(&df, "input_uri");
    uris.sort();
    assert_eq!(uris, vec![Some(input(1).uri), Some(input(2).uri)]);
    let remaining = f64_col(&df, "remaining__promo");
    assert!(remaining.iter().all(|r| r.is_some_and(|v| v <= 100.0)));
    // The second decision saw the first one's charge.
    let mut remaining: Vec<f64> = remaining.into_iter().flatten().collect();
    remaining.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(remaining[0], 100.0);
    assert!(remaining[1] < 100.0);
}

/// Unbudgeted campaigns and decisions without an input get null provenance
/// columns, not missing ones — and no pacing columns at all.
#[tokio::test]
async fn unbudgeted_campaign_exports_null_provenance() {
    let (wal, dir) = fresh("plain");
    let db = BanditDB::new(&wal, &dir);
    db.add_campaign(
        "p",
        vec!["a".into(), "b".into()],
        2,
        1.0,
        Algorithm::ThompsonSampling,
        None,
        None,
    )
    .await
    .unwrap();
    let (_, iid) = db.predict("p", vec![0.5, 0.5]).unwrap();
    db.reward(&iid, 1.0).await.unwrap();
    db.interact("p", "a", vec![0.2, 0.8], 0.0).await.unwrap();
    db.checkpoint().await.unwrap();

    let df = export(&dir, "p");
    assert_eq!(df.height(), 2);
    assert_eq!(str_col(&df, "input_uri"), vec![None, None]);
    assert!(df.get_column_names().iter().all(|c| !c.contains("__")));
    // `interact` logs no propensities; the predicted row does.
    let props = str_col(&df, "arm_propensities");
    assert_eq!(props.iter().filter(|p| p.is_some()).count(), 1);
    assert_eq!(
        str_col(&df, "reward_status"),
        vec![Some("rewarded".into()); 2]
    );
}

/// Records written before these fields existed still replay.
#[test]
fn old_records_without_provenance_still_parse() {
    let old = r#"{"Predicted":{"interaction_id":"i","campaign_id":"c","arm_id":"a","context":[0.5,0.5],"timestamp_secs":1,"arm_propensities":null,"is_reemit":false}}"#;
    let DbEvent::Predicted {
        budget, input_ref, ..
    } = serde_json::from_str(old).unwrap()
    else {
        panic!("not a Predicted record");
    };
    assert!(budget.is_empty());
    assert!(input_ref.is_none());
}
