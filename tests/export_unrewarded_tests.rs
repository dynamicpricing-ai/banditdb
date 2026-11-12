//! With `BANDITDB_EXPORT_UNREWARDED=1`, a decision whose reward never arrives is
//! exported with `reward` null and a `reward_status` saying why: `expired` (TTL)
//! or `evicted` (pending cache full). Without the flag nothing changes.
//!
//! The settings are environment variables read when a `BanditDB` is built, so
//! every test builds its instance under `ENV_LOCK`.

use banditdb::engine::{ArmFilter, WalMessage};
use banditdb::state::{Algorithm, DbEvent, ExpiryCause, InputRef, InteractionRecord};
use banditdb::BanditDB;
use polars::prelude::*;
use std::sync::Mutex;
use std::time::Duration;

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn fresh(name: &str) -> (String, String) {
    let dir = format!("/tmp/banditdb_export_unrewarded_{name}");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    (format!("{dir}/wal.jsonl"), dir)
}

/// Build an instance with exactly these settings.
fn open(wal: &str, dir: &str, vars: &[(&str, &str)]) -> BanditDB {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for key in [
        "BANDITDB_EXPORT_UNREWARDED",
        "BANDITDB_REWARD_TTL_SECS",
        "BANDITDB_MAX_PENDING_INTERACTIONS",
    ] {
        std::env::remove_var(key);
    }
    for (k, v) in vars {
        std::env::set_var(k, v);
    }
    BanditDB::new(wal, dir)
}

async fn create(db: &BanditDB) {
    db.add_campaign(
        "c",
        vec!["a".into(), "b".into()],
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
    )
    .await
    .unwrap();
}

fn input(n: usize) -> Option<InputRef> {
    Some(InputRef {
        uri: format!("postgres://crm/claims/{n}"),
        sha256: None,
        media_type: None,
    })
}

fn predict(db: &BanditDB, n: usize) -> String {
    db.predict_with_ref("c", vec![0.5, 0.5], &ArmFilter::default(), input(n))
        .unwrap()
        .1
}

/// (reward_status, reward, input_uri) per exported row, sorted.
fn rows(dir: &str) -> Vec<(String, Option<f64>, Option<String>)> {
    let Ok(entries) = std::fs::read_dir(format!("{dir}/exports")) else {
        return vec![];
    };
    let mut out = vec![];
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !(name.starts_with("c_") && name.ends_with(".parquet")) {
            continue;
        }
        let df = ParquetReader::new(std::fs::File::open(e.path()).unwrap())
            .finish()
            .unwrap();
        let status = df.column("reward_status").unwrap().str().unwrap().clone();
        let reward = df.column("reward").unwrap().f64().unwrap().clone();
        let uri = df.column("input_uri").unwrap().str().unwrap().clone();
        let rewarded_at = df.column("rewarded_at").unwrap().i64().unwrap().clone();
        for i in 0..df.height() {
            // rewarded_at is null exactly when reward is.
            assert_eq!(rewarded_at.get(i).is_some(), reward.get(i).is_some());
            out.push((
                status.get(i).unwrap().to_string(),
                reward.get(i),
                uri.get(i).map(str::to_string),
            ));
        }
    }
    out.sort_by(|a, b| a.2.cmp(&b.2));
    out
}

#[tokio::test]
async fn expired_decision_is_exported_with_null_reward() {
    let (wal, dir) = fresh("expired");
    let db = open(
        &wal,
        &dir,
        &[
            ("BANDITDB_EXPORT_UNREWARDED", "1"),
            ("BANDITDB_REWARD_TTL_SECS", "1"),
        ],
    );
    create(&db).await;
    let late = predict(&db, 1);
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let on_time = predict(&db, 2);
    db.reward(&on_time, 1.0).await.unwrap();
    db.checkpoint().await.unwrap();

    assert_eq!(
        rows(&dir),
        vec![
            (
                "expired".into(),
                None,
                Some("postgres://crm/claims/1".into())
            ),
            (
                "rewarded".into(),
                Some(1.0),
                Some("postgres://crm/claims/2".into())
            ),
        ]
    );
    assert!(db.interactions.get(&late).is_none());
}

#[tokio::test]
async fn evicted_decision_is_exported_as_evicted() {
    let (wal, dir) = fresh("evicted");
    let db = open(
        &wal,
        &dir,
        &[
            ("BANDITDB_EXPORT_UNREWARDED", "1"),
            ("BANDITDB_MAX_PENDING_INTERACTIONS", "5"),
        ],
    );
    create(&db).await;
    for n in 0..50 {
        predict(&db, n);
    }
    db.checkpoint().await.unwrap();

    let rows = rows(&dir);
    let evicted = rows.iter().filter(|r| r.0 == "evicted").count();
    assert!(evicted >= 40, "only {evicted} of 50 exported as evicted");
    assert!(rows.iter().all(|r| r.0 == "evicted" && r.1.is_none()));
    // The checkpoint carries the rest as pending, so they are not exported yet.
    assert_eq!(evicted as u64 + db.interactions.entry_count(), 50);
}

#[tokio::test]
async fn without_the_flag_nothing_unrewarded_is_exported() {
    let (wal, dir) = fresh("flag_off");
    let db = open(&wal, &dir, &[("BANDITDB_REWARD_TTL_SECS", "1")]);
    create(&db).await;
    predict(&db, 1);
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let iid = predict(&db, 2);
    db.reward(&iid, 1.0).await.unwrap();
    db.checkpoint().await.unwrap();

    assert_eq!(
        rows(&dir),
        vec![(
            "rewarded".into(),
            Some(1.0),
            Some("postgres://crm/claims/2".into())
        )]
    );
    // The segment this checkpoint rotated out holds everything logged so far.
    let wal_text = std::fs::read_to_string(format!("{dir}/wal_segment.1")).unwrap();
    assert!(wal_text.contains("Predicted"));
    assert!(!wal_text.contains("PredictionExpired"));
}

/// A checkpoint can carry a prediction as pending that expires right after.
/// Restarting must not restore it and export it a second time.
#[tokio::test]
async fn an_expiry_after_a_checkpoint_is_exported_once_across_a_restart() {
    let (wal, dir) = fresh("restart");
    {
        let db = open(
            &wal,
            &dir,
            &[
                ("BANDITDB_EXPORT_UNREWARDED", "1"),
                ("BANDITDB_REWARD_TTL_SECS", "1"),
            ],
        );
        create(&db).await;
        predict(&db, 1);
        db.checkpoint().await.unwrap();
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        // Any cache call runs the housekeeping that notices the expiry.
        db.interactions.run_pending_tasks();
        let (tx, rx) = tokio::sync::oneshot::channel();
        db.event_tx
            .send(WalMessage::Checkpoint { reply: tx })
            .await
            .unwrap();
        rx.await.unwrap();
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    let db = open(
        &wal,
        &dir,
        &[
            ("BANDITDB_EXPORT_UNREWARDED", "1"),
            ("BANDITDB_REWARD_TTL_SECS", "1"),
        ],
    );
    assert_eq!(db.interactions.iter().count(), 0, "pending entry restored");
    db.checkpoint().await.unwrap();
    db.checkpoint().await.unwrap();
    assert_eq!(
        rows(&dir),
        vec![(
            "expired".into(),
            None,
            Some("postgres://crm/claims/1".into())
        )]
    );
}

/// The new record round-trips in both WAL formats.
#[test]
fn prediction_expired_round_trips() {
    let event = DbEvent::PredictionExpired {
        interaction_id: "i".into(),
        cause: ExpiryCause::Evicted,
        timestamp_secs: 7,
        prediction: InteractionRecord {
            campaign_id: "c".into(),
            arm_id: "a".into(),
            context: vec![0.5, 0.5].into(),
            arm_propensities: None,
            timestamp_secs: 1,
            input_ref: input(1),
            budget: Vec::new(),
            logged: true,
            wal_epoch: 0,
        },
    };
    let json: DbEvent = serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap();
    let mp: DbEvent = rmp_serde::from_slice(&rmp_serde::to_vec_named(&event).unwrap()).unwrap();
    for back in [json, mp] {
        let DbEvent::PredictionExpired {
            cause, prediction, ..
        } = back
        else {
            panic!("wrong variant");
        };
        assert_eq!(cause, ExpiryCause::Evicted);
        assert_eq!(prediction.input_ref, input(1));
    }
}
