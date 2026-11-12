//! `input_ref`: a pointer to what a decision was made from, stored with it.
//!
//! It must reach the WAL on `Predicted`, survive replay and checkpoints in the
//! pending cache, and ride along on the reward's `unlogged_prediction` when the
//! best-effort `Predicted` record was dropped. URIs carrying credentials are
//! refused before anything is scored or logged.

use banditdb::engine::{ArmFilter, WalMessage};
use banditdb::state::{Algorithm, DbEvent, EngineError, InputRef, InteractionRecord};
use banditdb::BanditDB;
use ndarray::Array1;
use std::fs;
use std::sync::atomic::Ordering;

fn r(uri: &str) -> InputRef {
    InputRef {
        uri: uri.into(),
        sha256: None,
        media_type: None,
    }
}

fn full() -> InputRef {
    InputRef {
        uri: "s3://acme-inputs/c-4412/manifest.json".into(),
        sha256: Some("9F2C".repeat(16)),
        media_type: Some("application/json".into()),
    }
}

async fn setup(dir: &str) -> BanditDB {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    db.add_campaign(
        "c",
        vec!["A".into(), "B".into()],
        2,
        1.0,
        Algorithm::Linucb,
        None,
        None,
    )
    .await
    .unwrap();
    db
}

async fn flush(db: &BanditDB) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    db.event_tx
        .send(WalMessage::Checkpoint { reply: tx })
        .await
        .unwrap();
    rx.await.unwrap();
}

fn predictions(db: &BanditDB) -> u64 {
    db.campaigns.read()["c"]
        .arms
        .read()
        .values()
        .map(|a| a.prediction_count.load(Ordering::Relaxed))
        .sum()
}

// ── Validation ───────────────────────────────────────────────────────────────

#[test]
fn accepts_pointers_to_common_stores() {
    for uri in [
        "s3://acme-inputs/c-4412/manifest.json",
        "gs://bucket/path/obj.png",
        "postgres://crm/tickets/4412",
        "postgres://crm/tickets/4412?v=17",
        "mongodb://support/messages/65f1c2ab9d",
        "https://files.example.com/docs/a.pdf",
        "urn:uuid:6e8bc430-9c3a-11d9-9669-0800200c9a66",
    ] {
        assert!(r(uri).validated().is_ok(), "{uri} must be accepted");
    }
}

#[test]
fn normalises_sha256_to_lowercase() {
    let v = full().validated().unwrap();
    assert_eq!(v.sha256.unwrap(), "9f2c".repeat(16));
}

#[test]
fn rejects_credentials_in_uri() {
    for uri in [
        "postgres://admin:hunter2@db.internal/crm/tickets/1",
        "mongodb://user@host/db/coll/1",
        "https://b.s3.amazonaws.com/k?X-Amz-Algorithm=AWS4&X-Amz-Signature=abc",
        "https://acct.blob.core.windows.net/c/b?sv=2022&sig=abc",
        "https://example.com/doc?TOKEN=abc",
        "https://example.com/doc?a=1;access_token=abc",
        "https://example.com/doc?api_key=abc#frag",
    ] {
        match r(uri).validated() {
            Err(EngineError::BadRequest(m)) => assert!(m.contains("credentials"), "{uri}: {m}"),
            other => panic!("{uri} must be rejected, got {other:?}"),
        }
    }
}

#[test]
fn credential_names_only_match_whole_parameters() {
    // "tokens" and "signature_version" are not credentials.
    assert!(r("https://example.com/doc?tokens=3&signature_version=2")
        .validated()
        .is_ok());
}

#[test]
fn rejects_malformed_references() {
    let long = format!("s3://b/{}", "k".repeat(600));
    let cases: Vec<(InputRef, &str)> = vec![
        (r(""), "empty"),
        (r(&long), "limit is 512"),
        (r("no-scheme-here"), "scheme"),
        (r("1s3://b/k"), "scheme"),
        (r("s3://b/a key"), "whitespace"),
        (r("s3://b/k\n"), "whitespace"),
        (
            InputRef {
                sha256: Some("abc".into()),
                ..r("s3://b/k")
            },
            "64 hex",
        ),
        (
            InputRef {
                sha256: Some("z".repeat(64)),
                ..r("s3://b/k")
            },
            "64 hex",
        ),
        (
            InputRef {
                media_type: Some("json".into()),
                ..r("s3://b/k")
            },
            "MIME",
        ),
        (
            InputRef {
                media_type: Some("text/".into()),
                ..r("s3://b/k")
            },
            "MIME",
        ),
    ];
    for (input, want) in cases {
        match input.clone().validated() {
            Err(EngineError::BadRequest(m)) => assert!(m.contains(want), "{input:?}: {m}"),
            other => panic!("{input:?} must be rejected, got {other:?}"),
        }
    }
    let ok = InputRef {
        media_type: Some("text/plain; charset=utf-8".into()),
        ..r("s3://b/k")
    };
    assert!(ok.validated().is_ok());
}

#[tokio::test]
async fn rejected_ref_scores_nothing_and_logs_nothing() {
    let dir = "/tmp/banditdb_input_ref_rejected";
    let db = setup(dir).await;
    let bad = r("postgres://u:p@h/db/t/1");

    let res = db.predict_with_ref(
        "c",
        vec![0.5, 0.5],
        &ArmFilter::default(),
        Some(bad.clone()),
    );
    assert!(matches!(res, Err(EngineError::BadRequest(_))));
    let res = db
        .interact_with_ref("c", "A", vec![0.5, 0.5], 1.0, Some(bad))
        .await;
    assert!(matches!(res, Err(EngineError::BadRequest(_))));

    assert_eq!(predictions(&db), 0, "a rejected request must not count");
    flush(&db).await;
    let wal = fs::read_to_string(format!("{dir}/wal.jsonl")).unwrap();
    assert!(!wal.contains("Predicted"), "nothing may reach the WAL");
    assert!(
        !wal.contains("u:p@"),
        "the credential must never be written"
    );
}

// ── Storage ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn predict_stores_ref_in_wal_and_pending_record() {
    let dir = "/tmp/banditdb_input_ref_predict";
    let db = setup(dir).await;
    let (_, iid) = db
        .predict_with_ref("c", vec![0.5, 0.5], &ArmFilter::default(), Some(full()))
        .unwrap();
    let want = full().validated().unwrap();
    assert_eq!(
        db.interactions.get(&iid).unwrap().input_ref,
        Some(want.clone())
    );

    // A plain predict stores none, and the WAL record omits the field.
    let (_, plain) = db.predict("c", vec![0.5, 0.5]).unwrap();
    assert_eq!(db.interactions.get(&plain).unwrap().input_ref, None);

    flush(&db).await;
    let wal = fs::read_to_string(format!("{dir}/wal.jsonl")).unwrap();
    let with_ref = wal.lines().find(|l| l.contains(&iid)).unwrap();
    assert!(with_ref.contains(r#""input_ref":{"uri":"s3://acme-inputs/c-4412/manifest.json""#));
    let without = wal.lines().find(|l| l.contains(&plain)).unwrap();
    assert!(!without.contains("input_ref"));

    // Replay restores it, so a reward after a restart still has the pointer.
    drop(db);
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    assert_eq!(db.interactions.get(&iid).unwrap().input_ref, Some(want));
    assert_eq!(db.interactions.get(&plain).unwrap().input_ref, None);
}

#[tokio::test]
async fn interact_stores_ref_in_wal() {
    let dir = "/tmp/banditdb_input_ref_interact";
    let db = setup(dir).await;
    let iid = db
        .interact_with_ref(
            "c",
            "B",
            vec![0.5, 0.5],
            1.0,
            Some(r("postgres://crm/tickets/9")),
        )
        .await
        .unwrap();
    let wal = fs::read_to_string(format!("{dir}/wal.jsonl")).unwrap();
    let line = wal
        .lines()
        .find(|l| l.contains(&iid) && l.contains("Predicted"))
        .unwrap();
    assert!(line.contains(r#""input_ref":{"uri":"postgres://crm/tickets/9"}"#));
}

#[tokio::test]
async fn checkpoint_keeps_ref_on_in_flight_predictions() {
    let dir = "/tmp/banditdb_input_ref_checkpoint";
    let db = setup(dir).await;
    let (_, iid) = db
        .predict_with_ref("c", vec![0.5, 0.5], &ArmFilter::default(), Some(full()))
        .unwrap();
    db.checkpoint().await.unwrap();
    drop(db);

    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    assert_eq!(
        db.interactions.get(&iid).unwrap().input_ref,
        Some(full().validated().unwrap()),
        "an in-flight prediction restored from checkpoint.json must keep its ref"
    );
}

#[test]
fn unlogged_prediction_carries_ref_in_both_wal_formats() {
    // When Predicted is dropped under WAL pressure, the reward carries the
    // record instead. The ref must survive that path in JSON and msgpack.
    let record = InteractionRecord {
        campaign_id: "c".into(),
        arm_id: "A".into(),
        context: Array1::from_vec(vec![0.5, 0.5]),
        arm_propensities: None,
        timestamp_secs: 1,
        input_ref: Some(full()),
        budget: Vec::new(),
        logged: false,
        wal_epoch: 0,
    };
    let event = DbEvent::Rewarded {
        interaction_id: "i".into(),
        reward: 1.0,
        timestamp_secs: 2,
        unlogged_prediction: Some(record),
    };

    let json: DbEvent = serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap();
    let mp: DbEvent = rmp_serde::from_slice(&rmp_serde::to_vec_named(&event).unwrap()).unwrap();
    for back in [json, mp] {
        let DbEvent::Rewarded {
            unlogged_prediction: Some(rec),
            ..
        } = back
        else {
            panic!("unlogged_prediction lost");
        };
        assert_eq!(rec.input_ref, Some(full()));
    }
}

#[test]
fn predicted_round_trips_in_msgpack_with_and_without_ref() {
    for input_ref in [None, Some(full())] {
        let event = DbEvent::Predicted {
            interaction_id: "i".into(),
            campaign_id: "c".into(),
            arm_id: "A".into(),
            context: vec![0.5, 0.5],
            timestamp_secs: 1,
            arm_propensities: None,
            is_reemit: false,
            budget: Vec::new(),
            input_ref: input_ref.clone(),
        };
        let back: DbEvent =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&event).unwrap()).unwrap();
        let DbEvent::Predicted { input_ref: got, .. } = back else {
            panic!("wrong variant");
        };
        assert_eq!(got, input_ref);
    }
}

#[test]
fn old_records_without_ref_deserialise() {
    let line = r#"{"Predicted":{"interaction_id":"i","campaign_id":"c","arm_id":"A","context":[0.5,0.5],"timestamp_secs":1,"arm_propensities":null,"is_reemit":false}}"#;
    let DbEvent::Predicted { input_ref, .. } = serde_json::from_str(line).unwrap() else {
        panic!("wrong variant");
    };
    assert_eq!(input_ref, None);

    // A newer server may add fields to input_ref; this one must still replay it.
    let line = r#"{"Predicted":{"interaction_id":"i","campaign_id":"c","arm_id":"A","context":[0.5,0.5],"timestamp_secs":1,"arm_propensities":null,"is_reemit":false,"input_ref":{"uri":"s3://b/k","future_field":1}}}"#;
    let DbEvent::Predicted { input_ref, .. } = serde_json::from_str(line).unwrap() else {
        panic!("wrong variant");
    };
    assert_eq!(input_ref.unwrap().uri, "s3://b/k");
}
