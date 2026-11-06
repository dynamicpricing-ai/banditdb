//! `input_ref` through a binary (msgpack) WAL and a restart.
//!
//! `BANDITDB_WAL_FORMAT` is process-wide, so this file holds a single test.

use banditdb::engine::{ArmFilter, WalMessage};
use banditdb::state::{Algorithm, InputRef};
use banditdb::BanditDB;
use std::fs;

#[tokio::test]
async fn msgpack_wal_replays_input_ref() {
    std::env::set_var("BANDITDB_WAL_FORMAT", "msgpack");
    let dir = "/tmp/banditdb_input_ref_msgpack";
    let wal = format!("{dir}/wal.log");
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();

    let input_ref = InputRef {
        uri: "mongodb://support/messages/65f1c2ab9d".into(),
        sha256: Some("ab".repeat(32)),
        media_type: Some("text/plain".into()),
    };
    let db = BanditDB::new(&wal, dir);
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
    let (_, with_ref) = db
        .predict_with_ref(
            "c",
            vec![0.5, 0.5],
            &ArmFilter::default(),
            Some(input_ref.clone()),
        )
        .unwrap();
    let (_, plain) = db.predict("c", vec![0.5, 0.5]).unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    db.event_tx
        .send(WalMessage::Checkpoint { reply: tx })
        .await
        .unwrap();
    rx.await.unwrap();
    drop(db);

    assert!(
        fs::read(&wal).unwrap().starts_with(b"BDMP"),
        "WAL must be msgpack"
    );
    let db = BanditDB::new(&wal, dir);
    assert_eq!(
        db.interactions.get(&with_ref).unwrap().input_ref,
        Some(input_ref)
    );
    assert_eq!(db.interactions.get(&plain).unwrap().input_ref, None);
}
