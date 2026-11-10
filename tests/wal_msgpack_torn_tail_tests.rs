//! A binary WAL that ends in a partial frame must be cut back before the writer
//! appends to it. Otherwise the stale length prefix swallows the records written
//! after restart, and those acknowledged writes are lost at the next replay.
//!
//! `BANDITDB_WAL_FORMAT` is process-wide, so this file holds a single test.

use banditdb::state::Algorithm;
use banditdb::BanditDB;
use std::io::Write;
use std::time::Duration;

async fn create(db: &BanditDB, id: &str) {
    db.add_campaign(
        id,
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

#[tokio::test]
async fn a_write_after_a_torn_binary_frame_survives_the_next_restart() {
    std::env::set_var("BANDITDB_WAL_FORMAT", "msgpack");
    let dir = "/tmp/banditdb_wal_msgpack_torn";
    let wal = format!("{dir}/wal.log");
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();

    {
        let db = BanditDB::new(&wal, dir);
        create(&db, "one").await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let intact_len = std::fs::metadata(&wal).unwrap().len();

    // A frame that claims 200 bytes but was cut off after 10.
    let mut f = std::fs::OpenOptions::new().append(true).open(&wal).unwrap();
    f.write_all(&200u32.to_le_bytes()).unwrap();
    f.write_all(&[0xAB; 10]).unwrap();
    drop(f);

    {
        let db = BanditDB::new(&wal, dir);
        assert!(db.campaigns.read().contains_key("one"));
        create(&db, "two").await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        std::fs::read(format!("{wal}.damaged-tail")).unwrap().len(),
        14,
        "the removed bytes are kept for inspection"
    );
    assert!(std::fs::metadata(&wal).unwrap().len() > intact_len);

    let db = BanditDB::new(&wal, dir);
    let campaigns = db.campaigns.read();
    assert!(campaigns.contains_key("one"));
    assert!(
        campaigns.contains_key("two"),
        "a write acknowledged after the torn frame was lost"
    );
}
