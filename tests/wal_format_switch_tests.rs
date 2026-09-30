//! Changing `BANDITDB_WAL_FORMAT` on an existing data directory must not lose
//! acknowledged writes.
//!
//! The writer used to append in the configured format to whatever WAL was already
//! there, while recovery parses the whole file in the format its first bytes
//! declare and skips what it cannot parse. Either direction of the switch
//! silently dropped every record written after it. The writer now keeps the
//! file's format and converts at the next rotation.
//!
//! `BANDITDB_WAL_FORMAT` is process-wide, so this file holds a single test.

use banditdb::state::Algorithm;
use banditdb::BanditDB;
use std::fs;
use std::sync::atomic::Ordering;

const MAGIC: &[u8] = b"BDMP";

fn reward_count(db: &BanditDB) -> u64 {
    db.campaigns.read()["c"].arms.read()["a"].reward_count.load(Ordering::Relaxed)
}

fn open(wal: &str, dir: &str, format: &str) -> BanditDB {
    std::env::set_var("BANDITDB_WAL_FORMAT", format);
    BanditDB::new(wal, dir)
}

async fn interact_n(db: &BanditDB, n: usize) {
    for _ in 0..n {
        db.interact("c", "a", vec![0.5, 0.5], 1.0).await.unwrap();
    }
}

#[tokio::test]
async fn switching_wal_format_loses_nothing_and_converts_at_rotation() {
    for (from, to) in [("json", "msgpack"), ("msgpack", "json")] {
        let dir = format!("/tmp/banditdb_wal_format_switch_{from}");
        let wal = format!("{dir}/wal.log");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let is_msgpack = |path: &str| fs::read(path).unwrap().starts_with(MAGIC);

        let db = open(&wal, &dir, from);
        db.add_campaign("c", vec!["a".into()], 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
        interact_n(&db, 3).await;
        drop(db);

        // Restart with the other format configured and keep writing.
        let db = open(&wal, &dir, to);
        interact_n(&db, 4).await;
        drop(db);

        let db = open(&wal, &dir, to);
        assert_eq!(reward_count(&db), 7, "{from} -> {to}: acknowledged rewards lost after the switch");
        assert_eq!(is_msgpack(&wal), from == "msgpack",
            "{from} -> {to}: the WAL must stay in its existing format until rotation");

        // A checkpoint rotates the WAL into the configured format, tail included.
        interact_n(&db, 2).await;
        db.checkpoint().await.unwrap();
        interact_n(&db, 5).await;
        assert_eq!(is_msgpack(&wal), to == "msgpack",
            "{from} -> {to}: rotation must convert the WAL to the configured format");
        drop(db);

        let db = open(&wal, &dir, to);
        assert_eq!(reward_count(&db), 14, "{from} -> {to}: rewards lost across the converting rotation");
        drop(db);
        let _ = fs::remove_dir_all(&dir);
    }
}
