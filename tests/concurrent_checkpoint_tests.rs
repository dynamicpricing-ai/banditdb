//! Two `checkpoint()` calls must not overlap.
//!
//! The manual endpoint, the automatic task and shutdown can all call it. Nothing
//! serialised them, so the second rotation seeked to its own barrier offset in a
//! WAL the first rotation had already rewritten — discarding acknowledged rewards
//! — while both raced on `checkpoint.tmp`, the `.json` → `.prev` rename and the
//! generation counter.

use banditdb::state::{Algorithm, CheckpointData};
use banditdb::BanditDB;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn reward_count(db: &BanditDB) -> u64 {
    db.campaigns.read()["c"].arms.read()["a"].reward_count.load(Ordering::Relaxed)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overlapping_checkpoints_lose_nothing() {
    for round in 0..5 {
        let dir = format!("/tmp/banditdb_concurrent_checkpoints_{round}");
        let wal = format!("{dir}/wal.jsonl");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let db = Arc::new(BanditDB::new(&wal, &dir));
        db.add_campaign("c", vec!["a".into()], 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
        // Enough WAL that each checkpoint's export takes long enough to overlap.
        for _ in 0..20_000 {
            db.predict("c", vec![0.5, 0.5]).unwrap();
        }

        let stop = Arc::new(AtomicBool::new(false));
        let writers: Vec<_> = (0..4).map(|_| {
            let (db, stop) = (Arc::clone(&db), Arc::clone(&stop));
            tokio::spawn(async move {
                while !stop.load(Ordering::Relaxed) {
                    db.interact("c", "a", vec![0.5, 0.5], 1.0).await.unwrap();
                }
            })
        }).collect();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let (a, b) = (Arc::clone(&db), Arc::clone(&db));
        let (r1, r2) = tokio::join!(
            tokio::spawn(async move { a.checkpoint().await }),
            tokio::spawn(async move { b.checkpoint().await }),
        );
        stop.store(true, Ordering::Relaxed);
        for w in writers { w.await.unwrap(); }
        r1.unwrap().expect("first checkpoint failed");
        r2.unwrap().expect("second checkpoint failed");

        let cp: CheckpointData = serde_json::from_str(
            &fs::read_to_string(format!("{dir}/checkpoint.json")).unwrap()).unwrap();
        assert_eq!(cp.generation, 2, "round {round}: two checkpoints must produce generation 2");
        let first_record = fs::read_to_string(&wal).unwrap().lines().next().unwrap_or("").to_string();
        assert_eq!(first_record, r#"{"WalStart":{"generation":2}}"#,
            "round {round}: the WAL must continue from the checkpoint on disk");

        let live = reward_count(&db);
        drop(Arc::try_unwrap(db).ok().expect("writers still hold the db"));
        let db = BanditDB::new(&wal, &dir);
        assert_eq!(reward_count(&db), live, "round {round}: acknowledged rewards lost on restart");
        let _ = fs::remove_dir_all(&dir);
    }
}
