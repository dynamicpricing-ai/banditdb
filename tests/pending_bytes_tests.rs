//! The pending-interaction cache is bounded in bytes, not only in entries.
//!
//! It was capped at 100,000 entries whatever their size. Each entry holds its
//! context vector, so at d=4096 the cap meant ~3 GB of pending predictions —
//! shared by every tenant, and serialised into every checkpoint.

use banditdb::state::Algorithm;
use banditdb::BanditDB;
use std::fs;
use std::sync::atomic::Ordering;

#[tokio::test]
async fn wide_contexts_are_capped_by_bytes() {
    let budget: u64 = 4 * 1024 * 1024;
    std::env::set_var("BANDITDB_MAX_PENDING_BYTES", budget.to_string());
    let dir = "/tmp/banditdb_pending_bytes";
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);

    // d=1024: each pending entry carries an 8 KB context.
    let d = 1024;
    db.add_campaign("c", vec!["a".into(), "b".into()], d, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    for _ in 0..2000 {
        db.predict("c", vec![0.5; d]).unwrap();
    }
    db.interactions.run_pending_tasks();

    let held = db.interactions.entry_count();
    assert!(held * (d as u64 * 8) <= budget,
        "{held} pending entries of {} KB each exceed the {} MB byte budget", d * 8 / 1024, budget >> 20);
    assert!(held > 0, "recent predictions must still be held");
    assert!(db.interactions_evicted.load(Ordering::Relaxed) > 0, "evictions must be counted");
    let _ = fs::remove_dir_all(dir);
}
