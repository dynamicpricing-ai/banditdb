//! `BANDITDB_MAX_TOTAL_CAMPAIGN_BYTES` bounds the memory all campaigns reserve
//! together.
//!
//! The per-campaign and per-tenant limits each cap one thing; nothing capped the
//! sum, so many valid campaigns could still exhaust the process. The env var is
//! read at startup and is process-wide, so this file holds a single test.

use banditdb::state::{Algorithm, EngineError, WarmStart};
use banditdb::BanditDB;
use std::fs;
use std::sync::Arc;

// One LinUCB arm at d=256 reserves ~0.5 MB.
const D: usize = 256;

fn fresh(dir: &str) -> String {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    format!("{dir}/wal.jsonl")
}

fn open(wal: &str, dir: &str, ceiling: u64) -> BanditDB {
    std::env::set_var("BANDITDB_MAX_TOTAL_CAMPAIGN_BYTES", ceiling.to_string());
    BanditDB::new(wal, dir)
}

async fn create(db: &BanditDB, id: &str) -> Result<(), EngineError> {
    db.add_campaign(id, vec!["a".into()], D, 1.0, Algorithm::Linucb, None, None).await
}

fn is_ceiling_error(e: &EngineError) -> bool {
    matches!(e, EngineError::LimitExceeded(m) if m.contains("BANDITDB_MAX_TOTAL_CAMPAIGN_BYTES"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn campaigns_share_one_memory_ceiling() {
    // ── Filling up ───────────────────────────────────────────────────────────
    let dir = "/tmp/banditdb_total_budget";
    let wal = fresh(dir);
    let db = open(&wal, dir, 3 * 1024 * 1024);

    let mut admitted = 0;
    let refusal = loop {
        match create(&db, &format!("c{admitted}")).await {
            Ok(()) => admitted += 1,
            Err(e) => break e,
        }
        assert!(admitted < 50, "campaigns were never refused");
    };
    assert!(admitted >= 2, "campaigns that fit must be admitted");
    assert!(is_ceiling_error(&refusal), "expected the total ceiling to refuse, got {refusal:?}");

    // Growing an existing campaign draws on the same ceiling.
    let err = db.add_arm("c0", "b", None, &WarmStart::default()).await.unwrap_err();
    assert!(is_ceiling_error(&err), "adding an arm must count against the ceiling, got {err:?}");

    // Deleting a campaign gives its reservation back.
    db.delete_campaign("c1").await.unwrap();
    create(&db, "after_delete").await.expect("a deleted campaign's memory must be released");
    drop(db);

    // ── Restart under a lower ceiling ────────────────────────────────────────
    // Recovery must never refuse campaigns that were already admitted: lowering
    // the ceiling cannot be allowed to make an instance unrecoverable.
    let db = open(&wal, dir, 1024 * 1024);
    assert_eq!(db.campaigns.read().len(), admitted, "recovery must restore every campaign");
    let err = create(&db, "over").await.unwrap_err();
    assert!(is_ceiling_error(&err), "an over-ceiling instance admits nothing new, got {err:?}");
    drop(db);
    let _ = fs::remove_dir_all(dir);

    // ── Concurrent admission ─────────────────────────────────────────────────
    // Room for exactly three. The check and the reservation must be one step, or
    // concurrent creations all pass the check and overshoot together.
    let dir = "/tmp/banditdb_total_budget_race";
    let wal = fresh(dir);
    let db = Arc::new(open(&wal, dir, 3 * 540 * 1024));
    let attempts: Vec<_> = (0..10).map(|i| {
        let db = Arc::clone(&db);
        tokio::spawn(async move { create(&db, &format!("r{i}")).await.is_ok() })
    }).collect();
    let mut ok = 0;
    for a in attempts { if a.await.unwrap() { ok += 1; } }
    assert_eq!(ok, 3, "concurrent creations overshot the ceiling");
    let _ = fs::remove_dir_all(dir);
}
