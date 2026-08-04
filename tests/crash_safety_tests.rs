//! P0.2 acceptance tests — checkpoint durability and crash recovery.
//!
//! Background (docs/PRODUCTION_STAGE1.md §2.1): checkpointing is immediately
//! followed by WAL rotation, which discards every event the checkpoint subsumes.
//! That makes the checkpoint the sole record of prior state, so any window where it
//! is absent or unreadable is unrecoverable data loss. Two defects existed:
//!
//!   1. `fs::write` + `fs::rename` left both contents and rename in the page cache,
//!      so a power loss after rotation lost everything.
//!   2. Recovery collapsed a parse failure into "no checkpoint", started empty, and
//!      reported healthy — then overwrote the evidence at the next checkpoint.

use banditdb::{BanditDB, engine::CheckpointLoad};
use banditdb::state::Algorithm;
use std::fs;
use std::path::Path;

fn fresh(dir: &str) {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
}

/// Build a campaign, drive traffic, and checkpoint it.
async fn seeded_db(dir: &str, rewards: usize) -> BanditDB {
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    db.add_campaign("c", vec!["A".into(), "B".into()], 2, 1.0, Algorithm::Linucb, None, None)
        .unwrap();
    for i in 0..rewards {
        let ctx = vec![(i % 5) as f64 / 5.0, (i % 3) as f64 / 3.0];
        if let Ok((arm, iid)) = db.predict("c", ctx) {
            let _ = db.reward(&iid, if arm == "A" { 1.0 } else { 0.0 });
        }
    }
    db.checkpoint().await.expect("checkpoint");
    db
}

// ---------------------------------------------------------------------------
// Durability of the checkpoint write itself
// ---------------------------------------------------------------------------

#[tokio::test]
async fn checkpoint_retains_previous_generation() {
    let dir = "/tmp/banditdb_p02_generations";
    fresh(dir);
    let db = seeded_db(dir, 20).await;

    assert!(Path::new(&format!("{dir}/checkpoint.json")).exists());
    // First checkpoint has no predecessor to retain.
    assert!(!Path::new(&format!("{dir}/checkpoint.prev")).exists());

    db.checkpoint().await.expect("second checkpoint");
    assert!(
        Path::new(&format!("{dir}/checkpoint.prev")).exists(),
        "the superseded checkpoint must be retained as checkpoint.prev — it is the \
         only fallback once the WAL has been rotated past it"
    );

    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn checkpoint_leaves_no_temp_files_behind() {
    let dir = "/tmp/banditdb_p02_no_temps";
    fresh(dir);
    let db = seeded_db(dir, 10).await;
    db.checkpoint().await.expect("checkpoint");

    assert!(!Path::new(&format!("{dir}/checkpoint.tmp")).exists());
    assert!(!Path::new(&format!("{dir}/wal_rotation.tmp")).exists());
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// Recovery decision table — the branch that used to silently lose data
// ---------------------------------------------------------------------------

#[test]
fn empty_data_dir_is_fresh_not_corrupt() {
    let dir = "/tmp/banditdb_p02_fresh";
    fresh(dir);
    assert!(matches!(BanditDB::load_checkpoint(dir), CheckpointLoad::Fresh));
    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn corrupt_checkpoint_without_fallback_is_reported_corrupt() {
    let dir = "/tmp/banditdb_p02_corrupt";
    fresh(dir);
    let _db = seeded_db(dir, 20).await;

    // Truncated JSON — the classic outcome of a crash mid-write on the old code path.
    fs::write(format!("{dir}/checkpoint.json"), "{\"wal_offset\": 12, \"campai").unwrap();

    match BanditDB::load_checkpoint(dir) {
        CheckpointLoad::Corrupt(e) => assert!(e.contains("parse"), "unexpected reason: {e}"),
        other => panic!(
            "a truncated checkpoint with no fallback must report Corrupt so startup can \
             abort; got {other:?} — this is the silent-data-loss regression"
        ),
    }
    let _ = fs::remove_dir_all(dir);
}

#[tokio::test]
async fn corrupt_checkpoint_falls_back_to_previous_generation() {
    let dir = "/tmp/banditdb_p02_fallback";
    fresh(dir);
    let db = seeded_db(dir, 20).await;
    db.checkpoint().await.expect("second checkpoint creates .prev");
    drop(db);

    fs::write(format!("{dir}/checkpoint.json"), "not json at all").unwrap();

    match BanditDB::load_checkpoint(dir) {
        CheckpointLoad::Loaded(cp) => {
            assert!(cp.campaigns.contains_key("c"), "fallback checkpoint lost the campaign");
        }
        other => panic!("expected fallback to checkpoint.prev, got {other:?}"),
    }
    let _ = fs::remove_dir_all(dir);
}

/// Simulates a crash in the window between `rename(json -> prev)` and
/// `rename(tmp -> json)`: only the previous generation exists on disk.
#[tokio::test]
async fn crash_between_generation_renames_recovers_from_prev() {
    let dir = "/tmp/banditdb_p02_midrename";
    fresh(dir);
    let db = seeded_db(dir, 20).await;
    db.checkpoint().await.expect("second checkpoint");
    drop(db);

    // The crash window: checkpoint.json not yet in place.
    fs::remove_file(format!("{dir}/checkpoint.json")).unwrap();
    assert!(Path::new(&format!("{dir}/checkpoint.prev")).exists());

    match BanditDB::load_checkpoint(dir) {
        CheckpointLoad::Loaded(cp) => assert!(cp.campaigns.contains_key("c")),
        other => panic!("crash between renames must recover from checkpoint.prev, got {other:?}"),
    }
    let _ = fs::remove_dir_all(dir);
}

/// End-to-end: a corrupt current generation must not cost the campaign, and the
/// recovered database must be usable rather than merely present.
#[tokio::test]
async fn recovered_database_serves_predictions_after_fallback() {
    let dir = "/tmp/banditdb_p02_e2e";
    fresh(dir);
    let db = seeded_db(dir, 30).await;
    db.checkpoint().await.expect("second checkpoint");
    drop(db);

    fs::write(format!("{dir}/checkpoint.json"), "{ truncated").unwrap();

    let recovered = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    let campaigns = recovered.campaigns.read();
    assert!(campaigns.contains_key("c"), "campaign lost during fallback recovery");
    drop(campaigns);

    let (arm, _) = recovered.predict("c", vec![0.4, 0.6]).expect("predict after recovery");
    assert!(arm == "A" || arm == "B");
    let _ = fs::remove_dir_all(dir);
}
