//! Replay must survive a damaged WAL record without losing the records around
//! it, the Parquet export must include rewards that arrive after a checkpoint,
//! and an archived campaign must not learn.

use banditdb::state::{Algorithm, EngineError};
use banditdb::BanditDB;
use std::io::Write;
use std::sync::atomic::Ordering;
use std::time::Duration;

fn fresh(name: &str) -> (String, String) {
    let dir = format!("/tmp/banditdb_wal_robust_{name}");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    (format!("{dir}/wal.jsonl"), dir)
}

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

fn append(wal: &str, bytes: &[u8]) {
    let mut f = std::fs::OpenOptions::new().append(true).open(wal).unwrap();
    f.write_all(bytes).unwrap();
}

fn rewards(db: &BanditDB, campaign: &str) -> u64 {
    db.campaigns
        .read()
        .get(campaign)
        .unwrap()
        .arms
        .read()
        .values()
        .map(|a| a.reward_count.load(Ordering::Relaxed))
        .sum()
}

fn exported_rows(dir: &str, campaign: &str) -> usize {
    use polars::prelude::*;
    let Ok(entries) = std::fs::read_dir(format!("{dir}/exports")) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.starts_with(&format!("{campaign}_")) && n.ends_with(".parquet")
        })
        .map(|e| {
            ParquetReader::new(std::fs::File::open(e.path()).unwrap())
                .finish()
                .unwrap()
                .height()
        })
        .sum()
}

/// A line that is not valid UTF-8 used to end replay: every record after it was
/// dropped, and the next checkpoint rotated them away for good.
#[tokio::test]
async fn replay_continues_past_a_line_that_is_not_utf8() {
    let (wal, dir) = fresh("utf8");
    {
        let db = BanditDB::new(&wal, &dir);
        create(&db, "before").await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let created = std::fs::read_to_string(&wal).unwrap();
    let after = created.lines().next().unwrap().replace("before", "after");
    append(&wal, b"{\"garbage\":\"\xff\xfe\"}\n");
    append(&wal, format!("{after}\n").as_bytes());

    let db = BanditDB::new(&wal, &dir);
    let campaigns = db.campaigns.read();
    assert!(campaigns.contains_key("before"));
    assert!(
        campaigns.contains_key("after"),
        "the record after the damaged line was not replayed"
    );
}

/// A crash mid-write leaves a partial last line. The first record written after
/// restart must not be glued onto it, or that acknowledged write is unparseable
/// at the next replay.
#[tokio::test]
async fn a_write_after_a_torn_tail_survives_the_next_restart() {
    let (wal, dir) = fresh("torn_json");
    {
        let db = BanditDB::new(&wal, &dir);
        create(&db, "one").await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    append(&wal, b"{\"Rewarded\":{\"interaction_id\":\"abc");

    {
        let db = BanditDB::new(&wal, &dir);
        create(&db, "two").await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    let db = BanditDB::new(&wal, &dir);
    let campaigns = db.campaigns.read();
    assert!(campaigns.contains_key("one"));
    assert!(
        campaigns.contains_key("two"),
        "a write acknowledged after the torn tail was lost"
    );
}

/// A reward that arrives after a checkpoint used to be missing from the export:
/// its Predicted record had been rotated out of the WAL the export reads.
#[tokio::test]
async fn export_includes_a_reward_that_arrives_after_a_checkpoint() {
    let (wal, dir) = fresh("late_reward");
    let db = BanditDB::new(&wal, &dir);
    create(&db, "c").await;

    let (_, late) = db.predict("c", vec![0.5, 0.5]).unwrap();
    db.checkpoint().await.unwrap();
    assert_eq!(exported_rows(&dir, "c"), 0);

    db.reward(&late, 1.0).await.unwrap();
    // A prediction rewarded within one WAL segment is exported exactly once.
    let (_, prompt) = db.predict("c", vec![0.1, 0.9]).unwrap();
    db.reward(&prompt, 0.0).await.unwrap();
    db.checkpoint().await.unwrap();

    assert_eq!(exported_rows(&dir, "c"), 2);
    assert_eq!(rewards(&db, "c"), 2);
}

/// The same across a restart: the pending prediction comes back from the
/// checkpoint, and its reward must still reach the export and the model.
#[tokio::test]
async fn export_includes_a_reward_that_arrives_after_a_restart() {
    let (wal, dir) = fresh("late_reward_restart");
    let iid = {
        let db = BanditDB::new(&wal, &dir);
        create(&db, "c").await;
        let (_, iid) = db.predict("c", vec![0.5, 0.5]).unwrap();
        db.checkpoint().await.unwrap();
        iid
    };
    tokio::time::sleep(Duration::from_millis(200)).await;

    {
        let db = BanditDB::new(&wal, &dir);
        db.reward(&iid, 1.0).await.unwrap();
        assert_eq!(rewards(&db, "c"), 1);
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Replay of that reward from the WAL tail, then export.
    let db = BanditDB::new(&wal, &dir);
    assert_eq!(rewards(&db, "c"), 1);
    db.checkpoint().await.unwrap();
    assert_eq!(exported_rows(&dir, "c"), 1);
}

/// Archived means frozen: no rewards, no historical ingestion. The pending
/// interaction is kept, so the reward lands once the campaign is restored.
#[tokio::test]
async fn archived_campaign_does_not_learn() {
    let (wal, dir) = fresh("archived");
    let db = BanditDB::new(&wal, &dir);
    create(&db, "z").await;
    let (_, iid) = db.predict("z", vec![0.5, 0.5]).unwrap();
    db.archive_campaign("z").await.unwrap();

    assert!(matches!(
        db.reward(&iid, 1.0).await,
        Err(EngineError::Archived(_))
    ));
    assert!(matches!(
        db.interact("z", "a", vec![0.5, 0.5], 1.0).await,
        Err(EngineError::Archived(_))
    ));
    assert_eq!(rewards(&db, "z"), 0);

    db.restore_campaign("z").await.unwrap();
    db.reward(&iid, 1.0).await.unwrap();
    assert_eq!(rewards(&db, "z"), 1);
}

/// The rejected reward must not be in the WAL either, or a restart would apply it.
#[tokio::test]
async fn rejected_reward_on_archived_campaign_is_not_replayed() {
    let (wal, dir) = fresh("archived_replay");
    {
        let db = BanditDB::new(&wal, &dir);
        create(&db, "z").await;
        let (_, iid) = db.predict("z", vec![0.5, 0.5]).unwrap();
        db.archive_campaign("z").await.unwrap();
        assert!(db.reward(&iid, 1.0).await.is_err());
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let db = BanditDB::new(&wal, &dir);
    assert_eq!(rewards(&db, "z"), 0);
}
