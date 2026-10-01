//! A WAL write that fails must never be acknowledged, and must not leave a torn
//! or duplicated record behind.
//!
//! When retries ran out, the writer recorded the fatal error but still fell
//! through to its group-commit fsync. That fsync succeeds — it does not care that
//! the write failed — so every waiting caller was told `Ok(())`. Each retry also
//! rewrote the whole batch on top of whatever the failed attempt had already
//! written, leaving a torn line that recovery cannot parse, or duplicate records
//! that it would apply twice.
//!
//! The failure is real, not simulated: a file-size limit (`RLIMIT_FSIZE`, with
//! `SIGXFSZ` ignored) makes writes past the WAL's current end fail with `EFBIG`
//! while fsync keeps working. The limit is process-wide, so this file holds a
//! single test.

#![cfg(unix)]

use banditdb::engine::WalMessage;
use banditdb::state::Algorithm;
use banditdb::BanditDB;
use std::fs;
use std::sync::atomic::Ordering;

fn set_file_size_limit(limit: libc::rlim_t) {
    let lim = libc::rlimit {
        rlim_cur: limit,
        rlim_max: libc::RLIM_INFINITY,
    };
    // SAFETY: plain syscalls with a valid, fully initialised argument.
    unsafe {
        libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
        assert_eq!(
            libc::setrlimit(libc::RLIMIT_FSIZE, &lim),
            0,
            "setrlimit failed"
        );
    }
}

fn reward_count(db: &BanditDB) -> u64 {
    db.campaigns.read()["c"].arms.read()["a"]
        .reward_count
        .load(Ordering::Relaxed)
}

#[tokio::test]
async fn failed_wal_write_is_not_acknowledged_and_leaves_wal_intact() {
    let dir = "/tmp/banditdb_wal_write_failure";
    let wal = format!("{dir}/wal.jsonl");
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();

    let db = BanditDB::new(&wal, dir);
    db.add_campaign("c", vec!["a".into()], 2, 1.0, Algorithm::Linucb, None, None)
        .await
        .unwrap();
    let iid = db.predict("c", vec![0.5, 0.5]).unwrap().1;

    // Flush, then forbid the WAL from growing past its current end.
    let (tx, rx) = tokio::sync::oneshot::channel();
    db.event_tx
        .send(WalMessage::Checkpoint { reply: tx })
        .await
        .unwrap();
    let len = rx.await.unwrap();
    let before = fs::read(&wal).unwrap();
    set_file_size_limit(len);

    let res = db.reward(&iid, 1.0).await;
    set_file_size_limit(libc::RLIM_INFINITY);

    assert!(
        res.is_err(),
        "a reward whose WAL write failed was acknowledged: {res:?}"
    );
    assert!(
        !db.wal_healthy.load(Ordering::SeqCst),
        "writer must report itself unhealthy"
    );
    assert_eq!(
        fs::read(&wal).unwrap(),
        before,
        "a failed write must leave the WAL exactly as it was — no torn or repeated records"
    );

    // Recovery sees a clean log: the prediction is pending, the reward never happened,
    // and the caller's retry of the reward now succeeds.
    drop(db);
    let db2 = BanditDB::new(&wal, dir);
    assert_eq!(reward_count(&db2), 0);
    db2.reward(&iid, 1.0).await.unwrap();
    assert_eq!(reward_count(&db2), 1);
    let _ = fs::remove_dir_all(dir);
}
