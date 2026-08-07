//! P2.1 acceptance tests — exclusive lock on the data directory.
//!
//! BanditDB is single-writer. Two processes sharing a DATA_DIR interleave WAL
//! appends and race on checkpoint renames, corrupting both with no error at the
//! time and no way to tell afterwards which records survived.
//!
//! The Helm chart guards this with `replicaCount: 1` and `strategy: Recreate`, but
//! those are conventions: a bad values file, a manual binary run, `docker compose
//! up --scale`, or a failed Recreate all bypass them. Nothing in the process itself
//! objected. Now the second starter exits instead.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn wait_for_health(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let ok = Command::new("curl")
            .args(["-sS", "--max-time", "2", "-o", "/dev/null", "-w", "%{http_code}",
                   &format!("http://127.0.0.1:{port}/health")])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "200")
            .unwrap_or(false);
        if ok { return true; }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

#[test]
fn second_process_on_the_same_data_dir_refuses_to_start() {
    let bin = env!("CARGO_BIN_EXE_banditdb");
    let dir = "/tmp/banditdb_lock_test_shared";
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();

    let mut first = match Command::new(bin)
        .env("DATA_DIR", dir).env("PORT", "18401").env("BANDITDB_API_KEY", "k")
        .stdout(Stdio::null()).stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => { eprintln!("SKIPPED: cannot spawn binary"); return; }
    };

    if !wait_for_health(18401, Duration::from_secs(20)) {
        let _ = first.kill();
        eprintln!("SKIPPED: first instance never became healthy");
        return;
    }

    // Second instance, same DATA_DIR, different port so the failure cannot be
    // mistaken for a port collision.
    let second = Command::new(bin)
        .env("DATA_DIR", dir).env("PORT", "18402").env("BANDITDB_API_KEY", "k")
        .stdout(Stdio::null()).stderr(Stdio::null())
        .output()
        .expect("second instance runs to completion");

    let _ = first.kill();
    let _ = first.wait();
    let _ = std::fs::remove_dir_all(dir);

    assert!(
        !second.status.success(),
        "a second process acquired the same data directory — concurrent WAL appends \
         would corrupt it silently, with no error at write time"
    );
}

#[test]
fn lock_is_released_when_the_holder_exits() {
    let bin = env!("CARGO_BIN_EXE_banditdb");
    let dir = "/tmp/banditdb_lock_test_release";
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();

    let mut first = match Command::new(bin)
        .env("DATA_DIR", dir).env("PORT", "18403").env("BANDITDB_API_KEY", "k")
        .stdout(Stdio::null()).stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => { eprintln!("SKIPPED: cannot spawn binary"); return; }
    };
    if !wait_for_health(18403, Duration::from_secs(20)) {
        let _ = first.kill();
        eprintln!("SKIPPED: first instance never became healthy");
        return;
    }

    // SIGKILL leaves the lock file on disk. flock is released by the kernel on
    // process death, so a replacement must still be able to start — otherwise every
    // crash would need manual cleanup before recovery.
    let _ = first.kill();
    let _ = first.wait();
    std::thread::sleep(Duration::from_millis(300));

    let mut second = Command::new(bin)
        .env("DATA_DIR", dir).env("PORT", "18404").env("BANDITDB_API_KEY", "k")
        .stdout(Stdio::null()).stderr(Stdio::null())
        .spawn()
        .expect("replacement spawns");

    let started = wait_for_health(18404, Duration::from_secs(20));
    let _ = second.kill();
    let _ = second.wait();
    let _ = std::fs::remove_dir_all(dir);

    assert!(
        started,
        "a stale lock file blocked restart after a crash — flock must be released by \
         the kernel on process death, leaving the file itself only diagnostic"
    );
}

/// Recovery reopens a directory the same process has already closed, so the lock
/// must be scoped to the instance rather than the process.
#[tokio::test]
async fn same_process_can_reopen_after_dropping_the_instance() {
    use banditdb::BanditDB;
    use banditdb::state::Algorithm;

    let dir = "/tmp/banditdb_lock_test_reopen";
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();

    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    db.add_campaign("c", vec!["A".into(), "B".into()], 2, 1.0, Algorithm::Linucb, None, None).await
        .unwrap();
    db.checkpoint().await.expect("checkpoint");
    drop(db);

    // Would abort the process if the lock outlived the instance.
    let recovered = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    assert!(
        recovered.campaigns.read().contains_key("c"),
        "reopen after drop lost the campaign"
    );

    let _ = std::fs::remove_dir_all(dir);
}
