//! `BANDITDB_MAX_CAMPAIGN_BYTES` bounds a campaign however it reaches its size.
//!
//! The limit was checked at creation only, so a campaign created small could grow
//! past it one `add_arm` at a time. The env var is read at startup and is
//! process-wide, so this file holds a single test.

use banditdb::state::{Algorithm, EngineError, WarmStart};
use banditdb::BanditDB;
use std::fs;

#[tokio::test]
async fn add_arm_respects_the_campaign_byte_limit() {
    // One arm at d=256 reserves ~0.5 MB; allow a little under four.
    std::env::set_var("BANDITDB_MAX_CAMPAIGN_BYTES", (2 * 1024 * 1024).to_string());
    let dir = "/tmp/banditdb_arm_memory_limit";
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = BanditDB::new(&format!("{dir}/wal.jsonl"), dir);
    db.add_campaign("c", vec!["a0".into()], 256, 1.0, Algorithm::Linucb, None, None).await.unwrap();

    let mut refused = None;
    for i in 1..20 {
        match db.add_arm("c", &format!("a{i}"), None, &WarmStart::default()).await {
            Ok(()) => {}
            Err(e) => { refused = Some((i, e)); break; }
        }
    }
    let (at, err) = refused.expect("arms were added past BANDITDB_MAX_CAMPAIGN_BYTES");
    assert!(at > 1, "arms that fit must still be accepted");
    assert!(matches!(err, EngineError::LimitExceeded(_)), "expected LimitExceeded, got {err:?}");
    assert_eq!(db.campaigns.read()["c"].arms.read().len(), at, "the refused arm must not exist");
    let _ = fs::remove_dir_all(dir);
}
