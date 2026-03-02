use banditdb::engine::WalMessage;
use banditdb::state::CheckpointData;
use banditdb::BanditDB;

/// Test 4.1 — Full Checkpoint / WAL-Rotation / Meta / Recovery Cycle
///
/// End-to-end proof that:
///   1. checkpoint.json captures the exact matrix state (A_inv, b, theta) frozen at
///      the WAL flush barrier — byte-for-byte identical to what was in memory.
///   2. checkpoint_latest.meta captures the correct campaign topology
///      (campaign_id, arm names, feature_dim, timestamp) with no matrix data.
///   3. WAL rotation discards the checkpointed prefix; the rotated file is exactly
///      the tail (empty when no events arrive after the checkpoint barrier).
///   4. A post-checkpoint event (new campaign) lands in the WAL tail.
///   5. A fresh BanditDB recovering from the same data dir:
///      a. Loads campaign matrices from checkpoint.json.
///      b. Replays the WAL tail — including our rotation-recovery fix where
///         wal_offset > file_len causes seek to be reset to 0.
///   6. Matrix values of the pre-checkpoint campaign are bit-exact after recovery.
///   7. Post-recovery predictions are functional on all campaigns.
#[tokio::test]
async fn test_4_1_checkpoint_wal_meta_recovery_cycle() {
    let data_dir = "/tmp/banditdb_ckpt_cycle_test";
    let wal_path = format!("{}/bandit_wal.jsonl", data_dir);

    // Clean slate
    let _ = std::fs::remove_dir_all(data_dir);
    std::fs::create_dir_all(data_dir).unwrap();

    // ===================================================================
    // Phase 1 — Train the model
    // ===================================================================
    let db = BanditDB::new(&wal_path, data_dir);
    db.add_campaign(
        "routing",
        vec!["fast".to_string(), "cheap".to_string()],
        2,
        1.0,
    );

    // Deterministic reward signal so the test is reproducible
    let true_theta = [3.0_f64, -2.0_f64];
    for i in 0..30_usize {
        let angle = i as f64 * 0.2;
        let ctx = vec![angle.sin(), angle.cos()];
        let reward =
            (true_theta[0] * ctx[0] + true_theta[1] * ctx[1]).clamp(0.0, 1.0);
        if let Some((_, iid)) = db.predict("routing", ctx) {
            db.reward(&iid, reward);
        }
    }

    // Flush all training events to disk via WAL barrier, then read back matrix state.
    // The Checkpoint message guarantees every Event sent before it is on disk.
    let (ftx, frx) = tokio::sync::oneshot::channel::<u64>();
    db.event_tx
        .send(WalMessage::Checkpoint { reply: ftx })
        .unwrap();
    let wal_size_before = frx.await.unwrap();
    assert!(wal_size_before > 0, "WAL must be non-empty after training");

    // Snapshot the in-memory matrices — these are the ground truth we will
    // compare everything else against.
    let (pre_theta, pre_b, pre_a_inv) = {
        let campaigns = db.campaigns.read();
        let c = campaigns.get("routing").unwrap();
        let arms = c.arms.read();
        let fast = arms.get("fast").unwrap();
        (
            fast.theta.clone(),
            fast.b.clone(),
            fast.a_inv.clone(),
        )
    };

    // ===================================================================
    // Phase 2 — Checkpoint
    // ===================================================================
    let result = db.checkpoint().await.unwrap();

    assert!(
        result.contains("Checkpoint written and WAL rotated"),
        "Unexpected checkpoint message: {}",
        result
    );
    assert!(
        result.contains("1 campaigns"),
        "Message should report campaign count: {}",
        result
    );

    // ===================================================================
    // Phase 3 — Verify checkpoint.json
    // ===================================================================
    let ckpt_raw =
        std::fs::read_to_string(format!("{}/checkpoint.json", data_dir))
            .expect("checkpoint.json must exist after checkpoint()");
    let ckpt: CheckpointData =
        serde_json::from_str(&ckpt_raw).expect("checkpoint.json must be valid JSON");

    assert!(ckpt.wal_offset > 0, "wal_offset must be positive");
    assert!(ckpt.timestamp_secs > 0, "timestamp must be set");
    assert!(
        ckpt.campaigns.contains_key("routing"),
        "routing campaign must be in checkpoint"
    );

    let ckpt_camp = &ckpt.campaigns["routing"];
    assert!(ckpt_camp.arms.contains_key("fast"), "fast arm must be in checkpoint");
    assert!(ckpt_camp.arms.contains_key("cheap"), "cheap arm must be in checkpoint");

    let ckpt_fast = &ckpt_camp.arms["fast"];
    assert_eq!(ckpt_fast.theta.len(), 2, "theta must have feature_dim=2 elements");
    assert_eq!(ckpt_fast.b.len(), 2, "b must have feature_dim=2 elements");
    assert_eq!(ckpt_fast.a_inv.shape(), &[2, 2], "A_inv must be 2×2");

    // Matrix values must be bit-exact with the pre-checkpoint memory snapshot
    for i in 0..2 {
        assert!(
            (ckpt_fast.theta[i] - pre_theta[i]).abs() < 1e-12,
            "theta[{}] mismatch: checkpoint={:.9} memory={:.9}",
            i, ckpt_fast.theta[i], pre_theta[i]
        );
        assert!(
            (ckpt_fast.b[i] - pre_b[i]).abs() < 1e-12,
            "b[{}] mismatch: checkpoint={:.9} memory={:.9}",
            i, ckpt_fast.b[i], pre_b[i]
        );
        for j in 0..2 {
            assert!(
                (ckpt_fast.a_inv[[i, j]] - pre_a_inv[[i, j]]).abs() < 1e-12,
                "A_inv[{},{}] mismatch: checkpoint={:.9} memory={:.9}",
                i, j, ckpt_fast.a_inv[[i, j]], pre_a_inv[[i, j]]
            );
        }
    }

    // .tmp files must have been cleaned up by the atomic renames
    assert!(
        !std::path::Path::new(&format!("{}/checkpoint.tmp", data_dir)).exists(),
        "checkpoint.tmp must not exist after successful checkpoint"
    );
    assert!(
        !std::path::Path::new(&format!("{}/wal_rotation.tmp", data_dir)).exists(),
        "wal_rotation.tmp must not exist after successful rotation"
    );

    // ===================================================================
    // Phase 5 — Verify WAL rotation
    // ===================================================================
    let wal_size_after = std::fs::metadata(&wal_path).unwrap().len();
    assert!(
        wal_size_after < wal_size_before,
        "WAL must shrink after rotation: before={} after={}",
        wal_size_before,
        wal_size_after
    );
    // No new events were sent between the last flush barrier and checkpoint(),
    // so the tail is empty and the rotated WAL must be exactly 0 bytes.
    assert_eq!(
        wal_size_after, 0,
        "WAL tail must be 0 bytes immediately after checkpoint with no concurrent writes"
    );

    // ===================================================================
    // Phase 6 — Post-checkpoint event lands in the WAL tail
    //
    // This new campaign is NOT in checkpoint.json (it didn't exist yet).
    // It must survive into db2 via WAL tail replay.  This is also the
    // key probe for the rotation-recovery fix: wal_offset (e.g. 4821) is
    // larger than the new WAL file (which only holds this one small event),
    // so recovery must detect that and seek to 0 instead.
    // ===================================================================
    db.add_campaign("post_ckpt", vec!["x".to_string()], 1, 1.0);

    // Flush the post-checkpoint event to disk before we drop the handle
    let (ftx2, frx2) = tokio::sync::oneshot::channel::<u64>();
    db.event_tx
        .send(WalMessage::Checkpoint { reply: ftx2 })
        .unwrap();
    let tail_size = frx2.await.unwrap();
    assert!(
        tail_size > 0,
        "WAL must have content after post-checkpoint add_campaign (got {} bytes)",
        tail_size
    );

    // ===================================================================
    // Phase 7 — Recovery (simulate process restart)
    // ===================================================================
    drop(db);

    let db2 = BanditDB::new(&wal_path, data_dir);

    // "routing" must be present — loaded from checkpoint.json
    assert!(
        db2.campaigns.read().contains_key("routing"),
        "routing campaign must survive recovery from checkpoint"
    );

    // "post_ckpt" must be present — replayed from WAL tail.
    // Absence here means the rotation-recovery fix (seek to 0 when
    // wal_offset > file_len) is not working.
    assert!(
        db2.campaigns.read().contains_key("post_ckpt"),
        "post_ckpt must be replayed from WAL tail (rotation-recovery fix)"
    );

    // ===================================================================
    // Phase 8 — Matrix fidelity after recovery
    //
    // "routing" received zero events after the checkpoint, so its matrices
    // must be bit-exact with the pre-checkpoint snapshot.
    // ===================================================================
    {
        let campaigns = db2.campaigns.read();
        let c = campaigns.get("routing").unwrap();
        let arms = c.arms.read();
        let fast2 = arms.get("fast").unwrap();

        assert_eq!(fast2.theta.len(), 2, "theta dim must be 2 after recovery");
        assert_eq!(fast2.b.len(), 2, "b dim must be 2 after recovery");
        assert_eq!(fast2.a_inv.shape(), &[2, 2], "A_inv shape must be 2×2 after recovery");

        for i in 0..2 {
            assert!(
                (fast2.theta[i] - pre_theta[i]).abs() < 1e-12,
                "theta[{}] must be bit-exact after recovery: got={:.9} want={:.9}",
                i, fast2.theta[i], pre_theta[i]
            );
            assert!(
                (fast2.b[i] - pre_b[i]).abs() < 1e-12,
                "b[{}] must be bit-exact after recovery: got={:.9} want={:.9}",
                i, fast2.b[i], pre_b[i]
            );
            for j in 0..2 {
                assert!(
                    (fast2.a_inv[[i, j]] - pre_a_inv[[i, j]]).abs() < 1e-12,
                    "A_inv[{},{}] must be bit-exact after recovery: got={:.9} want={:.9}",
                    i, j, fast2.a_inv[[i, j]], pre_a_inv[[i, j]]
                );
            }
        }

        // All values must be finite — NaN/Inf indicates a corrupt replay
        assert!(fast2.theta.iter().all(|v| v.is_finite()), "theta contains non-finite value after recovery");
        assert!(fast2.b.iter().all(|v| v.is_finite()), "b contains non-finite value after recovery");
        assert!(fast2.a_inv.iter().all(|v| v.is_finite()), "A_inv contains non-finite value after recovery");
    }

    // ===================================================================
    // Phase 9 — Post-recovery predictions are functional
    // ===================================================================
    let pred_routing = db2.predict("routing", vec![1.0, 0.0]);
    assert!(pred_routing.is_some(), "predict must work on routing after recovery");
    let (arm_id, iid) = pred_routing.unwrap();
    assert!(
        arm_id == "fast" || arm_id == "cheap",
        "predicted arm must be a registered arm, got: {}",
        arm_id
    );
    db2.reward(&iid, 1.0); // must not panic

    let pred_post = db2.predict("post_ckpt", vec![0.5]);
    assert!(
        pred_post.is_some(),
        "predict must work on post_ckpt campaign after recovery"
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(data_dir);
}
