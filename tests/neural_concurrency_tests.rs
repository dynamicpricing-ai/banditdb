#![cfg(feature = "neural")]
//! P0.1 acceptance tests — prediction must never contend with neural retraining.
//!
//! Background (docs/PRODUCTION_STAGE1.md §2.1): `predict()` runs under `arms.read()`
//! and used to reach the MLP through `neural.lock()`. Retraining takes those in the
//! opposite order — `neural.lock()` then `arms.read()` — which deadlocks as soon as a
//! writer is queued, because parking_lot's task-fair RwLock blocks new readers behind
//! a pending writer:
//!
//!   predict  holds arms.read      waits neural.lock
//!   retrain  holds neural.lock    waits arms.read    (behind the pending writer)
//!   reward   waits arms.write                        (behind predict's read)
//!
//! The fix publishes an immutable `NeuralWeights` snapshot that readers clone, so
//! `predict` never touches the neural mutex at all.

use banditdb::BanditDB;
use banditdb::state::{Algorithm, NeuralLinUCBConfig};
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

const CTX_DIM: usize = 6;
const EMBED_DIM: usize = 8;

fn neural_cfg(retrain_every: usize, retrain_steps: usize) -> NeuralLinUCBConfig {
    NeuralLinUCBConfig {
        context_dim: CTX_DIM,
        embed_dim:   EMBED_DIM,
        hidden_dim:  64,
        hidden_layers: 2,
        retrain_every,
        retrain_steps,
        learning_rate: 1e-3,
        lambda: 1.0,
    }
}

fn setup(dir: &str, retrain_every: usize, retrain_steps: usize) -> Arc<BanditDB> {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let db = Arc::new(BanditDB::new(&format!("{dir}/wal.jsonl"), dir));
    let cfg = neural_cfg(retrain_every, retrain_steps);
    db.add_campaign(
        "c",
        vec!["A".to_string(), "B".to_string()],
        // Neural arm matrices live in embedding space. The HTTP layer substitutes
        // cfg.embed_dim here; the engine API takes the value literally.
        cfg.embed_dim,
        1.0,
        Algorithm::NeuralLinUCB(cfg),
        None,
        None,
    )
    .unwrap();
    db
}

fn ctx(i: usize) -> Vec<f64> {
    (0..CTX_DIM).map(|d| ((i + d) % 10) as f64 / 10.0).collect()
}

/// Deterministic proof of the P0.1 property.
///
/// Holds the neural mutex for the whole duration of a prediction. Before P0.1 the
/// prediction blocked on that mutex and this test would hang; after it, prediction
/// reads the published snapshot and returns immediately.
#[tokio::test]
async fn predict_does_not_block_while_neural_lock_is_held() {
    let dir = "/tmp/banditdb_p01_held_lock";
    let db = setup(dir, 100_000, 5);

    let (tx, rx) = std::sync::mpsc::channel();
    let db_predict = Arc::clone(&db);

    // Grab the neural mutex and keep it for 3 seconds, simulating a long retrain.
    let holder = std::thread::spawn({
        let db = Arc::clone(&db);
        move || {
            let campaigns = db.campaigns.read();
            let campaign = campaigns.get("c").unwrap();
            let _guard = campaign.neural.as_ref().unwrap().lock();
            std::thread::sleep(Duration::from_secs(3));
        }
    });

    // Give the holder time to acquire before predicting.
    std::thread::sleep(Duration::from_millis(200));

    std::thread::spawn(move || {
        let started = Instant::now();
        let result = db_predict.predict("c", ctx(1));
        let _ = tx.send((result.is_ok(), started.elapsed()));
    });

    match rx.recv_timeout(Duration::from_millis(1500)) {
        Ok((ok, elapsed)) => {
            assert!(ok, "predict returned an error while the neural lock was held");
            assert!(
                elapsed < Duration::from_millis(500),
                "predict took {elapsed:?} — it is still contending with the neural mutex"
            );
        }
        Err(_) => panic!(
            "predict did not return within 1.5s while the neural lock was held — \
             the prediction path is still acquiring the neural mutex (P0.1 regression)"
        ),
    }

    holder.join().unwrap();
    let _ = fs::remove_dir_all(dir);
}

/// A published snapshot must be a deep copy: retraining the live network must not
/// mutate weights that readers are already serving from.
#[tokio::test]
async fn snapshot_is_isolated_from_subsequent_training() {
    let dir = "/tmp/banditdb_p01_isolation";
    let db = setup(dir, 20, 50);

    let probe = ndarray::Array1::from_vec(ctx(3));

    let before = {
        let campaigns = db.campaigns.read();
        let campaign = campaigns.get("c").unwrap();
        let held = Arc::clone(&campaign.neural_weights.as_ref().unwrap().read());
        let embedding = held.embed(&probe);
        // Keep `held` alive across the retrain below to prove the Arc we already
        // handed out cannot be mutated underneath us.
        (held, embedding)
    };

    // Drive enough rewards to make a retrain due, then run one.
    for i in 0..60 {
        if let Ok((arm, iid)) = db.predict("c", ctx(i)) {
            let _ = db.reward(&iid, if arm == "A" { 1.0 } else { 0.0 }).await;
        }
    }
    db.checkpoint().await.expect("checkpoint");

    let (held, embedding_before) = before;
    let embedding_after = held.embed(&probe);
    assert_eq!(
        embedding_before.to_vec(), embedding_after.to_vec(),
        "a snapshot handed to a reader changed after retraining — it is not a deep copy"
    );

    // The campaign must now be publishing different weights than that stale snapshot.
    let republished = {
        let campaigns = db.campaigns.read();
        let campaign = campaigns.get("c").unwrap();
        let w = Arc::clone(&campaign.neural_weights.as_ref().unwrap().read());
        w.embed(&probe)
    };
    assert_ne!(
        embedding_before.to_vec(), republished.to_vec(),
        "retraining did not republish new weights to the prediction path"
    );

    let _ = fs::remove_dir_all(dir);
}

/// Full contention scenario: concurrent predicts, rewards, and retrains.
/// Before P0.1 this deadlocks; the watchdog also catches the P99 stall from a
/// retrain holding the mutex across gradient steps.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_predict_reward_retrain_stays_live() {
    let dir = "/tmp/banditdb_p01_stress";
    // Small retrain_every + real step count so retrains fire constantly during the run.
    let db = setup(dir, 25, 40);

    let stop = Arc::new(AtomicBool::new(false));
    let predicts = Arc::new(AtomicU64::new(0));
    let max_latency_us = Arc::new(AtomicU64::new(0));

    // reward() is async now; these workers are plain OS threads outside the
    // runtime, so they drive it through a runtime handle.
    let handle = tokio::runtime::Handle::current();
    let mut workers = Vec::new();
    for t in 0..3 {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        let predicts = Arc::clone(&predicts);
        let max_latency_us = Arc::clone(&max_latency_us);
        let handle = handle.clone();
        workers.push(std::thread::spawn(move || {
            let mut i = t * 1000;
            while !stop.load(Ordering::Relaxed) {
                let started = Instant::now();
                let outcome = db.predict("c", ctx(i));
                let elapsed = started.elapsed().as_micros() as u64;
                max_latency_us.fetch_max(elapsed, Ordering::Relaxed);
                if let Ok((arm, iid)) = outcome {
                    predicts.fetch_add(1, Ordering::Relaxed);
                    let _ = handle.block_on(db.reward(&iid, if arm == "A" { 1.0 } else { 0.0 }));
                }
                i += 1;
            }
        }));
    }

    // Force retrains concurrently with the traffic above.
    for _ in 0..6 {
        tokio::time::sleep(Duration::from_millis(250)).await;
        db.checkpoint().await.expect("checkpoint under concurrent load");
    }

    stop.store(true, Ordering::Relaxed);
    for w in workers {
        w.join().expect("worker thread panicked or deadlocked");
    }

    let total = predicts.load(Ordering::Relaxed);
    let worst = Duration::from_micros(max_latency_us.load(Ordering::Relaxed));
    assert!(total > 100, "only {total} predictions completed — traffic was starved");
    assert!(
        worst < Duration::from_secs(1),
        "worst prediction latency was {worst:?}; retraining is still stalling the hot path"
    );

    let _ = fs::remove_dir_all(dir);
}
