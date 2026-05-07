#![cfg(feature = "neural")]

use banditdb::BanditDB;
use banditdb::state::{Algorithm, NeuralLinUCBConfig, ProgressiveConfig};
use ndarray::Array1;
use std::fs;
use std::sync::atomic::Ordering;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn linear_reward(arm: &str, ctx: &[f64]) -> f64 {
    let best = if ctx[0] + ctx[1] > 1.0 { "A" } else { "B" };
    if arm == best { 1.0 } else { 0.0 }
}

#[allow(dead_code)]
fn xor_reward(arm: &str, ctx: &[f64]) -> f64 {
    let a = ctx[0] > 0.5;
    let b = ctx[1] > 0.5;
    let correct = if a ^ b { "B" } else { "A" };
    if arm == correct { 1.0 } else { 0.0 }
}

/// Small fast NeuralLinUCB challenger config for tests.
fn neural_cfg(retrain_every: usize) -> NeuralLinUCBConfig {
    NeuralLinUCBConfig {
        context_dim:   2,
        embed_dim:     8,
        hidden_dim:    32,
        hidden_layers: 2,
        retrain_every,
        retrain_steps: 50,
        learning_rate: 1e-3,
        lambda:        1.0,
    }
}

fn progressive_algo(min_obs: usize, required_wins: usize, step_bps: u32, retrain_every: usize) -> Algorithm {
    Algorithm::Progressive(ProgressiveConfig {
        base:          Box::new(Algorithm::Linucb),
        challenger:    Box::new(Algorithm::NeuralLinUCB(neural_cfg(retrain_every))),
        min_obs,
        required_wins,
        step_bps,
    })
}

// ---------------------------------------------------------------------------
// Test 1 — No spurious promotion on a problem LinUCB can already solve.
//
// On a linear reward function the base LinUCB converges quickly. The challenger
// starts with random MLP weights and only gets better after retraining. Because
// neither model consistently beats the other by >10%, challenger_traffic_bps
// must never reach the fully-promoted threshold of 9000 bp (90 %).
// ---------------------------------------------------------------------------
#[tokio::test]
#[cfg(feature = "neural")]
async fn test_no_spurious_promotion_on_linear() {
    let data_dir = "/tmp/banditdb_tourney_linear";
    let wal      = format!("{}/wal.jsonl", data_dir);
    let _ = fs::remove_dir_all(data_dir);
    fs::create_dir_all(data_dir).unwrap();

    let db = BanditDB::new(&wal, data_dir);
    db.add_campaign(
        "linear",
        vec!["A".to_string(), "B".to_string()],
        2,   // base arm matrices are 2-dimensional
        0.1,
        progressive_algo(10, 3, 1000, 20),
        None,
    ).unwrap();

    // Balanced contexts that span the decision boundary clearly
    let contexts: &[Vec<f64>] = &[
        vec![0.9, 0.9], // A
        vec![0.1, 0.1], // B
        vec![0.8, 0.3], // A
        vec![0.2, 0.7], // A
        vec![0.3, 0.2], // B
    ];

    for i in 1..=200 {
        for ctx in contexts {
            if let Ok((arm, iid)) = db.predict("linear", ctx.clone()) {
                let _ = db.reward(&iid, linear_reward(&arm, ctx));
            }
        }
        if i % 20 == 0 { let _ = db.checkpoint().await; }
    }

    let bps = db.campaigns.read().get("linear").unwrap()
        .challenger_traffic_bps.load(Ordering::Relaxed);

    assert!(
        bps < 9000,
        "Neural was fully promoted on a linear problem (traffic = {}bp). \
         Base LinUCB should hold its ground on a problem it can solve.",
        bps
    );

    let _ = fs::remove_dir_all(data_dir);
}

// ---------------------------------------------------------------------------
// Test 2 — Tournament state survives checkpoint + restart.
//
// challenger_traffic_bps and tournament_wins must both be bit-exact after a
// full checkpoint/drop/recover cycle. Arm matrices (base + challenger) must
// also survive — verified via a successful post-recovery prediction.
// ---------------------------------------------------------------------------
#[tokio::test]
#[cfg(feature = "neural")]
async fn test_checkpoint_recovery_preserves_tournament_state() {
    let data_dir = "/tmp/banditdb_tourney_recovery";
    let wal      = format!("{}/wal.jsonl", data_dir);
    let _ = fs::remove_dir_all(data_dir);
    fs::create_dir_all(data_dir).unwrap();

    let xor_data: &[(Vec<f64>, &str)] = &[
        (vec![0.0, 0.0], "A"),
        (vec![0.0, 1.0], "B"),
        (vec![1.0, 0.0], "B"),
        (vec![1.0, 1.0], "A"),
    ];

    // Phase 1: train until some tournament state has accumulated
    let (pre_bps, pre_wins) = {
        let db = BanditDB::new(&wal, data_dir);
        db.add_campaign(
            "camp",
            vec!["A".to_string(), "B".to_string()],
            2,
            0.1,
            progressive_algo(5, 1, 1000, 20), // required_wins=1 for fast progress
            None,
        ).unwrap();

        for i in 1..=100 {
            for (ctx, target) in xor_data {
                if let Ok((arm, iid)) = db.predict("camp", ctx.clone()) {
                    let _ = db.reward(&iid, if &arm == target { 1.0 } else { 0.0 });
                }
            }
            if i % 25 == 0 { let _ = db.checkpoint().await; }
        }
        db.checkpoint().await.unwrap();

        let campaigns = db.campaigns.read();
        let c = campaigns.get("camp").unwrap();
        (
            c.challenger_traffic_bps.load(Ordering::Relaxed),
            c.tournament_wins.load(Ordering::Relaxed),
        )
    };

    // Phase 2: recover and verify state is identical
    let db2      = BanditDB::new(&wal, data_dir);
    let campaigns = db2.campaigns.read();
    let c         = campaigns.get("camp").expect("campaign must survive recovery");

    assert_eq!(
        c.challenger_traffic_bps.load(Ordering::Relaxed), pre_bps,
        "challenger_traffic_bps mismatch after restart: got {} expected {}",
        c.challenger_traffic_bps.load(Ordering::Relaxed), pre_bps
    );
    assert_eq!(
        c.tournament_wins.load(Ordering::Relaxed), pre_wins,
        "tournament_wins mismatch after restart: got {} expected {}",
        c.tournament_wins.load(Ordering::Relaxed), pre_wins
    );

    drop(campaigns);
    assert!(db2.predict("camp", vec![0.0, 0.0]).is_ok(),
        "predictions must work post-recovery");

    let _ = fs::remove_dir_all(data_dir);
}

// ---------------------------------------------------------------------------
// Test 3 — Traffic ramps in single steps of step_bps — never jumps.
//
// With required_wins=1 each winning checkpoint earns exactly one step_bps
// increment. We sample challenger_traffic_bps after every checkpoint and assert
// the delta never exceeds step_bps. We also assert the hard 90 % cap holds and
// that neural earns at least some extra traffic on the XOR problem.
// ---------------------------------------------------------------------------
#[tokio::test]
#[cfg(feature = "neural")]
async fn test_gradual_traffic_ramp() {
    let data_dir = "/tmp/banditdb_tourney_ramp";
    let wal      = format!("{}/wal.jsonl", data_dir);
    let _ = fs::remove_dir_all(data_dir);
    fs::create_dir_all(data_dir).unwrap();

    const STEP: u32 = 1000;

    let db = BanditDB::new(&wal, data_dir);
    db.add_campaign(
        "xor",
        vec!["A".to_string(), "B".to_string()],
        2,
        0.1,
        progressive_algo(5, 1, STEP, 20), // required_wins=1 for fastest possible ramp
        None,
    ).unwrap();

    let xor_data: &[(Vec<f64>, &str)] = &[
        (vec![0.0, 0.0], "A"),
        (vec![0.0, 1.0], "B"),
        (vec![1.0, 0.0], "B"),
        (vec![1.0, 1.0], "A"),
    ];

    let mut prev_bps: u32 = 1000;

    for i in 1..=500 {
        for (ctx, target) in xor_data {
            if let Ok((arm, iid)) = db.predict("xor", ctx.clone()) {
                let _ = db.reward(&iid, if &arm == target { 1.0 } else { 0.0 });
            }
        }

        if i % 25 == 0 {
            let _ = db.checkpoint().await;
            let bps = db.campaigns.read().get("xor").unwrap()
                .challenger_traffic_bps.load(Ordering::Relaxed);

            // Traffic may increase by at most one step or decrease by at most one step
            // per checkpoint (required_wins=1 means one win = one step_bps change).
            let delta = if bps > prev_bps { bps - prev_bps } else { prev_bps - bps };
            assert!(
                delta <= STEP,
                "Traffic changed by more than step_bps in one checkpoint: {} → {} (Δ={}, step={})",
                prev_bps, bps, delta, STEP
            );
            assert!(bps <= 9000, "Traffic exceeded 9000bp hard cap: {}", bps);

            prev_bps = bps;
        }
    }

    let final_bps = db.campaigns.read().get("xor").unwrap()
        .challenger_traffic_bps.load(Ordering::Relaxed);
    assert!(
        final_bps > 1000,
        "Neural never earned any extra traffic on XOR after 500 iterations ({}bp). \
         The tournament is not promoting.",
        final_bps
    );

    let _ = fs::remove_dir_all(data_dir);
}

// ---------------------------------------------------------------------------
// Test 4 — Mean reward does not degrade across the transition.
//
// Splits 600 iterations into an early window and a late window. Post-ramp mean
// reward must not drop more than 10 pp below the pre-ramp baseline, and it
// should be measurably higher — neural solves XOR, linear cannot.
// ---------------------------------------------------------------------------
#[tokio::test]
#[cfg(feature = "neural")]
async fn test_reward_continuity_across_transition() {
    let data_dir = "/tmp/banditdb_tourney_continuity";
    let wal      = format!("{}/wal.jsonl", data_dir);
    let _ = fs::remove_dir_all(data_dir);
    fs::create_dir_all(data_dir).unwrap();

    let db = BanditDB::new(&wal, data_dir);
    db.add_campaign(
        "xor",
        vec!["A".to_string(), "B".to_string()],
        2,
        0.1,
        progressive_algo(5, 1, 1000, 20),
        None,
    ).unwrap();

    let xor_data: &[(Vec<f64>, &str)] = &[
        (vec![0.0, 0.0], "A"),
        (vec![0.0, 1.0], "B"),
        (vec![1.0, 0.0], "B"),
        (vec![1.0, 1.0], "A"),
    ];

    const TOTAL: usize = 600;
    const WINDOW: usize = 100;
    let mut per_iter_reward: Vec<f64> = Vec::with_capacity(TOTAL);

    for i in 1..=TOTAL {
        let mut sum = 0.0f64;
        let mut n   = 0usize;
        for (ctx, target) in xor_data {
            if let Ok((arm, iid)) = db.predict("xor", ctx.clone()) {
                let r = if &arm == target { 1.0 } else { 0.0 };
                let _ = db.reward(&iid, r);
                sum += r;
                n   += 1;
            }
        }
        if n > 0 { per_iter_reward.push(sum / n as f64); }

        if i % 25 == 0 { let _ = db.checkpoint().await; }
    }

    let pre_mean:  f64 = per_iter_reward[..WINDOW].iter().sum::<f64>() / WINDOW as f64;
    let post_mean: f64 = per_iter_reward[TOTAL - WINDOW..].iter().sum::<f64>() / WINDOW as f64;

    assert!(
        post_mean >= pre_mean - 0.10,
        "Reward degraded across tournament transition: pre={:.3} post={:.3} (drop > 10pp). \
         The transition introduced a performance cliff.",
        pre_mean, post_mean
    );
    assert!(
        post_mean > pre_mean,
        "Reward did not improve after tournament promotion: pre={:.3} post={:.3}. \
         Neural should outperform linear on XOR after sufficient training.",
        pre_mean, post_mean
    );

    let _ = fs::remove_dir_all(data_dir);
}

// ---------------------------------------------------------------------------
// Test 5 — Demotion fires when challenger degrades relative to base.
//
// Shadow learning makes both models equally adaptive to natural concept drift,
// so we simulate structural degradation by inverting the challenger arm thetas
// after training. This mirrors a neural model that over-fit a prior distribution
// and is now anti-correlated with the current reward function. The base LinUCB
// is unaffected (online Sherman-Morrison updates per reward).
//
// required_wins=1: one checkpoint where base wins SNIPS triggers a single
// demotion step. We assert challenger_traffic_bps drops below the seeded 7000.
// ---------------------------------------------------------------------------
#[tokio::test]
#[cfg(feature = "neural")]
async fn test_rollback_on_challenger_degradation() {
    let data_dir = "/tmp/banditdb_tourney_rollback";
    let wal      = format!("{}/wal.jsonl", data_dir);
    let _ = fs::remove_dir_all(data_dir);
    fs::create_dir_all(data_dir).unwrap();

    let db = BanditDB::new(&wal, data_dir);
    db.add_campaign(
        "rollback",
        vec!["A".to_string(), "B".to_string()],
        2,
        0.1,
        Algorithm::Progressive(ProgressiveConfig {
            base: Box::new(Algorithm::Linucb),
            challenger: Box::new(Algorithm::NeuralLinUCB(NeuralLinUCBConfig {
                context_dim:   2,
                embed_dim:     8,
                hidden_dim:    32,
                hidden_layers: 2,
                retrain_every: 100_000, 
                retrain_steps: 50,
                learning_rate: 1e-3,
                lambda:        1.0,
            })),
            min_obs:       5,
            required_wins: 1,   
            step_bps:      1000,
        }),
        None,
    ).unwrap();

    // Contexts where A is best and B is best
    let ctx_a = vec![0.9, 0.9];
    let ctx_b = vec![-0.9, -0.9];

    db.campaigns.read().get("rollback").unwrap()
        .challenger_traffic_bps.store(7000, Ordering::Relaxed);

    // Populate buffer with perfectly uniform logged data.
    // For each context, log both arms equally so any policy has full coverage.
    for _ in 0..10 {
        let _ = db.interact("rollback", "A", ctx_a.clone(), 1.0); // Correct
        let _ = db.interact("rollback", "B", ctx_a.clone(), 0.0); // Wrong
        
        let _ = db.interact("rollback", "B", ctx_b.clone(), 1.0); // Correct
        let _ = db.interact("rollback", "A", ctx_b.clone(), 0.0); // Wrong
    }

    // Force base to be perfect (always picks target).
    {
        let campaigns = db.campaigns.read();
        let c = campaigns.get("rollback").unwrap();
        let mut arms = c.arms.write();
        arms.get_mut("A").unwrap().theta = Array1::from_vec(vec![1.0, 1.0]);
        arms.get_mut("B").unwrap().theta = Array1::from_vec(vec![-1.0, -1.0]);
    }
    
    // Force challenger to be anti-perfect. Since we don't know the MLP features,
    // we can't easily set theta. But wait! We can just force the Challenger to
    // pick the WRONG arm by setting its theta dynamically in evaluate_tournament?
    // No, we can just set the base's Algorithm to Neural and Challenger to LinUCB?
    // No, Progressive requires challenger to be NeuralLinUCB in the current tests.
    // BUT we CAN set challenger's theta to force it. Wait, the MLP features are
    // positive. If we set theta_A = [1e6...] and theta_B = [-1e6...], Challenger
    // will ALWAYS pick A.
    // If Challenger always picks A, its SNIPS will average the rewards for A.
    // Rewards for A on ctx_a = 1.0. Rewards for A on ctx_b = 0.0.
    // So its SNIPS will be exactly 0.5!
    // Base's SNIPS will be 1.0 (it picks A on ctx_a and B on ctx_b).
    // 1.0 > 0.5 * 1.10 -> Base wins!
    {
        let campaigns = db.campaigns.read();
        let c = campaigns.get("rollback").unwrap();
        if let Some(c_arms) = &c.challenger_arms {
            let mut arms = c_arms.write();
            arms.get_mut("A").unwrap().theta = Array1::from_elem(8,  1e6_f64);
            arms.get_mut("B").unwrap().theta = Array1::from_elem(8, -1e6_f64);
        }
    }

    db.checkpoint().await.unwrap();

    let final_bps = db.campaigns.read().get("rollback").unwrap()
        .challenger_traffic_bps.load(Ordering::Relaxed);

    assert!(
        final_bps < 7000,
        "Demotion did not fire despite challenger losing SNIPS (Base=1.0, Chal=0.5) \
         (still at {}bp, started at 7000bp, required_wins=1).",
        final_bps
    );

    let _ = fs::remove_dir_all(data_dir);
}
