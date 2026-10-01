//! Ground Truth Constrained Bandit Simulation Test.
//!
//! Validates Contextual Bandits with Knapsacks (BwK) using BanditDB's Lagrangian pacing.
//!
//! Scenario:
//! - 2 Arms: "standard" (cost 0.0) and "premium" (cost 1.0).
//! - True parameter vectors:
//!   theta_standard = [0.20, 0.10]
//!   theta_premium  = [0.05, 0.95]
//! - Context: x = [1.0, z], where z ~ Uniform(0, 1).
//! - Uplift: Delta(z) = mu_premium(z) - mu_standard(z) = -0.15 + 0.85 * z.
//! - Horizon: T = 3,000 requests.
//! - Budget: B = 900.0 (alpha = 30% of horizon).
//! - Theoretical optimal policy:
//!   Allocate "premium" to the top 30% uplift contexts (z >= 0.70).
//!   Theoretical optimal shadow price: lambda* = Delta(0.70) = 0.445.
//!   Theoretical expected optimal reward per step: 0.42175.
//!   Theoretical fluid LP cumulative reward: OPT = 1,265.25.
//!
//! Compares:
//! 1. Ground Truth Fluid LP Optimal Policy (Oracle with lambda* = 0.445).
//! 2. BanditDB Paced Policy (Lagrangian Dual Descent + LinUCB + Hard Feasibility Mask).
//! 3. Unconstrained Greedy Bandit (FCFS - burns budget early and collapses).
//!
//! Audit fixes applied:
//! - Separate RNG streams so each policy faces identical contexts.
//! - FCFS reward feeds the arm BanditDB recorded, not the overridden arm.
//! - Regret assertion uses absolute magnitude, handles negative regret gracefully.
//! - Comment/value alignment on lambda convergence tolerance.

use banditdb::BanditDB;
use banditdb::state::{Algorithm, PacingConfig, ResourceConstraint};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

fn temp_paths(test_name: &str) -> (String, String) {
    let wal = format!("/tmp/banditdb_sim_{test_name}.jsonl");
    let dir = format!("/tmp/bdb_sim_{test_name}");
    let _ = std::fs::remove_file(&wal);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::create_dir_all(&dir);
    (wal, dir)
}

#[tokio::test]
async fn test_constrained_bandit_ground_truth_simulation() {
    let (wal, dir) = temp_paths("constrained_gt");
    let db = BanditDB::new(&wal, &dir);

    let horizon: u64 = 3_000;
    let budget: f64 = 900.0; // 30% of 3000

    // Ground truth parameters
    let theta_std = [0.20, 0.10];
    let theta_prem = [0.05, 0.95];

    // Theoretical optimal shadow price: lambda* = Delta(0.70) = -0.15 + 0.85 * 0.70 = 0.445
    let optimal_lambda = 0.445;

    // 1. Setup BanditDB campaign with Lagrangian pacing
    let mut arm_costs = HashMap::new();
    arm_costs.insert("standard".to_string(), 0.0);
    arm_costs.insert("premium".to_string(), 1.0);

    let pacing_cfg = PacingConfig {
        resources: vec![ResourceConstraint {
            name: "inventory".to_string(),
            budget,
            horizon,
            step_size: Some(0.04),
            lambda_max: Some(2.0),
            initial_lambda: Some(0.0),
            arm_costs: arm_costs.clone(),
        }],
        adaptive: true,
    };

    db.add_campaign_pacing(
        "sim_pacing",
        vec!["standard".to_string(), "premium".to_string()],
        2,
        0.5,
        Algorithm::Linucb,
        None,
        None,
        Some(pacing_cfg),
    ).await.unwrap();

    // 2. Setup Unconstrained FCFS campaign (no pacing, manual budget tracking)
    db.add_campaign(
        "sim_unconstrained",
        vec!["standard".to_string(), "premium".to_string()],
        2,
        0.5,
        Algorithm::Linucb,
        None,
        None,
    ).await.unwrap();

    // FIX(Bug 1): Use separate RNG streams so every policy faces the same context sequence.
    // A dedicated context RNG produces z_t, and each policy gets its own noise RNG.
    let mut ctx_rng = StdRng::seed_from_u64(42);
    let mut opt_noise_rng = StdRng::seed_from_u64(100);
    let mut bdb_noise_rng = StdRng::seed_from_u64(200);
    let mut fcfs_noise_rng = StdRng::seed_from_u64(300);

    let mut opt_cum_reward = 0.0;
    let mut opt_budget_rem = budget;
    let mut opt_prem_count = 0;

    let mut bdb_cum_reward = 0.0;
    let mut bdb_prem_count = 0;

    let mut fcfs_cum_reward = 0.0;
    let mut fcfs_budget_rem = budget;
    let mut fcfs_prem_count = 0;

    for _t in 0..horizon {
        let z: f64 = ctx_rng.gen_range(0.0..1.0);
        let ctx = vec![1.0, z];

        let mu_std = theta_std[0] * ctx[0] + theta_std[1] * ctx[1];
        let mu_prem = theta_prem[0] * ctx[0] + theta_prem[1] * ctx[1];

        // ── 1. Ground Truth Optimal Policy (Fluid LP) ─────────────────────
        let delta = mu_prem - mu_std;
        let opt_arm = if delta >= optimal_lambda && opt_budget_rem >= 1.0 {
            opt_budget_rem -= 1.0;
            opt_prem_count += 1;
            "premium"
        } else {
            "standard"
        };
        let opt_reward = if opt_arm == "premium" {
            (mu_prem + opt_noise_rng.gen_range(-0.02..0.02)).clamp(0.0, 1.0)
        } else {
            (mu_std + opt_noise_rng.gen_range(-0.02..0.02)).clamp(0.0, 1.0)
        };
        opt_cum_reward += opt_reward;

        // ── 2. BanditDB Paced Policy ──────────────────────────────────────
        let (bdb_arm, bdb_iid) = db.predict("sim_pacing", ctx.clone()).unwrap();
        let bdb_reward = if bdb_arm == "premium" {
            bdb_prem_count += 1;
            (mu_prem + bdb_noise_rng.gen_range(-0.02..0.02)).clamp(0.0, 1.0)
        } else {
            (mu_std + bdb_noise_rng.gen_range(-0.02..0.02)).clamp(0.0, 1.0)
        };
        bdb_cum_reward += bdb_reward;
        db.reward(&bdb_iid, bdb_reward).await.unwrap();

        // ── 3. Unconstrained Greedy FCFS Policy ───────────────────────────
        let (raw_arm, raw_iid) = db.predict("sim_unconstrained", ctx.clone()).unwrap();

        // FIX(Bug 3): When the budget is exhausted and the FCFS policy overrides
        // the arm to "standard", the reward fed back to BanditDB must still correspond
        // to the arm BanditDB recorded (raw_arm), not the overridden arm. Otherwise
        // BanditDB's LinUCB updates the wrong parameter matrix. We compute rewards
        // for both the "observed" arm (what the user actually did) and the "logged" arm
        // (what BanditDB thinks happened).
        let fcfs_budget_allows = raw_arm == "premium" && fcfs_budget_rem >= 1.0;
        let fcfs_arm = if fcfs_budget_allows {
            fcfs_budget_rem -= 1.0;
            fcfs_prem_count += 1;
            "premium"
        } else if raw_arm == "premium" {
            // Budget exhausted: the actual outcome is "standard" ...
            "standard"
        } else {
            "standard"
        };

        // Reward for regret accounting is based on the actually-executed arm.
        let fcfs_reward = if fcfs_arm == "premium" {
            (mu_prem + fcfs_noise_rng.gen_range(-0.02..0.02)).clamp(0.0, 1.0)
        } else {
            (mu_std + fcfs_noise_rng.gen_range(-0.02..0.02)).clamp(0.0, 1.0)
        };
        fcfs_cum_reward += fcfs_reward;

        // Reward fed back to LinUCB must correspond to the arm BanditDB recorded.
        let fcfs_reward_for_engine = if raw_arm == "premium" {
            (mu_prem + fcfs_noise_rng.gen_range(-0.02..0.02)).clamp(0.0, 1.0)
        } else {
            (mu_std + fcfs_noise_rng.gen_range(-0.02..0.02)).clamp(0.0, 1.0)
        };
        db.reward(&raw_iid, fcfs_reward_for_engine).await.unwrap();
    }

    // ── Inspect BanditDB Pacing Report ──────────────────────────────────
    let report = db.campaign_pacing_report("sim_pacing").unwrap().expect("report exists");
    let res = &report.resources[0];

    let bdb_consumed = res.consumed;
    let bdb_remaining = res.remaining;
    let bdb_final_lambda = res.lambda;
    let bdb_utilization = res.utilization;

    println!("============================================================");
    println!("     CONSTRAINED BANDIT GROUND TRUTH SIMULATION RESULTS     ");
    println!("============================================================");
    println!("Horizon (T):                  {horizon}");
    println!("Budget (B):                   {budget:.1}");
    println!("Theoretical lambda*:          {optimal_lambda:.3}");
    println!("BanditDB Final lambda:        {bdb_final_lambda:.3}");
    println!("------------------------------------------------------------");
    println!("Optimal Policy Reward:        {opt_cum_reward:.2} (Prem calls: {opt_prem_count})");
    println!("BanditDB Paced Reward:        {bdb_cum_reward:.2} (Prem calls: {bdb_prem_count})");
    println!("FCFS Greedy Reward:           {fcfs_cum_reward:.2} (Prem calls: {fcfs_prem_count})");
    println!("------------------------------------------------------------");
    println!("BanditDB Budget Consumed:     {bdb_consumed:.1} / {budget:.1}");
    println!("BanditDB Budget Remaining:    {bdb_remaining:.1}");
    println!("BanditDB Utilization:         {:.1}%", bdb_utilization * 100.0);
    println!("------------------------------------------------------------");
    let bdb_regret = opt_cum_reward - bdb_cum_reward;
    let fcfs_regret = opt_cum_reward - fcfs_cum_reward;
    let bdb_regret_pct = if opt_cum_reward > 0.0 { (bdb_regret / opt_cum_reward) * 100.0 } else { 0.0 };
    let fcfs_regret_pct = if opt_cum_reward > 0.0 { (fcfs_regret / opt_cum_reward) * 100.0 } else { 0.0 };
    println!("BanditDB Regret vs Optimal:   {bdb_regret:.2} ({bdb_regret_pct:.2}% gap)");
    println!("FCFS Regret vs Optimal:       {fcfs_regret:.2} ({fcfs_regret_pct:.2}% gap)");
    println!("============================================================");

    // ── Verification Assertions ─────────────────────────────────────────
    // 1. Budget integrity: BanditDB MUST NOT exceed initial budget.
    assert!(
        bdb_consumed <= budget,
        "Budget violation! Consumed {bdb_consumed} > Budget {budget}"
    );

    // 2. High budget utilization: Pacing should smoothly consume most of the budget (> 85%).
    assert!(
        bdb_utilization >= 0.85,
        "Under-utilization! Utilization was {:.1}%, expected >= 85%",
        bdb_utilization * 100.0
    );

    // 3. Significant performance superiority over unconstrained FCFS:
    assert!(
        bdb_cum_reward > fcfs_cum_reward,
        "BanditDB ({bdb_cum_reward}) should beat unconstrained FCFS ({fcfs_cum_reward})"
    );

    // FIX(Bug 5): Use absolute regret magnitude. BanditDB can beat the realized oracle
    // (the oracle also has stochastic noise), so bdb_regret can be negative. We check
    // the absolute gap is within 10%.
    // 4. Low regret against the theoretical Fluid LP upper bound (< 10% absolute regret gap):
    assert!(
        opt_cum_reward == 0.0 || bdb_regret.abs() / opt_cum_reward < 0.10,
        "Regret too high: {:.2}% absolute gap against optimal",
        if opt_cum_reward > 0.0 { (bdb_regret.abs() / opt_cum_reward) * 100.0 } else { 0.0 }
    );

    // FIX(Bug 4): Comment and value now match at 0.15.
    // 5. Dual variable convergence: lambda should converge near optimal lambda* (within 0.15):
    assert!(
        (bdb_final_lambda - optimal_lambda).abs() < 0.15,
        "Lambda did not converge near optimal {optimal_lambda}: got {bdb_final_lambda}"
    );
}
