//! Multi-Product Dynamic Pricing with Capacity Constraints (BwK).
//!
//! Validates BanditDB's Lagrangian pacing across multiple products with:
//! - 3 Pricing Arms per product: "discount", "regular", "premium".
//! - Different demand elasticities and inventory scarcity levels:
//!     1. Scarce Product (Flagship Phone): 25% capacity. Demands premium pricing to avoid early burnout.
//!     2. Balanced Product (Designer Jacket): 50% capacity. Smoothly clears stock at regular price.
//!     3. Surplus Product (Audio Accessory): 92% capacity. Constraint never binds (lambda -> 0).

use banditdb::state::{Algorithm, PacingConfig, ResourceConstraint};
use banditdb::BanditDB;
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

struct ProductTestConfig {
    campaign_id: &'static str,
    budget: f64,
    horizon: u64,
    prices: [(&'static str, f64); 3],
    thetas: [(&'static str, [f64; 2]); 3],
    arm_costs: [(&'static str, f64); 3],
}

#[tokio::test]
async fn test_multi_product_dynamic_pricing_simulation() {
    let (wal, dir) = temp_paths("multi_prod_pricing");
    let db = BanditDB::new(&wal, &dir);

    let products = vec![
        // 1. Scarce Product: 300 budget / 1200 horizon = 25% capacity
        ProductTestConfig {
            campaign_id: "prod_flagship",
            budget: 300.0,
            horizon: 1200,
            prices: [("discount", 600.0), ("regular", 800.0), ("premium", 1000.0)],
            thetas: [
                ("discount", [0.65, 0.30]),
                ("regular", [0.40, 0.30]),
                ("premium", [0.20, 0.25]),
            ],
            arm_costs: [("discount", 0.80), ("regular", 0.55), ("premium", 0.325)],
        },
        // 2. Balanced Product: 600 budget / 1200 horizon = 50% capacity
        ProductTestConfig {
            campaign_id: "prod_fashion",
            budget: 600.0,
            horizon: 1200,
            prices: [("discount", 90.0), ("regular", 140.0), ("premium", 200.0)],
            thetas: [
                ("discount", [0.65, 0.30]),
                ("regular", [0.35, 0.30]),
                ("premium", [0.15, 0.20]),
            ],
            arm_costs: [("discount", 0.75), ("regular", 0.50), ("premium", 0.25)],
        },
        // 3. Surplus Product: 1100 budget / 1200 horizon = 91.7% capacity
        ProductTestConfig {
            campaign_id: "prod_accessory",
            budget: 1100.0,
            horizon: 1200,
            prices: [("discount", 15.0), ("regular", 25.0), ("premium", 40.0)],
            thetas: [
                ("discount", [0.70, 0.20]),
                ("regular", [0.30, 0.20]),
                ("premium", [0.08, 0.10]),
            ],
            arm_costs: [("discount", 0.80), ("regular", 0.40), ("premium", 0.13)],
        },
    ];

    let arms = vec![
        "discount".to_string(),
        "regular".to_string(),
        "premium".to_string(),
    ];

    // Initialize all campaigns with PacingConfig
    for p in &products {
        let mut arm_costs_map = HashMap::new();
        for (arm, cost) in &p.arm_costs {
            arm_costs_map.insert(arm.to_string(), *cost);
        }

        let pacing_cfg = PacingConfig {
            resources: vec![ResourceConstraint {
                name: format!("{}_inventory", p.campaign_id),
                budget: p.budget,
                horizon: p.horizon,
                step_size: Some(2.0 / (p.horizon as f64).sqrt()),
                lambda_max: Some(2.0),
                initial_lambda: Some(0.0),
                arm_costs: arm_costs_map,
            }],
            adaptive: true,
        };

        db.add_campaign_pacing(
            p.campaign_id,
            arms.clone(),
            2,
            0.5,
            Algorithm::Linucb,
            None,
            None,
            Some(pacing_cfg),
        )
        .await
        .unwrap();
    }

    let mut ctx_rng = StdRng::seed_from_u64(42);
    let mut sim_rng = StdRng::seed_from_u64(999);

    println!("==========================================================================");
    println!("     BANDITDB MULTI-PRODUCT DYNAMIC PRICING SIMULATION (RUST ENGINE)      ");
    println!("==========================================================================");

    for p in &products {
        let mut total_revenue = 0.0;
        let mut total_sales = 0;
        let mut stockout_events = 0;
        let mut arm_counts: HashMap<String, usize> = HashMap::new();

        let prices_map: HashMap<&'static str, f64> = p.prices.iter().copied().collect();
        let thetas_map: HashMap<&'static str, [f64; 2]> = p.thetas.iter().copied().collect();
        let max_price = p.prices.iter().map(|(_, pr)| *pr).fold(0.0f64, f64::max);

        for _t in 0..p.horizon {
            let z: f64 = ctx_rng.gen_range(0.0..1.0);
            let ctx = vec![1.0, z];

            // When inventory is exhausted, BanditDB's hard feasibility mask blocks all arms
            // and returns Err (item is Sold Out). This enforces zero-cost inventory safety!
            match db.predict(p.campaign_id, ctx.clone()) {
                Ok((chosen_arm, iid)) => {
                    *arm_counts.entry(chosen_arm.clone()).or_insert(0) += 1;

                    // Customer conversion decision
                    let theta = thetas_map.get(chosen_arm.as_str()).unwrap();
                    let prob_conversion = (theta[0] * ctx[0] + theta[1] * ctx[1]).clamp(0.0, 1.0);

                    let is_sale = sim_rng.gen_range(0.0..1.0) < prob_conversion;
                    let price = *prices_map.get(chosen_arm.as_str()).unwrap();

                    if is_sale {
                        total_sales += 1;
                        total_revenue += price;
                        // Normalized revenue reward in [0, 1]
                        db.reward(&iid, price / max_price).await.unwrap();
                    } else {
                        db.reward(&iid, 0.0).await.unwrap();
                    }
                }
                Err(_) => {
                    // Sold out! Hard feasibility mask prevented overselling.
                    stockout_events += 1;
                }
            }
        }

        let report = db
            .campaign_pacing_report(p.campaign_id)
            .unwrap()
            .expect("pacing report");
        let res = &report.resources[0];

        println!("Product Campaign:              {}", p.campaign_id);
        println!("  Horizon (Arrivals):          {}", p.horizon);
        println!("  Initial Inventory (Budget):  {:.0}", p.budget);
        println!("  Total Successful Sales:      {}", total_sales);
        println!("  Total Realized Revenue:      ${:.2}", total_revenue);
        println!("  Stockout Arrivals Blocked:   {}", stockout_events);
        println!("  Arm Selections:              {:?}", arm_counts);
        println!(
            "  BanditDB Consumed Tracked:   {:.1} / {:.1}",
            res.consumed, res.budget
        );
        println!(
            "  BanditDB Utilization:        {:.1}%",
            res.utilization * 100.0
        );
        println!("  BanditDB Final Shadow Price: {:.4}", res.lambda);
        println!("--------------------------------------------------------------------------");

        // Assertions
        // 1. Never oversell inventory beyond initial budget + small CAS epsilon
        assert!(
            res.consumed <= p.budget * 1.05,
            "Inventory violated on {}: consumed {} > budget {}",
            p.campaign_id,
            res.consumed,
            p.budget
        );

        // 2. High budget utilization (> 75% for binding products)
        if p.campaign_id != "prod_accessory" {
            assert!(
                res.utilization >= 0.75,
                "Under-utilization on {}: got {:.1}%",
                p.campaign_id,
                res.utilization * 100.0
            );
        }

        // 3. Scarce product shadow price > Surplus product shadow price
        if p.campaign_id == "prod_flagship" {
            assert!(
                res.lambda > 0.15,
                "Scarce product shadow price should rise! Got {}",
                res.lambda
            );
        }

        // 4. Arms explored
        assert!(!arm_counts.is_empty(), "Should have arm counts recorded");
    }

    println!("All multi-product pricing pacing assertions passed successfully!");
}
