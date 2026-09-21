//! Dynamic arm lifecycle: adding arms to a live campaign, soft exclusion, and
//! warm-start priors.
//!
//! The invariants under test, in the order they matter operationally:
//!   1. A new arm can join a running campaign and is immediately selectable.
//!   2. A paused arm is never selected but keeps learning from in-flight rewards.
//!   3. Per-request filters narrow the candidate set *and* the logged propensities.
//!   4. A warm-started arm starts at the borrowed mean with a cold arm's uncertainty.
//!   5. All of it survives a restart, by WAL replay and by checkpoint.

use banditdb::engine::ArmFilter;
use banditdb::state::{Algorithm, ArmStatus, WarmStart};
use banditdb::BanditDB;
use std::sync::atomic::Ordering;

/// Each test gets its own data directory — `BanditDB::new` takes an exclusive lock
/// on it. Keyed off the WAL filename, which is already unique per test.
fn data_dir_for(wal: &str) -> String {
    let stem = std::path::Path::new(wal)
        .file_stem().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unnamed".to_string());
    let dir = format!("/tmp/bdb_{stem}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn arms(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

/// Train an arm toward a known θ by rewarding it directly through `interact`,
/// which bypasses selection — the point is to give the arm a θ worth borrowing.
async fn train(db: &BanditDB, campaign: &str, arm: &str, ctx: &[f64], reward: f64, times: usize) {
    for _ in 0..times {
        db.interact(campaign, arm, ctx.to_vec(), reward).await.expect("interact must succeed");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Adding arms
// ════════════════════════════════════════════════════════════════════════════

/// A cold arm added to a live campaign is present, active, and selectable.
#[tokio::test]
async fn test_add_arm_is_selectable() {
    let wal = "/tmp/banditdb_test_add_arm.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("catalog", arms(&["a"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    db.add_arm("catalog", "b", None, &WarmStart::None).await.expect("add_arm must succeed");

    {
        let campaigns = db.campaigns.read();
        let campaign  = campaigns.get("catalog").unwrap();
        let guard     = campaign.arms.read();
        let b = guard.get("b").expect("new arm must exist");
        assert_eq!(b.status(), ArmStatus::Active);
        assert_eq!(b.theta.len(), 2, "new arm must match the campaign feature dimension");
    }

    // Only 'b' is eligible for this request, so it has to be the choice.
    let (arm, _) = db.predict_filtered("catalog", vec![1.0, 0.0], &ArmFilter::include(arms(&["b"]))).unwrap();
    assert_eq!(arm, "b", "a newly added arm must be selectable");

    let _ = std::fs::remove_file(wal);
}

/// Duplicates, unknown campaigns, and malformed ids are all rejected.
#[tokio::test]
async fn test_add_arm_rejects_bad_input() {
    let wal = "/tmp/banditdb_test_add_arm_bad.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("c", arms(&["a"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();

    assert!(db.add_arm("c", "a", None, &WarmStart::None).await.is_err(), "duplicate arm must be rejected");
    assert!(db.add_arm("missing", "b", None, &WarmStart::None).await.is_err(), "unknown campaign must be rejected");
    assert!(db.add_arm("c", "bad id!", None, &WarmStart::None).await.is_err(), "malformed arm_id must be rejected");
    assert!(db.add_arm("c", "", None, &WarmStart::None).await.is_err(), "empty arm_id must be rejected");

    // An archived campaign is frozen — arms included.
    db.archive_campaign("c").await.unwrap();
    assert!(db.add_arm("c", "b", None, &WarmStart::None).await.is_err(), "archived campaign must reject new arms");

    let _ = std::fs::remove_file(wal);
}

// ════════════════════════════════════════════════════════════════════════════
// Warm-start priors
// ════════════════════════════════════════════════════════════════════════════

/// `WarmStart::Population` starts the new arm at the mean θ of the active arms,
/// while leaving it exactly as uncertain as a cold arm (strength 1.0 ⇒ A⁻¹ = I).
#[tokio::test]
async fn test_warm_start_population_matches_mean_theta() {
    let wal = "/tmp/banditdb_test_warm_population.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("shop", arms(&["a", "b"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    train(&db, "shop", "a", &[1.0, 0.0], 0.9, 30).await;
    train(&db, "shop", "b", &[1.0, 0.0], 0.1, 30).await;

    let expected: Vec<f64> = {
        let campaigns = db.campaigns.read();
        let guard = campaigns.get("shop").unwrap().arms.read();
        let a = &guard.get("a").unwrap().theta;
        let b = &guard.get("b").unwrap().theta;
        (0..2).map(|i| (a[i] + b[i]) / 2.0).collect()
    };

    db.add_arm("shop", "c", None, &WarmStart::Population { strength: 1.0 }).await.unwrap();

    let campaigns = db.campaigns.read();
    let guard = campaigns.get("shop").unwrap().arms.read();
    let c = guard.get("c").unwrap();

    for (i, want) in expected.iter().enumerate() {
        assert!(
            (c.theta[i] - want).abs() < 1e-9,
            "warm-started θ[{i}] = {} must equal the population mean {want}", c.theta[i]
        );
    }
    // Uncertainty is untouched at strength 1.0: the arm starts at a sensible guess
    // but is still explored like a new arm.
    assert!((c.a_inv[[0, 0]] - 1.0).abs() < 1e-9, "strength 1.0 must leave A_inv = I");
    assert!((c.a_inv[[1, 1]] - 1.0).abs() < 1e-9, "strength 1.0 must leave A_inv = I");

    let _ = std::fs::remove_file(wal);
}

/// A higher strength buys a firmer prior: the same mean, but proportionally less
/// remaining uncertainty (A⁻¹ = I/λ), so the arm explores less.
#[tokio::test]
async fn test_warm_start_strength_scales_uncertainty() {
    let wal = "/tmp/banditdb_test_warm_strength.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("s", arms(&["a"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    train(&db, "s", "a", &[1.0, 0.0], 0.8, 20).await;

    db.add_arm("s", "firm", None, &WarmStart::Population { strength: 10.0 }).await.unwrap();

    let campaigns = db.campaigns.read();
    let guard = campaigns.get("s").unwrap().arms.read();
    let firm  = guard.get("firm").unwrap();
    let a     = guard.get("a").unwrap();

    assert!((firm.a_inv[[0, 0]] - 0.1).abs() < 1e-9, "strength 10 must give A_inv = I/10");
    for i in 0..2 {
        assert!((firm.theta[i] - a.theta[i]).abs() < 1e-9, "the mean must not depend on strength");
    }

    let _ = std::fs::remove_file(wal);
}

/// `WarmStart::Group` borrows only from arms in the same group, and fails loudly
/// when the group has no members to borrow from.
#[tokio::test]
async fn test_warm_start_group_scopes_to_group_members() {
    let wal = "/tmp/banditdb_test_warm_group.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("cat", arms(&["shoe_1"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    // shoe_1 predates groups, so label it by adding the rest of the catalogue around it.
    db.add_arm("cat", "shoe_2", Some("shoes".into()), &WarmStart::None).await.unwrap();
    db.add_arm("cat", "hat_1",  Some("hats".into()),  &WarmStart::None).await.unwrap();

    train(&db, "cat", "shoe_2", &[1.0, 0.0], 0.9, 30).await;
    train(&db, "cat", "hat_1",  &[1.0, 0.0], 0.1, 30).await;

    let shoe_theta: Vec<f64> = {
        let campaigns = db.campaigns.read();
        let guard = campaigns.get("cat").unwrap().arms.read();
        guard.get("shoe_2").unwrap().theta.to_vec()
    };

    db.add_arm("cat", "shoe_3", Some("shoes".into()), &WarmStart::Group { strength: 1.0 })
        .await.expect("group warm start must succeed");

    {
        let campaigns = db.campaigns.read();
        let guard = campaigns.get("cat").unwrap().arms.read();
        let shoe_3 = guard.get("shoe_3").unwrap();
        assert_eq!(shoe_3.group.as_deref(), Some("shoes"));
        for i in 0..2 {
            assert!(
                (shoe_3.theta[i] - shoe_theta[i]).abs() < 1e-9,
                "shoe_3 must borrow from shoes only — got {:?}, expected {:?}", shoe_3.theta, shoe_theta
            );
        }
    }

    // An empty group has nothing to lend, and silently cold-starting would hide that.
    assert!(
        db.add_arm("cat", "bag_1", Some("bags".into()), &WarmStart::Group { strength: 1.0 }).await.is_err(),
        "warm start from an empty group must be rejected"
    );
    // So does a group warm start with no group declared at all.
    assert!(
        db.add_arm("cat", "bag_2", None, &WarmStart::Group { strength: 1.0 }).await.is_err(),
        "group warm start without a group must be rejected"
    );

    let _ = std::fs::remove_file(wal);
}

/// `WarmStart::Arms` borrows from an explicit list — including from an arm that was
/// just paused, which is the "replace this creative with a fresh one" case.
#[tokio::test]
async fn test_warm_start_from_named_arms_including_paused() {
    let wal = "/tmp/banditdb_test_warm_named.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("c", arms(&["old", "other"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    train(&db, "c", "old", &[1.0, 0.0], 0.9, 30).await;

    let old_theta: Vec<f64> = {
        let campaigns = db.campaigns.read();
        let guard = campaigns.get("c").unwrap().arms.read();
        guard.get("old").unwrap().theta.to_vec()
    };

    db.set_arm_status("c", "old", ArmStatus::Retired).await.unwrap();
    db.add_arm("c", "new", None, &WarmStart::Arms { arms: arms(&["old"]), strength: 1.0 })
        .await.expect("explicit warm start must see retired arms");

    {
        let campaigns = db.campaigns.read();
        let guard = campaigns.get("c").unwrap().arms.read();
        let new = guard.get("new").unwrap();
        for (i, want) in old_theta.iter().enumerate() {
            assert!((new.theta[i] - want).abs() < 1e-9, "new arm must inherit the retired arm's θ");
        }
    }

    assert!(
        db.add_arm("c", "x", None, &WarmStart::Arms { arms: arms(&["ghost"]), strength: 1.0 }).await.is_err(),
        "warm start from an unknown arm must be rejected"
    );
    assert!(
        db.add_arm("c", "y", None, &WarmStart::Population { strength: 0.0 }).await.is_err(),
        "strength 0 must be rejected — it would divide by zero building A_inv"
    );

    let _ = std::fs::remove_file(wal);
}

// ════════════════════════════════════════════════════════════════════════════
// Soft exclusion
// ════════════════════════════════════════════════════════════════════════════

/// A paused arm is never selected, and reactivating it restores everything it knew.
#[tokio::test]
async fn test_paused_arm_is_never_selected() {
    let wal = "/tmp/banditdb_test_pause.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("p", arms(&["a", "b"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    db.set_arm_status("p", "b", ArmStatus::Paused).await.unwrap();

    for i in 0..50 {
        let ctx = vec![(i as f64 * 0.1).sin().abs(), (i as f64 * 0.1).cos().abs()];
        let (arm, _) = db.predict("p", ctx).unwrap();
        assert_eq!(arm, "a", "a paused arm must never be selected");
    }

    let theta_while_paused: Vec<f64> = {
        let campaigns = db.campaigns.read();
        let guard = campaigns.get("p").unwrap().arms.read();
        guard.get("b").unwrap().theta.to_vec()
    };

    db.set_arm_status("p", "b", ArmStatus::Active).await.unwrap();
    let campaigns = db.campaigns.read();
    let guard = campaigns.get("p").unwrap().arms.read();
    let b = guard.get("b").unwrap();
    assert_eq!(b.status(), ArmStatus::Active, "reactivation must restore eligibility");
    assert_eq!(b.theta.to_vec(), theta_while_paused, "pausing must not touch the arm's matrices");

    let _ = std::fs::remove_file(wal);
}

/// The reason exclusion is soft: rewards for predictions made *before* the pause
/// still arrive, and the arm must still learn from them.
#[tokio::test]
async fn test_paused_arm_still_learns_from_in_flight_rewards() {
    let wal = "/tmp/banditdb_test_pause_learns.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("f", arms(&["a", "b"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();

    // Prediction in flight, then the arm is pulled from rotation.
    let (arm, iid) = db.predict_filtered("f", vec![1.0, 0.0], &ArmFilter::include(arms(&["b"]))).unwrap();
    assert_eq!(arm, "b");
    db.set_arm_status("f", "b", ArmStatus::Paused).await.unwrap();

    db.reward(&iid, 1.0).await.expect("a reward for an in-flight prediction must still be accepted");

    let campaigns = db.campaigns.read();
    let guard = campaigns.get("f").unwrap().arms.read();
    let b = guard.get("b").unwrap();
    assert_eq!(b.reward_count.load(Ordering::Relaxed), 1, "a paused arm must still absorb its in-flight reward");
    assert!(b.theta.iter().any(|v| v.abs() > 1e-12), "the reward must have moved θ");

    let _ = std::fs::remove_file(wal);
}

/// Pausing the last active arm would make every prediction fail, with no way back
/// except another API call. Refuse it.
#[tokio::test]
async fn test_cannot_pause_last_active_arm() {
    let wal = "/tmp/banditdb_test_last_arm.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("l", arms(&["a", "b"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    db.set_arm_status("l", "a", ArmStatus::Paused).await.unwrap();

    assert!(
        db.set_arm_status("l", "b", ArmStatus::Paused).await.is_err(),
        "pausing the last active arm must be refused"
    );
    assert!(db.predict("l", vec![1.0, 0.0]).is_ok(), "the campaign must still serve predictions");
    assert!(
        db.set_arm_status("l", "ghost", ArmStatus::Paused).await.is_err(),
        "an unknown arm must be rejected"
    );

    let _ = std::fs::remove_file(wal);
}

// ════════════════════════════════════════════════════════════════════════════
// Per-request filtering and propensities
// ════════════════════════════════════════════════════════════════════════════

/// A request-level exclusion removes the arm from selection *and* from the logged
/// propensities — the logged policy has to be the policy that ran, or off-policy
/// evaluation is reasoning about a fiction.
#[tokio::test]
async fn test_request_filter_shapes_propensities() {
    let wal = "/tmp/banditdb_test_filter_prop.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("r", arms(&["a", "b", "c"]), 3, 1.0, Algorithm::Linucb, None, None).await.unwrap();

    let (arm, iid) = db.predict_filtered(
        "r", vec![1.0, 0.0, 0.0], &ArmFilter::exclude(arms(&["c"])),
    ).unwrap();
    assert_ne!(arm, "c", "an excluded arm must not be selected");

    let record = db.interactions.get(iid.as_str()).expect("interaction must be pending");
    let props  = record.arm_propensities.as_ref().expect("LinUCB always logs propensities");
    assert_eq!(props.len(), 2, "propensities must cover the eligible arms only");
    assert!(!props.contains_key("c"), "an excluded arm must not appear in the propensity map");
    let sum: f64 = props.values().sum();
    assert!((sum - 1.0).abs() < 1e-9, "propensities over the eligible set must sum to 1, got {sum}");

    let _ = std::fs::remove_file(wal);
}

/// Thompson sampling estimates propensities by Monte Carlo; those draws must also
/// be restricted to the eligible set.
#[tokio::test]
async fn test_ts_propensities_respect_filter() {
    let wal = "/tmp/banditdb_test_filter_ts.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("t", arms(&["a", "b", "c"]), 3, 1.0, Algorithm::ThompsonSampling, None, None).await.unwrap();

    for _ in 0..20 {
        let (arm, iid) = db.predict_filtered(
            "t", vec![1.0, 0.0, 0.0], &ArmFilter::include(arms(&["a", "b"])),
        ).unwrap();
        assert!(arm == "a" || arm == "b", "selection must stay inside the allow-list, got {arm}");

        let record = db.interactions.get(iid.as_str()).unwrap();
        let props  = record.arm_propensities.as_ref().unwrap();
        assert_eq!(props.len(), 2, "TS propensities must cover the eligible arms only");
        let sum: f64 = props.values().sum();
        assert!((sum - 1.0).abs() < 1e-9, "TS propensities must sum to 1, got {sum}");
    }

    let _ = std::fs::remove_file(wal);
}

/// Excluding everything is a client bug, not a reason to serve an ineligible arm.
#[tokio::test]
async fn test_empty_eligible_set_is_an_error() {
    let wal = "/tmp/banditdb_test_filter_empty.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("e", arms(&["a", "b"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();

    assert!(
        db.predict_filtered("e", vec![1.0, 0.0], &ArmFilter::exclude(arms(&["a", "b"]))).is_err(),
        "excluding every arm must fail rather than fall back to an ineligible arm"
    );
    assert!(
        db.predict_filtered("e", vec![1.0, 0.0], &ArmFilter::include(arms(&["ghost"]))).is_err(),
        "an allow-list naming no real arm must fail"
    );
    // A filter can only narrow the active set, never widen it past a pause.
    db.set_arm_status("e", "b", ArmStatus::Paused).await.unwrap();
    assert!(
        db.predict_filtered("e", vec![1.0, 0.0], &ArmFilter::include(arms(&["b"]))).is_err(),
        "a filter must not resurrect a paused arm"
    );

    let _ = std::fs::remove_file(wal);
}

// ════════════════════════════════════════════════════════════════════════════
// Durability
// ════════════════════════════════════════════════════════════════════════════

/// Added arms, their groups, their priors, and their statuses all survive a
/// restart — first through WAL replay, then through a checkpoint.
#[tokio::test]
async fn test_arm_lifecycle_survives_restart() {
    let wal = "/tmp/banditdb_test_arm_recovery.jsonl";
    let _ = std::fs::remove_file(wal);
    let data_dir = data_dir_for(wal);

    let theta_before: Vec<f64>;
    {
        let db = BanditDB::new(wal, &data_dir);
        db.add_campaign("rec", arms(&["a", "b"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
        train(&db, "rec", "a", &[1.0, 0.0], 0.9, 30).await;

        db.add_arm("rec", "c", Some("grp".into()), &WarmStart::Population { strength: 2.0 }).await.unwrap();
        db.set_arm_status("rec", "b", ArmStatus::Paused).await.unwrap();

        let campaigns = db.campaigns.read();
        let guard = campaigns.get("rec").unwrap().arms.read();
        theta_before = guard.get("c").unwrap().theta.to_vec();
    }

    // 1. Replay from the WAL.
    {
        let db = BanditDB::new(wal, &data_dir);
        let campaigns = db.campaigns.read();
        let guard = campaigns.get("rec").unwrap().arms.read();
        let c = guard.get("c").expect("an added arm must survive WAL replay");
        assert_eq!(c.group.as_deref(), Some("grp"), "the group must survive replay");
        assert_eq!(c.theta.to_vec(), theta_before, "the warm-start prior must replay identically");
        assert!((c.a_inv[[0, 0]] - 0.5).abs() < 1e-9, "strength 2 must replay as A_inv = I/2");
        assert_eq!(guard.get("b").unwrap().status(), ArmStatus::Paused, "the pause must survive replay");
        assert_eq!(guard.get("a").unwrap().status(), ArmStatus::Active);
    }

    // 2. Checkpoint, then replay from the checkpoint instead of the log.
    {
        let db = BanditDB::new(wal, &data_dir);
        db.checkpoint().await.expect("checkpoint must succeed");
    }
    {
        let db = BanditDB::new(wal, &data_dir);
        let campaigns = db.campaigns.read();
        let guard = campaigns.get("rec").unwrap().arms.read();
        let c = guard.get("c").expect("an added arm must survive a checkpoint");
        assert_eq!(c.group.as_deref(), Some("grp"), "the group must survive checkpointing");
        assert_eq!(guard.get("b").unwrap().status(), ArmStatus::Paused, "the pause must survive checkpointing");
    }

    let _ = std::fs::remove_file(wal);
}

/// Progressive runs base and challenger side by side, and shadow learning updates
/// both on every reward. An arm that exists in only one of the two maps would be
/// dropped by whichever model is missing it, so `add_arm` must reach both.
#[tokio::test]
async fn test_add_arm_reaches_progressive_challenger() {
    use banditdb::state::ProgressiveConfig;

    let wal = "/tmp/banditdb_test_arm_progressive.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    let algo = Algorithm::Progressive(ProgressiveConfig {
        base:          Box::new(Algorithm::Linucb),
        challenger:    Box::new(Algorithm::ThompsonSampling),
        min_obs:       100,
        required_wins: 3,
        step_bps:      1000,
    });
    db.add_campaign("prog", arms(&["a", "b"]), 2, 1.0, algo, None, None).await.unwrap();
    train(&db, "prog", "a", &[1.0, 0.0], 0.9, 20).await;

    db.add_arm("prog", "c", None, &WarmStart::Population { strength: 1.0 }).await.unwrap();
    db.set_arm_status("prog", "b", ArmStatus::Paused).await.unwrap();

    let campaigns = db.campaigns.read();
    let campaign  = campaigns.get("prog").unwrap();
    let challenger = campaign.challenger_arms.as_ref().expect("Progressive must have challenger arms").read();

    assert!(challenger.contains_key("c"), "a new arm must reach the challenger model too");
    assert_eq!(challenger.get("b").unwrap().status(), ArmStatus::Paused,
        "a status change must apply to both models — otherwise the challenger keeps serving a paused arm");
    // The challenger's prior is resolved against the challenger's own θ, not the
    // base model's. Shadow learning trained both, but the two models hold different
    // estimates, and seeding one from the other would leak base state into the
    // challenger the tournament is meant to compare it against.
    let expected: Vec<f64> = {
        let a = &challenger.get("a").unwrap().theta;
        let b = &challenger.get("b").unwrap().theta;
        (0..2).map(|i| (a[i] + b[i]) / 2.0).collect()
    };
    let c = &challenger.get("c").unwrap().theta;
    for i in 0..2 {
        assert!((c[i] - expected[i]).abs() < 1e-9,
            "challenger prior must come from challenger θ — got {c:?}, expected {expected:?}");
    }

    let _ = std::fs::remove_file(wal);
}

/// Diagnostics and the report describe the live policy: entropy and the leading
/// arm ignore arms that can no longer be served, but their history stays visible.
#[tokio::test]
async fn test_reporting_separates_active_from_historical() {
    let wal = "/tmp/banditdb_test_arm_reporting.jsonl";
    let _ = std::fs::remove_file(wal);
    let db = BanditDB::new(wal, &data_dir_for(wal));

    db.add_campaign("rep", arms(&["winner", "loser"]), 2, 1.0, Algorithm::Linucb, None, None).await.unwrap();
    train(&db, "rep", "winner", &[1.0, 0.0], 1.0, 40).await;
    train(&db, "rep", "loser",  &[1.0, 0.0], 0.0, 40).await;

    db.set_arm_status("rep", "winner", ArmStatus::Paused).await.unwrap();

    let diag = db.campaign_diagnostics("rep").unwrap();
    assert_eq!(diag.arm_count, 2, "a paused arm is still an arm");
    assert_eq!(diag.active_arm_count, 1, "only one arm can still be served");
    assert_eq!(diag.arm_stats.get("winner").unwrap().status, ArmStatus::Paused);

    let report = db.campaign_report("rep").unwrap();
    assert_eq!(report.leading_arm.as_deref(), Some("loser"),
        "the leader must be an arm the policy can actually pick");
    assert_eq!(report.arms.get("winner").unwrap().status, ArmStatus::Paused);
    assert!(report.arms.get("winner").unwrap().predictions > 0,
        "a paused arm keeps its history in the report");

    let _ = std::fs::remove_file(wal);
}
