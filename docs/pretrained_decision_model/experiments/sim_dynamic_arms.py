"""Dynamic-arm simulation: fashion catalogue whose inventory churns.

Two campaigns run on ONE seeded request stream with common random numbers:

    warm — the launched arms are warm-started from named sibling arms
    cold — identical in every way, launched cold

Two arms launch at the same step: one into a strong group, one into a weak group,
so the experiment sees both the favourable and the unfavourable case.

Everything else (contexts, stock-outs, pauses, reward coin flips) is shared, so
any difference between them is attributable to the warm start alone.

Regret is measured against the best *eligible* arm at each step. Scoring it
against the globally best arm would invent phantom regret during the stock-out
and pause windows and make the whole comparison a lie.
"""

import os, sys, time, json, math
import numpy as np
import requests

DB     = os.getenv("DB_URL", "http://localhost:8099")
KEY    = os.getenv("BANDITDB_API_KEY", "sim-admin-key")
H      = {"X-Api-Key": KEY, "Content-Type": "application/json"}
DIM    = 8
STEPS  = 12_000
SEED   = int(os.getenv("SIM_SEED", "7"))

# Event schedule (step index)
STOCKOUT_AT   = 3_000    # sneaker_a unavailable for ~20% of shoppers, per request
PAUSE_AT      = 4_500    # winter: sandals out of rotation
LAUNCH_AT     = 6_000    # sneaker_new joins
REACTIVATE_AT = 9_000    # spring: sandals back
RETIRE_AT     = 10_500   # boot_b retired

BASE_ARMS = {
    "sneaker_a": "shoes",   "sneaker_b": "shoes",
    "boot_a":    "boots",   "boot_b":    "boots",
    "sandal_a":  "sandals", "sandal_b":  "sandals",
}
NEW_ARMS = {"sneaker_new": ["sneaker_a", "sneaker_b"],   # mediocre group
            "boot_new":    ["boot_a"]}                    # strong group

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
# Context: [season_sin, season_cos, price_sensitivity, mobile, session_depth,
#           returning, discount_affinity, bias]
# Arm payoff is theta·x clipped to [0,1]. Season features flip which group wins.

TRUE = {
    "sneaker_a": np.array([ 0.05,  0.10, -0.05, 0.10, 0.05, 0.05, 0.00, 0.30]),
    "sneaker_b": np.array([ 0.05,  0.08, -0.03, 0.08, 0.04, 0.06, 0.02, 0.28]),
    "boot_a":    np.array([-0.18,  0.02,  0.02, 0.02, 0.03, 0.04, 0.05, 0.34]),
    "boot_b":    np.array([-0.10, -0.02,  0.00, 0.00, 0.01, 0.01, 0.02, 0.18]),
    "sandal_a":  np.array([ 0.22, -0.02,  0.03, 0.03, 0.02, 0.03, 0.06, 0.30]),
    "sandal_b":  np.array([ 0.20, -0.01,  0.02, 0.02, 0.02, 0.02, 0.05, 0.28]),
}
# The premise of a group warm start: a new sneaker behaves like the other
# sneakers. Its truth is the group mean plus a little idiosyncrasy.
TRUE["sneaker_new"] = (TRUE["sneaker_a"] + TRUE["sneaker_b"]) / 2 + np.array(
    [0.0, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.03])
TRUE["boot_new"] = TRUE["boot_a"] + np.array(
    [0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.02])


def payoff(arm, x):
    return float(np.clip(TRUE[arm] @ x, 0.0, 1.0))


# ---------------------------------------------------------------------------
# One shared request stream (common random numbers)
# ---------------------------------------------------------------------------
def build_stream():
    ctxs, coins, stockouts = [], [], []
    for t in range(STEPS):
        # A year of seasonality across the run.
        phase = 2 * math.pi * t / STEPS
        x = np.array([
            math.sin(phase), math.cos(phase),
            rng.uniform(0, 1), float(rng.random() < 0.6),
            rng.uniform(0, 1), float(rng.random() < 0.35),
            rng.uniform(0, 1), 1.0,
        ])
        x = x / np.linalg.norm(x)          # unit L2 — see README
        ctxs.append(x)
        coins.append(rng.random())          # shared Bernoulli draw
        stockouts.append(rng.random() < 0.20)
    return ctxs, coins, stockouts


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
S = requests.Session()

def api(method, path, **kw):
    r = S.request(method, f"{DB}{path}", headers=H, timeout=30, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
    return r

def create_campaign(cid):
    api("POST", "/campaign", json={
        "campaign_id": cid, "arms": list(BASE_ARMS), "feature_dim": DIM,
        "alpha": 1.0, "algorithm": "linucb", "decay_half_life_hours": 72.0,
    })
    # Label the founding arms with their groups by re-adding? No — groups are set
    # at add time, so the founding six are ungrouped. The launch therefore warm
    # starts from an explicit list of the sneaker arms, which is the same borrow.

def add_arm(cid, arm, sources, warm):
    body = {"arm_id": arm, "group": "shoes" if arm.startswith("sneaker") else "boots"}
    body["warm_start"] = ({"from": "arms", "arms": sources, "strength": 1.0}
                          if warm else {"from": "none"})
    api("POST", f"/campaign/{cid}/arms", json=body)

def arm_theta(cid, arm):
    info = api("GET", f"/campaign/{cid}").json()
    return np.array(info["arms"][arm]["theta"])

def set_status(cid, arm, status):
    api("POST", f"/campaign/{cid}/arms/{arm}/status", json={"status": status})

def diagnostics(cid):
    return api("GET", f"/campaign/{cid}/diagnostics").json()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(cid, warm, ctxs, coins, stockouts, trace):
    live    = dict(BASE_ARMS)             # arms that exist, client-side mirror
    paused  = set()
    reward_sum = 0.0
    regret_sum = 0.0
    post_launch_regret = 0.0
    stats = {a: {"impressions": 0, "payoff": 0.0} for a in NEW_ARMS}
    theta_err = []          # (steps_since_launch, arm, L2 error vs truth)
    probes = [100, 250, 500, 1000, 2000]

    for t in range(STEPS):
        # --- fire scheduled events -----------------------------------------
        if t == PAUSE_AT:
            for a in ("sandal_a", "sandal_b"):
                set_status(cid, a, "paused"); paused.add(a)
        if t == LAUNCH_AT:
            for a, src in NEW_ARMS.items():
                add_arm(cid, a, src, warm)
                live[a] = "shoes" if a.startswith("sneaker") else "boots"
                theta_err.append({"since": 0, "arm": a,
                                  "err": float(np.linalg.norm(arm_theta(cid, a) - TRUE[a]))})
        if t > LAUNCH_AT and (t - LAUNCH_AT) in probes:
            for a in NEW_ARMS:
                theta_err.append({"since": t - LAUNCH_AT, "arm": a,
                                  "err": float(np.linalg.norm(arm_theta(cid, a) - TRUE[a]))})
        # Decay only runs at checkpoint, so a paused arm's posterior widens only
        # if checkpoints actually happen. Without these the reactivation story
        # would be a claim the run never tested.
        if t > 0 and t % 2000 == 0:
            api("POST", "/checkpoint")
        if t == REACTIVATE_AT:
            for a in ("sandal_a", "sandal_b"):
                set_status(cid, a, "active"); paused.discard(a)
        if t == RETIRE_AT:
            set_status(cid, "boot_b", "retired"); paused.add("boot_b")

        x = ctxs[t]
        excluded = []
        if t >= STOCKOUT_AT and stockouts[t]:
            excluded = ["sneaker_a"]

        eligible = [a for a in live if a not in paused and a not in excluded]

        body = {"campaign_id": cid, "context": x.tolist()}
        if excluded:
            body["exclude_arms"] = excluded
        r = api("POST", "/predict", json=body).json()
        arm, iid = r["arm_id"], r["interaction_id"]

        p = payoff(arm, x)
        y = 1.0 if coins[t] < p else 0.0
        api("POST", "/reward", json={"interaction_id": iid, "reward": y})

        best = max(payoff(a, x) for a in eligible)
        reward_sum += p
        regret_sum += best - p

        if t >= LAUNCH_AT:
            post_launch_regret += best - p
            if arm in stats:
                stats[arm]["impressions"] += 1
                stats[arm]["payoff"] += p

        if t % 500 == 0 or t == STEPS - 1:
            d = diagnostics(cid)
            naive = naive_entropy(d["arm_stats"])
            trace.append({
                "step": t, "cid": cid,
                "entropy_active": d["selection_entropy"],
                "entropy_naive":  naive,
                "active": d["active_arm_count"], "total": d["arm_count"],
                "reward_rate": reward_sum / (t + 1),
            })

    return {
        "reward_sum": reward_sum, "regret_sum": regret_sum,
        "post_launch_regret": post_launch_regret,
        "new_arms": stats, "theta_err": theta_err,
    }


def naive_entropy(arm_stats):
    """Entropy over EVERY arm, including paused ones — what the metric used to be."""
    counts = [s["predictions"] for s in arm_stats.values()]
    total  = sum(counts)
    if total == 0 or len(counts) < 2:
        return 1.0
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0) / math.log(len(counts))


def main():
    ctxs, coins, stockouts = build_stream()
    trace = []
    out = {}
    TAG = os.getenv("SIM_TAG", "x")
    for cid, warm in ((f"warm_{TAG}", True), (f"cold_{TAG}", False)):
        create_campaign(cid)
        t0 = time.time()
        out[cid] = run(cid, warm, ctxs, coins, stockouts, trace)
        out[cid]["seconds"] = time.time() - t0
        print(f"  {cid} done in {out[cid]['seconds']:.0f}s", flush=True)

    for cid in list(out):
        out[cid]["report"] = api("GET", f"/campaign/{cid}/report").json()
        out[cid]["diag"]   = diagnostics(cid)

    json.dump({"runs": out, "trace": trace,
               "truth": {k: v.tolist() for k, v in TRUE.items()}},
              open(sys.argv[1], "w"))
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
