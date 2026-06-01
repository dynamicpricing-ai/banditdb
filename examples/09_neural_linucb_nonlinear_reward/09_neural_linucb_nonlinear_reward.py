"""examples/09_neural_linucb_nonlinear_reward.py

Demonstrates NeuralLinUCB's advantage over LinUCB on a reward landscape that
linear models cannot express.

True reward depends on the XOR interaction of the first two context features:
  arm_a wins when (x0 > 0.5) == (x1 > 0.5)  — the diagonal quadrants
  arm_b wins in the off-diagonal quadrants
  arm_c is mediocre everywhere

A 4-dim context is sent; the other two features are pure noise.  LinUCB must
find a linear separator in [x0, x1] space — none exists for an XOR pattern.
NeuralLinUCB learns x0*x1-style interactions through its hidden layers.

Both campaigns run against the same server with identical context sequences
so the comparison is fair.  /checkpoint is called every CHECKPOINT_EVERY steps
to trigger the NeuralLinUCB retrain (Algorithm 2 + re-accumulation).

Requirements:
    pip install requests matplotlib numpy
    cargo build --features neural   # server must be compiled with neural support
    ./target/debug/banditdb         # server running on :8080

Run:
    python examples/09_neural_linucb_nonlinear_reward.py
"""

import math
import os
import sys
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_URL          = "http://localhost:8080"
LINUCB_ID       = "nl_demo_linucb"
NEURAL_ID       = "nl_demo_neural"
ARMS            = ["arm_a", "arm_b", "arm_c"]
CONTEXT_DIM     = 4          # raw context sent to the server
N_ITERATIONS    = 2_000
CHECKPOINT_EVERY = 100       # call /checkpoint (and retrain) every N rewards
RANDOM_SEED     = 42

# ---------------------------------------------------------------------------
# Ground-truth reward function — nonlinear (XOR pattern)
# ---------------------------------------------------------------------------
# arm_a is optimal in the diagonal quadrants  (x0 high & x1 high, or both low)
# arm_b is optimal in the off-diagonal quadrants
# arm_c is always mediocre
# No linear combination of [x0, x1, x2, x3] separates diagonal from off-diagonal.

rng = random.Random(RANDOM_SEED)

def oracle_reward(arm, context):
    x0, x1 = context[0], context[1]
    diagonal = (x0 > 0.5) == (x1 > 0.5)
    base = {"arm_a": 0.80 if diagonal     else 0.20,
            "arm_b": 0.80 if not diagonal else 0.20,
            "arm_c": 0.50}[arm]
    return max(0.0, min(1.0, base + rng.gauss(0, 0.08)))

def expected_best(context):
    """Expected reward of the oracle-optimal arm (no noise)."""
    x0, x1 = context[0], context[1]
    return 0.80  # oracle always picks the high-reward arm in expectation

# ---------------------------------------------------------------------------
# 1. Create campaigns (idempotent — safe to re-run)
# ---------------------------------------------------------------------------
requests.post(f"{DB_URL}/campaign", json={
    "campaign_id": LINUCB_ID,
    "arms":        ARMS,
    "feature_dim": CONTEXT_DIM,
    "alpha":       1.0,
    "algorithm":   "linucb",
})

resp = requests.post(f"{DB_URL}/campaign", json={
    "campaign_id": NEURAL_ID,
    "arms":        ARMS,
    "feature_dim": 0,       # ignored for NeuralLinUCB (arm matrices use embed_dim)
    "alpha":       1.0,
    "algorithm": {
        "neural_lin_ucb": {
            "context_dim":   CONTEXT_DIM,
            "embed_dim":     8,
            "hidden_dim":    64,
            "hidden_layers": 2,
            "retrain_every": CHECKPOINT_EVERY,
            "retrain_steps": 200,
            "learning_rate": 0.003,
            "lambda":        0.1,
        }
    },
})

if resp.status_code == 422:
    print("ERROR: Server rejected NeuralLinUCB campaign.")
    print("Ensure the server was compiled with:  cargo build --features neural")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Run N_ITERATIONS of the predict → reward loop for both campaigns
# ---------------------------------------------------------------------------
def predict(campaign_id, context):
    r = requests.post(f"{DB_URL}/predict", json={"campaign_id": campaign_id, "context": context})
    body = r.json()
    return body["arm_id"], body["interaction_id"]

def reward_interaction(iid, value):
    requests.post(f"{DB_URL}/reward", json={"interaction_id": iid, "reward": round(value, 6)})

arm_counts       = {alg: {a: 0 for a in ARMS} for alg in ["linucb", "neural"]}
cum_reward       = {"linucb": 0.0, "neural": 0.0}
cum_optimal      = 0.0
cum_rewards_log  = {"linucb": [], "neural": []}
cum_optimal_log  = []
arm_pct_log      = {"linucb": {a: [] for a in ARMS}, "neural": {a: [] for a in ARMS}}
checkpoint_steps = []
iterations       = []

print(f"Running {N_ITERATIONS} iterations — checkpoint every {CHECKPOINT_EVERY} steps …")

for i in range(1, N_ITERATIONS + 1):
    context = [rng.random() for _ in range(CONTEXT_DIM)]

    linucb_arm, linucb_iid = predict(LINUCB_ID, context)
    neural_arm, neural_iid = predict(NEURAL_ID, context)

    linucb_r = oracle_reward(linucb_arm, context)
    neural_r = oracle_reward(neural_arm, context)

    reward_interaction(linucb_iid, linucb_r)
    reward_interaction(neural_iid, neural_r)

    # Accumulate stats
    arm_counts["linucb"][linucb_arm] += 1
    arm_counts["neural"][neural_arm] += 1
    cum_reward["linucb"] += linucb_r
    cum_reward["neural"] += neural_r
    cum_optimal += expected_best(context)

    iterations.append(i)
    cum_rewards_log["linucb"].append(cum_reward["linucb"])
    cum_rewards_log["neural"].append(cum_reward["neural"])
    cum_optimal_log.append(cum_optimal)
    for a in ARMS:
        arm_pct_log["linucb"][a].append(arm_counts["linucb"][a] / i)
        arm_pct_log["neural"][a].append(arm_counts["neural"][a] / i)

    # Checkpoint: persists state + triggers NeuralLinUCB retrain (Algorithm 2)
    if i % CHECKPOINT_EVERY == 0:
        requests.post(f"{DB_URL}/checkpoint")
        checkpoint_steps.append(i)
        print(f"  [{i:>5}]  checkpoint  |"
              f"  LinUCB regret: {cum_optimal - cum_reward['linucb']:.1f}"
              f"  Neural regret: {cum_optimal - cum_reward['neural']:.1f}")

print("Done.  Rendering chart …")

# ---------------------------------------------------------------------------
# 3. Four-panel comparison chart
# ---------------------------------------------------------------------------
COLORS  = {"arm_a": "#2196F3", "arm_b": "#FF9800", "arm_c": "#9C27B0"}
ALGO_C  = {"linucb": "#F44336", "neural": "#4CAF50"}
WINDOW  = 50

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "BanditDB · LinUCB vs NeuralLinUCB — XOR Nonlinear Reward Landscape",
    fontsize=13, fontweight="bold", y=0.99,
)

def add_checkpoints(ax, y_pos=None):
    for s in checkpoint_steps:
        ax.axvline(s, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

# ── Panel A: LinUCB arm selection ─────────────────────────────────────────
ax = axes[0, 0]
ax.stackplot(
    iterations,
    [[v * 100 for v in arm_pct_log["linucb"][a]] for a in ARMS],
    labels=[f"{a}" for a in ARMS],
    colors=[COLORS[a] for a in ARMS],
    alpha=0.75,
)
add_checkpoints(ax)
ax.set_title("LinUCB — arm selection", fontsize=10)
ax.set_ylabel("Selection share (%)")
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_xlabel("Iteration")

# ── Panel B: NeuralLinUCB arm selection ──────────────────────────────────
ax = axes[0, 1]
ax.stackplot(
    iterations,
    [[v * 100 for v in arm_pct_log["neural"][a]] for a in ARMS],
    labels=[f"{a}" for a in ARMS],
    colors=[COLORS[a] for a in ARMS],
    alpha=0.75,
)
add_checkpoints(ax)
ax.set_title(f"NeuralLinUCB — arm selection  (retrain every {CHECKPOINT_EVERY} steps)", fontsize=10)
ax.set_ylabel("Selection share (%)")
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_xlabel("Iteration")

# ── Panel C: Cumulative regret comparison ─────────────────────────────────
ax = axes[1, 0]
regret_linucb = [cum_optimal_log[i] - cum_rewards_log["linucb"][i] for i in range(len(iterations))]
regret_neural = [cum_optimal_log[i] - cum_rewards_log["neural"][i]  for i in range(len(iterations))]
ax.plot(iterations, regret_linucb, label="LinUCB",       color=ALGO_C["linucb"], linewidth=1.5)
ax.plot(iterations, regret_neural, label="NeuralLinUCB", color=ALGO_C["neural"], linewidth=1.5)
add_checkpoints(ax)
ax.set_title("Cumulative regret — lower is better", fontsize=10)
ax.set_xlabel("Iteration")
ax.set_ylabel("Cumulative regret")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel D: Per-step regret (smoothed) ───────────────────────────────────
ax = axes[1, 1]
def smoothed(series, w=WINDOW):
    return [sum(series[max(0, i-w):i+1]) / min(i+1, w) for i in range(len(series))]

instant_linucb = [cum_optimal_log[i] - cum_rewards_log["linucb"][i]
                  - (cum_optimal_log[i-1] - cum_rewards_log["linucb"][i-1] if i > 0 else 0)
                  for i in range(len(iterations))]
instant_neural = [cum_optimal_log[i] - cum_rewards_log["neural"][i]
                  - (cum_optimal_log[i-1] - cum_rewards_log["neural"][i-1] if i > 0 else 0)
                  for i in range(len(iterations))]

ax.plot(iterations, smoothed(instant_linucb), label="LinUCB",       color=ALGO_C["linucb"], linewidth=1.5)
ax.plot(iterations, smoothed(instant_neural), label="NeuralLinUCB", color=ALGO_C["neural"], linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
add_checkpoints(ax)
ax.set_title(f"Per-step regret ({WINDOW}-step avg)  — grey lines: retrain checkpoints", fontsize=10)
ax.set_xlabel("Iteration")
ax.set_ylabel(f"Regret / step")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(SCRIPT_DIR, "09_neural_linucb_nonlinear_reward.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved → {out}")

# Print final summary
print("\n── Final statistics ─────────────────────────────────────────")
print(f"{'Algorithm':<18} {'Cum reward':>12} {'Cum regret':>12} {'arm_a %':>8} {'arm_b %':>8} {'arm_c %':>8}")
for label, key in [("LinUCB", "linucb"), ("NeuralLinUCB", "neural")]:
    reg   = cum_optimal - cum_reward[key]
    a_pct = arm_counts[key]["arm_a"] / N_ITERATIONS * 100
    b_pct = arm_counts[key]["arm_b"] / N_ITERATIONS * 100
    c_pct = arm_counts[key]["arm_c"] / N_ITERATIONS * 100
    print(f"{label:<18} {cum_reward[key]:>12.1f} {reg:>12.1f} {a_pct:>7.1f}% {b_pct:>7.1f}% {c_pct:>7.1f}%")

plt.show()
