"""examples/02_convergence_simulation.py

Runs a synthetic convergence simulation against a live BanditDB server
and renders a three-panel asymptotic convergence chart.

Every iteration follows the exact same workflow a production application
would use: POST /predict → compute reward locally → POST /reward.
There is no special "simulate" endpoint — this is the real developer API.

Install dependencies:
    pip install requests matplotlib numpy

Run (server must be up on port 8080):
    python examples/02_convergence_simulation.py
"""

import math
import os
import random
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Resolve paths relative to this script, regardless of where it is invoked from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DB_URL       = "http://localhost:8080"
CAMPAIGN_ID  = "convergence_demo"
FEATURE_DIM  = 3
N_ITERATIONS = 1_000
ALPHA        = 1.0          # LinUCB exploration parameter (passed at campaign creation time via alpha)
RANDOM_SEED  = 42

# ---------------------------------------------------------------------------
# Ground-truth weights — the oracle knows the "true" reward function for
# each arm.  The bandit must *discover* which arm is best via exploration.
#
# Expected oracle reward = sigmoid(true_theta · E[context])
#   arm_a : sigmoid((2.0+1.5-0.5)×0.5) = sigmoid(1.5) ≈ 0.82   ← dominant
#   arm_b : sigmoid((0.5+0.5+0.5)×0.5) = sigmoid(0.75) ≈ 0.68
#   arm_c : sigmoid((-0.5+0.5+0.5)×0.5) = sigmoid(0.25) ≈ 0.56
# ---------------------------------------------------------------------------
ARMS = ["arm_a", "arm_b", "arm_c"]
TRUE_THETAS = {
    "arm_a": [ 2.0,  1.5, -0.5],
    "arm_b": [ 0.5,  0.5,  0.5],
    "arm_c": [-0.5,  0.5,  0.5],
}

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def oracle_reward(arm, context):
    """Synthetic reward: sigmoid(true_theta[arm] · context), in (0, 1)."""
    dot = sum(w * x for w, x in zip(TRUE_THETAS[arm], context))
    return sigmoid(dot)

def optimal_reward(context):
    """Reward the oracle-best arm would have received for this context."""
    return max(oracle_reward(arm, context) for arm in ARMS)

# ---------------------------------------------------------------------------
# 1. Register the campaign (idempotent — safe to re-run)
# ---------------------------------------------------------------------------
requests.post(f"{DB_URL}/campaign", json={
    "campaign_id": CAMPAIGN_ID,
    "arms":        ARMS,
    "feature_dim": FEATURE_DIM,
})

# ---------------------------------------------------------------------------
# 2. Run N_ITERATIONS of the standard predict → reward loop
# ---------------------------------------------------------------------------
rng = random.Random(RANDOM_SEED)

arm_counts          = {a: 0 for a in ARMS}
cum_reward          = 0.0
cum_optimal         = 0.0
arm_selection_pct   = {a: [] for a in ARMS}
cumulative_reward   = []
optimal_cumulative  = []
iterations          = []

print(f"Running {N_ITERATIONS} iterations …")

for i in range(1, N_ITERATIONS + 1):
    # Draw a random context from Uniform[0, 1]^FEATURE_DIM
    context = [rng.random() for _ in range(FEATURE_DIM)]

    # ── Ask BanditDB which arm to use ──────────────────────────────────────
    resp = requests.post(f"{DB_URL}/predict", json={
        "campaign_id": CAMPAIGN_ID,
        "context":     context,
    })
    body           = resp.json()
    chosen_arm     = body["arm_id"]
    interaction_id = body["interaction_id"]

    # ── Compute the oracle reward for the chosen arm ───────────────────────
    reward = oracle_reward(chosen_arm, context)

    # ── Tell BanditDB what happened (this updates the LinUCB matrices) ─────
    requests.post(f"{DB_URL}/reward", json={
        "interaction_id": interaction_id,
        "reward":         round(reward, 6),
    })

    # ── Record statistics ──────────────────────────────────────────────────
    arm_counts[chosen_arm] += 1
    cum_reward  += reward
    cum_optimal += optimal_reward(context)

    iterations.append(i)
    cumulative_reward.append(cum_reward)
    optimal_cumulative.append(cum_optimal)
    for arm in ARMS:
        arm_selection_pct[arm].append(arm_counts[arm] / i)

print("Done.  Rendering chart …")

# ---------------------------------------------------------------------------
# 3. Three-panel asymptotic convergence chart
# ---------------------------------------------------------------------------
COLORS = {"arm_a": "#2196F3", "arm_b": "#FF9800", "arm_c": "#9C27B0"}

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig.suptitle(
    "BanditDB · LinUCB Asymptotic Convergence  (real /predict + /reward API)",
    fontsize=13, fontweight="bold", y=0.98,
)

# ── Panel 1: Arm selection share ─────────────────────────────────────────────
ax1 = axes[0]
ax1.stackplot(
    iterations,
    [[v * 100 for v in arm_selection_pct[a]] for a in ARMS],
    labels=[f"{a}  (%)" for a in ARMS],
    colors=[COLORS[a] for a in ARMS],
    alpha=0.75,
)
ax1.set_ylabel("Arm Selection Share (%)")
ax1.set_ylim(0, 100)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.legend(loc="upper right", fontsize=9)
ax1.set_title("Arm Selection — bandit converges on the dominant arm", fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# ── Panel 2: Cumulative reward vs oracle optimum ──────────────────────────────
ax2 = axes[1]
ax2.plot(iterations, optimal_cumulative, label="Oracle optimum",
         color="#4CAF50", linewidth=1.5, linestyle="--")
ax2.plot(iterations, cumulative_reward,  label="BanditDB actual",
         color="#F44336", linewidth=1.5)
ax2.fill_between(iterations, cumulative_reward, optimal_cumulative,
                 alpha=0.12, color="#F44336", label="Cumulative regret")
ax2.set_ylabel("Cumulative Reward")
ax2.legend(loc="upper left", fontsize=9)
ax2.set_title("Cumulative Reward vs Oracle Optimum — regret gap narrows", fontsize=10)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

# ── Panel 3: Instantaneous regret per step ────────────────────────────────────
# Smooth with a rolling 50-step mean to make the trend visible.
window = 50
instant_regret = [
    optimal_cumulative[i] - cumulative_reward[i]
    - (optimal_cumulative[i - 1] - cumulative_reward[i - 1] if i > 0 else 0)
    for i in range(len(iterations))
]
smoothed = [
    sum(instant_regret[max(0, i - window):i + 1]) / min(i + 1, window)
    for i in range(len(iterations))
]

ax3 = axes[2]
ax3.plot(iterations, smoothed, color="#607D8B", linewidth=1.5)
ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax3.set_xlabel("Iteration")
ax3.set_ylabel(f"Regret / step  ({window}-step avg)")
ax3.set_title("Instantaneous Regret — approaches zero as exploitation dominates", fontsize=10)
ax3.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.97])

output_path = os.path.join(SCRIPT_DIR, "convergence_chart.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Chart saved → {output_path}")
plt.show()
