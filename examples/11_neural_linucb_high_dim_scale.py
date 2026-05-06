"""examples/11_neural_linucb_high_dim_scale.py

Demonstrates the memory and convergence benefits of NeuralLinUCB when contexts
are high-dimensional — the regime that matters for LLM-native applications.

Scenario: an agent is choosing between 10 retrieval strategies (arms) given
a 256-dim query embedding (simulating the output of a small sentence encoder).

LinUCB per-arm state: a 256×256 matrix = 65,536 f64 values = 512 KB per arm
                      → 10 arms = 5.1 MB just for the A⁻¹ matrices

NeuralLinUCB: 256-dim context → 16-dim embedding
              per-arm state: 16×16 matrix = 256 f64 values = 2 KB per arm
              → 10 arms = 20 KB  (256× smaller)
              + shared network weights ≈ 70 KB total

Both algorithms are run in parallel on the same reward landscape.
The script prints per-request timing and final memory footprint comparison,
then plots regret curves and arm convergence.

Requirements:
    pip install requests matplotlib numpy
    cargo build --features neural
    ./target/debug/banditdb

Run:
    python examples/11_neural_linucb_high_dim_scale.py
"""

import os
import sys
import time
import random
import math
import requests
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DB_URL           = "http://localhost:8080"
LINUCB_ID        = "scale_linucb"
NEURAL_ID        = "scale_neural"
CONTEXT_DIM      = 256
EMBED_DIM        = 16
HIDDEN_DIM       = 64
N_ARMS           = 10
N_ITERATIONS     = 1_500
CHECKPOINT_EVERY = 300
RANDOM_SEED      = 99
ARMS             = [f"strategy_{i}" for i in range(N_ARMS)]

rng = random.Random(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Reward landscape: each arm has a "preferred" direction in 256-dim space.
# Reward = cosine similarity between context and arm's ideal direction.
# Nonlinear because the optimal arm depends on orientation in a high-dim space.
# ---------------------------------------------------------------------------
np_rng = np.random.default_rng(RANDOM_SEED)
ARM_DIRECTIONS = {arm: np_rng.standard_normal(CONTEXT_DIM) for arm in ARMS}
for arm in ARMS:
    ARM_DIRECTIONS[arm] /= np.linalg.norm(ARM_DIRECTIONS[arm])

def oracle_reward(arm, context):
    ctx = np.array(context)
    ctx_norm = ctx / (np.linalg.norm(ctx) + 1e-8)
    cos_sim = float(np.dot(ARM_DIRECTIONS[arm], ctx_norm))
    # Map cosine similarity [-1, 1] → [0, 1] with some noise
    base = (cos_sim + 1.0) / 2.0
    return max(0.0, min(1.0, base + rng.gauss(0, 0.05)))

def expected_optimal(context):
    ctx = np.array(context) / (np.linalg.norm(context) + 1e-8)
    best = max(float(np.dot(ARM_DIRECTIONS[a], ctx)) for a in ARMS)
    return (best + 1.0) / 2.0

def random_unit_context():
    v = np_rng.standard_normal(CONTEXT_DIM)
    v = v / np.linalg.norm(v)
    return v.tolist()

# ---------------------------------------------------------------------------
# Memory footprint calculation
# ---------------------------------------------------------------------------
def linucb_state_bytes(context_dim, n_arms):
    # a_inv (dim×dim) + b (dim,) + theta (dim,) per arm, f64 = 8 bytes
    per_arm = (context_dim * context_dim + 2 * context_dim) * 8
    return per_arm * n_arms

def neural_state_bytes(context_dim, embed_dim, hidden_dim, hidden_layers, n_arms):
    # Network weights (f32 = 4 bytes)
    params = context_dim * hidden_dim + hidden_dim               # first layer w + b
    for _ in range(hidden_layers - 1):
        params += hidden_dim * hidden_dim + hidden_dim
    params += hidden_dim * embed_dim + embed_dim                 # output layer
    network_bytes = params * 4

    # Per-arm state in embed_dim space (f64 = 8 bytes)
    per_arm = (embed_dim * embed_dim + 2 * embed_dim) * 8
    arm_bytes = per_arm * n_arms

    return network_bytes + arm_bytes

linucb_mem = linucb_state_bytes(CONTEXT_DIM, N_ARMS)
neural_mem = neural_state_bytes(CONTEXT_DIM, EMBED_DIM, HIDDEN_DIM, 2, N_ARMS)

print("── Memory footprint comparison ─────────────────────────────────────────")
print(f"  Context dim:   {CONTEXT_DIM}")
print(f"  Embed dim:     {EMBED_DIM}")
print(f"  Arms:          {N_ARMS}")
print(f"  LinUCB state:  {linucb_mem / 1024:.1f} KB  ({linucb_mem / 1024 / 1024:.2f} MB)")
print(f"  NeuralLinUCB:  {neural_mem / 1024:.1f} KB  ({neural_mem / 1024 / 1024:.2f} MB)")
print(f"  Reduction:     {linucb_mem / neural_mem:.1f}×")
print()

# ---------------------------------------------------------------------------
# 1. Create campaigns
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
    "feature_dim": 0,
    "alpha":       1.0,
    "algorithm": {
        "neural_lin_ucb": {
            "context_dim":   CONTEXT_DIM,
            "embed_dim":     EMBED_DIM,
            "hidden_dim":    HIDDEN_DIM,
            "hidden_layers": 2,
            "retrain_every": CHECKPOINT_EVERY,
            "retrain_steps": 100,
            "learning_rate": 0.001,
            "lambda":        1.0,
        }
    },
})

if resp.status_code == 422:
    print("ERROR: Server rejected NeuralLinUCB campaign.")
    print("Ensure the server was compiled with:  cargo build --features neural")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Run simulation
# ---------------------------------------------------------------------------
cum_reward   = {"linucb": 0.0, "neural": 0.0}
cum_optimal  = 0.0
cum_reward_log = {"linucb": [], "neural": []}
cum_opt_log  = []
latency_log  = {"linucb": [], "neural": []}
arm_counts   = {"linucb": {a: 0 for a in ARMS}, "neural": {a: 0 for a in ARMS}}
checkpoint_steps = []
iterations   = []

print(f"Running {N_ITERATIONS} iterations with {CONTEXT_DIM}-dim contexts …")

for i in range(1, N_ITERATIONS + 1):
    context = random_unit_context()

    results = {}
    for alg, cid in [("linucb", LINUCB_ID), ("neural", NEURAL_ID)]:
        t0 = time.perf_counter()
        r  = requests.post(f"{DB_URL}/predict", json={"campaign_id": cid, "context": context})
        latency_log[alg].append((time.perf_counter() - t0) * 1000)
        body = r.json()
        results[alg] = (body["arm_id"], body["interaction_id"])

    for alg, (arm, iid) in results.items():
        r_val = oracle_reward(arm, context)
        requests.post(f"{DB_URL}/reward", json={"interaction_id": iid, "reward": round(r_val, 6)})
        arm_counts[alg][arm] += 1
        cum_reward[alg] += r_val

    cum_optimal += expected_optimal(context)
    iterations.append(i)
    for alg in ("linucb", "neural"):
        cum_reward_log[alg].append(cum_reward[alg])
    cum_opt_log.append(cum_optimal)

    if i % CHECKPOINT_EVERY == 0:
        requests.post(f"{DB_URL}/checkpoint")
        checkpoint_steps.append(i)
        lin_lat = sum(latency_log["linucb"][-CHECKPOINT_EVERY:]) / CHECKPOINT_EVERY
        neu_lat = sum(latency_log["neural"][-CHECKPOINT_EVERY:]) / CHECKPOINT_EVERY
        print(f"  [{i:>5}]  checkpoint  |"
              f"  avg latency — LinUCB: {lin_lat:.1f} ms  Neural: {neu_lat:.1f} ms  |"
              f"  regret — LinUCB: {cum_optimal - cum_reward['linucb']:.1f}"
              f"  Neural: {cum_optimal - cum_reward['neural']:.1f}")

print("Done.  Rendering charts …")

# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------
WINDOW = 50
ALGO_C = {"linucb": "#F44336", "neural": "#2196F3"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"BanditDB · LinUCB vs NeuralLinUCB — {CONTEXT_DIM}-dim Context, {N_ARMS} Arms",
    fontsize=13, fontweight="bold", y=0.99,
)

def add_ckpts(ax):
    for s in checkpoint_steps:
        ax.axvline(s, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

def smoothed(series, w=WINDOW):
    return [sum(series[max(0, k-w):k+1]) / min(k+1, w) for k in range(len(series))]

# ── Panel A: Cumulative regret ────────────────────────────────────────────
ax = axes[0, 0]
for alg in ("linucb", "neural"):
    regret = [cum_opt_log[k] - cum_reward_log[alg][k] for k in range(len(iterations))]
    ax.plot(iterations, regret, label=alg.replace("linucb", "LinUCB").replace("neural", "NeuralLinUCB"),
            color=ALGO_C[alg], linewidth=1.5)
add_ckpts(ax)
ax.set_title("Cumulative regret (lower = better)", fontsize=10)
ax.set_xlabel("Iteration")
ax.set_ylabel("Cumulative regret")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel B: Per-step regret smoothed ────────────────────────────────────
ax = axes[0, 1]
for alg in ("linucb", "neural"):
    regret = [cum_opt_log[k] - cum_reward_log[alg][k]
              - (cum_opt_log[k-1] - cum_reward_log[alg][k-1] if k > 0 else 0)
              for k in range(len(iterations))]
    ax.plot(iterations, smoothed(regret), label=alg.replace("linucb", "LinUCB").replace("neural", "NeuralLinUCB"),
            color=ALGO_C[alg], linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
add_ckpts(ax)
ax.set_title(f"Per-step regret ({WINDOW}-step avg)", fontsize=10)
ax.set_xlabel("Iteration")
ax.set_ylabel("Regret / step")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel C: Predict latency comparison ───────────────────────────────────
ax = axes[1, 0]
sm_lin = smoothed(latency_log["linucb"], w=100)
sm_neu = smoothed(latency_log["neural"], w=100)
ax.plot(iterations, sm_lin, label=f"LinUCB ({CONTEXT_DIM}×{CONTEXT_DIM} matrices)", color=ALGO_C["linucb"], linewidth=1.3)
ax.plot(iterations, sm_neu, label=f"NeuralLinUCB ({EMBED_DIM}×{EMBED_DIM} matrices + MLP)", color=ALGO_C["neural"], linewidth=1.3)
add_ckpts(ax)
ax.set_title("Predict latency — 100-step rolling average", fontsize=10)
ax.set_xlabel("Iteration")
ax.set_ylabel("Latency (ms)")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel D: Memory footprint bar chart ──────────────────────────────────
ax = axes[1, 1]
labels   = ["LinUCB\n(per arm)", "NeuralLinUCB\n(per arm)", "NeuralLinUCB\n(total incl. network)"]
sizes_kb = [
    linucb_state_bytes(CONTEXT_DIM, 1) / 1024,
    neural_mem / N_ARMS / 1024,
    neural_mem / 1024,
]
colors = [ALGO_C["linucb"], ALGO_C["neural"], "#1565C0"]
bars = ax.bar(labels, sizes_kb, color=colors, alpha=0.80, width=0.5)
for bar, val in zip(bars, sizes_kb):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f} KB", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title(
    f"State memory per configuration\n"
    f"Context dim={CONTEXT_DIM}, {N_ARMS} arms, embed dim={EMBED_DIM}",
    fontsize=10,
)
ax.set_ylabel("Memory (KB)")
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(SCRIPT_DIR, "11_neural_linucb_high_dim_scale.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved → {out}")

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
print("\n── Final summary ────────────────────────────────────────────────────────")
print(f"  Iterations:        {N_ITERATIONS}")
print(f"  Context dim:       {CONTEXT_DIM}")
print(f"  Arms:              {N_ARMS}")
print(f"  LinUCB state:      {linucb_mem / 1024:.1f} KB")
print(f"  NeuralLinUCB:      {neural_mem / 1024:.1f} KB  ({linucb_mem / neural_mem:.1f}× smaller)")
print()
print(f"  {'Algorithm':<20} {'Cum reward':>12} {'Cum regret':>12} {'Avg latency':>14}")
for alg in ("linucb", "neural"):
    regret = cum_optimal - cum_reward[alg]
    avg_lat = sum(latency_log[alg]) / len(latency_log[alg])
    label = "LinUCB" if alg == "linucb" else "NeuralLinUCB"
    print(f"  {label:<20} {cum_reward[alg]:>12.1f} {regret:>12.1f} {avg_lat:>12.2f} ms")

plt.show()
