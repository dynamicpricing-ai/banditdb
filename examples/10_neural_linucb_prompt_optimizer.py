"""examples/10_neural_linucb_prompt_optimizer.py

NeuralLinUCB for LLM prompt template selection — a real agentic use case.

An agent receives incoming queries of 4 types (technical, creative, factual,
conversational).  Each query is encoded as an 8-dim embedding (simulating what
a sentence encoder would produce).  The agent must learn which of 5 prompt
templates produces the highest output quality for each query type.

The reward function is nonlinear:
  "structured" scores high for technical queries, low for others
  "creative"   scores high for creative queries
  "concise"    scores high for factual queries
  "friendly"   scores high for conversational queries
  "generic"    mediocre across all query types

LinUCB cannot find a global linear rule because "which template is best" depends
on a learned categorisation of the query embedding — exactly what the neural
network learns as its hidden representation.

/checkpoint is called every CHECKPOINT_EVERY steps to trigger Algorithm 2 (gradient
descent on the shared network) + re-accumulation of per-arm LinUCB statistics.

The final heatmap shows which template the bandit learned to route to each query
type — the diagonal should dominate after convergence.

Requirements:
    pip install requests matplotlib numpy
    cargo build --features neural
    ./target/debug/banditdb         # server on :8080

Run:
    python examples/10_neural_linucb_prompt_optimizer.py
"""

import os
import sys
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DB_URL           = "http://localhost:8080"
CAMPAIGN_ID      = "prompt_optimizer"
N_ITERATIONS     = 3_000
CHECKPOINT_EVERY = 300
RANDOM_SEED      = 7

TEMPLATES   = ["structured", "creative", "concise", "friendly", "generic"]
QUERY_TYPES = ["technical", "creative", "factual", "conversational"]
CONTEXT_DIM = 8    # simulated sentence-encoder output dimension

# Ground-truth: which template is optimal for each query type
BEST_TEMPLATE = {
    "technical":      "structured",
    "creative":       "creative",
    "factual":        "concise",
    "conversational": "friendly",
}

# Query type cluster centers in 8-dim embedding space
# Two features per query type are "on"; the rest are noise
QUERY_CENTERS = {
    "technical":      [0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1],
    "creative":       [0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1],
    "factual":        [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.1],
    "conversational": [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8],
}

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
rng = random.Random(RANDOM_SEED)

def make_context(query_type):
    """Return a noisy 8-dim embedding for the given query type."""
    center = QUERY_CENTERS[query_type]
    return [max(0.0, min(1.0, c + rng.gauss(0, 0.12))) for c in center]

def oracle_reward(template, query_type):
    """Nonlinear reward: depends on (template, query_type) match, not linear in context."""
    if BEST_TEMPLATE.get(query_type) == template:
        base = 0.85          # best match
    elif template == "generic":
        base = 0.50          # mediocre everywhere
    else:
        base = 0.20          # wrong template
    return max(0.0, min(1.0, base + rng.gauss(0, 0.10)))

def expected_optimal(_query_type):
    return 0.85

# ---------------------------------------------------------------------------
# 1. Create campaign
# ---------------------------------------------------------------------------
resp = requests.post(f"{DB_URL}/campaign", json={
    "campaign_id": CAMPAIGN_ID,
    "arms":        TEMPLATES,
    "feature_dim": 0,
    "alpha":       1.0,
    "algorithm": {
        "neural_lin_ucb": {
            "context_dim":   CONTEXT_DIM,
            "embed_dim":     8,
            "hidden_dim":    32,
            "hidden_layers": 2,
            "retrain_every": CHECKPOINT_EVERY,
            "retrain_steps": 150,
            "learning_rate": 0.001,
            "lambda":        1.0,
        }
    },
})

if resp.status_code == 422:
    print("ERROR: Server rejected NeuralLinUCB campaign.")
    print("Ensure the server was compiled with:  cargo build --features neural")
    sys.exit(1)
elif resp.status_code not in (200, 201, 409):
    print(f"Campaign creation returned {resp.status_code}: {resp.text}")

# ---------------------------------------------------------------------------
# 2. Simulate N_ITERATIONS of the predict → reward loop
# ---------------------------------------------------------------------------

# Track selection counts per (query_type, template) for the final heatmap
selection_matrix = {qt: {t: 0 for t in TEMPLATES} for qt in QUERY_TYPES}

cum_reward      = 0.0
cum_optimal     = 0.0
cum_rewards_log = []
cum_regret_log  = []
reward_by_qtype = {qt: [] for qt in QUERY_TYPES}
checkpoint_steps = []
iterations      = []

# Rolling reward per query type for the convergence panel
WINDOW = 100
reward_window = {qt: [] for qt in QUERY_TYPES}
rolling_reward = {qt: [] for qt in QUERY_TYPES}

print(f"Running {N_ITERATIONS} iterations — checkpoint every {CHECKPOINT_EVERY} steps …")

for i in range(1, N_ITERATIONS + 1):
    query_type = rng.choice(QUERY_TYPES)
    context    = make_context(query_type)

    resp = requests.post(f"{DB_URL}/predict", json={
        "campaign_id": CAMPAIGN_ID,
        "context":     context,
    })
    body           = resp.json()
    chosen         = body["arm_id"]
    interaction_id = body["interaction_id"]

    r = oracle_reward(chosen, query_type)
    requests.post(f"{DB_URL}/reward", json={"interaction_id": interaction_id, "reward": round(r, 6)})

    # Track statistics
    selection_matrix[query_type][chosen] += 1
    cum_reward  += r
    cum_optimal += expected_optimal(query_type)

    iterations.append(i)
    cum_rewards_log.append(cum_reward)
    cum_regret_log.append(cum_optimal - cum_reward)

    for qt in QUERY_TYPES:
        reward_window[qt].append(r if qt == query_type else None)

    for qt in QUERY_TYPES:
        vals = [v for v in reward_window[qt][-WINDOW:] if v is not None]
        rolling_reward[qt].append(sum(vals) / len(vals) if vals else 0.0)

    if i % CHECKPOINT_EVERY == 0:
        requests.post(f"{DB_URL}/checkpoint")
        checkpoint_steps.append(i)
        print(f"  [{i:>5}]  checkpoint  |  cum regret: {cum_optimal - cum_reward:.1f}")

print("Done.  Rendering charts …")

# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "BanditDB · NeuralLinUCB — Prompt Template Optimiser",
    fontsize=13, fontweight="bold", y=0.99,
)

TEMPLATE_COLORS = {
    "structured": "#2196F3",
    "creative":   "#FF9800",
    "concise":    "#4CAF50",
    "friendly":   "#E91E63",
    "generic":    "#9E9E9E",
}
QT_COLORS = {
    "technical":      "#1565C0",
    "creative":       "#E65100",
    "factual":        "#2E7D32",
    "conversational": "#880E4F",
}

def add_ckpts(ax):
    for s in checkpoint_steps:
        ax.axvline(s, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

# ── Panel A: Selection heatmap (query_type × template) ───────────────────
ax = axes[0, 0]
matrix = np.array([[selection_matrix[qt][t] for t in TEMPLATES] for qt in QUERY_TYPES], dtype=float)
# Normalise each row to show proportions
row_sums = matrix.sum(axis=1, keepdims=True)
matrix_norm = np.divide(matrix, row_sums, where=row_sums > 0)

im = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(TEMPLATES)))
ax.set_yticks(range(len(QUERY_TYPES)))
ax.set_xticklabels(TEMPLATES, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(QUERY_TYPES, fontsize=9)
ax.set_title("Template selection heatmap — diagonal = learned specialisation", fontsize=10)
plt.colorbar(im, ax=ax, label="Selection share per query type")

# Annotate cells with percentages
for r in range(len(QUERY_TYPES)):
    for c in range(len(TEMPLATES)):
        pct = matrix_norm[r, c] * 100
        color = "white" if matrix_norm[r, c] > 0.6 else "black"
        ax.text(c, r, f"{pct:.0f}%", ha="center", va="center", fontsize=8, color=color)

# Mark the ground-truth optimal template for each query type
for r, qt in enumerate(QUERY_TYPES):
    opt_c = TEMPLATES.index(BEST_TEMPLATE[qt])
    ax.add_patch(plt.Rectangle((opt_c - 0.5, r - 0.5), 1, 1,
                                fill=False, edgecolor="gold", linewidth=2.5))

from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="none", edgecolor="gold", linewidth=2, label="Ground-truth optimal")],
          loc="lower right", fontsize=8)

# ── Panel B: Cumulative reward vs oracle ─────────────────────────────────
ax = axes[0, 1]
oracle_line = [cum_optimal / (i + 1) * (i + 1) for i in range(len(iterations))]
cum_opt_line = [(i + 1) * 0.85 for i in range(len(iterations))]
ax.plot(iterations, cum_opt_line,   label="Oracle optimum",   color="#4CAF50", linestyle="--", linewidth=1.5)
ax.plot(iterations, cum_rewards_log, label="NeuralLinUCB",   color="#2196F3", linewidth=1.5)
ax.fill_between(iterations, cum_rewards_log, cum_opt_line, alpha=0.12, color="#F44336", label="Regret")
add_ckpts(ax)
ax.set_title("Cumulative reward vs oracle", fontsize=10)
ax.set_xlabel("Iteration")
ax.set_ylabel("Cumulative reward")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel C: Rolling reward per query type ────────────────────────────────
ax = axes[1, 0]
for qt in QUERY_TYPES:
    ax.plot(iterations, rolling_reward[qt], label=qt, color=QT_COLORS[qt], linewidth=1.4)
ax.axhline(0.85, color="black", linewidth=0.8, linestyle="--", label="Optimal (0.85)")
ax.axhline(0.50, color="grey",  linewidth=0.8, linestyle=":",  label="Generic  (0.50)")
add_ckpts(ax)
ax.set_title(f"Rolling reward by query type ({WINDOW}-step avg)\ngrey lines = retrain checkpoints", fontsize=10)
ax.set_xlabel("Iteration")
ax.set_ylabel(f"Avg reward (last {WINDOW} steps of that query type)")
ax.set_ylim(0.0, 1.0)
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel D: Cumulative regret ────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(iterations, cum_regret_log, color="#F44336", linewidth=1.5)
add_ckpts(ax)
ax.set_title("Cumulative regret — slope decreases after each retrain", fontsize=10)
ax.set_xlabel("Iteration")
ax.set_ylabel("Cumulative regret")
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(SCRIPT_DIR, "10_neural_linucb_prompt_optimizer.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved → {out}")

# ---------------------------------------------------------------------------
# 4. Print routing table
# ---------------------------------------------------------------------------
print("\n── Learned routing table (% of selections) ─────────────────────────────────")
header = f"{'Query type':<20}" + "".join(f"{t:>13}" for t in TEMPLATES)
print(header)
print("-" * len(header))
for qt in QUERY_TYPES:
    total = sum(selection_matrix[qt].values())
    row = f"{qt:<20}"
    for t in TEMPLATES:
        pct = selection_matrix[qt][t] / max(total, 1) * 100
        star = "*" if BEST_TEMPLATE.get(qt) == t else " "
        row += f"{pct:>11.1f}%{star}"
    print(row)
print("\n* = ground-truth optimal template for that query type")

plt.show()
