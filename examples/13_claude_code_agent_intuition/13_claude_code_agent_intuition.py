"""examples/13_claude_code_agent_intuition.py

Claude Code Agent Intuition — NeuralLinUCB on 256-dim agent-state embeddings.

── The problem ──────────────────────────────────────────────────────────────────────

An AI coding agent (Claude Code, Cursor, Devin) faces a decision before every
non-trivial task: which problem-solving strategy to use first?  Today, every
agent follows the same static ReAct loop regardless of whether it is fixing a
typo or untangling a 10-year-old payment system.  The result is wasted tool
calls, unnecessary file reads, and — most expensively — wrong first moves that
compound into longer, more expensive sessions.

The insight: the *type of task* is legible from the agent's cognitive state at
decision time, even before a single tool is called.  Task description, codebase
familiarity, error complexity, conversation momentum, and attempt history form a
rich context that a neural bandit can learn to map onto effective strategies.

── What the context vector captures ─────────────────────────────────────────────────

In production, the agent assembles a text snapshot of its current cognitive state
and encodes it with a lightweight sentence transformer:

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 22 MB, 384-dim, CPU-fast

    def embed_agent_state(state) -> list[float]:
        text = (
            f"Task: {state.task_description}\\n"
            f"Files read so far: {state.files_read_count} "
            f"({', '.join(state.files_read[:3])})\\n"
            f"Last error: {state.last_error or 'none'}\\n"
            f"Prior attempts: {state.attempt_count}\\n"
            f"User expertise signal: {state.expertise_estimate}\\n"
            f"Codebase familiarity: {state.familiarity_score:.2f}\\n"
            f"Conversation turns: {state.turn_count}\\n"
        )
        return encoder.encode(text, normalize_embeddings=True).tolist()

    # → 384-dim unit vector, fed directly to /predict as `context`

In this simulation, 256-dim unit vectors cluster around archetype centres with
Gaussian noise (σ = 0.25), matching the distributional geometry of real sentence
embeddings.  Drop in the SentenceTransformer call and the rest is unchanged.

── Task archetypes ──────────────────────────────────────────────────────────────────

  simple_bugfix        Clear exception, known file, single-function scope.
  complex_bugfix       Vague error, large blast radius, unfamiliar call chain.
  new_feature          Greenfield scope, clear spec, familiar patterns nearby.
  legacy_refactor      Old code, zero tests, high coupling, high risk.
  unfamiliar_codebase  First time in this repo; no mental model yet.
  pattern_repeat       Same class of fix seen many times before.

── Arms (problem-solving strategies) ───────────────────────────────────────────────

  explore_first       Read broadly before touching anything.
  test_driven         Write a failing test first, then implement.
  minimal_surgical    Smallest possible diff; one function, one file.
  clarify_before_act  Ask the developer a scoped question before acting.
  grep_pattern_match  Find similar solved patterns; adapt and apply.
  full_rearchitect    Redesign the affected area; accept the scope.
  delegate_subagent   Spawn a specialised subagent for this task type.

── The key insight ──────────────────────────────────────────────────────────────────

Developers' most common instinct on a bug — "make the smallest change possible" —
is actively harmful for complex bugs.  The bandit learns this from outcome data:
minimal_surgical on complex_bugfix earns 0.28, explore_first earns 0.85.

No rule can encode this because "complex" is latent in the embedding, not a label
the developer provides.  The neural layer learns to separate task types in the
256-dim embedding space; the LinUCB layer learns per-strategy confidence bounds.

The simulation runs 1 500 decision points — 300 developers × ~5 sessions each.
Each session reward reflects whether the strategy led to an accepted PR without
follow-up revisions (1.0 = accepted clean, 0.0 = reverted or abandoned).

── Requirements ─────────────────────────────────────────────────────────────────────

    pip install requests matplotlib numpy
    cargo build --features neural
    BANDITDB_API_KEY=<key> ./target/debug/banditdb   # server on :8080

Run:
    python examples/13_claude_code_agent_intuition.py
"""

import os
import sys
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DB_URL           = "http://localhost:8080"
API_KEY          = os.getenv("BANDITDB_API_KEY", "")
HEADERS          = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
CAMPAIGN_ID      = "claude_code_agent_intuition"
CONTEXT_DIM      = 256
EMBED_DIM        = 24
N_ITERATIONS     = 1_500    # 300 developers × ~5 decision points each
CHECKPOINT_EVERY = 150
RANDOM_SEED      = 42

ARMS = [
    "explore_first",
    "test_driven",
    "minimal_surgical",
    "clarify_before_act",
    "grep_pattern_match",
    "full_rearchitect",
    "delegate_subagent",
]

ARCHETYPES = [
    "simple_bugfix",
    "complex_bugfix",
    "new_feature",
    "legacy_refactor",
    "unfamiliar_codebase",
    "pattern_repeat",
]

# Ground-truth reward: did this strategy produce an accepted PR without revisions?
# Row = archetype, Col = arm (same order as ARMS list above).
# Hard rows:
#   complex_bugfix:      minimal_surgical=0.28 — instinct is wrong, explore=0.85
#   unfamiliar_codebase: jumping straight to code (minimal/rearchitect) fails badly
#   legacy_refactor:     surgical change leaves hidden coupling bugs (0.22)
REWARD_BASE = {
    "simple_bugfix": {
        "explore_first":     0.40,
        "test_driven":       0.62,
        "minimal_surgical":  0.88,
        "clarify_before_act":0.30,
        "grep_pattern_match":0.65,
        "full_rearchitect":  0.18,
        "delegate_subagent": 0.35,
    },
    "complex_bugfix": {
        "explore_first":     0.85,
        "test_driven":       0.52,
        "minimal_surgical":  0.28,
        "clarify_before_act":0.48,
        "grep_pattern_match":0.42,
        "full_rearchitect":  0.62,
        "delegate_subagent": 0.72,
    },
    "new_feature": {
        "explore_first":     0.45,
        "test_driven":       0.85,
        "minimal_surgical":  0.30,
        "clarify_before_act":0.55,
        "grep_pattern_match":0.40,
        "full_rearchitect":  0.42,
        "delegate_subagent": 0.52,
    },
    "legacy_refactor": {
        "explore_first":     0.60,
        "test_driven":       0.38,
        "minimal_surgical":  0.22,
        "clarify_before_act":0.48,
        "grep_pattern_match":0.42,
        "full_rearchitect":  0.85,
        "delegate_subagent": 0.62,
    },
    "unfamiliar_codebase": {
        "explore_first":     0.52,
        "test_driven":       0.28,
        "minimal_surgical":  0.18,
        "clarify_before_act":0.88,
        "grep_pattern_match":0.45,
        "full_rearchitect":  0.20,
        "delegate_subagent": 0.62,
    },
    "pattern_repeat": {
        "explore_first":     0.28,
        "test_driven":       0.45,
        "minimal_surgical":  0.58,
        "clarify_before_act":0.18,
        "grep_pattern_match":0.88,
        "full_rearchitect":  0.22,
        "delegate_subagent": 0.32,
    },
}

# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------
linucb_bytes = (CONTEXT_DIM ** 2 + 2 * CONTEXT_DIM) * 8 * len(ARMS)
neural_bytes = (EMBED_DIM   ** 2 + 2 * EMBED_DIM)   * 8 * len(ARMS)
net_bytes    = (CONTEXT_DIM * 64 + 64 +
                64 * 64 + 64 +
                64 * EMBED_DIM + EMBED_DIM) * 4
neural_total = neural_bytes + net_bytes

print("── Memory footprint ─────────────────────────────────────────────────────────")
print(f"  Context dim:    {CONTEXT_DIM}   (256-dim sentence-transformer embedding)")
print(f"  Embed dim:      {EMBED_DIM}   (neural compression)")
print(f"  Arms:           {len(ARMS)}")
print(f"  LinUCB (naive): {linucb_bytes / 1024 / 1024:.1f} MB")
print(f"  NeuralLinUCB:   {neural_total / 1024:.0f} KB   ({linucb_bytes // neural_total}× smaller)")
print()

# ---------------------------------------------------------------------------
# Context generation — simulates sentence-transformer output per archetype.
# Each archetype clusters around a random unit-vector centre in 256-dim space.
# σ = 0.25 matches real intra-archetype embedding variability.
# ---------------------------------------------------------------------------
np_rng = np.random.default_rng(RANDOM_SEED)
rng    = random.Random(RANDOM_SEED)

ARCHETYPE_CENTERS = {a: np_rng.standard_normal(CONTEXT_DIM) for a in ARCHETYPES}
for a in ARCHETYPES:
    ARCHETYPE_CENTERS[a] /= np.linalg.norm(ARCHETYPE_CENTERS[a])

def make_context(archetype: str) -> list:
    raw = ARCHETYPE_CENTERS[archetype] + np_rng.standard_normal(CONTEXT_DIM) * 0.25
    return (raw / np.linalg.norm(raw)).tolist()

def oracle_reward(arm: str, archetype: str) -> float:
    base = REWARD_BASE[archetype][arm]
    return max(0.0, min(1.0, base + rng.gauss(0, 0.08)))

def expected_optimal(archetype: str) -> float:
    return max(REWARD_BASE[archetype].values())

# ---------------------------------------------------------------------------
# Create campaign
# ---------------------------------------------------------------------------
resp = requests.post(f"{DB_URL}/campaign", headers=HEADERS, json={
    "campaign_id": CAMPAIGN_ID,
    "arms":        ARMS,
    "feature_dim": 0,
    "alpha":       1.0,
    "algorithm": {
        "neural_lin_ucb": {
            "context_dim":   CONTEXT_DIM,
            "embed_dim":     EMBED_DIM,
            "hidden_dim":    64,
            "hidden_layers": 2,
            "retrain_every": CHECKPOINT_EVERY,
            "retrain_steps": 200,
            "learning_rate": 0.001,
            "lambda":        1.0,
        }
    },
    "metadata": {
        "use_case":        "ai_agent_strategy_selection",
        "embedding_model": "all-MiniLM-L6-v2",
        "context_source":  "agent_cognitive_state_at_decision_point",
        "reward_signal":   "pr_accepted_without_revision",
    },
})

if resp.status_code == 422:
    print("ERROR: NeuralLinUCB not compiled in.")
    print("Rebuild with:  cargo build --features neural")
    sys.exit(1)
elif resp.status_code not in (200, 201, 409):
    print(f"Campaign creation failed {resp.status_code}: {resp.text}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Simulation loop — 1 500 decision points
# ---------------------------------------------------------------------------
selection_matrix = {a: {arm: 0 for arm in ARMS} for a in ARCHETYPES}

# For the "unlearning the patch instinct" panel:
# Track minimal_surgical vs explore_first rates on complex_bugfix specifically
complex_minimal_counts  = 0
complex_explore_counts  = 0
complex_total           = 0
complex_minimal_rate    = []
complex_explore_rate    = []
INSIGHT_WINDOW          = 100

cum_reward      = 0.0
cum_optimal     = 0.0
cum_reward_log  = []
cum_regret_log  = []
iterations      = []
checkpoint_steps = []

WINDOW = 100
reward_window  = {a: [] for a in ARCHETYPES}
rolling_reward = {a: [] for a in ARCHETYPES}

print(f"Running {N_ITERATIONS} iterations "
      f"({N_ITERATIONS // 5} simulated developers × ~5 sessions each)")
print(f"Context: {CONTEXT_DIM}-dim sentence-transformer embeddings")
print(f"Checkpoint + neural retrain every {CHECKPOINT_EVERY} steps")
print()
print(f"{'Step':>6}  {'Event':^12}  {'Cum regret':>12}  "
      f"{'minimal/complex':>16}  {'explore/complex':>16}")
print("─" * 72)

for i in range(1, N_ITERATIONS + 1):
    archetype = rng.choice(ARCHETYPES)
    context   = make_context(archetype)

    resp = requests.post(f"{DB_URL}/predict", headers=HEADERS,
                         json={"campaign_id": CAMPAIGN_ID, "context": context})
    body = resp.json()
    arm  = body["arm_id"]
    iid  = body["interaction_id"]

    r = oracle_reward(arm, archetype)
    requests.post(f"{DB_URL}/reward", headers=HEADERS,
                  json={"interaction_id": iid, "reward": round(r, 4)})

    selection_matrix[archetype][arm] += 1

    if archetype == "complex_bugfix":
        complex_total          += 1
        complex_minimal_counts += (arm == "minimal_surgical")
        complex_explore_counts += (arm == "explore_first")

    complex_minimal_rate.append(complex_minimal_counts / max(complex_total, 1))
    complex_explore_rate.append(complex_explore_counts / max(complex_total, 1))

    for a in ARCHETYPES:
        reward_window[a].append(r if a == archetype else None)
        vals = [v for v in reward_window[a][-WINDOW:] if v is not None]
        rolling_reward[a].append(sum(vals) / len(vals) if vals else 0.0)

    cum_reward  += r
    cum_optimal += expected_optimal(archetype)
    iterations.append(i)
    cum_reward_log.append(cum_reward)
    cum_regret_log.append(cum_optimal - cum_reward)

    if i % CHECKPOINT_EVERY == 0:
        requests.post(f"{DB_URL}/checkpoint", headers=HEADERS)
        checkpoint_steps.append(i)
        m_rate = complex_minimal_counts / max(complex_total, 1)
        e_rate = complex_explore_counts / max(complex_total, 1)
        print(f"{i:>6}  {'checkpoint':^12}  "
              f"{cum_optimal - cum_reward:>12.1f}  "
              f"{m_rate:>15.1%}  {e_rate:>15.1%}")

print()
print("Done.  Rendering charts …")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
ARM_COLORS = {
    "explore_first":      "#2196F3",
    "test_driven":        "#4CAF50",
    "minimal_surgical":   "#FF9800",
    "clarify_before_act": "#9C27B0",
    "grep_pattern_match": "#00BCD4",
    "full_rearchitect":   "#F44336",
    "delegate_subagent":  "#607D8B",
}
ARCHETYPE_COLORS = {
    "simple_bugfix":        "#4CAF50",
    "complex_bugfix":       "#F44336",
    "new_feature":          "#2196F3",
    "legacy_refactor":      "#FF9800",
    "unfamiliar_codebase":  "#9C27B0",
    "pattern_repeat":       "#00BCD4",
}

def add_ckpts(ax):
    for s in checkpoint_steps:
        ax.axvline(s, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

def smoothed(series, w=80):
    return [sum(series[max(0, k - w):k + 1]) / min(k + 1, w)
            for k in range(len(series))]

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    "BanditDB · NeuralLinUCB — Claude Code Agent Intuition\n"
    f"256-dim sentence-transformer embeddings · {len(ARMS)} strategies · "
    f"{N_ITERATIONS:,} decision points ({N_ITERATIONS // 5} developers)",
    fontsize=12, fontweight="bold", y=0.99,
)

# ── Panel A: Strategy routing heatmap ────────────────────────────────────────
ax = axes[0, 0]

matrix   = np.array(
    [[selection_matrix[a][arm] for arm in ARMS] for a in ARCHETYPES], dtype=float
)
row_sums = matrix.sum(axis=1, keepdims=True)
norm_m   = np.divide(matrix, row_sums, where=row_sums > 0)

optimal_arm = {a: max(REWARD_BASE[a], key=REWARD_BASE[a].get) for a in ARCHETYPES}

cmap = LinearSegmentedColormap.from_list("bdb", ["#18181b", "#1e3a5f", "#2196F3"])
im   = ax.imshow(norm_m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ARMS)))
ax.set_yticks(range(len(ARCHETYPES)))
ax.set_xticklabels([a.replace("_", "\n") for a in ARMS], fontsize=8)
ax.set_yticklabels([a.replace("_", " ") for a in ARCHETYPES], fontsize=9)
ax.set_title("Strategy routing heatmap\ngold border = ground-truth optimal strategy", fontsize=10)
plt.colorbar(im, ax=ax, label="Selection share per task archetype")

for r, a in enumerate(ARCHETYPES):
    for c, arm in enumerate(ARMS):
        pct   = norm_m[r, c] * 100
        color = "white" if norm_m[r, c] > 0.5 else "#a1a1aa"
        ax.text(c, r, f"{pct:.0f}%", ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")
    opt_c = ARMS.index(optimal_arm[a])
    ax.add_patch(plt.Rectangle((opt_c - 0.5, r - 0.5), 1, 1,
                                fill=False, edgecolor="gold", linewidth=2.5))

# ── Panel B: Cumulative reward vs oracle ─────────────────────────────────────
ax = axes[0, 1]
mean_opt  = sum(expected_optimal(a) for a in ARCHETYPES) / len(ARCHETYPES)
oracle_log = [(i + 1) * mean_opt for i in range(len(iterations))]

ax.plot(iterations, oracle_log,     label="Oracle (always optimal)",
        color="#4CAF50", linestyle="--", linewidth=1.5)
ax.plot(iterations, cum_reward_log, label="NeuralLinUCB",
        color="#2196F3", linewidth=1.5)
ax.fill_between(iterations, cum_reward_log, oracle_log,
                alpha=0.15, color="#F44336", label="Regret")
add_ckpts(ax)
ax.set_title("Cumulative reward vs oracle\ngrey lines = neural retrain checkpoints", fontsize=10)
ax.set_xlabel("Decision point")
ax.set_ylabel("Cumulative reward (accepted PRs)")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel C: "Unlearning the patch instinct on complex bugs" ─────────────────
ax = axes[1, 0]
sm_minimal = smoothed(complex_minimal_rate, w=INSIGHT_WINDOW)
sm_explore = smoothed(complex_explore_rate, w=INSIGHT_WINDOW)
baseline   = 1 / len(ARMS)

ax.plot(iterations, sm_explore, label="explore_first  (optimal: 0.85 reward)",
        color=ARM_COLORS["explore_first"],     linewidth=2.0)
ax.plot(iterations, sm_minimal, label="minimal_surgical  (wrong: 0.28 reward)",
        color=ARM_COLORS["minimal_surgical"],  linewidth=2.0, linestyle="--")
ax.axhline(baseline, color="grey", linewidth=0.8, linestyle=":",
           label=f"Random baseline ({baseline:.0%})")
add_ckpts(ax)
ax.set_title(
    "\"Unlearning the patch instinct\" — complex_bugfix only\n"
    "Model learns explore_first beats minimal_surgical on complex bugs",
    fontsize=10,
)
ax.set_xlabel("Decision point")
ax.set_ylabel(f"Selection rate ({INSIGHT_WINDOW}-step rolling avg)")
ax.set_ylim(0, 0.75)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel D: Per-archetype rolling reward (convergence) ──────────────────────
ax = axes[1, 1]
for a in ARCHETYPES:
    sm = smoothed(rolling_reward[a], w=WINDOW)
    opt = expected_optimal(a)
    ax.plot(iterations, sm,
            label=f"{a.replace('_', ' ')} (opt={opt:.2f})",
            color=ARCHETYPE_COLORS[a], linewidth=1.5)

add_ckpts(ax)
ax.set_title(f"Per-archetype rolling reward ({WINDOW}-step window)\n"
             "All archetypes should converge toward their optimal", fontsize=10)
ax.set_xlabel("Decision point")
ax.set_ylabel("Rolling avg reward")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "13_claude_code_agent_intuition.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved → {out}")
plt.show()
