"""examples/12_neural_linucb_banking_financial_health.py

Proactive Financial Health Interventions — NeuralLinUCB on 768-dim transaction embeddings.

── The problem ──────────────────────────────────────────────────────────────────────

Banks send financial nudges (overdraft alerts, savings prompts, debt consolidation
suggestions) based on rigid rules: "if balance < 200 → send overdraft warning".
Rules are context-blind: the same message goes to a customer who is structurally
cash-stressed and one who just had an unusually large one-off payment. The result
is messages that are tone-deaf, ignored, or — in the worst case — drive churn.

── Why 768 dimensions ───────────────────────────────────────────────────────────────

The context is the embedding of a customer's last 30 transaction descriptions —
the text of what they actually spent money on — run through a financial language
model (FinBERT / financial sentence encoder). This produces a 768-dim vector that
captures spending patterns, stress signals, and momentum in a form no hand-crafted
feature set can fully replicate.

In production make_context() would be replaced by:

    from transformers import pipeline
    encoder = pipeline("feature-extraction", model="yiyanghkust/finbert-tone")

    def embed_transactions(transactions: list[str]) -> list[float]:
        text = " | ".join(transactions[-30:])
        embedding = encoder(text)[0][0]          # CLS token, 768-dim
        norm = sum(x**2 for x in embedding)**0.5
        return [x / norm for x in embedding]

Typical transaction narratives by financial state:

  cash_stressed:         "Payday advance £200 | Late fee £35 | ATM £40 |
                          Utility direct debit returned | Uber Eats £8.50 | ..."
  saving_momentum:       "Auto-transfer to savings £150 | Meal prep box £42 |
                          Gym cancellation | Library | Spotify cancel | ..."
  debt_heavy:            "Min payment Barclaycard £25 | Min payment HSBC £30 |
                          Cash advance £100 | Interest charge £18 | ..."
  financially_comfortable: "Vanguard ISA £500 | Waitrose £120 | Flight LHR-JFK |
                             Restaurant £85 | Salary credit £4,800 | ..."

── Arms ─────────────────────────────────────────────────────────────────────────────

  savings_nudge             "You've spent less than usual this month. Move £50 to
                             savings?" — reinforces momentum; tone-deaf when stressed.
  overdraft_warning         "Your balance may drop below £0 in 3 days based on your
                             patterns." — high value for stressed customers; alarming
                             and confusing for comfortable ones.
  debt_consolidation_prompt "You're paying interest on 3 cards. A consolidation loan
                             could save you £180/year." — high value for debt-heavy;
                             deeply wrong for savers.
  spend_insight             "Your dining spend is up 40% vs your 3-month average." —
                             useful awareness across states, never the best choice.
  investment_prompt         "You have £800 sitting in your current account. Consider
                             a cash ISA." — excellent for comfortable; predatory-feeling
                             for stressed customers.
  no_message                Stay silent. The bandit must learn when NOT to interrupt.

── The innovation ───────────────────────────────────────────────────────────────────

The reward for sending investment_prompt to a cash_stressed customer is 0.03 —
it is actively harmful. The reward for staying silent (no_message) with the same
customer is 0.28 — better than four of the five active arms.

A rule-based system cannot learn this. A standard ML model trained offline cannot
adapt in real time as customer states shift. NeuralLinUCB operating on transaction
embeddings learns the abstain boundary contextually — without being told which
customers to leave alone.

This is the genuine gap: banks have credit decisioning models (LogReg, GBM) but
nothing that closes the feedback loop on behavioural nudges in real time.

── Regulatory note ──────────────────────────────────────────────────────────────────

BanditDB logs softmax propensity scores at every predict call. These are the audit
trail for "why did this customer receive this message" — directly usable for fair
treatment reviews and FCA Consumer Duty reporting.

Requirements:
    pip install requests matplotlib numpy
    cargo build --features neural
    ./target/debug/banditdb    # server on :8080

Run:
    python examples/12_neural_linucb_banking_financial_health.py
"""

import os
import sys
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DB_URL           = "http://localhost:8080"
CAMPAIGN_ID      = "financial_health_nudge"
CONTEXT_DIM      = 768
EMBED_DIM        = 32
N_ITERATIONS     = 3_000
CHECKPOINT_EVERY = 300
RANDOM_SEED      = 17

ARMS = [
    "savings_nudge",
    "overdraft_warning",
    "debt_consolidation_prompt",
    "spend_insight",
    "investment_prompt",
    "no_message",
]

STATES = [
    "cash_stressed",
    "saving_momentum",
    "debt_heavy",
    "financially_comfortable",
]

# ---------------------------------------------------------------------------
# 30-day behavioural change score (0–1).
# Captures: did the customer take the prompted action, not churn,
# and show improved financial health indicators within 30 days.
#
# The hard row is cash_stressed: investment_prompt scores 0.03 (predatory),
# no_message scores 0.28 (at least does no harm). The bandit must discover
# this without being told — the only signal is the 768-dim context and the
# reward it eventually receives.
# ---------------------------------------------------------------------------
REWARD_BASE = {
    "cash_stressed": {
        "overdraft_warning":          0.72,
        "no_message":                 0.28,
        "debt_consolidation_prompt":  0.38,
        "spend_insight":              0.22,
        "savings_nudge":              0.09,
        "investment_prompt":          0.03,
    },
    "saving_momentum": {
        "savings_nudge":              0.82,
        "investment_prompt":          0.58,
        "spend_insight":              0.48,
        "no_message":                 0.15,
        "debt_consolidation_prompt":  0.12,
        "overdraft_warning":          0.08,
    },
    "debt_heavy": {
        "debt_consolidation_prompt":  0.80,
        "overdraft_warning":          0.44,
        "spend_insight":              0.38,
        "no_message":                 0.18,
        "savings_nudge":              0.14,
        "investment_prompt":          0.04,
    },
    "financially_comfortable": {
        "investment_prompt":          0.76,
        "savings_nudge":              0.62,
        "spend_insight":              0.44,
        "no_message":                 0.22,
        "overdraft_warning":          0.06,
        "debt_consolidation_prompt":  0.07,
    },
}

# ---------------------------------------------------------------------------
# Memory footprint — print before the simulation starts
# ---------------------------------------------------------------------------
linucb_bytes  = (CONTEXT_DIM ** 2 + 2 * CONTEXT_DIM) * 8 * len(ARMS)
neural_bytes  = (EMBED_DIM   ** 2 + 2 * EMBED_DIM)   * 8 * len(ARMS)
network_bytes = (CONTEXT_DIM * 128 + 128 +
                 128 * 128    + 128 +
                 128 * EMBED_DIM + EMBED_DIM) * 4   # f32 weights
neural_total  = neural_bytes + network_bytes

print("── Memory footprint ─────────────────────────────────────────────────────────")
print(f"  Context dim:       {CONTEXT_DIM}   (768-dim FinBERT embedding)")
print(f"  Arms:              {len(ARMS)}")
print(f"  LinUCB state:      {linucb_bytes / 1024 / 1024:.1f} MB  "
      f"({CONTEXT_DIM}×{CONTEXT_DIM}×{len(ARMS)} matrices)")
print(f"  NeuralLinUCB:      {neural_total / 1024:.0f} KB  "
      f"(32×32×{len(ARMS)} + MLP weights)")
print(f"  Reduction:         {linucb_bytes / neural_total:.0f}×")
print()

# ---------------------------------------------------------------------------
# Context generation — simulates FinBERT embeddings of transaction narratives.
# Each financial state clusters around a random centre in 768-dim space.
# Noise std ≈ 0.25 reflects real intra-state transaction variability.
# ---------------------------------------------------------------------------
np_rng = np.random.default_rng(RANDOM_SEED)
rng    = random.Random(RANDOM_SEED)

SEGMENT_CENTERS = {s: np_rng.standard_normal(CONTEXT_DIM) for s in STATES}
for s in STATES:
    SEGMENT_CENTERS[s] /= np.linalg.norm(SEGMENT_CENTERS[s])

def make_context(state: str) -> list:
    raw = SEGMENT_CENTERS[state] + np_rng.standard_normal(CONTEXT_DIM) * 0.25
    return (raw / np.linalg.norm(raw)).tolist()

def oracle_reward(arm: str, state: str) -> float:
    base = REWARD_BASE[state][arm]
    return max(0.0, min(1.0, base + rng.gauss(0, 0.08)))

def expected_optimal(state: str) -> float:
    return max(REWARD_BASE[state].values())

# ---------------------------------------------------------------------------
# Create campaign
# ---------------------------------------------------------------------------
resp = requests.post(f"{DB_URL}/campaign", json={
    "campaign_id": CAMPAIGN_ID,
    "arms":        ARMS,
    "feature_dim": 0,
    "alpha":       1.0,
    "algorithm": {
        "neural_lin_ucb": {
            "context_dim":   CONTEXT_DIM,
            "embed_dim":     EMBED_DIM,
            "hidden_dim":    128,
            "hidden_layers": 2,
            "retrain_every": CHECKPOINT_EVERY,
            "retrain_steps": 150,
            "learning_rate": 0.001,
            "lambda":        1.0,
        }
    },
    "metadata": {
        "channel":             "mobile_push",
        "regulatory_context":  "financial_wellness",
        "message_frequency_cap": 3,
        "embedding_model":     "yiyanghkust/finbert-tone",
    },
})

if resp.status_code == 422:
    print("ERROR: Server rejected NeuralLinUCB campaign.")
    print("Compile with:  cargo build --features neural")
    sys.exit(1)
elif resp.status_code not in (200, 201, 409):
    print(f"Unexpected status {resp.status_code}: {resp.text}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
selection_matrix = {s: {a: 0 for a in ARMS} for s in STATES}

# For the "silent when stressed" insight: track no_message selection
# specifically for cash_stressed vs all other states
nomsg_counts  = {"cash_stressed": 0, "other": 0}
nomsg_totals  = {"cash_stressed": 0, "other": 0}
nomsg_rate_cs = []   # rolling no_message rate for cash_stressed
nomsg_rate_ot = []   # rolling no_message rate for others
NM_WINDOW     = 150

cum_reward      = 0.0
cum_optimal     = 0.0
cum_reward_log  = []
cum_regret_log  = []
checkpoint_steps = []
iterations      = []

WINDOW = 100
reward_window  = {s: [] for s in STATES}
rolling_reward = {s: [] for s in STATES}

print(f"Running {N_ITERATIONS} iterations — {CONTEXT_DIM}-dim context — "
      f"checkpoint every {CHECKPOINT_EVERY} steps …")
print(f"{'Iter':>6}  {'Checkpoint':^12}  {'Cum regret':>12}  "
      f"{'no_msg (stressed)':>18}  {'no_msg (other)':>14}")
print("─" * 72)

for i in range(1, N_ITERATIONS + 1):
    state   = rng.choice(STATES)
    context = make_context(state)

    resp = requests.post(f"{DB_URL}/predict",
                         json={"campaign_id": CAMPAIGN_ID, "context": context})
    body  = resp.json()
    arm   = body["arm_id"]
    iid   = body["interaction_id"]

    r = oracle_reward(arm, state)
    requests.post(f"{DB_URL}/reward",
                  json={"interaction_id": iid, "reward": round(r, 4)})

    # Track selection matrix
    selection_matrix[state][arm] += 1

    # Track no_message behaviour
    key = "cash_stressed" if state == "cash_stressed" else "other"
    nomsg_totals[key] += 1
    if arm == "no_message":
        nomsg_counts[key] += 1

    # Rolling no_message rates — one data point per iteration, keyed by current state
    nomsg_rate_cs.append(nomsg_counts["cash_stressed"] / max(nomsg_totals["cash_stressed"], 1))
    nomsg_rate_ot.append(nomsg_counts["other"]         / max(nomsg_totals["other"],         1))

    # Rolling reward per state
    for s in STATES:
        reward_window[s].append(r if s == state else None)
        vals = [v for v in reward_window[s][-WINDOW:] if v is not None]
        rolling_reward[s].append(sum(vals) / len(vals) if vals else 0.0)

    cum_reward  += r
    cum_optimal += expected_optimal(state)
    iterations.append(i)
    cum_reward_log.append(cum_reward)
    cum_regret_log.append(cum_optimal - cum_reward)

    if i % CHECKPOINT_EVERY == 0:
        requests.post(f"{DB_URL}/checkpoint")
        checkpoint_steps.append(i)
        cs_rate = nomsg_counts["cash_stressed"] / max(nomsg_totals["cash_stressed"], 1)
        ot_rate = nomsg_counts["other"]         / max(nomsg_totals["other"],         1)
        print(f"{i:>6}  {'checkpoint':^12}  "
              f"{cum_optimal - cum_reward:>12.1f}  "
              f"{cs_rate:>17.1%}  {ot_rate:>13.1%}")

print()
print("Done.  Rendering charts …")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
ARM_COLORS = {
    "savings_nudge":             "#4CAF50",
    "overdraft_warning":         "#F44336",
    "debt_consolidation_prompt": "#FF9800",
    "spend_insight":             "#9C27B0",
    "investment_prompt":         "#2196F3",
    "no_message":                "#607D8B",
}
STATE_COLORS = {
    "cash_stressed":          "#F44336",
    "saving_momentum":        "#4CAF50",
    "debt_heavy":             "#FF9800",
    "financially_comfortable":"#2196F3",
}

def add_ckpts(ax):
    for s in checkpoint_steps:
        ax.axvline(s, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

def smoothed(series, w=100):
    return [sum(series[max(0, k - w):k + 1]) / min(k + 1, w)
            for k in range(len(series))]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "BanditDB · NeuralLinUCB — Proactive Financial Health Interventions\n"
    f"768-dim transaction embeddings · {len(ARMS)} message arms · {N_ITERATIONS:,} customers",
    fontsize=12, fontweight="bold", y=0.99,
)

# ── Panel A: Routing heatmap ──────────────────────────────────────────────
ax = axes[0, 0]

# Normalise each row to selection share
matrix = np.array(
    [[selection_matrix[s][a] for a in ARMS] for s in STATES], dtype=float
)
row_sums = matrix.sum(axis=1, keepdims=True)
norm_m   = np.divide(matrix, row_sums, where=row_sums > 0)

# Highlight the ground-truth optimal arm per state
optimal_arm = {s: max(REWARD_BASE[s], key=REWARD_BASE[s].get) for s in STATES}

cmap = LinearSegmentedColormap.from_list("bdb", ["#18181b", "#1e3a5f", "#2196F3"])
im   = ax.imshow(norm_m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ARMS)))
ax.set_yticks(range(len(STATES)))
ax.set_xticklabels([a.replace("_", "\n") for a in ARMS], fontsize=8)
ax.set_yticklabels([s.replace("_", " ") for s in STATES], fontsize=9)
ax.set_title("Message routing heatmap\ngold border = ground-truth optimal arm", fontsize=10)
plt.colorbar(im, ax=ax, label="Selection share per customer state")

for r, s in enumerate(STATES):
    for c, a in enumerate(ARMS):
        pct   = norm_m[r, c] * 100
        color = "white" if norm_m[r, c] > 0.5 else "#a1a1aa"
        ax.text(c, r, f"{pct:.0f}%", ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")
    opt_c = ARMS.index(optimal_arm[s])
    ax.add_patch(plt.Rectangle((opt_c - 0.5, r - 0.5), 1, 1,
                                fill=False, edgecolor="gold", linewidth=2.5))

# ── Panel B: Cumulative reward vs oracle ─────────────────────────────────
ax = axes[0, 1]
oracle_log = [(i + 1) * sum(expected_optimal(s) for s in STATES) / len(STATES)
              for i in range(len(iterations))]
ax.plot(iterations, oracle_log,     label="Oracle (always optimal)",
        color="#4CAF50", linestyle="--", linewidth=1.5)
ax.plot(iterations, cum_reward_log, label="NeuralLinUCB",
        color="#2196F3", linewidth=1.5)
ax.fill_between(iterations, cum_reward_log, oracle_log,
                alpha=0.15, color="#F44336", label="Regret")
add_ckpts(ax)
ax.set_title("Cumulative reward vs oracle\ngrey lines = retrain checkpoints", fontsize=10)
ax.set_xlabel("Customer")
ax.set_ylabel("Cumulative reward")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel C: "Silent when stressed" — no_message selection by state ───────
ax = axes[1, 0]
sm_cs = smoothed(nomsg_rate_cs, w=NM_WINDOW)
sm_ot = smoothed(nomsg_rate_ot, w=NM_WINDOW)

ax.plot(iterations, sm_cs, label="cash_stressed customers",
        color=STATE_COLORS["cash_stressed"], linewidth=1.8)
ax.plot(iterations, sm_ot, label="all other states",
        color="#607D8B", linewidth=1.5, linestyle="--")
ax.axhline(1 / len(ARMS), color="grey", linewidth=0.8, linestyle=":",
           label=f"Random baseline ({1/len(ARMS):.0%})")
add_ckpts(ax)
ax.set_title(
    "\"Silent when stressed\" — no_message selection rate\n"
    "Model learns to abstain for cash-stressed customers",
    fontsize=10,
)
ax.set_xlabel("Customer")
ax.set_ylabel(f"no_message rate ({NM_WINDOW}-step rolling avg)")
ax.set_ylim(0, 0.65)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel D: Rolling reward per financial state ────────────────────────────
ax = axes[1, 1]
for s in STATES:
    optimal = expected_optimal(s)
    ax.plot(iterations, rolling_reward[s],
            label=f"{s.replace('_', ' ')} (opt={optimal:.2f})",
            color=STATE_COLORS[s], linewidth=1.5)
add_ckpts(ax)
ax.set_title(
    f"Rolling reward by customer state ({WINDOW}-step avg)\n"
    "Convergence toward per-state optimal after each retrain",
    fontsize=10,
)
ax.set_xlabel("Customer")
ax.set_ylabel(f"Avg reward (last {WINDOW} of that state)")
ax.set_ylim(0.0, 1.0)
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(SCRIPT_DIR, "12_neural_linucb_banking_financial_health.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved → {out}")

# ---------------------------------------------------------------------------
# Routing table + key insight
# ---------------------------------------------------------------------------
print("\n── Learned routing table ────────────────────────────────────────────────────")
arm_labels = [a[:12].ljust(12) for a in ARMS]
print(f"{'State':<26}" + "".join(f"{l:>14}" for l in arm_labels))
print("─" * (26 + 14 * len(ARMS)))
for s in STATES:
    total = sum(selection_matrix[s].values())
    row   = f"{s.replace('_', ' '):<26}"
    for a in ARMS:
        pct  = selection_matrix[s][a] / max(total, 1) * 100
        star = "★" if a == optimal_arm[s] else " "
        row += f"{pct:>12.1f}%{star}"
    print(row)
print("\n★ = ground-truth optimal arm for that customer state")

cs_final = nomsg_counts["cash_stressed"] / max(nomsg_totals["cash_stressed"], 1)
ot_final = nomsg_counts["other"]         / max(nomsg_totals["other"],         1)
print(f"\n── \"Silent when stressed\" ───────────────────────────────────────────────────")
print(f"  no_message rate — cash_stressed:   {cs_final:.1%}")
print(f"  no_message rate — all other states:{ot_final:.1%}")
print(f"  (random baseline: {1/len(ARMS):.1%})")
if cs_final > ot_final * 1.3:
    print("  ✓ Model learned to stay silent more often for stressed customers")
else:
    print("  (more iterations or retrains needed to see full separation)")

print(f"\n── Memory ───────────────────────────────────────────────────────────────────")
print(f"  LinUCB at {CONTEXT_DIM}-dim:   {linucb_bytes / 1024 / 1024:.1f} MB")
print(f"  NeuralLinUCB:       {neural_total / 1024:.0f} KB   ({linucb_bytes / neural_total:.0f}× smaller)")

plt.show()
