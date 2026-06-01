"""examples/14_neural_linucb_churn_prevention.py

B2B SaaS Churn Prevention — NeuralLinUCB on 1024-dim Voyage-3 embeddings.

── The problem ──────────────────────────────────────────────────────────────────

Every B2B SaaS churn tool today does the same thing: score risk, fire a playbook.
Gainsight, ChurnZero, Totango — they all hand high-risk accounts to static rule
engines.  "Risk > 0.7 → send discount email."  The playbooks do not learn.

The result: a new user who is confused gets a discount they don't need.  A
healthy power user gets an unsolicited CSM call that signals you're worried.
A disengaged enterprise account gets a feature tour when what they need is an
executive conversation.

The critical gap between every commercial churn tool and this demo: they predict
risk, but do not optimise which intervention to use for which customer.

── The context: multimodal Voyage-3 embedding ───────────────────────────────────

In production, replace make_context() with the following.  Fuse usage signals,
recent email/Slack excerpts, and support history into a single text string, then
embed with Voyage-3 (1024-dim):

    import voyageai
    vc = voyageai.Client()  # VOYAGE_API_KEY

    def embed_customer_signals(customer: dict) -> list[float]:
        text = (
            f"Usage: {customer['usage_summary']}\\n"
            f"Days since last login: {customer['days_since_login']}\\n"
            f"Feature adoption: {customer['feature_adoption_pct']:.0%}\\n"
            f"Recent email: {customer['last_email_snippet']}\\n"
            f"Support: {customer['support_summary']}\\n"
            f"NPS score: {customer['nps_score']}\\n"
            f"Tier: {customer['tier']}, ARR: ${customer['arr']:,}\\n"
            f"Tenure: {customer['tenure_months']} months\\n"
        )
        result = vc.embed([text], model="voyage-3", input_type="document")
        return result.embeddings[0]  # 1024-dim unit vector

Typical text per segment (what Voyage-3 receives):

  power_user:
    "Usage: 47 API calls/day, all 3 core workflows active. Days since last
     login: 0. Feature adoption: 89%. Recent email: 'Thanks for the quick
     support — you guys are the best.' Support: 0 tickets last 30 days.
     NPS: 9. Enterprise, $85k ARR, 24 months."

  disengaged_enterprise:
    "Usage: weekly logins, down from daily 6 months ago. Days since last
     login: 12. Feature adoption: 23% (was 67%). Recent email: 'We've been
     busy internally, haven't had bandwidth to dig in.' Support: 1 unresolved
     ticket 45 days ago. NPS: 7 (was 9). Enterprise, $120k ARR, 36 months."

  new_user_confused:
    "Usage: 3 logins total. Days since last login: 4. Feature adoption: 8%,
     completed onboarding step 1 of 5. Recent email: 'I'm not sure where to
     start — the dashboard is a bit overwhelming.' Support: 2 tickets: How do
     I connect my data source? / What is a campaign? NPS: 6. Starter, $2.4k
     ARR, 1 month."

  price_sensitive:
    "Usage: active, 35% feature adoption. Days since last login: 1. Opened
     pricing page 4 times last week. Recent email: 'We're evaluating options
     at renewal. Can you match competitor pricing?' Support: 0 tickets.
     NPS: 6. Growth, $18k ARR, 8 months."

  at_risk_churning:
    "Usage: down 78% last 30 days. Days since last login: 18. Recent email:
     'We've decided to move in a different direction. Can you help with data
     export?' Support: 1 ticket — How do I export my data? NPS: 4.
     Growth, $24k ARR, 14 months."

── Arms ─────────────────────────────────────────────────────────────────────────

  no_action         Stay silent. Critical for power users — unsolicited
                    outreach signals you're worried, which creates doubt.

  discount_offer    20% off next 3 months. Direct answer to price concerns.
                    Wasteful (and slightly insulting) for healthy accounts.

  csm_call          Human CSM outreach within 24 hours. High-touch, expensive.
                    Powerful for at-risk accounts; intrusive for healthy ones.

  feature_tour      Personalized 5-email sequence on underused features.
                    Perfect for new users who don't know the product.
                    Misses the point entirely for price-sensitive accounts.

  executive_qbr     VP/C-level Quarterly Business Review invite. Signals
                    strategic partnership. Essential for disengaged enterprise;
                    overkill for a confused new user or a $2k ARR account.

── The key insights the bandit must discover without being told ──────────────────

  1. power_user     → no_action = 0.90   (don't interrupt healthy customers)
     power_user     → discount_offer = 0.15  (patronising — creates doubt)

  2. disengaged_enterprise → executive_qbr = 0.82  (needs senior attention)
     disengaged_enterprise → feature_tour = 0.42   (they know the features)

  3. new_user_confused → feature_tour = 0.84  (they just need onboarding)
     new_user_confused → discount_offer = 0.18  (price is not the problem)

  4. price_sensitive → discount_offer = 0.78  (it's about price — answer it)
     price_sensitive → feature_tour = 0.32    (they know the features)

  5. at_risk_churning → no_action = 0.04  (they will churn without intervention)
     at_risk_churning → csm_call = 0.68   (human touch at last minute)

These mappings live only in the reward signal.  No labels, no rules.

── Memory advantage: NeuralLinUCB vs naive LinUCB ───────────────────────────────

  Context dim:  1024 (Voyage-3)
  Naive LinUCB: 1024² × 5 arms × 8 bytes = 40 MB  (grows as dim²)
  NeuralLinUCB: MLP compresses 1024 → 64, LinUCB on 64-dim = 776 KB total (53×)

Requirements:
    pip install requests matplotlib numpy
    cargo build --features neural
    BANDITDB_API_KEY=<key> ./target/debug/banditdb   # server on :8080

Run:
    python examples/14_neural_linucb_churn_prevention.py
"""

import os
import sys
import time
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DB_URL           = "http://localhost:8080"
API_KEY          = os.getenv("BANDITDB_API_KEY", "")
HEADERS          = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
CAMPAIGN_ID      = "churn_prevention_v5"
CONTEXT_DIM      = 1024   # Voyage-3 output dimension
EMBED_DIM        = 64     # MLP compresses 1024 → 64; LinUCB operates here
N_ITERATIONS     = 5_000
CHECKPOINT_EVERY = 250
RANDOM_SEED      = 31

ARMS = [
    "no_action",
    "discount_offer",
    "csm_call",
    "feature_tour",
    "executive_qbr",
]

SEGMENTS = [
    "power_user",
    "disengaged_enterprise",
    "new_user_confused",
    "price_sensitive",
    "at_risk_churning",
]

# ---------------------------------------------------------------------------
# Reward matrix: probability of retention within 30 days given intervention.
# Reward = 1.0 means customer retained; 0.0 means churned.
# These values are the ground truth the bandit must discover from signal alone.
# ---------------------------------------------------------------------------
REWARD_BASE = {
    "power_user": {
        "no_action":       0.90,  # healthy — leave them alone
        "feature_tour":    0.48,
        "csm_call":        0.35,
        "executive_qbr":   0.22,
        "discount_offer":  0.15,  # patronising — creates doubt
    },
    "disengaged_enterprise": {
        "executive_qbr":   0.82,  # needs senior attention
        "csm_call":        0.58,
        "feature_tour":    0.42,  # they know the features
        "discount_offer":  0.28,
        "no_action":       0.18,
    },
    "new_user_confused": {
        "feature_tour":    0.84,  # they need onboarding
        "csm_call":        0.55,
        "no_action":       0.22,
        "discount_offer":  0.18,  # price is not the issue
        "executive_qbr":   0.14,
    },
    "price_sensitive": {
        "discount_offer":  0.78,  # it's about price — answer it directly
        "executive_qbr":   0.48,
        "csm_call":        0.45,
        "feature_tour":    0.32,  # they know the features
        "no_action":       0.12,
    },
    "at_risk_churning": {
        "csm_call":        0.68,  # human touch at the last minute
        "discount_offer":  0.58,
        "executive_qbr":   0.52,
        "feature_tour":    0.30,
        "no_action":       0.04,  # they will churn without intervention
    },
}

# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------
linucb_bytes  = (CONTEXT_DIM ** 2 + 2 * CONTEXT_DIM) * 8 * len(ARMS)
neural_bytes  = (EMBED_DIM   ** 2 + 2 * EMBED_DIM)   * 8 * len(ARMS)
network_bytes = (CONTEXT_DIM * 128 + 128 +
                 128 * 128   + 128 +
                 128 * EMBED_DIM + EMBED_DIM) * 4
neural_total  = neural_bytes + network_bytes

print("── Memory footprint ─────────────────────────────────────────────────────────")
print(f"  Context dim:       {CONTEXT_DIM}   (1024-dim Voyage-3 multimodal embedding)")
print(f"  Embed dim:         {EMBED_DIM}     (neural compression target)")
print(f"  Arms:              {len(ARMS)}")
print(f"  LinUCB (naive):    {linucb_bytes / 1024 / 1024:.1f} MB  "
      f"({CONTEXT_DIM}×{CONTEXT_DIM}×{len(ARMS)} matrices)")
print(f"  NeuralLinUCB:      {neural_total / 1024:.0f} KB  "
      f"({EMBED_DIM}×{EMBED_DIM}×{len(ARMS)} + MLP weights)")
print(f"  Reduction:         {linucb_bytes / neural_total:.0f}×")
print()

# ---------------------------------------------------------------------------
# Context generation — simulates Voyage-3 embeddings of customer signal text.
# Each segment clusters around a random unit-vector centre in 1024-dim space.
# σ = 0.25 matches real intra-segment variability in sentence embedding space.
# In production: replace with embed_customer_signals() using voyageai.Client().
# ---------------------------------------------------------------------------
np_rng = np.random.default_rng(RANDOM_SEED)
rng    = random.Random(RANDOM_SEED)

SEGMENT_CENTERS = {s: np_rng.standard_normal(CONTEXT_DIM) for s in SEGMENTS}
for s in SEGMENTS:
    SEGMENT_CENTERS[s] /= np.linalg.norm(SEGMENT_CENTERS[s])

def make_context(segment: str) -> list:
    raw = SEGMENT_CENTERS[segment] + np_rng.standard_normal(CONTEXT_DIM) * 0.25
    return (raw / np.linalg.norm(raw)).tolist()

def post(url, json_body):
    for attempt in range(8):
        r = requests.post(url, headers=HEADERS, json=json_body)
        if r.status_code == 429:
            wait = 1.5 ** attempt
            time.sleep(wait)
            continue
        return r
    return r

CSM_COST = 0.15  # $200/call normalised — subtracted from csm_call reward

def oracle_reward(arm: str, segment: str) -> float:
    base = REWARD_BASE[segment][arm]
    if arm == "csm_call":
        base = max(0.0, base - CSM_COST)
    return max(0.0, min(1.0, base + rng.gauss(0, 0.08)))

def expected_optimal(segment: str) -> float:
    return max(
        (v - CSM_COST if a == "csm_call" else v)
        for a, v in REWARD_BASE[segment].items()
    )

# ---------------------------------------------------------------------------
# Create campaign
# ---------------------------------------------------------------------------
resp = post(f"{DB_URL}/campaign", {
    "campaign_id": CAMPAIGN_ID,
    "arms":        ARMS,
    "feature_dim": 0,       # ignored for NeuralLinUCB — arm_dim = embed_dim
    "alpha":       0.5,
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
        "use_case":        "b2b_saas_churn_prevention",
        "embedding_model": "voyage-3",
        "context_source":  "usage_signals_email_support_crm_fused",
        "reward_signal":   "account_retained_within_30_days",
        "arms_strategy":   "intervention_type",
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
selection_matrix = {s: {a: 0 for a in ARMS} for s in SEGMENTS}

# "Don't disturb the healthy" tracker: no_action rate for power_user vs others
noact_counts = {"power_user": 0, "other": 0}
noact_totals = {"power_user": 0, "other": 0}
noact_rate_pu = []
noact_rate_ot = []
NOACT_WINDOW  = 150

cum_reward     = 0.0
cum_optimal    = 0.0
cum_reward_log = []
cum_regret_log = []
checkpoint_steps = []
iterations     = []

WINDOW = 100
reward_window  = {s: [] for s in SEGMENTS}
rolling_reward = {s: [] for s in SEGMENTS}

optimal_arm = {
    s: max(ARMS, key=lambda a: REWARD_BASE[s][a] - (CSM_COST if a == "csm_call" else 0))
    for s in SEGMENTS
}

print(f"Running {N_ITERATIONS} iterations — {CONTEXT_DIM}-dim context — "
      f"checkpoint every {CHECKPOINT_EVERY} steps …")
print(f"{'Iter':>6}  {'Checkpoint':^12}  {'Cum regret':>12}  "
      f"{'no_action (power)':>18}  {'no_action (other)':>17}")
print("─" * 74)

for i in range(1, N_ITERATIONS + 1):
    segment = rng.choice(SEGMENTS)
    context = make_context(segment)

    resp = post(f"{DB_URL}/predict",
                {"campaign_id": CAMPAIGN_ID, "context": context})
    body = resp.json()
    if "arm_id" not in body:
        print(f"  WARN step {i}: predict returned {resp.status_code} {body}", flush=True)
        continue
    arm  = body["arm_id"]
    iid  = body["interaction_id"]

    r = oracle_reward(arm, segment)
    post(f"{DB_URL}/reward",
         {"interaction_id": iid, "reward": round(r, 4)})

    selection_matrix[segment][arm] += 1

    key = "power_user" if segment == "power_user" else "other"
    noact_totals[key] += 1
    if arm == "no_action":
        noact_counts[key] += 1

    noact_rate_pu.append(noact_counts["power_user"] / max(noact_totals["power_user"], 1))
    noact_rate_ot.append(noact_counts["other"]      / max(noact_totals["other"],      1))

    for s in SEGMENTS:
        reward_window[s].append(r if s == segment else None)
        vals = [v for v in reward_window[s][-WINDOW:] if v is not None]
        rolling_reward[s].append(sum(vals) / len(vals) if vals else 0.0)

    cum_reward  += r
    cum_optimal += expected_optimal(segment)
    iterations.append(i)
    cum_reward_log.append(cum_reward)
    cum_regret_log.append(cum_optimal - cum_reward)

    if i % CHECKPOINT_EVERY == 0:
        post(f"{DB_URL}/checkpoint", {})
        checkpoint_steps.append(i)
        pu_rate = noact_counts["power_user"] / max(noact_totals["power_user"], 1)
        ot_rate = noact_counts["other"]      / max(noact_totals["other"],      1)
        print(f"{i:>6}  {'checkpoint':^12}  "
              f"{cum_optimal - cum_reward:>12.1f}  "
              f"{pu_rate:>17.1%}  {ot_rate:>16.1%}")

print()
print("Done.  Rendering charts …")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
ARM_COLORS = {
    "no_action":       "#607D8B",
    "discount_offer":  "#F44336",
    "csm_call":        "#FF9800",
    "feature_tour":    "#4CAF50",
    "executive_qbr":   "#9C27B0",
}

SEGMENT_COLORS = {
    "power_user":            "#4CAF50",
    "disengaged_enterprise": "#9C27B0",
    "new_user_confused":     "#2196F3",
    "price_sensitive":       "#FF9800",
    "at_risk_churning":      "#F44336",
}

def add_ckpts(ax):
    for s in checkpoint_steps:
        ax.axvline(s, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

def smoothed(series, w=100):
    return [sum(series[max(0, k - w):k + 1]) / min(k + 1, w)
            for k in range(len(series))]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "BanditDB · NeuralLinUCB — B2B SaaS Churn Prevention\n"
    f"1024-dim Voyage-3 multimodal embeddings · {len(ARMS)} intervention arms · "
    f"{N_ITERATIONS:,} accounts",
    fontsize=12, fontweight="bold", y=0.99,
)

# ── Panel A: Intervention routing heatmap ─────────────────────────────────
ax = axes[0, 0]

matrix  = np.array(
    [[selection_matrix[s][a] for a in ARMS] for s in SEGMENTS], dtype=float
)
row_sums = matrix.sum(axis=1, keepdims=True)
norm_m   = np.divide(matrix, row_sums, where=row_sums > 0)

cmap = LinearSegmentedColormap.from_list("bdb", ["#18181b", "#1e3a5f", "#2196F3"])
im   = ax.imshow(norm_m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ARMS)))
ax.set_yticks(range(len(SEGMENTS)))
ax.set_xticklabels([a.replace("_", "\n") for a in ARMS], fontsize=8)
ax.set_yticklabels([s.replace("_", "\n") for s in SEGMENTS], fontsize=8)
ax.set_title("Intervention routing heatmap\ngold border = ground-truth optimal arm", fontsize=10)
plt.colorbar(im, ax=ax, label="Selection share per customer segment")

for r, s in enumerate(SEGMENTS):
    for c, a in enumerate(ARMS):
        pct   = norm_m[r, c] * 100
        color = "white" if norm_m[r, c] > 0.5 else "#a1a1aa"
        ax.text(c, r, f"{pct:.0f}%", ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")
    opt_c = ARMS.index(optimal_arm[s])
    ax.add_patch(plt.Rectangle((opt_c - 0.5, r - 0.5), 1, 1,
                                fill=False, edgecolor="gold", linewidth=2.5))

# ── Panel B: Cumulative reward vs oracle ──────────────────────────────────
ax = axes[0, 1]
oracle_log = [(i + 1) * sum(expected_optimal(s) for s in SEGMENTS) / len(SEGMENTS)
              for i in range(len(iterations))]
ax.plot(iterations, oracle_log,     label="Oracle (always optimal)",
        color="#4CAF50", linestyle="--", linewidth=1.5)
ax.plot(iterations, cum_reward_log, label="NeuralLinUCB",
        color="#2196F3", linewidth=1.5)
ax.fill_between(iterations, cum_reward_log, oracle_log,
                alpha=0.15, color="#F44336", label="Regret")
add_ckpts(ax)
ax.set_title("Cumulative retention vs oracle\ngrey lines = neural retrain checkpoints", fontsize=10)
ax.set_xlabel("Account")
ax.set_ylabel("Cumulative accounts retained")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel C: "Don't disturb the healthy" ─────────────────────────────────
ax = axes[1, 0]
sm_pu = smoothed(noact_rate_pu, w=NOACT_WINDOW)
sm_ot = smoothed(noact_rate_ot, w=NOACT_WINDOW)

ax.plot(iterations, sm_pu,
        label="power_user segment",
        color=SEGMENT_COLORS["power_user"], linewidth=1.8)
ax.plot(iterations, sm_ot,
        label="all other segments",
        color="#607D8B", linewidth=1.5, linestyle="--")
ax.axhline(1 / len(ARMS), color="grey", linewidth=0.8, linestyle=":",
           label=f"Random baseline ({1/len(ARMS):.0%})")
add_ckpts(ax)
ax.set_title(
    '"Don\'t disturb the healthy" — no_action selection rate\n'
    "Model learns power users need silence, not intervention",
    fontsize=10,
)
ax.set_xlabel("Account")
ax.set_ylabel(f"no_action rate ({NOACT_WINDOW}-step rolling avg)")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel D: Rolling retention rate per segment ───────────────────────────
ax = axes[1, 1]
for s in SEGMENTS:
    optimal = expected_optimal(s)
    ax.plot(iterations, rolling_reward[s],
            label=f"{s.replace('_', ' ')} (opt={optimal:.2f})",
            color=SEGMENT_COLORS[s], linewidth=1.5)
add_ckpts(ax)
ax.set_title(
    f"Rolling retention rate by customer segment ({WINDOW}-step avg)\n"
    "Convergence toward per-segment optimal after each retrain",
    fontsize=10,
)
ax.set_xlabel("Account")
ax.set_ylabel(f"Avg retention (last {WINDOW} of that segment)")
ax.set_ylim(0.0, 1.0)
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(SCRIPT_DIR, "14_neural_linucb_churn_prevention.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved → {out}")

# ---------------------------------------------------------------------------
# Routing table + key insights
# ---------------------------------------------------------------------------
print("\n── Learned intervention routing table ───────────────────────────────────────")
arm_labels = [a[:14].ljust(14) for a in ARMS]
print(f"{'Segment':<28}" + "".join(f"{l:>15}" for l in arm_labels))
print("─" * (28 + 15 * len(ARMS)))
for s in SEGMENTS:
    total = sum(selection_matrix[s].values())
    row   = f"{s.replace('_', ' '):<28}"
    for a in ARMS:
        pct  = selection_matrix[s][a] / max(total, 1) * 100
        star = "★" if a == optimal_arm[s] else " "
        row += f"{pct:>13.1f}%{star}"
    print(row)
print("\n★ = ground-truth optimal arm for that customer segment")

# Key insight report
pu_final = noact_counts["power_user"] / max(noact_totals["power_user"], 1)
ot_final = noact_counts["other"]      / max(noact_totals["other"],      1)
print(f"\n── \"Don't disturb the healthy\" ──────────────────────────────────────────────")
print(f"  no_action rate — power_user:    {pu_final:.1%}")
print(f"  no_action rate — all others:    {ot_final:.1%}")
print(f"  (random baseline: {1/len(ARMS):.1%})")
if pu_final > ot_final * 1.5:
    print("  ✓ Model learned to leave healthy accounts alone")
else:
    print("  (more iterations needed — try increasing N_ITERATIONS)")

churn_rate = selection_matrix["at_risk_churning"]["no_action"] / max(
    sum(selection_matrix["at_risk_churning"].values()), 1
)
print(f"\n── \"Act on churning accounts\" ────────────────────────────────────────────────")
print(f"  no_action rate — at_risk_churning: {churn_rate:.1%}")
print(f"  (random baseline: {1/len(ARMS):.1%})")
if churn_rate < 1 / len(ARMS) * 0.6:
    print("  ✓ Model learned to intervene on at-risk accounts")

print(f"\n── Memory ───────────────────────────────────────────────────────────────────")
print(f"  Naive LinUCB at {CONTEXT_DIM}-dim:  {linucb_bytes / 1024 / 1024:.1f} MB")
print(f"  NeuralLinUCB:            {neural_total / 1024:.0f} KB   ({linucb_bytes // neural_total}× smaller)")
print(f"\n── Production swap-in ───────────────────────────────────────────────────────")
print("  Replace make_context(segment) with embed_customer_signals(customer)")
print("  using voyageai.Client().embed([text], model='voyage-3')")
print("  Context vector shape: (1024,) — drop-in replacement, no other changes.")

# plt.show()  — suppressed for non-interactive runs
