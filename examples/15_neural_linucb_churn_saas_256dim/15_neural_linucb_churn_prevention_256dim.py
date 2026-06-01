"""examples/15_neural_linucb_churn_prevention_256dim.py

B2B SaaS Churn Prevention — NeuralLinUCB on 256-dim sentence embeddings.

Improvements over example 14 (1024-dim Voyage-3):
  • context_dim = 256 — 16× cheaper retrains, 14× less memory
  • embed_dim = 32   — better compression ratio for 256-dim context
  • retrain_every = 100 — 100 retrains over 10K iterations vs 40
  • 6 arms: adds in_app_nudge (automated $5/account middle option)
  • 10 000 iterations — full convergence expected

── Production embedding (256-dim) ──────────────────────────────────────────────

Replace make_context() with sentence-transformers. CPU-only, no API key:

    from sentence_transformers import SentenceTransformer
    import numpy as np

    encoder = SentenceTransformer("all-MiniLM-L6-v2")   # 22 MB, 384-dim output

    # Project 384 → 256 once at startup (fit PCA on a sample batch)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=256, random_state=0)
    # pca.fit(encoder.encode(sample_texts))  # fit on 500+ samples

    def embed_customer_signals(customer: dict) -> list[float]:
        text = (
            f"Usage: {customer['usage_summary']}\\n"
            f"Days since last login: {customer['days_since_login']}\\n"
            f"Feature adoption: {customer['feature_adoption_pct']:.0%}\\n"
            f"Recent email: {customer['last_email_snippet']}\\n"
            f"Support: {customer['support_summary']}\\n"
            f"NPS: {customer['nps_score']}\\n"
            f"Tier: {customer['tier']}, ARR: ${customer['arr']:,}\\n"
        )
        emb = encoder.encode([text], normalize_embeddings=True)[0]   # 384-dim
        emb256 = pca.transform(emb.reshape(1, -1))[0]               # 256-dim
        return (emb256 / np.linalg.norm(emb256)).tolist()

Alternative: OpenAI text-embedding-3-small with dimensions=256 (native 256-dim output,
no PCA needed — drop-in replacement, one API call per customer event):

    from openai import OpenAI
    client = OpenAI()

    def embed_customer_signals(customer: dict) -> list[float]:
        text = ...  # same fused text as above
        r = client.embeddings.create(model="text-embedding-3-small",
                                     input=text, dimensions=256)
        return r.data[0].embedding   # already 256-dim unit vector

── Arms (6) ─────────────────────────────────────────────────────────────────────

  no_action         Stay silent. Critical for healthy accounts.
  in_app_nudge      Automated in-app banner or 3-email sequence. $5/account.
                    Middle option between silence and human outreach.
  discount_offer    20% off next 3 months. Direct answer to price concerns.
  csm_call          Human CSM outreach. $200/account. High-touch.
  feature_tour      Personalised 5-email onboarding/feature sequence.
  executive_qbr     VP/C-level Quarterly Business Review. Strategic signal.

── Key insight: in_app_nudge closes the gap ─────────────────────────────────────

Without in_app_nudge, the bandit's only options between "do nothing" and
"spend $200 on a CSM call" are binary. This forces the LinUCB layer to over-invest
in csm_call because no cheap alternative exists.

With in_app_nudge, the bandit discovers:
  • power_user:  no_action=0.90, in_app_nudge=0.25  → silence still best
  • new_user:    feature_tour=0.84, in_app_nudge=0.65 → automated onboarding works
  • at_risk:     csm_call (cost-adj)=0.53, discount=0.58 → discount wins on net value
  • disengaged:  executive_qbr=0.82, in_app_nudge=0.35 → enterprise needs humans

Cost-adjusted reward (csm_call costs $200 ≈ 0.15 reward units):
  at_risk_churning optimal → discount_offer (0.58 > csm_call 0.53)
  all others     → unchanged from raw rewards

── Memory advantage ─────────────────────────────────────────────────────────────

  Context dim:  256 (vs 1024 in example 14)
  Naive LinUCB: 256² × 6 arms × 8 bytes = 3.1 MB
  NeuralLinUCB: 32² × 6 + MLP weights  = ~220 KB  (14× smaller)
  vs example 14: retrains 16× faster, memory 3.5× less

Requirements:
    pip install requests matplotlib numpy
    cargo build --features neural
    BANDITDB_API_KEY=<key> ./target/debug/banditdb

Run:
    BANDITDB_API_KEY=<key> python examples/15_neural_linucb_churn_prevention_256dim.py
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
API_KEY          = os.getenv("BANDITDB_API_KEY", "BDB2H3a7JYS134GBevNhJV5WeOKCi0ehBjj")
HEADERS          = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
CAMPAIGN_ID      = "churn_256_v2"
CONTEXT_DIM      = 256    # sentence-transformer compatible (MiniLM / OpenAI 3-small)
EMBED_DIM        = 32     # 8× compression; LinUCB operates here
N_ITERATIONS     = 50_000
CHECKPOINT_EVERY = 500    # progress logging interval
RANDOM_SEED      = 31

ARMS = [
    "no_action",
    "in_app_nudge",     # new: automated $5/account middle option
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
# Reward matrix: P(retained within 30 days | intervention).
# in_app_nudge fills the gap between silence and expensive human outreach.
# ---------------------------------------------------------------------------
REWARD_BASE = {
    "power_user": {
        "no_action":       0.90,  # healthy — silence is the right answer
        "feature_tour":    0.48,
        "in_app_nudge":    0.25,  # slight annoyance for a happy customer
        "csm_call":        0.35,
        "executive_qbr":   0.22,
        "discount_offer":  0.15,
    },
    "disengaged_enterprise": {
        "executive_qbr":   0.82,  # needs senior-level attention
        "csm_call":        0.58,
        "feature_tour":    0.42,
        "in_app_nudge":    0.35,  # too low-touch for enterprise accounts
        "discount_offer":  0.28,
        "no_action":       0.18,
    },
    "new_user_confused": {
        "feature_tour":    0.84,  # personalised onboarding works best
        "in_app_nudge":    0.65,  # automated sequence works nearly as well
        "csm_call":        0.55,
        "no_action":       0.22,
        "discount_offer":  0.18,
        "executive_qbr":   0.14,
    },
    "price_sensitive": {
        "discount_offer":  0.78,  # price is the issue — answer it directly
        "executive_qbr":   0.48,
        "csm_call":        0.45,
        "in_app_nudge":    0.42,  # value messaging helps
        "feature_tour":    0.32,
        "no_action":       0.12,
    },
    "at_risk_churning": {
        "csm_call":        0.68,  # human touch — best raw score
        "discount_offer":  0.58,  # best after cost adjustment
        "executive_qbr":   0.52,
        "in_app_nudge":    0.42,
        "feature_tour":    0.30,
        "no_action":       0.04,  # they churn without intervention
    },
}

CSM_COST = 0.15  # $200/call normalised — subtracted at reward time

# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------
linucb_bytes  = (CONTEXT_DIM ** 2 + 2 * CONTEXT_DIM) * 8 * len(ARMS)
neural_bytes  = (EMBED_DIM   ** 2 + 2 * EMBED_DIM)   * 8 * len(ARMS)
network_bytes = (CONTEXT_DIM * 128 + 128 +
                 128 * 64    + 64  +
                 64  * EMBED_DIM + EMBED_DIM) * 4
neural_total  = neural_bytes + network_bytes

print("── Memory footprint ─────────────────────────────────────────────────────────")
print(f"  Context dim:       {CONTEXT_DIM}   (256-dim sentence embedding)")
print(f"  Embed dim:         {EMBED_DIM}     ({CONTEXT_DIM}→{EMBED_DIM} compression, 8×)")
print(f"  Arms:              {len(ARMS)}")
print(f"  LinUCB (naive):    {linucb_bytes / 1024:.0f} KB  "
      f"({CONTEXT_DIM}×{CONTEXT_DIM}×{len(ARMS)} matrices)")
print(f"  NeuralLinUCB:      {neural_total / 1024:.0f} KB  "
      f"({EMBED_DIM}×{EMBED_DIM}×{len(ARMS)} + MLP weights)")
print(f"  Reduction:         {linucb_bytes / neural_total:.0f}×")
print(f"  Retrains:          every 100 rewards × {N_ITERATIONS//100} = "
      f"{N_ITERATIONS//100} total", flush=True)
print()

# ---------------------------------------------------------------------------
# Context generation — simulates 256-dim sentence-transformer embeddings.
# Production: replace with embed_customer_signals() + all-MiniLM-L6-v2.
# ---------------------------------------------------------------------------
np_rng = np.random.default_rng(RANDOM_SEED)
rng    = random.Random(RANDOM_SEED)

SEGMENT_CENTERS = {s: np_rng.standard_normal(CONTEXT_DIM) for s in SEGMENTS}
for s in SEGMENTS:
    SEGMENT_CENTERS[s] /= np.linalg.norm(SEGMENT_CENTERS[s])

def make_context(segment: str) -> list:
    raw = SEGMENT_CENTERS[segment] + np_rng.standard_normal(CONTEXT_DIM) * 0.25
    return (raw / np.linalg.norm(raw)).tolist()

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

optimal_arm = {
    s: max(ARMS, key=lambda a: REWARD_BASE[s][a] - (CSM_COST if a == "csm_call" else 0))
    for s in SEGMENTS
}

# ---------------------------------------------------------------------------
# HTTP helper — retries on 429 with exponential backoff
# ---------------------------------------------------------------------------
def post(url: str, json_body: dict) -> requests.Response:
    for attempt in range(8):
        r = requests.post(url, headers=HEADERS, json=json_body)
        if r.status_code != 429:
            return r
        time.sleep(1.5 ** attempt)
    return r

# ---------------------------------------------------------------------------
# Create campaign
# ---------------------------------------------------------------------------
resp = post(f"{DB_URL}/campaign", {
    "campaign_id": CAMPAIGN_ID,
    "arms":        ARMS,
    "feature_dim": 0,
    "alpha":       0.5,
    "algorithm": {
        "neural_lin_ucb": {
            "context_dim":   CONTEXT_DIM,
            "embed_dim":     EMBED_DIM,
            "hidden_dim":    128,
            "hidden_layers": 2,
            "retrain_every": 100,   # 100 retrains over 10K iterations
            "retrain_steps": 200,
            "learning_rate": 0.001,
            "lambda":        1.0,
        }
    },
    "metadata": {
        "use_case":        "b2b_saas_churn_prevention",
        "embedding_model": "all-MiniLM-L6-v2 (384→256 PCA) or text-embedding-3-small (256)",
        "context_source":  "usage_signals_email_support_crm_fused",
        "reward_signal":   "account_retained_within_30_days",
        "cost_adjustment": "csm_call -0.15 (~$200/account)",
    },
})

if resp.status_code == 422:
    print("ERROR: NeuralLinUCB not compiled. Run: cargo build --features neural")
    sys.exit(1)
elif resp.status_code not in (200, 201, 409):
    print(f"Unexpected status {resp.status_code}: {resp.text}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
selection_matrix = {s: {a: 0 for a in ARMS} for s in SEGMENTS}

# "Don't disturb the healthy" tracker
noact_counts = {"power_user": 0, "other": 0}
noact_totals = {"power_user": 0, "other": 0}
noact_rate_pu = []
noact_rate_ot = []
NOACT_WINDOW  = 300

cum_reward     = 0.0
cum_optimal    = 0.0
cum_reward_log = []
cum_regret_log = []
checkpoint_steps = []
iterations     = []

WINDOW = 200
reward_window  = {s: [] for s in SEGMENTS}
rolling_reward = {s: [] for s in SEGMENTS}

print(f"Running {N_ITERATIONS:,} iterations — {CONTEXT_DIM}-dim context — "
      f"retrain every 100 rewards — checkpoint log every {CHECKPOINT_EVERY} …", flush=True)
print(f"{'Iter':>6}  {'':12}  {'Cum regret':>12}  "
      f"{'no_action (power)':>18}  {'no_action (other)':>17}")
print("─" * 74, flush=True)

for i in range(1, N_ITERATIONS + 1):
    segment = rng.choice(SEGMENTS)
    context = make_context(segment)

    resp = post(f"{DB_URL}/predict",
                {"campaign_id": CAMPAIGN_ID, "context": context})
    body = resp.json()
    if "arm_id" not in body:
        print(f"  WARN step {i}: {resp.status_code} {body}", flush=True)
        continue
    arm = body["arm_id"]
    iid = body["interaction_id"]

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
              f"{pu_rate:>17.1%}  {ot_rate:>16.1%}", flush=True)

print(flush=True)
print("Done.  Rendering charts …", flush=True)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
ARM_COLORS = {
    "no_action":       "#607D8B",
    "in_app_nudge":    "#34D399",
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
        ax.axvline(s, color="grey", linestyle=":", linewidth=0.8, alpha=0.4)

def smoothed(series, w=300):
    return [sum(series[max(0, k - w):k + 1]) / min(k + 1, w)
            for k in range(len(series))]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "BanditDB · NeuralLinUCB — B2B SaaS Churn Prevention (256-dim)\n"
    f"sentence-transformer embeddings · {len(ARMS)} arms incl. in_app_nudge · "
    f"{N_ITERATIONS:,} accounts · cost-adjusted rewards",
    fontsize=11, fontweight="bold", y=0.99,
)

# ── Panel A: Intervention routing heatmap ────────────────────────────────
ax = axes[0, 0]
matrix   = np.array(
    [[selection_matrix[s][a] for a in ARMS] for s in SEGMENTS], dtype=float
)
row_sums = matrix.sum(axis=1, keepdims=True)
norm_m   = np.divide(matrix, row_sums, where=row_sums > 0)
cmap = LinearSegmentedColormap.from_list("bdb", ["#18181b", "#1e3a5f", "#2196F3"])
im   = ax.imshow(norm_m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ARMS)))
ax.set_yticks(range(len(SEGMENTS)))
ax.set_xticklabels([a.replace("_", "\n") for a in ARMS], fontsize=7)
ax.set_yticklabels([s.replace("_", "\n") for s in SEGMENTS], fontsize=8)
ax.set_title("Intervention routing heatmap\ngold = ground-truth optimal arm", fontsize=10)
plt.colorbar(im, ax=ax, label="Selection share per segment")
for r, s in enumerate(SEGMENTS):
    for c, a in enumerate(ARMS):
        pct   = norm_m[r, c] * 100
        color = "white" if norm_m[r, c] > 0.5 else "#a1a1aa"
        ax.text(c, r, f"{pct:.0f}%", ha="center", va="center",
                fontsize=7, color=color, fontweight="bold")
    opt_c = ARMS.index(optimal_arm[s])
    ax.add_patch(plt.Rectangle((opt_c - 0.5, r - 0.5), 1, 1,
                                fill=False, edgecolor="gold", linewidth=2.5))

# ── Panel B: Cumulative reward vs oracle ─────────────────────────────────
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
ax.set_title("Cumulative retention vs oracle\ngrey = checkpoint / retrain", fontsize=10)
ax.set_xlabel("Account")
ax.set_ylabel("Cumulative accounts retained")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel C: "Don't disturb the healthy" ─────────────────────────────────
ax = axes[1, 0]
sm_pu = smoothed(noact_rate_pu, w=NOACT_WINDOW)
sm_ot = smoothed(noact_rate_ot, w=NOACT_WINDOW)
ax.plot(iterations, sm_pu, label="power_user",
        color=SEGMENT_COLORS["power_user"], linewidth=1.8)
ax.plot(iterations, sm_ot, label="all other segments",
        color="#607D8B", linewidth=1.5, linestyle="--")
ax.axhline(1 / len(ARMS), color="grey", linewidth=0.8, linestyle=":",
           label=f"Random ({1/len(ARMS):.0%})")
add_ckpts(ax)
ax.set_title(
    '"Don\'t disturb the healthy" — no_action rate\n'
    "Separation = model learned power users need silence",
    fontsize=10,
)
ax.set_xlabel("Account")
ax.set_ylabel(f"no_action rate ({NOACT_WINDOW}-step rolling avg)")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel D: Rolling retention per segment ───────────────────────────────
ax = axes[1, 1]
for s in SEGMENTS:
    ax.plot(iterations, rolling_reward[s],
            label=f"{s.replace('_', ' ')} (opt={expected_optimal(s):.2f})",
            color=SEGMENT_COLORS[s], linewidth=1.5)
add_ckpts(ax)
ax.set_title(
    f"Rolling retention by segment ({WINDOW}-step avg)\n"
    "Convergence toward per-segment optimal",
    fontsize=10,
)
ax.set_xlabel("Account")
ax.set_ylabel(f"Avg retention (last {WINDOW})")
ax.set_ylim(0.0, 1.0)
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(SCRIPT_DIR, "15_neural_linucb_churn_prevention_256dim.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved → {out}", flush=True)

# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------
print("\n── Learned intervention routing table ───────────────────────────────────────")
arm_labels = [a[:13].ljust(13) for a in ARMS]
print(f"{'Segment':<28}" + "".join(f"{l:>14}" for l in arm_labels))
print("─" * (28 + 14 * len(ARMS)))
for s in SEGMENTS:
    total = sum(selection_matrix[s].values())
    row   = f"{s.replace('_', ' '):<28}"
    for a in ARMS:
        pct  = selection_matrix[s][a] / max(total, 1) * 100
        star = "★" if a == optimal_arm[s] else " "
        row += f"{pct:>12.1f}%{star}"
    print(row)
print("\n★ = ground-truth optimal (cost-adjusted)")

# Performance vs baselines
model_avg  = cum_reward_log[-1] / N_ITERATIONS
random_avg = sum(
    sum((v - CSM_COST if a == "csm_call" else v) for v in seg.values()) / len(ARMS)
    for seg in REWARD_BASE.values()
) / len(SEGMENTS)
oracle_avg = sum(expected_optimal(s) for s in SEGMENTS) / len(SEGMENTS)

print(f"\n── Performance ──────────────────────────────────────────────────────────────")
print(f"  Oracle avg reward:  {oracle_avg:.3f}")
print(f"  Model avg reward:   {model_avg:.3f}  ({(model_avg/oracle_avg)*100:.0f}% of oracle)")
print(f"  Random avg reward:  {random_avg:.3f}")
print(f"  vs random:          {((model_avg - random_avg)/random_avg)*100:+.0f}%")

pu_final = noact_counts["power_user"] / max(noact_totals["power_user"], 1)
ot_final = noact_counts["other"]      / max(noact_totals["other"],      1)
print(f"\n── \"Don't disturb the healthy\" ──────────────────────────────────────────────")
print(f"  no_action — power_user:  {pu_final:.1%}")
print(f"  no_action — all others:  {ot_final:.1%}")
print(f"  random baseline:         {1/len(ARMS):.1%}")
if pu_final > ot_final * 1.5:
    print("  ✓ Model learned to leave healthy accounts alone")
else:
    print("  — separation not yet clear (expected at 10K with 100 retrains)")

print(f"\n── Memory ───────────────────────────────────────────────────────────────────")
print(f"  Naive LinUCB: {linucb_bytes/1024:.0f} KB")
print(f"  NeuralLinUCB: {neural_total/1024:.0f} KB  ({linucb_bytes//neural_total}× smaller)")
print(f"\n── Retrain cadence ──────────────────────────────────────────────────────────")
print(f"  {N_ITERATIONS // 100} retrains over {N_ITERATIONS:,} iterations  "
      f"(every 100 rewards, {200} gradient steps each)")
