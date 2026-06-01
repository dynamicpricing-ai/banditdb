# 06 · Law Firm Client Intake Routing (Business Development)

> A real-time intake engine that learns which response to give each inbound enquiry — book a consultation, send a self-serve intake form, refer to a partner firm, or decline — based on matter value, complexity, conflict risk, and current firm capacity.

---

## Problem

A mid-size law firm receives dozens of inbound enquiries per week. Four responses are available, and the wrong one costs money in two directions:

- **Over-invest in a low-value enquiry** → a senior partner spends two hours on a matter that bills $2,000.
- **Under-invest in a high-value enquiry** → a promising client receives a generic intake form, feels unvalued, and signs with a competitor.
- **Miss a conflict** → the firm takes on a matter it should have declined, triggering a professional liability issue.
- **Decline when a referral was the right move** → lost referral fee income and damaged relationship with a partner firm.

A rule-based triage system hard-codes thresholds: "if estimated value > $100K, schedule a consultation." It cannot learn that a specific practice area consistently over-estimates upfront value, that a particular referral source sends matters that reliably close, or that Friday afternoon enquiries from individual clients almost never convert. BanditDB discovers all of these patterns from the reward signal — without a single rule.

---

## Arms

| Arm | Description |
|-----|-------------|
| `schedule_consultation` | Highest-touch response. A partner or senior associate contacts the prospect directly. Correct for complex, high-value, low-conflict enquiries where capacity allows. |
| `send_intake_form` | Self-serve pre-qualification. Efficient for moderate-value or straightforward matters. Filters out low-intent enquiries with minimal partner time. |
| `refer_to_partner_firm` | Route to a trusted partner firm. Appropriate when the matter is outside the firm's specialty, capacity is constrained, or the matter is valuable but misaligned with current strategy. Generates referral fee income and relationship capital. |
| `decline` | Politely close the enquiry. Correct for matters below the firm's minimum threshold, high conflict risk, or resource-intensive cases unlikely to reach engagement. |

---

## Context Vector (`feature_dim = 5`)

| Index | Feature | Range | How to compute |
|-------|---------|-------|----------------|
| 0 | `case_value_norm` | `[0.0, 1.0]` | `estimated_matter_value / 1_000_000` — cap at $1M; initial estimate from the intake conversation or referral note |
| 1 | `matter_complexity` | `[0.0, 1.0]` | Categorical mapped to float: simple drafting `0.1`, employment dispute `0.3`, commercial litigation `0.5`, M&A `0.8`, complex multi-party `1.0` |
| 2 | `org_size_norm` | `[0.0, 1.0]` | Individual `0.0`, startup `0.2`, SME `0.4`, mid-market `0.7`, large corporation `1.0` |
| 3 | `conflict_risk` | `[0.0, 1.0]` | Output of the firm's automated conflict-check system; `0.0` = no conflicts found, `1.0` = direct conflict confirmed |
| 4 | `capacity_norm` | `[0.0, 1.0]` | Available partner hours this week / maximum partner hours; `1.0` = fully available, `0.0` = fully booked |

---

## Reward Design

The reward is recorded 30 days after the intake decision and measures how well the chosen arm performed given what actually happened.

```
reward = min(1.0, billed_revenue / target_revenue)   if matter was signed
reward = 0.4                                          if referral completed and fee confirmed
reward = 0.2                                          if decline was correct (conflict confirmed, or matter fell below threshold on full assessment)
reward = 0.0                                          if no outcome (prospect ghosted, early termination, or wrong routing caused a lost engagement)
```

This design rewards all four arms proportionally to their actual outcome — not just conversions. A correct `decline` scores `0.2` because it preserved partner capacity. A correct `refer_to_partner_firm` scores `0.4` because it generated referral income without consuming capacity. The bandit learns that the right arm for a given context is the one that consistently maximises this blended score — not simply the one with the highest ceiling.

---

## Code

```python
from banditdb import Client

db = Client("http://localhost:8080", api_key="your-secret-key")

# ── 1. Create the campaign (run once at startup) ──────────────────────────────
db.create_campaign(
    campaign_id="client_intake",
    arms=["schedule_consultation", "send_intake_form",
          "refer_to_partner_firm", "decline"],
    feature_dim=5,
)

# ── 2. A new enquiry arrives. Build the context vector. ───────────────────────
# Context: [case_value_norm, matter_complexity, org_size_norm, conflict_risk, capacity_norm]
enquiry_context = [0.65, 0.70, 0.30, 0.05, 0.80]
#                  ^      ^      ^     ^     ^
#                  $650K  commercial  SME  low conflict  80% capacity free

# ── 3. Ask BanditDB how to route this enquiry ─────────────────────────────────
action, interaction_id = db.predict("client_intake", enquiry_context)
print(f"Routing: {action}")  # e.g., "schedule_consultation"

# ── 4. Route the enquiry, then record the outcome 30 days later ───────────────
route_enquiry(enquiry_id, action)

# 30 days later: matter signed, $680K billed against $600K target
billed_revenue = 680_000
target_revenue = 600_000
reward = min(1.0, billed_revenue / target_revenue)  # → 1.0 (capped on over-performance)

db.reward(interaction_id, reward)
```

---

## What the Model Learns

After several hundred intake decisions with outcomes recorded, the model converges on routing patterns that experienced partners recognise but struggle to systematise:

| Enquiry Profile | Context | Arm that emerges | Why the bandit learns this |
|---|---|---|---|
| High-value, complex, low conflict, capacity available | `[0.85, 0.80, 0.60, 0.05, 0.90]` | `schedule_consultation` | Large complex matters with available partners and no conflicts are the highest-value intakes. Direct contact converts them at a significantly higher rate. |
| Low-value, simple, individual client | `[0.05, 0.10, 0.00, 0.10, 0.60]` | `send_intake_form` | Simple low-value matters are pre-qualified efficiently through self-serve. Partner time is preserved for complex cases. |
| Outside specialty, mid-value | `[0.40, 0.50, 0.30, 0.20, 0.70]` | `refer_to_partner_firm` | Matters outside the firm's core practice areas generate referral fee income and relationship capital without consuming capacity. |
| High conflict risk, any value | `[0.70, 0.60, 0.50, 0.85, 0.80]` | `decline` | Even valuable matters become liabilities when conflict risk is high. The bandit quickly learns that `conflict_risk > 0.6` predicts poor outcomes regardless of case value. |
| High-value, complex, fully booked | `[0.90, 0.85, 0.70, 0.10, 0.05]` | `refer_to_partner_firm` | When capacity is at zero, even strong matters are better referred than handled at reduced quality. The interaction between `case_value_norm` and `capacity_norm` is non-obvious — a rule-based system would need an explicit override branch. |

---

## Convergence Estimate

**~600 intake decisions** — at a 50% outcome rate within 30 days, yielding ~300 reward signals — to measurably outperform random intake routing.

| Phase | Intake decisions | What the model knows |
|---|---|---|
| Exploration | 0–150 | Arms selected roughly equally. Routing is functionally random. |
| Early signal | 150–350 | High conflict risk → `decline` pattern begins to emerge. High-value + high capacity → `schedule_consultation` preference appears. |
| Measurable lift | ~600 | Cumulative revenue from BanditDB-routed intakes exceeds random routing. |
| Stable routing | ~1,200 | All four arms are reliably triggered by the appropriate context combinations. Capacity interaction effects — high-value matters deferred when capacity is low — become visible. |

**Assumptions:** 50% of enquiries produce an outcome within 30 days. Set `BANDITDB_REWARD_TTL_SECS` to at least `2592000` (30 days). At 30% outcome rate — realistic for cold inbound enquiries — enroll approximately 1,000 decisions to collect 300 reward signals.

The multi-valued reward (`0.0 / 0.2 / 0.4 / 1.0`) provides richer signal than a binary flag. Even correct declines and referrals contribute positive signal, so the model does not falsely learn that `schedule_consultation` is always best simply because it has the highest ceiling reward.

---

## Related Examples

- [`03_ecommerce_upsell_optimization.md`](./03_ecommerce_upsell_optimization.md) — binary conversion reward with user-state context
- [`04_adaptive_clinical_trials.md`](./04_adaptive_clinical_trials.md) — delayed rewards and high-stakes routing
