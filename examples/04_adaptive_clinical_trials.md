# 04 · Adaptive Clinical Trials (High-Stakes Routing)

> A real-time patient routing engine that continuously shifts new enrolments toward the most effective treatment based on each patient's specific biomarker profile — without waiting months for interim analysis.

---

## Problem

Traditional randomised controlled trials (RCTs) assign patients to treatment arms with fixed probabilities (e.g. 33% / 33% / 33%) for the entire duration of the study. This creates a critical ethical and statistical problem:

- **Patients kept on a poor arm** → weeks or months of exposure to an inferior treatment while the evidence accumulates.
- **Over-powered studies** → enrolment continues long after the optimal treatment has become statistically obvious.

Adaptive trial designs allow arm probabilities to shift as evidence accrues. BanditDB implements this correctly, re-weighting each new patient assignment in sub-milliseconds using the biomarkers available at the time of enrolment.

---

## Arms

| Arm | Description |
|-----|-------------|
| `drug_A` | First candidate treatment. |
| `drug_B` | Second candidate treatment. |
| `placebo` | Control arm — required for regulatory approval and baseline comparison. |

---

## Context Vector (`feature_dim = 4`)

| Index | Feature | Range | How to compute |
|-------|---------|-------|----------------|
| 0 | `age_normalized` | `[0.0, 1.0]` | `(patient_age - min_age) / (max_age - min_age)` across the trial population |
| 1 | `blood_pressure_normalized` | `[0.0, 1.0]` | `(systolic_bp - 80) / (200 - 80)` — maps the clinical range 80–200 mmHg to [0,1] |
| 2 | `biomarker_A_present` | `{0.0, 1.0}` | `1.0` if lab result is positive for Biomarker A |
| 3 | `biomarker_B_present` | `{0.0, 1.0}` | `1.0` if lab result is positive for Biomarker B |

---

## Reward Design

The reward represents the patient's treatment response at the follow-up assessment window.

```
reward = clinical_improvement_score / max_possible_improvement
```

- **`clinical_improvement_score`** — a validated composite endpoint (e.g. HAMD-17 reduction, tumour shrinkage %, symptom checklist delta).
- **`max_possible_improvement`** — the theoretical maximum score on that scale.
- **Result** — naturally in `[0, 1]`. A patient who reaches full remission scores `1.0`; a patient who shows no change or deteriorates scores `0.0`.

The delayed reward capability of BanditDB is essential here: the follow-up window may be days or weeks after the initial enrolment prediction. BanditDB holds the context in its TTL cache until the reward arrives.

---

## Code

```python
from banditdb import Client

db = Client("http://localhost:8080", api_key="your-secret-key")

# ── 1. Register the trial campaign (run once at trial initialisation) ──────────
db.create_campaign(
    campaign_id="trial_alpha",
    arms=["drug_A", "drug_B", "placebo"],
    feature_dim=4,
)

# ── 2. A new patient is enrolled. Build their biomarker context vector. ────────
# Context: [age_normalized, blood_pressure_normalized, biomarker_A, biomarker_B]
patient_features = [0.65, 0.80, 1.0, 0.0]
#                   ^      ^     ^    ^
#                   52yr   145   +A   -B

# ── 3. BanditDB calculates the UCB for this specific patient profile ───────────
treatment, interaction_id = db.predict("trial_alpha", patient_features)
print(f"Assigning patient to: {treatment}")  # e.g., "drug_A"

# ── 4. Three weeks later: follow-up assessment ────────────────────────────────
# Patient's HAMD-17 score improved from 24 to 8 (range 0–52; lower is better)
improvement = 24 - 8          # = 16 points of improvement
max_possible = 24             # baseline score is the ceiling of possible improvement
reward = improvement / max_possible   # = 0.667

db.reward(interaction_id, reward=round(reward, 4))
# BanditDB will now route future patients with similar biomarkers (Biomarker A+, B-)
# and similar age/BP profiles toward drug_A with higher confidence.
```

---

## What the Model Learns

As the trial progresses, the LinUCB weights converge toward treatment-biomarker interactions that may not have been hypothesised at trial design time:

| Patient Profile | Context | Arm that emerges | Why the bandit learns this |
|---|---|---|---|
| Biomarker A+, younger | `[0.3, 0.5, 1.0, 0.0]` | `drug_A` | Strong response signal in A+ patients under 45. |
| Biomarker B+, older, hypertensive | `[0.7, 0.8, 0.0, 1.0]` | `drug_B` | Drug B shows consistent efficacy in this subgroup; Drug A may interact with hypertension. |
| Both biomarkers absent | `[0.5, 0.4, 0.0, 0.0]` | `placebo`-heavy weighting | Neither active treatment outperforms control in biomarker-negative patients — a critical signal that the drug is a targeted therapy, not a broad one. |

This is the canonical strength of adaptive trials: the model can surface subgroup heterogeneity (a drug that works only for Biomarker A+ patients) that a fixed-allocation RCT would bury in the average.

> **Regulatory note:** Adaptive designs using BanditDB should be pre-registered with the relevant ethics board and statistical analysis plan filed before enrolment begins. The WAL export (Apache Parquet) provides a full audit trail of every allocation decision and its context vector for regulatory submission.

---

## Related Examples

- [`01_dynamic_pricing_sell_through.md`](./01_dynamic_pricing_sell_through.md) — market-state context (temporal bandit)
- [`03_ecommerce_upsell_optimization.md`](./03_ecommerce_upsell_optimization.md) — user-state context with binary conversion reward
