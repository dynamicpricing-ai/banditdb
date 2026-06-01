# 05 · Sleep Improvement (curl Walkthrough)

> A personalised sleep intervention engine that learns which environmental adjustment — temperature, light, or noise — produces the greatest sleep quality improvement for each individual's profile.

---

## Problem

One-size-fits-all sleep advice ignores the fact that individual responses vary significantly by physiology and lifestyle. A 25-year-old male athlete and a 60-year-old sedentary woman will respond differently to the same environmental change. BanditDB learns those differences automatically — routing each user to the intervention most likely to work for *their* profile, and improving with every reported outcome.

---

## Arms

| Arm | Description |
|-----|-------------|
| `decrease_temperature` | Lower bedroom temperature to the optimal sleep range (16–19°C). |
| `decrease_light` | Eliminate residual light sources; blackout blinds or sleep mask. |
| `decrease_noise` | Reduce ambient noise; white noise machine or ear plugs. |

---

## Context Vector (`feature_dim = 5`)

| Index | Feature | Range | How to compute |
|-------|---------|-------|----------------|
| 0 | `sex` | `{0.0, 1.0}` | `0.0` = male, `1.0` = female |
| 1 | `age_norm` | `[0.0, 1.0]` | `age / 100` |
| 2 | `weight_norm` | `[0.0, 1.0]` | `weight_kg / 150` |
| 3 | `activity` | `[0.0, 1.0]` | Self-reported activity level: `0.0` = sedentary, `1.0` = daily intense exercise |
| 4 | `bedtime_norm` | `[0.0, 1.0]` | `hour_of_day / 24` (e.g. 23:00 → `0.96`) |

---

## Reward Design

```
reward = (sleep_quality_after - sleep_quality_before) / sleep_quality_before
         clamped to [0.0, 1.0]
```

Measured via a standardised questionnaire (e.g. Pittsburgh Sleep Quality Index) or wearable sleep score, collected the following morning. A score of `1.0` means the intervention doubled sleep quality; `0.0` means no improvement or deterioration.

The delayed reward capability of BanditDB is essential here: the reward arrives hours after the prediction (next morning), and BanditDB holds the context in its TTL cache until it does.

---

## curl Walkthrough

**Create** the campaign (run once):
```bash
curl -s -X POST http://localhost:8080/campaign \
  -H "Content-Type: application/json" \
  -d '{"campaign_id":"sleep","arms":["decrease_temperature","decrease_light","decrease_noise"],"feature_dim":5}'
```
> `"Campaign Created"`

**List** campaigns:
```bash
curl -s http://localhost:8080/campaigns
```
> `[{"campaign_id":"sleep","alpha":1.0,"algorithm":"linucb","arm_count":3}]`

**Predict** — female, 35yo, 75 kg, moderately active, bedtime 23:00:
```bash
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"campaign_id":"sleep","context":[1.0, 0.35, 0.50, 0.60, 0.96]}'
```
> `{"arm_id":"decrease_temperature","interaction_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890"}`

**Reward** — sleep quality improved by 80% (use `interaction_id` from the predict response):
```bash
curl -s -X POST http://localhost:8080/reward \
  -H "Content-Type: application/json" \
  -d '{"interaction_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","reward":0.80}'
```
> `"OK"`

> If the server is running with `BANDITDB_API_KEY` set, add `-H "X-Api-Key: <your-key>"` to each request.

---

## What the Model Learns

After enough predict→reward cycles the model converges on profile-specific interventions:

| User Profile | Context | Arm that emerges | Signal |
|---|---|---|---|
| Young active male, early bedtime | `[0.0, 0.25, 0.53, 0.90, 0.92]` | `decrease_temperature` | Athletic bodies run hot; cooling has outsized impact. |
| Older sedentary female, late bedtime | `[1.0, 0.60, 0.60, 0.10, 0.98]` | `decrease_light` | Late light exposure suppresses melatonin more severely in older, less active individuals. |
| Urban dweller, any profile | `[*, *, *, *, *]` (high noise baseline) | `decrease_noise` | If the user's implicit noise feature correlates with poor outcomes on other arms, noise reduction rises. |

---

## Convergence Estimate

**~300 rewarded outcomes** to measurably outperform random arm assignment.

| Phase | Rewarded outcomes | What the model knows |
|---|---|---|
| Exploration | 0–100 | Arms selected roughly equally. Functionally identical to random. |
| Early signal | 100–200 | Temperature vs noise distinction begins to emerge for extreme profiles (young active male vs older sedentary female). |
| Measurable lift | ~300 | Cumulative reward from BanditDB exceeds what uniform random assignment would have delivered. |
| Stable routing | ~500 | Per-profile recommendations are consistent. New users with unseen feature combinations still trigger exploration. |

**Assumptions:** The reward is continuous and nearly always observed — any participant who submits a morning sleep score closes the loop. This estimate assumes ≥ 70% next-morning reporting compliance. At 50% compliance, enroll approximately 430 participants to collect 300 rewarded outcomes.

The main risk is not sample size but **reward delay**: if participants forget to report the following morning, BanditDB cannot update. Set `BANDITDB_REWARD_TTL_SECS` to at least 64800 (18 hours) to hold the context in cache until the report arrives.

---

## Related Examples

- [`04_adaptive_clinical_trials.md`](./04_adaptive_clinical_trials.md) — delayed rewards in a high-stakes medical context
- [`03_ecommerce_upsell_optimization.md`](./03_ecommerce_upsell_optimization.md) — binary conversion reward with user-state context
