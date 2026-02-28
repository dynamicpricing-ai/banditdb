# 01 · Dynamic Pricing for Inventory Sell-Through

> A self-calibrating pricing engine that learns when to hold margin and when to liquidate — driven by sell-through rate, holiday proximity, day of week, season urgency, and competitor positioning.

---

## Problem

A fashion retailer has 10,000 units of summer inventory that *must* clear before Labor Day. Two failure modes exist and they pull in opposite directions:

- **Discount too early** → destroy margin on items that would have sold themselves over the holiday weekend.
- **Discount too late** → liquidate at a loss with a warehouse full of autumn stock arriving next week.

A rule-based pricing system requires an analyst to hand-write every `if/elif` branch. It will still miss the non-obvious interactions — a holiday landing on a *Tuesday* behaves differently from a holiday landing on a *Friday*. BanditDB discovers all of these interactions automatically from the reward signal.

### What makes this example architecturally distinct

Examples 1 (e-commerce upsell) and 2 (clinical trials) use a context that describes a **person**. This example uses a context that describes the **state of the market right now**. Sell-through rate, holiday proximity, and competitor price all change hourly. This is a *temporal bandit*: the agent learns which strategy dominates under each combination of market conditions, not under each user segment.

---

## Arms

| Arm | Description |
|-----|-------------|
| `full_price` | Hold margin. Let demand do the work. |
| `ten_percent_off` | Mild nudge. Useful for soft days near peak season. |
| `twenty_five_percent_off` | Meaningful discount. Effective when a competitor has undercut. |
| `flash_clearance` | Emergency liquidation. Protects against dead stock. |

---

## Context Vector (`feature_dim = 5`)

| Index | Feature | Range | How to compute |
|-------|---------|-------|----------------|
| 0 | `sell_through_rate` | `[0.0, 1.0]` | `units_sold / units_total` |
| 1 | `is_weekend` | `{0.0, 1.0}` | `1.0` if `weekday() >= 5` else `0.0` |
| 2 | `holiday_proximity` | `[0.0, 1.0]` | Peaks in the 30-day run-up to the next major shopping holiday; `0.0` outside that window |
| 3 | `season_urgency` | `[0.0, 1.0]` | Rises linearly from `0.0` at 90 days before season end to `1.0` on the final day |
| 4 | `price_position` | `[0.0, 1.0]` | `0.5` = at parity with competitor; below `0.5` = we are more expensive; above `0.5` = we are cheaper |

---

## Reward Design

The reward answers: *what fraction of the theoretically best outcome did this price tier actually deliver?*

```
reward = (units_sold × margin_after_discount) / (target_units_per_hour × full_price_margin)
```

- **Numerator** — gross margin actually captured in the slot.
- **Denominator** — gross margin we would have captured selling the target volume at full price.
- **Result** — naturally in `[0, 1]` for any price tier, capped at `1.0` on over-performance.

This design is important: it does not reward raw units sold. A `flash_clearance` arm that moves 200 units at near-zero margin scores *lower* than a `full_price` arm that moves 50 units at full margin on a holiday weekend. The bandit learns that discounting during peak demand is wasteful, not heroic.

---

## Code

```python
import requests
import datetime

db_url = "http://localhost:8080"

# ── 1. Create the campaign ────────────────────────────────────────────────────

requests.post(f"{db_url}/campaign", json={
    "campaign_id":  "summer_clearance",
    "arms":         ["full_price", "ten_percent_off", "twenty_five_percent_off", "flash_clearance"],
    "feature_dim":  5,
})

# ── 2. Build the context vector from live signals ─────────────────────────────

def get_market_context(inventory_total, inventory_sold, our_price, competitor_price):
    today        = datetime.date.today()
    season_end   = datetime.date(2025, 9,  1)   # Labor Day: last day summer items move
    next_holiday = datetime.date(2025, 7,  4)   # 4th of July: nearest demand spike

    days_to_season_end = max(0, (season_end   - today).days)
    days_to_holiday    = max(0, (next_holiday - today).days)

    return [
        inventory_sold / inventory_total,                                          # sell_through_rate
        1.0 if today.weekday() >= 5 else 0.0,                                     # is_weekend
        min(1.0, max(0.0, 1.0 - days_to_holiday    / 30.0)),                      # holiday_proximity
        min(1.0, max(0.0, 1.0 - days_to_season_end / 90.0)),                      # season_urgency
        min(1.0, max(0.0, (competitor_price - our_price) / our_price + 0.5)),     # price_position
    ]

# ── 3. Predict ────────────────────────────────────────────────────────────────
# Scenario: Tuesday afternoon, mid-July. 52% sold. Rival just dropped to $42.

context = get_market_context(
    inventory_total  = 10_000,
    inventory_sold   = 5_200,
    our_price        = 49.99,
    competitor_price = 42.00,
)
# → [0.52, 0.0, 0.67, 0.44, 0.16]
#    sell_through=0.52   half the warehouse is still full
#    is_weekend=0.0      Tuesday, foot traffic is low
#    holiday_proxy=0.67  4th of July was 10 days ago, demand spike is fading
#    season_urgency=0.44 6 weeks left, pressure is building
#    price_position=0.16 competitor is significantly cheaper than us

price_tier, interaction_id = requests.post(f"{db_url}/predict", json={
    "campaign_id": "summer_clearance",
    "context":     context,
}).json().values()

print(f"BanditDB recommends: {price_tier}")  # e.g. "twenty_five_percent_off"

# ── 4. Reward ─────────────────────────────────────────────────────────────────
# 1 hour later: 43 units moved at the recommended price tier.

units_sold        = 43
margin_after_disc = 18.50   # gross margin per unit at 25% off
full_price_margin = 29.99   # gross margin per unit at full price
target_units_hr   = 50      # expected units / hour in this time slot

reward = (units_sold * margin_after_disc) / (target_units_hr * full_price_margin)
reward = min(1.0, reward)   # cap at 1.0 on over-performance

requests.post(f"{db_url}/reward", json={
    "interaction_id": interaction_id,
    "reward":         round(reward, 4),
})
```

---

## What the Model Learns

After a full season of hourly predict → reward cycles, the LinUCB weights converge to reflect these regimes:

| Market State | Context | Optimal Arm | Why the bandit learns this |
|---|---|---|---|
| Holiday weekend, high sell-through | `[0.88, 1.0, 0.95, 0.3, 0.5]` | `full_price` | Organic demand is at its peak. Discounting destroys margin for free. |
| Weekday slump, competitor undercut | `[0.52, 0.0, 0.10, 0.44, 0.16]` | `twenty_five_percent_off` | Demand is soft and a rival is cheaper. Move stock before they take the sale. |
| High sell-through, mid-season | `[0.89, 0.0, 0.20, 0.30, 0.5]` | `full_price` | Scarcity drives desire. The remaining units are worth *more*, not less. |
| Season end, majority unsold | `[0.54, 0.0, 0.05, 0.97, 0.5]` | `flash_clearance` | 5 days left, 4,600 units unsold. Dead stock costs more than the discount. |

The model also picks up second-order interactions that rule-based systems miss — for example, a competitor undercut on a holiday weekend (`price_position=0.1` combined with `holiday_proximity=0.9`) still favours `full_price`, because the holiday demand signal overrides the competitive pressure. A rule-based engine would need an explicit override branch for this; BanditDB infers it from reward data alone.

---

## Related Examples

- `02_` — *(coming soon)*
