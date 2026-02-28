# 03 · E-Commerce Upsell Optimization (Personalization)

> A real-time checkout personalizer that learns which incentive closes each type of shopper — mobile vs. desktop, new vs. returning, high-value cart vs. impulse buy.

---

## Problem

At checkout a retailer has three levers to pull: offer a 10% discount, offer free shipping, or show nothing at all. The wrong choice costs money in two directions:

- **Discount to a shopper who would have converted anyway** → margin given away for free.
- **Show nothing to a price-sensitive new visitor** → cart abandonment.

Traditional A/B testing wastes traffic by holding one third of users on a provably worse experience for weeks until statistical significance is reached. BanditDB continuously shifts traffic toward the winning arm for each user segment from the very first interaction.

---

## Arms

| Arm | Description |
|-----|-------------|
| `ten_percent_off` | Percentage discount applied at checkout. Effective for price-sensitive first-time buyers. |
| `free_shipping` | Removes shipping cost. High perceived value, especially for mobile shoppers with smaller carts. |
| `no_offer` | Show nothing. Correct for high-intent returning customers who will convert regardless. |

---

## Context Vector (`feature_dim = 3`)

| Index | Feature | Range | How to compute |
|-------|---------|-------|----------------|
| 0 | `is_mobile` | `{0.0, 1.0}` | `1.0` if the request `User-Agent` is a mobile device |
| 1 | `cart_value_normalized` | `[0.0, 1.0]` | `cart_value / max_expected_cart_value` (e.g. divide by $500) |
| 2 | `is_returning_customer` | `{0.0, 1.0}` | `1.0` if the user has completed at least one previous purchase |

---

## Reward Design

The reward answers: *did showing this offer cause a purchase?*

```
reward = 1.0   if the user completed the purchase
reward = 0.0   if the user abandoned the cart
```

This is the simplest possible reward signal — a binary conversion event. For advanced use, weight the reward by net margin recovered versus the cost of the discount offered (similar to the dynamic pricing example).

---

## Code

```python
from banditdb import Client

db = Client("http://localhost:8080", api_key="your-secret-key")

# ── 1. Create the campaign (run once at startup) ───────────────────────────────
db.create_campaign(
    campaign_id="checkout_upsell",
    arms=["ten_percent_off", "free_shipping", "no_offer"],
    feature_dim=3,
)

# ── 2. A user reaches checkout. Build their context vector. ───────────────────
# Context: [is_mobile, cart_value_normalized, is_returning_customer]
user_context = [1.0, 0.85, 0.0]
#               ^               ^              ^
#               mobile device   $425 cart      first-time buyer

# ── 3. Ask BanditDB what to show them ─────────────────────────────────────────
offer, interaction_id = db.predict("checkout_upsell", user_context)
print(f"Displaying: {offer}")  # e.g., "free_shipping"

# ── 4. The user completes the purchase — close the feedback loop ───────────────
db.reward(interaction_id, reward=1.0)

# ── Or: the user abandons the cart ────────────────────────────────────────────
# db.reward(interaction_id, reward=0.0)
```

---

## What the Model Learns

After thousands of checkout interactions, the LinUCB weights converge to reflect these patterns:

| Shopper Profile | Context | Optimal Arm | Why the bandit learns this |
|---|---|---|---|
| Mobile, first-time, small cart | `[1.0, 0.15, 0.0]` | `free_shipping` | Shipping cost feels large relative to a small cart on mobile. Removing it overcomes the psychological barrier. |
| Desktop, first-time, large cart | `[0.0, 0.90, 0.0]` | `ten_percent_off` | High-value new buyers respond to a dollar amount saved, not a flat shipping waiver. |
| Any device, returning customer | `[*, *, 1.0]` | `no_offer` | Returning customers have demonstrated intent. Discounting cannibalises margin from a sale that was already happening. |
| Mobile, first-time, mid cart | `[1.0, 0.50, 0.0]` | `free_shipping` | Consistent with the small-cart case — free shipping outperforms discount for mobile-first shoppers. |

The model also discovers interaction effects that a rule-based system would miss: a returning customer with an unusually large cart (a gift purchase, for example) may still respond to a 10% discount. BanditDB infers this from reward data without any explicit rule.

---

## Related Examples

- [`01_dynamic_pricing_sell_through.md`](./01_dynamic_pricing_sell_through.md) — market-state context (temporal bandit) rather than user-state context
- [`04_adaptive_clinical_trials.md`](./04_adaptive_clinical_trials.md) — high-stakes routing with biomarker context vectors
