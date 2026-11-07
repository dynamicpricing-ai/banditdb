# 03 · E-Commerce Upsell Optimization (Personalization)

> A real-time checkout personalizer that learns which incentive closes each type of shopper — mobile vs. desktop, new vs. returning, high-value cart vs. impulse buy — while staying inside a fixed promo budget.

---

## Problem

At checkout a retailer has three levers to pull: offer a 10% discount, offer free shipping, or show nothing at all. The wrong choice costs money in two directions:

- **Discount to a shopper who would have converted anyway** → margin given away for free.
- **Show nothing to a price-sensitive new visitor** → cart abandonment.

There is a third constraint: **the promo budget is finite**. Finance approves, say, $100,000 of discounts and shipping subsidies for the quarter. An unconstrained optimizer will happily give an incentive to every shopper it predicts might convert, burn the budget in the first few weeks, and leave the rest of the quarter with nothing to offer.

Traditional A/B testing wastes traffic by holding one third of users on a provably worse experience for weeks until statistical significance is reached. BanditDB continuously shifts traffic toward the winning arm for each user segment from the very first interaction — and, with a budget attached, spends incentives only where they buy the most extra conversions.

---

## Arms

| Arm | Description |
|-----|-------------|
| `ten_percent_off` | Percentage discount applied at checkout. Effective for price-sensitive first-time buyers. |
| `free_shipping` | Removes shipping cost. High perceived value, especially for mobile shoppers with smaller carts. |
| `no_offer` | Show nothing. Correct for high-intent returning customers who will convert regardless. Costs nothing. |

---

## Context Vector (`feature_dim = 3`)

| Index | Feature | Range | How to compute |
|-------|---------|-------|----------------|
| 0 | `is_mobile` | `{0.0, 1.0}` | `1.0` if the request `User-Agent` is a mobile device |
| 1 | `cart_value_normalized` | `[0.0, 1.0]` | `cart_value / max_expected_cart_value` (e.g. divide by $500) |
| 2 | `is_returning_customer` | `{0.0, 1.0}` | `1.0` if the user has completed at least one previous purchase |

---

## Budget Constraint

One resource, `promo_budget`, measured in **promo units** where 1 unit ≈ $25 (the average 10% discount on a $250 cart):

| Arm | Cost (units) | Approx. $ |
|-----|--------------|-----------|
| `ten_percent_off` | `1.0` | ~$25 discount |
| `free_shipping` | `0.3` | ~$7.50 label cost |
| `no_offer` | free (omitted) | $0 |

**Sizing:** a $100,000 quarterly promo budget = 4,000 units. The store expects ~120,000 checkouts per quarter, so `horizon=120_000` decisions. That allows ~0.033 units per checkout on average — roughly one discount in 30 checkouts, or one free-shipping offer in 9.

Why units and not dollars: the shadow price (λ) is subtracted from an arm's score *per unit of cost*, and the score is on the reward scale (0–1 conversion). Keeping cost near 1 keeps λ in a range the default step size learns quickly. Costs in raw dollars (25.0) would need a much smaller `step_size`.

How it works:

- Each decision, BanditDB subtracts `λ × cost` from each arm's score. λ starts at 0 and rises when spend runs ahead of pace, falls back when it lags.
- With `adaptive=True` (default) the target rate is *remaining budget / remaining decisions*, so an early overspend is corrected rather than carried forward.
- An arm whose cost exceeds the remaining budget is masked. Once the budget is exhausted, only `no_offer` is served.

---

## Reward Design

The reward answers: *did showing this offer cause a purchase?*

```
reward = 1.0   if the user completed the purchase
reward = 0.0   if the user abandoned the cart
```

This is the simplest possible reward signal — a binary conversion event. The budget handles cost, so the reward does not need to subtract the discount: λ prices it in. If you'd rather optimise margin directly, weight the reward by net margin (similar to the dynamic pricing example) — but keep it on a 0–1 scale so it stays comparable to `λ × cost`.

---

## Code

```python
from banditdb import Client, PacingConfig, ResourceConstraint

db = Client("http://localhost:8080", api_key="your-secret-key")

# ── 1. Create the campaign (run once per budget period) ───────────────────────
# $100k promo budget for the quarter, in $25 units, over ~120k checkouts.
db.create_campaign(
    campaign_id="checkout_upsell_2026q4",
    arms=["ten_percent_off", "free_shipping", "no_offer"],
    feature_dim=3,
    pacing=PacingConfig(resources=[
        ResourceConstraint(
            name="promo_budget",
            budget=4_000,           # 4,000 units × $25 = $100,000
            horizon=120_000,        # expected checkouts this quarter
            arm_costs={
                "ten_percent_off": 1.0,
                "free_shipping":   0.3,
                # no_offer omitted → free
            },
        ),
    ]),
)

# ── 2. A user reaches checkout. Build their context vector. ───────────────────
# Context: [is_mobile, cart_value_normalized, is_returning_customer]
user_context = [1.0, 0.85, 0.0]
#               ^               ^              ^
#               mobile device   $425 cart      first-time buyer

# ── 3. Ask BanditDB what to show them ─────────────────────────────────────────
offer, interaction_id = db.predict("checkout_upsell_2026q4", user_context)
print(f"Displaying: {offer}")  # e.g., "free_shipping"

# ── 4. The user completes the purchase — close the feedback loop ───────────────
db.reward(interaction_id, reward=1.0)

# ── Or: the user abandons the cart ────────────────────────────────────────────
# db.reward(interaction_id, reward=0.0)

# ── 5. Watch the budget (dashboard / cron) ────────────────────────────────────
for r in db.pacing("checkout_upsell_2026q4")["resources"]:
    print(f"{r['name']}: {r['utilization']:.0%} spent "
          f"after {r['decisions']:,}/{r['horizon']:,} checkouts, "
          f"λ={r['lambda']:.3f}, exhausted={r['is_exhausted']}")
```

**Things to know about the budget:**

- **Spend is recorded at `predict`, not at `reward`.** Showing the offer costs the units whether or not the shopper buys. So the budget caps *offers shown*, not discounts redeemed. If ~50% of discounted carts convert, a 4,000-unit budget caps actual discount spend near $50,000 — size it with your redemption rate in mind, or budget for offers shown and treat it as a ceiling.
- **Cost is fixed per arm.** A 10% discount on a $480 cart costs more than on a $60 cart, but both are charged 1.0 unit. Use the average, or split into campaigns by cart band if the spread matters.
- **Budgets never reset and pacing is immutable.** Create a new campaign each quarter (hence the `_2026q4` suffix).
- **Spell `arm_costs` keys exactly.** The Python SDK rejects keys that aren't arms; the raw HTTP API doesn't, and a typo leaves the real arm free.

---

## What the Model Learns

After thousands of checkout interactions, the LinUCB weights converge to reflect these patterns:

| Shopper Profile | Context | Optimal Arm | Why the bandit learns this |
|---|---|---|---|
| Mobile, first-time, small cart | `[1.0, 0.15, 0.0]` | `free_shipping` | Shipping cost feels large relative to a small cart on mobile. Removing it overcomes the psychological barrier. |
| Desktop, first-time, large cart | `[0.0, 0.90, 0.0]` | `ten_percent_off` | High-value new buyers respond to a dollar amount saved, not a flat shipping waiver. |
| Any device, returning customer | `[*, *, 1.0]` | `no_offer` | Returning customers have demonstrated intent. Discounting cannibalises margin from a sale that was already happening. |
| Mobile, first-time, mid cart | `[1.0, 0.50, 0.0]` | `free_shipping` | Consistent with the small-cart case — free shipping outperforms discount for mobile-first shoppers. |

### Under the budget

When the budget binds, λ rises and every incentive has to earn back `λ × cost` in extra conversion probability. Incentives are reserved for shoppers where they change the outcome most:

| Shopper Profile | Unconstrained | Budget binding (λ > 0) | Why |
|---|---|---|---|
| Desktop, first-time, large cart | `ten_percent_off` | `ten_percent_off` | Largest lift from the discount; still clears the shadow price. |
| Mobile, first-time, small cart | `free_shipping` | `free_shipping` | Cheap (0.3 units) and high lift — the last incentive to be cut. |
| Desktop, first-time, mid cart | `ten_percent_off` | `free_shipping` | Discount's extra lift over free shipping no longer covers the extra 0.7 units. |
| Returning customer | `no_offer` | `no_offer` | Low lift, so it never justified spend. |

As spend runs ahead of pace λ goes up and the marginal segments drop to cheaper arms; when spend lags, λ falls and they get offers again. The budget lasts the whole quarter instead of running out in week three.

The model also discovers interaction effects that a rule-based system would miss: a returning customer with an unusually large cart (a gift purchase, for example) may still respond to a 10% discount. BanditDB infers this from reward data without any explicit rule.

---

## Convergence Estimate

**~1,500 checkout interactions** — at a 50% completion rate, yielding ~750 binary reward signals — to measurably outperform random assignment.

| Phase | Checkout interactions | What the model knows |
|---|---|---|
| Exploration | 0–300 | Arms selected roughly equally per user segment. No meaningful pattern yet. |
| Early signal | 300–800 | Mobile vs desktop distinction begins to appear. Returning customer preference for `no_offer` starts to emerge. |
| Measurable lift | ~1,500 | Cumulative revenue from BanditDB exceeds uniform random assignment. |
| Stable routing | ~3,000 | All three shopper segments (mobile/new, desktop/new, returning) are reliably routed. |

**Assumptions:** 50% checkout completion rate (purchase or abandon events are always observed). If your completion rate is lower — for example 30%, because many sessions time out before a final action — scale proportionally: you will need approximately 2,500 interactions to collect 750 reward signals.

**Budget and convergence:** λ moves from the first decision, so pacing works before the conversion model has converged. Early in the quarter, though, the model doesn't yet know which segments benefit most, so spend is spread more evenly. With `adaptive=True` the later, better-informed decisions get the remaining budget.

The binary reward (1.0 / 0.0) is the highest-noise signal across these four examples. Every interaction either converts or it does not — there is no partial signal. This is why convergence requires more total interactions than examples with continuous rewards, even though the feature space here is the simplest (3 features).

---

## Related Examples

- [`01_dynamic_pricing_sell_through.md`](../01_dynamic_pricing_sell_through/01_dynamic_pricing_sell_through.md) — market-state context (temporal bandit) rather than user-state context
- [`24_trial_conversion_budget`](../24_trial_conversion_budget/trial_conversion_pacing.py) — runnable paced-vs-blind simulation with three budgets (demos, compute, discounts)
- [`22_collections_next_best_action`](../22_collections_next_best_action/collections_pacing.py) — pacing a scarce human resource
- [`04_adaptive_clinical_trials.md`](../04_adaptive_clinical_trials/04_adaptive_clinical_trials.md) — high-stakes routing with biomarker context vectors
