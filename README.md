
<div align="center">
  
  <h1>🎰 BanditDB</h1>
  <p><b>The ultra-fast, drop-in Personalization & Reinforcement Learning Database.</b></p>

  [![Rust](https://img.shields.io/badge/Rust-1.93+-orange.svg)](https://www.rust-lang.org)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![Docker](https://img.shields.io/badge/Docker-Ready-blue)](#)

  <br />
</div>

## 💡 What is BanditDB?

Standard databases store *what happened*. **BanditDB stores *how to act*.**

BanditDB is a lightweight, lock-free database written in Rust, purpose-built for **Contextual Bandits** and real-time Reinforcement Learning. It handles the complex linear algebra of exploration vs. exploitation (LinUCB, Thompson Sampling) entirely in memory, abstracting the math behind a dead-simple developer API.

### The Problem It Solves
Building a self-learning personalization engine today requires a massive data plumbing nightmare: stitching together Kafka (for event streaming), Redis (for state), a Python worker (for matrix math), and Postgres (for logs). 

**BanditDB replaces all of that with a single 50MB Rust binary.** 

### What's In It For Developers?
* **Zero-Math Machine Learning:** You just pass an array of user features, and BanditDB tells you what to show them.
* **Instantaneous Learning:** When a user clicks, buys, or engages, you send a `reward`. The matrices update in sub-microseconds. The very next user gets a mathematically optimized experience.
* **Built-in Delayed Rewards:** The database features a highly concurrent TTL cache that automatically remembers the user's context while waiting for their future reward.
* **Data Science Ready:** All interactions are event-sourced to a Write-Ahead Log (WAL) and instantly exportable to heavily compressed **Apache Parquet** files for offline model training in Pandas/Polars.
* **Amnesia-Free AI Agents:** Built-in Model Context Protocol (MCP) support allows autonomous LLMs to use the database as a "shared intuition engine."

---

## 🚀 Installation

BanditDB consists of the **Rust Engine** (deployed via Docker) and the **Python SDK** (installed via pip).

### 1. Start the Database Engine
Create a `docker-compose.yml` file and start the server:

```yaml
version: '3.8'
services:
  banditdb:
    image: simeonai/banditdb:latest # Or build from source!
    ports:
      - "8080:8080"
    volumes:
      - banditdb_data:/data
    environment:
      - DATA_DIR=/data
volumes:
  banditdb_data:
```
```bash
docker compose up -d
```

### 2. Install the Developer SDK
```bash
pip install banditdb-python
```

---

## 📖 Quick Start & Core Examples

BanditDB is endlessly versatile. Here are four real-world ways developers are using it today.

### Example 1: E-Commerce Upsell Optimization (Personalization)
**The Goal:** Maximize revenue by offering the right incentive at checkout. Should we offer a 10% discount, free shipping, or nothing at all? Traditional A/B testing wastes traffic. BanditDB learns in real-time.

```python
from banditdb import Client

db = Client("http://localhost:8080")

# 1. Initialize the Campaign (Cold Start)
db.create_campaign("checkout_upsell", ["10_percent_off", "free_shipping", "no_offer"], feature_dim=3)

# 2. A user reaches checkout. Define their context.
# Context:[is_mobile, cart_value_normalized, is_returning_customer]
user_context =[1.0, 0.85, 0.0]

# 3. Ask BanditDB what to show them!
offer_to_show, interaction_id = db.predict("checkout_upsell", user_context)
print(f"Displaying: {offer_to_show}") # e.g., "free_shipping"

# 4. If the user completes the purchase, reward the database!
# The DB instantly updates its linear algebra matrices.
db.reward(interaction_id, reward=120.50) # Reward = Cart Value
```

### Example 2: Adaptive Clinical Trials (High-Stakes Routing)
**The Goal:** In clinical trials, keeping patients on poorly performing treatments for months is dangerous. BanditDB can adaptively route patients to the most effective treatments based on their specific biomarkers in real-time.

```python
# Context:[age_normalized, blood_pressure, biomarker_A_present, biomarker_B_present]
patient_features =[0.65, 0.8, 1.0, 0.0]

db.create_campaign("trial_alpha",["drug_A", "drug_B", "placebo"], feature_dim=4)

# BanditDB calculates the Upper Confidence Bound (UCB) for this specific patient
treatment, interaction_id = db.predict("trial_alpha", patient_features)

# 3 weeks later... the patient's condition improves significantly.
# We send a reward. BanditDB will now confidently route future patients 
# with similar biomarkers to this successful treatment.
db.reward(interaction_id, reward=1.0)
```

---

## 🧠 Example 3: The AI "Hive Mind" (Model Context Protocol)

Standard AI Agents (like Claude or AutoGPT) are amnesiacs. If they choose a slow, expensive LLM for a simple task and fail, they will make the exact same mistake tomorrow. 

**BanditDB acts as a shared Intuition Engine for AI Swarms.** Using the built-in MCP server, agents can ask BanditDB which LLM to route a prompt to based on context and price.

### Running the MCP Server
If you have the Python package installed, you can start the MCP tool natively:
```bash
banditdb-mcp
```
Add it to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "banditdb": {
      "command": "banditdb-mcp",
      "args":[]
    }
  }
}
```

### How the AI Uses It: Dynamic LLM Routing
When you prompt an AI Agent to perform a task, it autonomously interacts with BanditDB using its internal tools:

1. **The Agent observes the task:** 
   *Context vector:* `[prompt_length, math_complexity, budget_cents]` -> `[0.2, 0.9, 0.1]`
2. **The Agent asks BanditDB via MCP:** 
   *"Which model should I route this to? `gpt-4o`, `claude-3-haiku`, or `llama-3`?"*
3. **BanditDB Predicts:** 
   *"Try `claude-3-haiku`."*
4. **The Agent executes the task.** If Haiku fails at the complex math, the Agent autonomously calls the MCP tool to report a `0.0` reward.
5. **The Result:** The Swarm learns. The next time *any* agent in your network encounters a high-math-complexity prompt, BanditDB's updated matrices will route them to `gpt-4o` instead.

---

## 🏷️ Example 4: Dynamic Pricing for Inventory Sell-Through

**The Goal:** A fashion retailer has 10,000 units of summer inventory that *must* clear before Labor Day. Discount too early and you destroy margin on items that would have sold themselves. Discount too late and you're liquidating at a loss with a warehouse full of autumn stock arriving next week.

This example is architecturally different from Examples 1 and 2: the context does not describe a *user* — it describes the *state of the market right now*. Sell-through rate, days to the next holiday, inventory pressure, and competitor positioning all change hourly. BanditDB learns which pricing strategy wins under each combination of these signals, turning your pricing engine into a self-calibrating market observer.

```python
import requests
import datetime

db_url = "http://localhost:8080"

# Four pricing strategies: from hold-margin to emergency liquidation.
requests.post(f"{db_url}/campaign", json={
    "campaign_id":  "summer_clearance",
    "arms":         ["full_price", "ten_percent_off", "twenty_five_percent_off", "flash_clearance"],
    "feature_dim":  5,
})

def get_market_context(inventory_total, inventory_sold, our_price, competitor_price):
    today        = datetime.date.today()
    season_end   = datetime.date(2025, 9,  1)   # Labor Day: last day summer items move
    next_holiday = datetime.date(2025, 7,  4)   # 4th of July: next demand spike

    days_to_season_end = max(0, (season_end   - today).days)
    days_to_holiday    = max(0, (next_holiday - today).days)

    return [
        inventory_sold / inventory_total,                             # sell_through_rate  [0.0 → 1.0]
        1.0 if today.weekday() >= 5 else 0.0,                        # is_weekend         {0.0, 1.0}
        min(1.0, max(0.0, 1.0 - days_to_holiday    / 30.0)),         # holiday_proximity  spikes in the 30-day run-up
        min(1.0, max(0.0, 1.0 - days_to_season_end / 90.0)),         # season_urgency     rises steeply in the final 90 days
        min(1.0, max(0.0, (competitor_price - our_price) / our_price + 0.5)),  # price_position  0.5 = at parity
    ]

# ── Scenario: Tuesday afternoon, mid-July. 52% sold. Rival just dropped to $42. ──

context = get_market_context(
    inventory_total  = 10_000,
    inventory_sold   = 5_200,
    our_price        = 49.99,
    competitor_price = 42.00,
)
# Resulting context vector: [0.52, 0.0, 0.67, 0.44, 0.16]
#   sell_through=0.52  → half the warehouse is still full
#   is_weekend=0.0     → it's Tuesday, foot traffic is low
#   holiday_proxy=0.67 → 4th of July was 10 days ago, demand spike is fading
#   season_urgency=0.44 → 6 weeks left, pressure is building
#   price_position=0.16 → competitor is significantly cheaper

price_tier, interaction_id = requests.post(f"{db_url}/predict", json={
    "campaign_id": "summer_clearance",
    "context":     context,
}).json().values()

print(f"BanditDB recommends: {price_tier}")  # e.g. "twenty_five_percent_off"

# ── 1 hour later: 43 units moved at the recommended price tier. ──────────────

# Reward design: what fraction of the theoretical best outcome did we achieve?
# Best case = selling the target volume (50 units/hr) at full-price margin ($29.99/unit).
# This naturally normalizes the reward to [0, 1] across all price tiers.
units_sold          = 43
margin_after_disc   = 18.50   # gross margin per unit at 25% off
full_price_margin   = 29.99   # gross margin per unit at full price
target_units_hr     = 50      # expected units/hr for this time slot

reward = (units_sold * margin_after_disc) / (target_units_hr * full_price_margin)
reward = min(1.0, reward)     # cap at 1.0 on over-performance

requests.post(f"{db_url}/reward", json={
    "interaction_id": interaction_id,
    "reward":         round(reward, 4),
})
```

**What BanditDB learns over a full season:**

| Market State | Context Snapshot | Learned Optimal Arm | Why |
|---|---|---|---|
| Holiday weekend, low inventory | `[0.88, 1.0, 0.95, 0.3, 0.5]` | `full_price` | Organic demand is at its peak. Discounting is pure margin destruction. |
| Weekday slump, competitor undercut | `[0.52, 0.0, 0.1, 0.44, 0.16]` | `twenty_five_percent_off` | Demand is soft and a rival is cheaper. Move stock before they take the sale. |
| High sell-through, mid-season | `[0.89, 0.0, 0.2, 0.3, 0.5]` | `full_price` | Scarcity is doing the selling. Remaining units are worth more, not less. |
| End of season, unsold majority | `[0.54, 0.0, 0.05, 0.97, 0.5]` | `flash_clearance` | 5 days to season end with 4,600 units remaining. Dead stock costs more than the discount. |

> **The compounding insight:** a traditional rule-based system requires a pricing analyst to hand-write every one of these `if/elif` branches — and they still miss the non-obvious interactions (e.g., a holiday on a *Tuesday* behaves differently from a holiday on a *Friday*). BanditDB discovers all of them automatically from the reward signal.

---

## 📊 The Data Science Escape Hatch

We know Data Scientists hate black boxes. While BanditDB learns instantly in memory, it uses **Event Sourcing** to write every interaction to a disk WAL. 

With a single HTTP call, BanditDB natively compiles this log into heavily compressed **Apache Parquet** files, allowing your ML team to perform Offline Policy Evaluation (OPE) instantly.

```python
import polars as pl
import requests

# Export the database memory to a Data Lake
requests.get("http://localhost:8080/export")

# Load instantly into Polars or Pandas
df = pl.read_parquet("/data/bandit_logs_latest.parquet")

# View the exact prediction histories, contexts, and propensity scores
predictions = df.select(pl.col("Predicted")).unnest("Predicted").drop_nulls()
print(predictions)
```

---

## 🏗 Architecture Under the Hood

*   **Compute:** Rust + `ndarray` using SIMD-accelerated Sherman-Morrison rank-1 matrix updates. Matrix inversion is mathematically bypassed for $O(d^2)$ latency.
*   **State:** Lock-free concurrency using `parking_lot` RwLocks. Tens of thousands of predictions can be read concurrently across all CPU cores without blocking.
*   **Memory:** Delayed rewards are mapped to historical context vectors via `moka`, a blazing-fast concurrent TTL cache that prevents OOM crashes.
*   **Durability:** Asynchronous MPSC channels pipe interactions to a JSON-lines WAL for perfect crash recovery without impacting API latency.

## 🤝 Contributing
BanditDB is an open-source project. Whether you want to add Proximal Policy Optimization (PPO) for Version 2, optimize the SIMD math routines, or build SDKs for Go and TypeScript, PRs are welcome! 

Visit [banditdb.com](https://banditdb.com) to read the full documentation.