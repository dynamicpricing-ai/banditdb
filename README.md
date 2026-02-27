
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

BanditDB is endlessly versatile. Here are three real-world ways developers are using it today.

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