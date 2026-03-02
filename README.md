
<div align="center">

  <h1>🎰 BanditDB</h1>
  <p><b>The lightning-fast, drop-in auto-optimizer for Developers and AI Agents.</b></p>

  [![Rust](https://img.shields.io/badge/Rust-1.93+-orange.svg)](https://www.rust-lang.org)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
  [![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
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
* **Instantaneous Learning:** When a user clicks, buys, or engages, you send a `reward`. The matrices update in microseconds. The very next user gets a mathematically optimized experience.
* **Built-in Delayed Rewards:** The database features a highly concurrent TTL cache that automatically remembers the user's context while waiting for their future reward.
* **Data Science Ready:** All interactions are event-sourced to a Write-Ahead Log (WAL) and instantly exportable to heavily compressed **Apache Parquet** files for offline model training in Pandas/Polars.
* **Amnesia-Free AI Agents:** The Python SDK ships with `banditdb-mcp`, a Model Context Protocol server that exposes `get_intuition` and `record_outcome` as native tools. Add it to your `claude_desktop_config.json` and your agent swarm starts learning autonomously — no application code required.

---

## 🚀 Installation

BanditDB consists of the **Rust Engine** (deployed via Docker) and the **Python SDK** (installed via pip).

### 1. Start the Database Engine
Create a `docker-compose.yml` file and start the server:

```yaml
version: '3.8'
services:
  banditdb:
    image: simeonlukov/banditdb:latest # Or build from source!
    ports:
      - "8080:8080"
    volumes:
      - banditdb_data:/data
    environment:
      - DATA_DIR=/data
      - BANDITDB_API_KEY=your-secret-key        # Remove to disable auth (dev mode)
      - BANDITDB_REWARD_TTL_SECS=86400          # How long to remember a context while waiting for its reward (seconds)
      # - BANDITDB_CHECKPOINT_INTERVAL=10000   # Auto-checkpoint after every N rewarded events
      # - BANDITDB_MAX_WAL_SIZE_MB=50          # Auto-checkpoint when WAL exceeds this size (recommended for edge deployments)
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

## 🔌 API Reference

All endpoints accept and return `application/json`. When `BANDITDB_API_KEY` is set, every request except `/health` must include the header `X-Api-Key: <key>`.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | No | Returns `{"status":"ok"}`. Always public — safe for load balancer probes. |
| `GET` | `/campaigns` | Yes | List all live campaigns with their `alpha` and arm count. |
| `GET` | `/campaign/:id` | Yes | Full diagnostic for one campaign: per-arm `theta`, `theta_norm`, `prediction_count`, `reward_count`, and campaign-level totals. Returns 404 if not found. |
| `POST` | `/campaign` | Yes | Create a new campaign. Body: `{"campaign_id","arms","feature_dim","alpha"}`. `alpha` is optional (default `1.0`) — lower values (e.g. `0.1`) exploit learned knowledge more aggressively; higher values (e.g. `3.0`) keep exploring uncertain arms longer. |
| `DELETE` | `/campaign/:id` | Yes | Delete a campaign and write a `CampaignDeleted` event to the WAL. Returns 404 if not found. |
| `POST` | `/predict` | Yes | Given a context vector, returns the optimal arm and an interaction ID. Body: `{"campaign_id","context"}` |
| `POST` | `/reward` | Yes | Close the feedback loop. Body: `{"interaction_id","reward"}`. Reward must be normalised to `[0, 1]`. |
| `POST` | `/checkpoint` | Yes | Flush the WAL, snapshot all campaign matrices to `checkpoint.json`, write completed prediction→reward pairs to per-campaign Parquet files in `exports/`, and rotate the WAL. Returns a summary string. |
| `GET` | `/export` | Yes | List the per-campaign Parquet files available in the `exports/` directory. Files are created by `POST /checkpoint`. |

Error responses are always structured: `{"error": "<message>"}` with an appropriate HTTP status code.

---

## 📖 Quick Start

### AI Agent LLM Routing (Cost Optimisation)
**The Goal:** AI agents blindly routing every task to GPT-4o spend 60× more than necessary. Simple summarisation, classification, and Q&A tasks are solved equally well by a model that costs $0.25/1M tokens instead of $15.00. BanditDB learns — from every agent in your swarm — exactly which model wins for which type of task.

```python
from banditdb import Client

db = Client("http://localhost:8080", api_key="your-secret-key")

# 1. Register the routing campaign once
db.create_campaign("llm_routing", ["gpt-4o", "claude-haiku", "llama-3"], feature_dim=3)

# 2. An agent receives a task. Build its context vector.
# Context: [prompt_length_normalized, math_complexity, budget_cents_normalized]
task_context = [0.2, 0.9, 0.1]

# 3. Ask BanditDB which model to use
model, interaction_id = db.predict("llm_routing", task_context)
print(f"Routing to: {model}")  # e.g., "claude-haiku"

# 4. Run the task. Report success or failure as a reward.
# Every agent in the swarm contributes to — and benefits from — this shared memory.
db.reward(interaction_id, reward=1.0)  # 0.0 if the model failed the task
```

### Native Agent Tool Use (MCP)

If you are running an AI agent rather than writing application code, `banditdb-mcp` exposes the same predict→reward loop as native MCP tools that any Claude-based agent can call directly.

```bash
pip install banditdb-python
```

Add the server to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "banditdb": {
      "command": "banditdb-mcp",
      "env": {
        "BANDITDB_URL": "http://localhost:8080",
        "BANDITDB_API_KEY": "your-secret-key"
      }
    }
  }
}
```

The agent now has five tools available:

| Tool | Arguments | What it does |
|------|-----------|--------------|
| `create_campaign` | `campaign_id`, `arms`, `feature_dim`, `alpha=1.0` | Create a new decision campaign |
| `list_campaigns` | — | List all active campaigns with arm count and alpha |
| `campaign_diagnostics` | `campaign_id` | Per-arm `theta_norm`, prediction count, reward rate — use when a campaign isn't learning |
| `get_intuition` | `campaign_id`, `context` | Returns the recommended action and an `interaction_id` to save |
| `record_outcome` | `interaction_id`, `reward` | Reports success (1.0) or failure (0.0) and updates the model |

Every agent in a swarm shares the same BanditDB instance, so the learned model improves with every interaction across the entire fleet.

For more domain-specific walkthroughs — e-commerce personalisation, dynamic pricing — browse the [`examples/`](./examples/) directory.

---

## 📊 The Data Science Escape Hatch

We know Data Scientists hate black boxes. While BanditDB learns instantly in memory, it uses **Event Sourcing** to write every interaction to a disk WAL.

`POST /checkpoint` compiles completed prediction→reward pairs into Snappy-compressed **Apache Parquet** files — one file per campaign — allowing your ML team to perform Offline Policy Evaluation (OPE) directly in Pandas or Polars.

Every prediction will eventually appear in the Parquet file even if its reward arrives hours later. BanditDB re-emits in-flight interactions at each checkpoint so delayed rewards are always captured in a future cycle.

```python
import polars as pl
import requests

HEADERS = {"X-Api-Key": "your-secret-key"}

# Checkpoint: snapshot models, export Parquet, rotate the WAL.
# Call this on a schedule or trigger it from your application.
requests.post("http://localhost:8080/checkpoint", headers=HEADERS)

# One Parquet file per campaign, written to {DATA_DIR}/exports/
# Mount the volume and read directly, or copy out of the container.
df = pl.read_parquet("/data/exports/llm_routing.parquet")

# Flat schema — one row per completed prediction→reward pair:
# interaction_id | arm_id | reward | predicted_at | rewarded_at | feature_0 | feature_1 | ...
print(df.head())
print(df.columns)
```

To list which Parquet files have been written so far:

```bash
curl -s http://localhost:8080/export -H "X-Api-Key: your-secret-key"
# e.g. "Parquet files in /data/exports: [\"llm_routing.parquet\"]"
```

### Inspecting Campaigns via the WAL

The WAL is plain JSONL — every campaign lifecycle event is human-readable on disk.

```bash
# See all campaigns ever created
grep "CampaignCreated" /data/bandit_wal.jsonl | jq '.CampaignCreated.campaign_id'

# See which campaigns have been deleted
grep "CampaignDeleted" /data/bandit_wal.jsonl | jq '.CampaignDeleted.campaign_id'
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