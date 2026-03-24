
<div align="center">

  <h1>🎰 BanditDB</h1>
  <p><b>The Intuition Database.</b></p>
  <!-- <p>Your app learns from every interaction and makes smarter decisions — automatically.</p> -->

  [![Rust](https://img.shields.io/badge/Rust-1.93+-orange.svg)](https://www.rust-lang.org)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
  [![Docker](https://img.shields.io/badge/Docker-Ready-blue)](#)

  <br />
</div>

## What is BanditDB?

An in-memory decision database that learns from feedback.

Under the hood, BanditDB is a database written in Rust that runs **Contextual Bandit** algorithms — **LinUCB** and **Linear Thompson Sampling**. Predictions are served via concurrent reads across all CPU cores; rewards trigger microsecond write locks held only for the duration of a matrix update. BanditDB provides entire backup and restore via rotational Write-Ahead Log + Checkpoint and exports complete Parquet files. Recovery on restart is automatic.


### High-level features

* **Your app gets smarter automatically.** Pass a context vector, get a decision. Send a reward when it works. BanditDB does the rest.
* **Instant learning.** Matrices update in microseconds. No batch jobs, no retraining, no lag.
* **Built-in delayed rewards.** A concurrent TTL cache remembers the user's context while waiting for their future reward — purchases, conversions, or outcomes that arrive hours later.
* **Data Science escape hatch.** Every interaction is event-sourced and exportable for Offline Policy Evaluation.
* **Python SDK.** ships with `banditdb-mcp`, a MCP server that gives agents native `get_intuition` and `record_outcome` tools. Your agent swarm builds shared intuition — autonomously.

---

## 🚀 Installation

BanditDB consists of the **Rust Engine** (~11MB native binary for Linux, macOS, and Windows — also available as a Docker image) and the **Python SDK** (installed via pip).

### 1. Start the Database Engine

Three ways to get the engine running — pick whichever fits your environment:

**Binary (fastest, no Docker required):**
```bash
curl -fsSL https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/scripts/install.sh | sh
banditdb
```

**Docker:**
```bash
docker run -d -p 8080:8080 simeonlukov/banditdb:latest
```


```bash
curl http://localhost:8080/health   # {"status":"ok"}
```

**Production (with persistence and auth):**
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

## API Reference

All endpoints accept and return `application/json`. When `BANDITDB_API_KEY` is set, every request except `/health` must include the header `X-Api-Key: <key>`.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | No | Returns `{"status":"ok"}`. Always public — safe for load balancer probes. |
| `GET` | `/campaigns` | Yes | List all live campaigns with their `alpha` and arm count. |
| `GET` | `/campaign/:id` | Yes | Full diagnostic for one campaign: per-arm `theta`, `theta_norm`, `prediction_count`, `reward_count`, and campaign-level totals. Returns 404 if not found. |
| `POST` | `/campaign` | Yes | Create a new campaign. Body: `{"campaign_id","arms","feature_dim","alpha","algorithm"}`. `alpha` is optional (default `1.0`). `algorithm` is optional (default `"linucb"`) — also accepts `"thompson_sampling"` for Linear Thompson Sampling, which samples from the posterior instead of adding a UCB bonus. |
| `DELETE` | `/campaign/:id` | Yes | Delete a campaign and write a `CampaignDeleted` event to the WAL. Returns 404 if not found. |
| `POST` | `/predict` | Yes | Given a context vector, returns the optimal arm and an interaction ID. Body: `{"campaign_id","context"}` |
| `POST` | `/reward` | Yes | Close the feedback loop. Body: `{"interaction_id","reward"}`. Reward must be normalised to `[0, 1]`. |
| `POST` | `/checkpoint` | Yes | Flush the WAL, snapshot all campaign matrices to `checkpoint.json`, write completed prediction→reward pairs to per-campaign Parquet files in `exports/`, and rotate the WAL. Returns a summary string. |
| `GET` | `/export` | Yes | List the per-campaign Parquet files available in the `exports/` directory. Files are created by `POST /checkpoint`. |

Error responses are always structured: `{"error": "<message>"}` with an appropriate HTTP status code.

---

## Live Sandbox

Want to try BanditDB without installing anything? A public sandbox runs at **`https://sandbox.banditdb.com`** with three pre-loaded demo campaigns. It resets nightly at 03:00 UTC.

| Campaign | Arms | Context |
|----------|------|---------|
| `sleep` | `decrease_temperature`, `decrease_light`, `decrease_noise` | `[sex, age/100, weight_kg/150, activity_0–1, bedtime_hour/24]` |
| `prompt_strategy` | `zero_shot`, `chain_of_thought`, `few_shot`, `structured_output` | `[task_complexity, domain, input_length_norm, session_turn_norm, user_expertise]` |
| `client_intake` | `schedule_consultation`, `send_intake_form`, `refer_to_partner_firm`, `decline` | `[case_value_norm, matter_complexity, org_size_norm, conflict_risk, capacity_norm]` |

```bash
# List live campaigns
curl -s https://sandbox.banditdb.com/campaigns -H "X-Api-Key: banditdb-demo"

# Get a prediction from the sleep campaign
curl -s -X POST https://sandbox.banditdb.com/predict \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: banditdb-demo" \
  -d '{"campaign_id":"sleep","context":[1.0, 0.35, 0.50, 0.60, 0.96]}'
```

---

## Try It — CURL

A complete predict→reward cycle for a **sleep improvement** campaign.
Arms: `decrease_temperature`, `decrease_light`, `decrease_noise`.
Context vector: `[sex, age/100, weight_kg/150, activity_0–1, bedtime_hour/24]`

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

**Predict** — female, 35yo, 75 kg, moderately active, bedtime 23:00: - see above how the context was calculated out of the actual values to fit in [0-1] interval.
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

After enough predict→reward cycles the model converges: patients with similar profiles are routed to the arm that consistently produced the highest improvement ratio.

---

## Quick Start

### Sleep Improvement
**The Goal:** One-size-fits-all sleep advice ignores individual physiology. A 25-year-old male athlete and a 60-year-old sedentary woman respond differently to the same environmental change. BanditDB learns those differences automatically — routing each participant to the intervention most likely to work for *their* profile, improving with every reported outcome.

```python
from banditdb import Client

db = Client("http://localhost:8080", api_key="your-secret-key")

# 1. Create the campaign once at startup
db.create_campaign(
    "sleep",
    arms=["decrease_temperature", "decrease_light", "decrease_noise"],
    feature_dim=5,
)

# 2. A participant is ready for tonight's intervention. Build their context vector.
# Context: [sex, age/100, weight_kg/150, activity_0–1, bedtime_hour/24]
context = [
    1.0,   # female
    0.35,  # age 35
    0.50,  # 75 kg
    0.60,  # moderately active
    0.96,  # bedtime 23:00
]

# 3. Ask BanditDB which intervention to apply
arm, interaction_id = db.predict("sleep", context)
print(f"Tonight's intervention: {arm}")  # e.g., "decrease_temperature"

# 4. Apply the intervention, then reward the next morning with a PSQI-based score
apply_intervention(user_id, arm)

# next morning: sleep quality improved from 62 → 79
score_before = 62
score_after  = 79
reward = (score_after - score_before) / score_before  # → 0.27

db.reward(interaction_id, reward)
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
| `create_campaign` | `campaign_id`, `arms`, `feature_dim`, `alpha=1.0`, `algorithm="linucb"` | Create a new decision campaign. Set `algorithm="thompson_sampling"` for natural Bayesian exploration — no alpha-sweep needed. |
| `list_campaigns` | — | List all active campaigns with arm count and alpha |
| `campaign_diagnostics` | `campaign_id` | Per-arm `theta_norm`, prediction count, reward rate — use when a campaign isn't learning |
| `get_intuition` | `campaign_id`, `context` | Returns the recommended action and an `interaction_id` to save |
| `record_outcome` | `interaction_id`, `reward` | Reports success (1.0) or failure (0.0) and updates the model |

Every agent in a swarm shares the same BanditDB instance, so the learned model improves with every interaction across the entire fleet.

For more domain-specific walkthroughs — e-commerce personalisation, dynamic pricing — browse the [`examples/`](./examples/) directory.

---

## Choosing an Algorithm

BanditDB supports two contextual bandit algorithms. Both share identical per-arm state (A⁻¹, b, θ), so switching is a single field in the campaign creation call.

| Algorithm | `algorithm` value | How it explores | When to use |
|-----------|------------------|-----------------|-------------|
| **LinUCB** | `"linucb"` (default) | Adds a deterministic UCB bonus: `θ·x + α·√(x·A⁻¹·x)` | When you want predictable, tunable exploration. Sweep `alpha` offline to find the right exploration level. |
| **Linear Thompson Sampling** | `"thompson_sampling"` | Samples θ̃ from the posterior N(θ, α²·A⁻¹) and scores by θ̃·x | When you want natural Bayesian exploration with no sweep. `alpha=1.0` is the principled default — it equals the natural posterior width. Concurrent users automatically diversify arm coverage. |

```python
# LinUCB (default) — same as before
db.create_campaign("ucb_offers", ["offer_a", "offer_b"], feature_dim=3, alpha=1.5)

# Thompson Sampling — natural posterior exploration
db.create_campaign("ts_offers", ["offer_a", "offer_b"], feature_dim=3,
                   algorithm="thompson_sampling")
```

The `algorithm` field is stored in both the WAL and checkpoint files. Old WAL records and checkpoints without an `algorithm` field recover as `"linucb"` automatically.

---

## The Data Science Escape Hatch

We know Data Scientists hate black boxes. While BanditDB learns instantly in memory, it uses **Event Sourcing** to write every interaction to a disk WAL.

`POST /checkpoint` compiles completed prediction→reward pairs into Snappy-compressed **Apache Parquet** files — one file per campaign — allowing your ML team to perform Offline Policy Evaluation (OPE) directly in Pandas or Polars.

Every prediction will eventually appear in the Parquet file even if its reward arrives hours later. BanditDB re-emits in-flight interactions at each checkpoint so delayed rewards are always captured in a future cycle.

Each row includes a **`propensity`** column — the softmax-normalised probability that the logging policy selected the chosen arm given the context (LinUCB campaigns; `null` for Thompson Sampling). This is the `P(a | x)` term required by Inverse Propensity Scoring (IPS) estimators. It answers the question you can't answer without it: *"what would my cumulative reward have been with `alpha=2.0` instead of `1.0`?"* — without running a live experiment.

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
# interaction_id | arm_id | reward | predicted_at | rewarded_at | propensity | feature_0 | feature_1 | ...
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

## How Recovery Works

BanditDB survives crashes and restarts automatically. No manual intervention required. Here is exactly what happens under the hood so you can reason about data loss windows, design your backup strategy, and diagnose recovery edge cases.

### The Two Files

| File | Purpose |
|------|---------|
| `{DATA_DIR}/checkpoint.json` | Snapshot of all campaign matrices (A⁻¹, b, θ, counts) at a specific WAL byte offset |
| `{DATA_DIR}/bandit_wal.jsonl` | Append-only event log: `CampaignCreated`, `Predicted`, `Rewarded`, `CampaignDeleted` |

On startup, BanditDB restores itself in two phases:

### Phase 1 — Load the Checkpoint

If `checkpoint.json` exists, BanditDB reads it and restores all campaign matrices directly into memory. This is instant — no replaying, just deserialisation. The checkpoint also records the WAL byte offset at which it was taken.

If no checkpoint exists, BanditDB starts from an empty state and replays the entire WAL from byte 0.

### Phase 2 — Replay the WAL Tail

BanditDB opens `bandit_wal.jsonl`, seeks to the checkpoint's byte offset, and replays every event written after that point. This catches up on anything that happened since the last checkpoint — recent campaigns, predictions, and rewards.

One edge case: when a checkpoint is taken, the WAL is **rotated** (the pre-checkpoint prefix is discarded and only the tail is kept). After rotation, the stored byte offset in `checkpoint.json` will exceed the current file size. BanditDB detects this and seeks to byte 0 instead, replaying the entire (now-short) tail correctly.

```
Startup sequence:

  checkpoint.json found?
  ├── YES → restore all matrices from snapshot
  │         → open WAL, seek to checkpoint.wal_offset
  │         →   if offset > file size (post-rotation): seek to 0
  │         → replay events from that position
  └── NO  → open WAL, replay from byte 0
```

### What Is the Data Loss Window?

**Everything in the WAL is durable.** The WAL writer task flushes after every write burst and `fsync`s before acknowledging a checkpoint. If you crash between two checkpoints, BanditDB replays the WAL tail on the next start — your model state is fully recovered.

The only unrewarded data at risk is **in-flight predictions** (interactions that were predicted but not yet rewarded at the moment of a crash). These live in the Moka TTL cache in memory. They are not in the WAL until a reward arrives. After a crash, those interaction IDs are lost. Any reward sent for them after restart will return 404.

To mitigate this: checkpoint frequently. BanditDB re-emits in-flight predictions into the WAL tail at each checkpoint, so any reward that arrives before the next crash will be captured.

### What Does `POST /checkpoint` Actually Do?

1. **Flush barrier** — sends a flush message through the WAL channel; the writer drains all pending events and `fsync`s to disk before responding with the confirmed byte offset.
2. **Snapshot** — reads all campaign matrices under a read lock and serialises them to `checkpoint.tmp`, then atomically renames it to `checkpoint.json`.
3. **Parquet export** — reads the WAL from byte 0 to the flush offset, joins `Predicted` + `Rewarded` events on `interaction_id`, and appends matched pairs to per-campaign Parquet files in `{DATA_DIR}/exports/`. Unmatched (in-flight) predictions are **re-emitted** into the WAL tail.
4. **WAL rotation** — truncates the WAL to only the tail (events after the checkpoint offset). Pre-checkpoint history is no longer needed for recovery.

### Parquet Files: Analytics Only, Not Recovery

The Parquet files in `exports/` are **not** used for recovery. They are analytics exports. Losing them does not affect model state. Recovery uses only `checkpoint.json` + `bandit_wal.jsonl`.

### Recommended Production Setup

```bash
# Auto-checkpoint every 10,000 rewards (model stays durable within 10K events)
BANDITDB_CHECKPOINT_INTERVAL=10000

# Or cap WAL size regardless of event count (useful on edge deployments)
BANDITDB_MAX_WAL_SIZE_MB=50

# Back up the two recovery files on a schedule
cp /data/checkpoint.json  /backup/checkpoint-$(date +%s).json
cp /data/bandit_wal.jsonl /backup/wal-$(date +%s).jsonl
```

To move BanditDB to a new host: copy `checkpoint.json` and `bandit_wal.jsonl` to the same `DATA_DIR` on the new machine and start. Recovery is automatic.

---

## 📈 Benchmark

BanditDB is validated against the **MovieLens 100K** dataset using the standard [replay method](https://arxiv.org/abs/1003.5956) (Li et al. 2010) — the same unbiased offline estimator used in the original LinUCB paper.

**Setup:** 92,698 ratings → 6 genre arms (Drama, Comedy, Action, Romance, Thriller, Adventure). 
The final optimized context uses a **44-Dimensional vector** combining binned demographics, causal user history (like rates + exposure counts), feature-crossing (age × genre history), and movie era preferences. The model uses **Dynamic Per-User Alpha Decay** to balance exploration across the 90/10 chronological split.

| Metric | Value |
|--------|-------|
| Train interactions | 83,428 |
| Test interactions | 9,270 |
| Replay match rate | 7.3% |
| Random baseline avg reward | 0.5606 |
| BanditDB avg reward | 0.6984 |
| **Lift over random** | **+24.6%** |

Reproduce the benchmark:
```bash
python benchmark/movielens/convert.py   # download & convert MovieLens 100K
docker cp benchmark/data/movielens_train.jsonl <container>:/data/bandit_wal.jsonl
docker exec <container> rm -f /data/checkpoint.json
docker compose restart
python benchmark/movielens/evaluate_improved.py  # run heavily optimized causal evaluation loop
```

Use `benchmark/movielens/offline_sweep.py` or create your own custom scripts to sweep alpha, reward type, and feature sets offline before deploying entirely to a live BanditDB workflow.

### Throughput Benchmark

To measure predictions-per-second on your own hardware:

```bash
docker run -d -p 8080:8080 simeonlukov/banditdb:latest
python benchmark/throughput/bench.py
```

Sweeps concurrency levels 1→128 and reports p50/p99 latency and RPS at each level. On a commodity 4-core machine over loopback, peak throughput is **~10,000 predictions/second** at concurrency=64 with a p50 latency of ~300µs per request.

---

## 🏗 Architecture Under the Hood

*   **Compute:** Rust + `ndarray` using SIMD-accelerated Sherman-Morrison rank-1 matrix updates. Matrix inversion is mathematically bypassed for $O(d^2)$ latency.
*   **State:** `parking_lot` RwLocks — concurrent reads for predictions across all CPU cores; μs write locks for reward updates only.
*   **Memory:** Delayed rewards are mapped to historical context vectors via `moka`, a blazing-fast concurrent TTL cache that prevents OOM crashes.
*   **Durability:** Asynchronous MPSC channels pipe interactions to a JSON-lines WAL for perfect crash recovery without impacting API latency.

## 🤝 Contributing
BanditDB is an open-source project. Whether you want to add new model for Version 2 or build SDKs for Go and TypeScript, PRs are welcome! 

Visit [banditdb](https://dynamicpricing-ai.github.io/banditdb/) to read the full documentation.