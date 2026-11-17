
<div align="center">

  <h1>🎰 BanditDB</h1>

  [![CI](https://github.com/dynamicpricing-ai/banditdb/actions/workflows/ci.yml/badge.svg)](https://github.com/dynamicpricing-ai/banditdb/actions/workflows/ci.yml)
  [![Rust](https://img.shields.io/badge/Rust-1.93+-orange.svg)](https://www.rust-lang.org)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://pypi.org/project/banditdb-python/)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
  [![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://hub.docker.com/r/simeonlukov/banditdb)

  <br />
</div>

## What is BanditDB?

An in-memory decision database that learns from feedback.

BanditDB is a database written in Rust that runs **contextual bandit** algorithms: **LinUCB**, **Linear Thompson Sampling**, **NeuralLinUCB**, **NeuralThompsonSampling**, and a **Progressive Tournament** that selects the better algorithm as your data grows. Predictions are served via concurrent reads across all CPU cores. A reward takes a write lock only for the microseconds of an O(d²) Sherman–Morrison update. Every decision goes to a write-ahead log, and recovery on restart is automatic.

* **Five algorithms in one binary**, chosen with one field at campaign creation.
* **Two HTTP calls:** `POST /predict` returns which decision to take, `POST /reward` teaches the model what happened.
* **Delayed rewards:** outcomes can arrive hours or days after the decision.
* **Budgets:** give arms a cost and BanditDB paces the spend across the window (Lagrangian pacing).
* **Live catalogues:** add, warm-start, pause and retire arms without disturbing what the others learned.
* **Know when you're done:** `GET /campaign/:id/report` returns 95% CIs per arm and a `converged` flag. Diagnostics detect a campaign that has silently stopped exploring.
* **Data science escape hatch:** propensity-logged Parquet exports, and IPS / SNIPS / Doubly Robust off-policy evaluation in the SDK.
* **Production:** reader / writer / admin keys, multi-tenancy, Prometheus metrics, fsync-durable rewards.
* **Clients:** Python and TypeScript SDKs, an MCP server for AI agents, or plain HTTP.

**Documentation: [banditdb.com/docs](https://banditdb.com/docs/)**

---

## Get started

### BanditDB Cloud (hosted)

Nothing to install. Sign in to the **[BanditDB Console](https://console.banditdb.com)** with Google, name your organization, and create an API key. Your code then calls `https://api.banditdb.com` with that key.

In the console, the **Playground** lets you sketch a campaign (its choices, the signals that describe each case, what success looks like, an optional budget). It shows the code in curl, Python, TypeScript and MCP as you type. **Make campaign** creates it on your tenant. AI apps connect to `https://mcp.banditdb.com/mcp` and sign in with OAuth.

Limits and prices: [banditdb.com/pricing](https://banditdb.com/pricing/). Details: [Installation → BanditDB Cloud](https://banditdb.com/docs/installation/#cloud).

### Self-host

```bash
# Binary (Linux needs glibc >= 2.39, i.e. Ubuntu 24.04+; otherwise use Docker or build from source)
curl -fsSL https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/scripts/install.sh | sh
banditdb

# Docker
docker run -d -p 8080:8080 -v banditdb_data:/data simeonlukov/banditdb:latest

# From source (add --features neural for NeuralLinUCB / NeuralThompsonSampling)
cargo build --release && ./target/release/banditdb
```

The Windows `.exe` is built without the neural algorithms. Docker Compose, Helm, persistence and every environment variable: [Installation](https://banditdb.com/docs/installation/) and [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

### SDKs

```bash
pip install banditdb-python      # Python
npm install banditdb-js          # TypeScript / JavaScript
```

---

## Quick start

A complete create → predict → reward → report cycle. Against Cloud, use `https://api.banditdb.com` and your key. Against a local engine with no keys configured, drop the header.

```bash
export BANDITDB_URL=http://localhost:8080
export BANDITDB_API_KEY=your-key

# Create (once)
curl -s -X POST $BANDITDB_URL/campaign -H "X-Api-Key: $BANDITDB_API_KEY" -H "Content-Type: application/json" \
  -d '{"campaign_id": "sleep",
       "arms": ["decrease_temperature", "decrease_light", "decrease_noise"],
       "feature_dim": 5,
       "metadata": {"features": ["sex", "age_norm", "weight_norm", "activity", "bedtime_norm"]}}'

# Predict: which intervention for this person tonight?
curl -s -X POST $BANDITDB_URL/predict -H "X-Api-Key: $BANDITDB_API_KEY" -H "Content-Type: application/json" \
  -d '{"campaign_id": "sleep", "context": [1.0, 0.35, 0.50, 0.60, 0.96]}'
# {"arm_id":"decrease_temperature","interaction_id":"a1b2c3d4-..."}

# Reward: how well it worked, in [0, 1]
curl -s -X POST $BANDITDB_URL/reward -H "X-Api-Key: $BANDITDB_API_KEY" -H "Content-Type: application/json" \
  -d '{"interaction_id": "a1b2c3d4-...", "reward": 0.80}'

# Is it done?
curl -s $BANDITDB_URL/campaign/sleep/report -H "X-Api-Key: $BANDITDB_API_KEY"
```

The same in Python:

```python
from banditdb import Client

db = Client("http://localhost:8080", api_key="your-key")
db.create_campaign("offers", ["a", "b"], features=["age", "is_mobile"])
arm, interaction_id = db.predict("offers", {"age": 0.34, "is_mobile": 1.0})
db.reward(interaction_id, 1.0)
```

> **Normalise your contexts.** Scale each context vector to unit L2 norm (`normalize_context()` in the SDK, or name your `features` and the SDK does it). The exploration term grows with `‖x‖`. Unnormalised input still converges, only far more slowly, with no error. On UCI shuttle this alone cut cumulative regret from 2,026 to 709. [Why](https://banditdb.com/docs/contexts/#normalise).

---

## Algorithms

| Algorithm | `algorithm` value | When to use | Based on |
|---|---|---|---|
| LinUCB | `"linucb"` (default) | Linear reward, fast convergence, lowest cost. | Li et al. 2010 |
| Linear Thompson Sampling | `"thompson_sampling"` | Bayesian exploration. Concurrent users diversify arm coverage. | Agrawal & Goyal 2013 |
| NeuralLinUCB | `{"neural_lin_ucb": {…}}` | Non-linear reward. MLP embedding plus LinUCB in embedding space. Conservative early. | Xu et al. 2022 |
| NeuralThompsonSampling | `{"neural_thompson_sampling": {…}}` | Same embedding, posterior draws instead of UCB. Better long-run convergence. | Zhang et al. 2021 |
| Progressive | `{"progressive": {…}}` | Unsure. A linear base and a neural challenger run in shadow, and traffic moves to whichever wins repeated SNIPS evaluations. | — |

Configs, the tournament rules and when to pick which: [Choosing an Algorithm](https://banditdb.com/docs/algorithms/).

---

## Documentation

| Topic | Where |
|---|---|
| Installation, Cloud & Playground, Run, Quick Start | [banditdb.com/docs](https://banditdb.com/docs/) |
| Every endpoint | [API Reference](https://banditdb.com/api/) · [`docs/openapi.yaml`](docs/openapi.yaml) (also served at `GET /openapi.yaml`) |
| Algorithms, NeuralLinUCB, model architecture | [Algorithms](https://banditdb.com/docs/algorithms/) · [NeuralLinUCB](https://banditdb.com/docs/neural-linucb/) · [Model Architecture](https://banditdb.com/docs/model-architecture/) |
| Budgets and pacing | [Budgets & Costs](https://banditdb.com/docs/budgets/) · [`docs/lagrangian_pacing_mathematics.md`](docs/lagrangian_pacing_mathematics.md) |
| Adding, warm-starting, pausing arms | [Dynamic Arms](https://banditdb.com/docs/dynamic-arms/) |
| Context normalisation, metadata, `decision_model`, `input_ref` | [Contexts, Metadata & Inputs](https://banditdb.com/docs/contexts/) |
| API keys, roles, multi-tenancy | [Access Control & Tenants](https://banditdb.com/docs/access-control/) |
| Convergence, selection entropy, health, metrics | [Observability & Monitoring](https://banditdb.com/docs/observability/) |
| Parquet export, off-policy evaluation | [Data Science Escape Hatch](https://banditdb.com/docs/data-science/) |
| Durability and recovery | [How Recovery Works](https://banditdb.com/docs/recovery/) · [`docs/HA_RUNBOOK.md`](docs/HA_RUNBOOK.md) |
| Environment variables, metrics, alerts | [`docs/OPERATIONS.md`](docs/OPERATIONS.md) |
| Measured scale limits and the SLA they support | [`docs/PRODUCTION_STAGE1.md`](docs/PRODUCTION_STAGE1.md) |
| Errors and status codes | [Errors](https://banditdb.com/docs/errors/) |
| AI agents (MCP) | [MCP](https://banditdb.com/docs/mcp/) |
| Worked examples | [`examples/`](examples/) |

---

## Benchmarks

| Benchmark | Result |
|---|---|
| Yahoo! R6A vs Open Bandit Pipeline | BanditDB-LinUCB **+61.5%** estimated click-through lift over the logging policy (DRos) |
| MovieLens-100K vs Open Bandit Pipeline | BanditDB-ThompsonSampling **+22.4%** 5-star rate (DRos) |
| UCI shuttle ([Neural TS paper](https://arxiv.org/abs/2010.00827) setup) | LinUCB cumulative regret **709 vs the published 966.6**, untuned |
| Throughput, one 4-core node | **~10,000 predictions/s** (p50 ≈ 300 µs), **~4,400 fsync-durable rewards/s** at concurrency 32 |

Methodology, all policies, confidence intervals and how to reproduce: [Benchmarks](https://banditdb.com/docs/benchmarks/) · [`benchmark/uci/README.md`](benchmark/uci/README.md).

---

## Architecture

* **Compute:** Rust + `ndarray`. Sherman–Morrison rank-one updates keep each reward O(d²) with no matrix inversion.
* **State:** `parking_lot` RwLocks: concurrent reads for predictions, microsecond write locks for reward updates.
* **Delayed rewards:** predictions are matched to their rewards via `moka`, a concurrent TTL cache.
* **Durability:** a group-committed write-ahead log (JSONL or MessagePack), checkpoints and WAL rotation. `/reward` returns only after fsync.
* **Neural:** `candle` MLPs retrained off the request path. Predictions read an immutable weight snapshot.
* **Auth:** constant-time API key comparison.

## Contributing

BanditDB is open source under Apache 2.0. PRs are welcome.

---

## References

The formulas BanditDB implements come from these papers:

1. **LinUCB:** Li, Chu, Langford, Schapire. *A Contextual-Bandit Approach to Personalized News Article Recommendation.* WWW 2010. [arXiv:1003.0146](https://arxiv.org/abs/1003.0146)
2. **Linear Thompson Sampling:** Agrawal, Goyal. *Thompson Sampling for Contextual Bandits with Linear Payoffs.* ICML 2013. [arXiv:1209.3352](https://arxiv.org/abs/1209.3352)
3. **NeuralLinUCB:** Xu, Wen, Zhao, Gu. *Neural Contextual Bandits with Deep Representation and Shallow Exploration.* ICLR 2022. [arXiv:2012.01780](https://arxiv.org/abs/2012.01780)
4. **NeuralThompsonSampling:** Zhang, Zhou, Li, Gu. *Neural Thompson Sampling.* ICLR 2021. [arXiv:2010.00827](https://arxiv.org/abs/2010.00827)
5. **Half-life decay (D-LinUCB):** Russac, Vernade, Cappé. *Weighted Linear Bandits for Non-Stationary Environments.* NeurIPS 2019. [arXiv:1909.09146](https://arxiv.org/abs/1909.09146)
6. **Budget pacing (dual descent on shadow prices):** Balseiro, Lu, Mirrokni. *The Best of Many Worlds: Dual Mirror Descent for Online Allocation Problems.* Operations Research 71, 2023. [arXiv:2011.10124](https://arxiv.org/abs/2011.10124)
