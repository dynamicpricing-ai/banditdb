
<div align="center">

  <h1>🎰 BanditDB</h1>

  [![Rust](https://img.shields.io/badge/Rust-1.93+-orange.svg)](https://www.rust-lang.org)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
  [![Docker](https://img.shields.io/badge/Docker-Ready-blue)](#)

  <br />
</div>

## What is BanditDB?

An in-memory decision database that learns from feedback.

BanditDB is a database written in Rust that runs **Contextual Bandit** algorithms — **LinUCB**, **Linear Thompson Sampling**, **NeuralLinUCB**, **NeuralThompsonSampling** (Zhang et al. 2021), and a **Progressive Tournament** that autonomously selects the best algorithm as your data grows. Predictions are served via concurrent reads across all CPU cores; rewards trigger microsecond write locks held only for the duration of a matrix update. Full backup and restore via a rotational Write-Ahead Log + Checkpoint. Recovery on restart is automatic.

* **Five algorithms in one binary:** LinUCB, Thompson Sampling, NeuralLinUCB, NeuralThompsonSampling, and Progressive Tournament (autonomous algorithm selection via SNIPS-weighted shadow learning).
* **Two HTTP calls:** `POST /predict` returns which decision to take, `POST /reward` updates the model.
* **Built-in convergence signal:** `GET /campaign/:id/report` returns 95% CI bounds per arm and a `converged` flag — know exactly when to stop the experiment.
* Delayed rewards handled via a TTL cache — rewards can arrive hours or days after the prediction.
* Every interaction is appended to a Write-Ahead Log and exportable to Parquet for offline analysis.
* RBAC with reader / writer / admin roles; optional multi-tenancy with namespace isolation.
* Python SDK (with MCP server for AI agents) and JavaScript / TypeScript SDK.
* IPS and Doubly Robust estimators for offline policy evaluation.

---

## Algorithms

| Algorithm | When to use |
|-----------|-------------|
| `"linucb"` | Default. Linear reward signal, fast convergence, lowest computational cost. |
| `"thompson_sampling"` | Bayesian exploration. Better empirical performance when rewards are highly stochastic. |
| `{"neural_lin_ucb": {...}}` | Non-linear reward function. MLP embedding + LinUCB in embedding space. Retrains at checkpoint. Conservative exploration — good when early mistakes are costly. |
| `{"neural_thompson_sampling": {...}}` | Same MLP embedding as NeuralLinUCB, but Thompson Sampling draws instead of UCB bounds. More diverse early exploration — better long-run convergence at 20K+ interactions. |
| `{"progressive": {...}}` | Unsure which algorithm fits? BanditDB runs a base and a challenger in parallel and shifts traffic to the winner. |

### NeuralLinUCB

Learns a deep embedding of the context vector, then applies LinUCB in the embedding space. Use when the reward function is non-linear in raw features and early stability is important. The MLP retrains at every `POST /checkpoint` call.

```python
from banditdb import Client, NeuralLinUCBConfig

db = Client("http://localhost:8080")
cfg = NeuralLinUCBConfig(
    context_dim=10,   # must match feature_dim
    embed_dim=32,     # arm matrix dimension
    hidden_dim=128,
    hidden_layers=2,
    retrain_every=200,
)
db.create_campaign("prices", ["10", "15", "20"], feature_dim=10, algorithm=cfg)
```

### NeuralThompsonSampling

Same MLP architecture and retrain procedure as NeuralLinUCB, but exploration uses Thompson Sampling: at predict time, arm weights are sampled from the posterior `w ~ N(θ, σ²A⁻¹)` instead of computing a UCB bound. This produces natural diversity across concurrent requests — different users in the same segment get different arms, generating richer coverage earlier.

**Choose NeuralThompsonSampling when:**
- You have large homogeneous segments (many users, same context cluster)
- Long-run convergence matters more than early-step stability
- You want to match the [NeuralTS paper (Zhang et al. 2021)](https://arxiv.org/abs/2010.00827) benchmark

```python
from banditdb import Client, NeuralLinUCBConfig

db = Client("http://localhost:8080")
cfg = NeuralLinUCBConfig(          # same config struct as NeuralLinUCB
    context_dim=128,
    embed_dim=16,
    hidden_dim=64,
    hidden_layers=2,
    retrain_every=100,
    retrain_steps=200,
)
db.create_campaign(
    "finmedia_churn",
    ["no_action", "personalized_digest", "premium_trial",
     "analyst_report_gift", "discount_offer", "account_manager_call"],
    feature_dim=128,
    algorithm={"neural_thompson_sampling": cfg},
)
```

**NeuralLinUCB vs NeuralThompsonSampling at a glance:**

| | NeuralLinUCB | NeuralThompsonSampling |
|---|---|---|
| Exploration | UCB upper bound (deterministic) | Posterior sample (stochastic) |
| Early regret | Lower | Higher |
| Convergence at 20K+ | Slower | Faster |
| Best for | Production safety | Maximum long-run performance |
| Paper | Li et al. 2010 + neural ext. | Zhang et al. 2021 |

### Progressive Tournament

Runs two algorithms side-by-side in shadow mode. Every reward updates both models. At each checkpoint, both are evaluated with SNIPS: if the challenger wins `required_wins` consecutive rounds by > 10%, one `step_bps` of traffic (default 10%) shifts to the challenger — and vice-versa. Traffic is bounded: 10% floor to 90% ceiling.

```python
from banditdb import Client, NeuralLinUCBConfig, ProgressiveConfig

db = Client("http://localhost:8080")
cfg = ProgressiveConfig(
    base="linucb",
    challenger=NeuralLinUCBConfig(context_dim=10, embed_dim=32),
    min_obs=100,
    required_wins=3,
    step_bps=1000,   # 10% per confirmed win run
)
db.create_campaign("prices", ["10", "15", "20"], feature_dim=10, algorithm=cfg)
```

Watch the tournament live:
```bash
curl http://localhost:8080/campaign/prices/diagnostics
# → { "challenger_traffic_pct": 30.0, "tournament_win_streak": 2, ... }
```

---

## Installation

BanditDB consists of the **Rust Engine** (~11 MB native binary — also available as a Docker image and Helm chart), the **Python SDK** (pip), and the **JavaScript / TypeScript SDK** (npm, zero runtime dependencies).

### 1. Start the Database Engine

**Binary (fastest, no Docker required):**
```bash
curl -fsSL https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/scripts/install.sh | sh
banditdb
curl http://localhost:8080/health   # {"status":"ok"}
```

**Docker:**
```bash
docker run -d -p 8080:8080 simeonlukov/banditdb:latest
```

**Production (with persistence and RBAC):**
```yaml
# docker-compose.yml
version: '3.8'
services:
  banditdb:
    image: simeonlukov/banditdb:latest
    ports:
      - "8080:8080"
    volumes:
      - banditdb_data:/data
    environment:
      - DATA_DIR=/data
      # Role-based auth — format: <key>=<role>  or  <key>=<role>:<tenant_id>
      # Roles: admin (all), writer (predict/reward), reader (GET only)
      - BANDITDB_API_KEYS=admin-key=admin,app-key=writer,dash-key=reader
      - BANDITDB_REWARD_TTL_SECS=86400
      # - BANDITDB_CHECKPOINT_INTERVAL=10000   # auto-checkpoint after N rewarded events
      # - BANDITDB_MAX_WAL_SIZE_MB=50          # auto-checkpoint when WAL exceeds N MB
      # - BANDITDB_WAL_FORMAT=msgpack          # binary WAL (smaller, faster I/O)
      # - BANDITDB_TENANT_MODE=true            # strict tenant isolation
volumes:
  banditdb_data:
```
```bash
docker compose up -d
```

**Kubernetes (Helm):**
```bash
helm install banditdb ./helm/banditdb \
  --set auth.apiKeys="admin-key=admin,app-key=writer" \
  --set persistence.size=10Gi \
  --set ingress.enabled=true \
  --set ingress.host=banditdb.example.com
```

The Helm chart at `helm/banditdb/` includes a PersistentVolumeClaim, Secret for API keys, liveness/readiness probes, and a `Recreate` deployment strategy (required for single-writer WAL safety).

### 2. Install the SDKs

**Python:**
```bash
pip install banditdb-python
```

**JavaScript / TypeScript (zero runtime dependencies):**
```bash
npm install banditdb-js
```

---

## API Reference

All endpoints accept and return `application/json`. When `BANDITDB_API_KEYS` is set, every request except `/health` and `/metrics` must include `X-Api-Key: <key>`. Use `BANDITDB_API_KEY` (singular) for a single admin key in development.

| Method | Endpoint | Min Role | Description |
|--------|----------|----------|-------------|
| `GET` | `/health` | — | Returns `{"status":"ok"\|"degraded", "campaigns":{...}}`. Always public — safe for load balancer probes. Entropy collapse raises overall status to `"degraded"`. |
| `GET` | `/metrics` | — | Prometheus text-format metrics. Public unless `BANDITDB_METRICS_PUBLIC=false`. |
| `GET` | `/openapi.yaml` | — | OpenAPI 3.1 specification (this API). |
| `GET` | `/campaigns` | reader | List all campaigns with algorithm, arm count, and metadata. |
| `GET` | `/campaign/:id` | reader | Full per-arm state: theta vectors, reward counts, campaign-level totals. |
| `POST` | `/campaign` | admin | Create a campaign. Body: `{campaign_id, arms, feature_dim, alpha?, algorithm?, metadata?}`. |
| `DELETE` | `/campaign/:id` | admin | Permanently delete a campaign. Irreversible — use `/archive` for soft-delete. |
| `POST` | `/campaign/:id/archive` | admin | Soft-delete. Frozen campaign preserves all data; recoverable via `/restore`. |
| `POST` | `/campaign/:id/restore` | admin | Restore an archived campaign to active status. |
| `GET` | `/campaign/:id/report` | reader | Convergence report: mean reward per arm with 95% CI, leading arm, `converged` flag. |
| `GET` | `/campaign/:id/diagnostics` | reader | Operator diagnostics: theta norms, A_inv bounds, tournament traffic %, neural buffer size, **selection entropy with collapse detection**. |
| `POST` | `/predict` | writer | Select the best arm for a context vector. Returns `{arm_id, interaction_id}`. |
| `POST` | `/batch_predict` | writer | Predict for up to 100 campaign/context pairs in one call. Per-item failures inline. |
| `POST` | `/reward` | writer | Record the outcome. Body: `{interaction_id, reward}`. Reward must be in `[0, 1]`. |
| `POST` | `/checkpoint` | admin | Flush WAL, write Parquet shards, run neural retrain + tournament eval, rotate WAL. |
| `GET` | `/export` | reader | List per-campaign Parquet files in the `exports/` directory. |

Error responses are always `{"error": "<message>"}` with an appropriate HTTP status code.

### Convergence Report

`GET /campaign/:id/report` answers "is this campaign done?":

```json
{
  "campaign_id": "prices",
  "leading_arm": "15",
  "converged": true,
  "overall_reward_rate": 0.74,
  "arms": {
    "10": { "traffic_share": 0.18, "predictions": 312,  "rewards": 89,  "mean_reward": 0.61, "reward_lower_ci": 0.55, "reward_upper_ci": 0.67 },
    "15": { "traffic_share": 0.64, "predictions": 1104, "rewards": 408, "mean_reward": 0.79, "reward_lower_ci": 0.76, "reward_upper_ci": 0.82 },
    "20": { "traffic_share": 0.18, "predictions": 319,  "rewards": 98,  "mean_reward": 0.62, "reward_lower_ci": 0.56, "reward_upper_ci": 0.68 }
  }
}
```

| `converged` | Meaning |
|-------------|---------|
| `true` | Leading arm's 95% CI lower bound exceeds every other arm's upper bound — statistically significant. Stop. |
| `false` | Leading arm is ahead but CIs overlap — keep collecting data. |
| `null` | Fewer than 30 rewards per arm — insufficient data. |

Cross-validate with `causal_analysis()` (Python SDK): if `arm.traffic_share` matches the causal assignment percentages, the bandit has converged to the correct causal structure.

---

### Entropy Alerting

`GET /campaign/:id/diagnostics` includes a live selection entropy signal that detects when a campaign has stopped exploring — silently behaving like a static policy.

```json
{
  "selection_entropy": 0.09,
  "entropy_status": "critical",
  "entropy_trend": "falling",
  "converged": false,
  "likely_cause": "recent_collapse",
  "suggested_action": "Entropy dropped since last checkpoint. Check reward pipeline for bugs or recent config changes."
}
```

| `entropy_status` | Meaning |
|-----------------|---------|
| `ok` | Entropy is healthy, or the campaign has statistically converged (Guard 1), or fewer than 500 predictions (Guard 2). |
| `warning` | Entropy < 0.4 — one arm is absorbing most traffic without a convergence signal. |
| `critical` | Entropy < 0.2 — near-total collapse. Likely cause and suggested action are always included. |

| `entropy_trend` | Meaning |
|----------------|---------|
| `stable` | No significant change since last checkpoint. |
| `falling` | Dropped > 0.1 since last checkpoint — recent event (bug, deploy, cohort shift). |
| `recovering` | Recovered > 0.1 since last checkpoint. |
| `unknown` | No checkpoint has been written yet. |

`GET /health` aggregates entropy across all active campaigns:

```json
{
  "status": "degraded",
  "campaigns": {
    "prices":  { "entropy": 0.09, "status": "critical" },
    "banners": { "entropy": 0.71, "status": "ok" }
  }
}
```

HTTP status is `200` for entropy issues (data quality, not service availability) and `503` only when the WAL writer is unavailable.

---

## RBAC & Multi-Tenancy

### Role-Based Access Control

Set multiple API keys with roles via `BANDITDB_API_KEYS` (comma-separated `key=role` pairs):

```
BANDITDB_API_KEYS=admin-key=admin,app-key=writer,dash-key=reader
```

| Role | Allowed endpoints |
|------|------------------|
| `admin` | All operations |
| `writer` | `POST /predict`, `POST /batch_predict`, `POST /reward` |
| `reader` | All `GET` endpoints |

### Multi-Tenancy

Append a tenant ID to the key definition: `<key>=<role>:<tenant_id>`:

```
BANDITDB_API_KEYS=team-a-key=admin:team-a,team-b-key=admin:team-b
BANDITDB_TENANT_MODE=true   # strict: tenants cannot see each other's campaigns
```

Campaign IDs are automatically namespaced — `team-a` creating `prices` stores it as `team-a/prices` internally, transparent to the API caller. With `BANDITDB_TENANT_MODE=true`, cross-tenant reads return 404.

---

## Live Sandbox

A public sandbox runs at **`https://sandbox.banditdb.com`** with three pre-loaded demo campaigns. It resets nightly at 03:00 UTC.

| Campaign | Arms | Context |
|----------|------|---------|
| `sleep` | `decrease_temperature`, `decrease_light`, `decrease_noise` | `[sex, age/100, weight_kg/150, activity_0–1, bedtime_hour/24]` |
| `prompt_strategy` | `zero_shot`, `chain_of_thought`, `few_shot`, `structured_output` | `[task_complexity, domain, input_length_norm, session_turn_norm, user_expertise]` |
| `client_intake` | `schedule_consultation`, `send_intake_form`, `refer_to_partner_firm`, `decline` | `[case_value_norm, matter_complexity, org_size_norm, conflict_risk, capacity_norm]` |

```bash
curl -s https://sandbox.banditdb.com/campaigns -H "X-Api-Key: banditdb-demo"

curl -s -X POST https://sandbox.banditdb.com/predict \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: banditdb-demo" \
  -d '{"campaign_id":"sleep","context":[1.0, 0.35, 0.50, 0.60, 0.96]}'
```

---

## Try It — CURL

A complete predict→reward→report cycle for a **sleep improvement** campaign.

**Create** (run once):
```bash
curl -s -X POST http://localhost:8080/campaign \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": "sleep",
    "arms": ["decrease_temperature","decrease_light","decrease_noise"],
    "feature_dim": 5,
    "metadata": {
      "owner": "wellness-team",
      "features": ["sex","age_norm","weight_norm","activity","bedtime_norm"]
    }
  }'
```

**Predict:**
```bash
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"campaign_id":"sleep","context":[1.0, 0.35, 0.50, 0.60, 0.96]}'
```
> `{"arm_id":"decrease_temperature","interaction_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890"}`

**Reward** (sleep quality improved 80%):
```bash
curl -s -X POST http://localhost:8080/reward \
  -H "Content-Type: application/json" \
  -d '{"interaction_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","reward":0.80}'
```
> `"OK"`

**Check convergence:**
```bash
curl -s http://localhost:8080/campaign/sleep/report
```

---

## Documentation

Full documentation — Quick Start, Algorithm guide, Data Science / OPE, and Recovery — is at:

**[https://banditdb.com/docs.html](https://banditdb.com/docs.html)**

---

## Benchmark

BanditDB is validated against the **MovieLens 100K** dataset using the standard [replay method](https://arxiv.org/abs/1003.5956) (Li et al. 2010) — the same unbiased offline estimator used in the original LinUCB paper.

**Setup:** 92,698 ratings → 6 genre arms. 44-dimensional context with binned demographics, causal user history, feature-crossing, and era preferences. Dynamic Per-User Alpha Decay.

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
python benchmark/movielens/convert.py
docker cp benchmark/data/movielens_train.jsonl <container>:/data/bandit_wal.jsonl
docker exec <container> rm -f /data/checkpoint.json
docker compose restart
python benchmark/movielens/evaluate_improved.py
```

### Throughput

Peak throughput on a commodity 4-core machine over loopback: **~10,000 predictions/second** at concurrency=64 with p50 latency of ~300µs.

```bash
docker run -d -p 8080:8080 simeonlukov/banditdb:latest
python benchmark/throughput/bench.py
```

---

## Architecture

* **Compute:** Rust + `ndarray` using SIMD-accelerated Sherman-Morrison rank-1 matrix updates. Matrix inversion is mathematically bypassed for $O(d^2)$ latency.
* **State:** `parking_lot` RwLocks — concurrent reads for predictions across all CPU cores; µs write locks for reward updates only.
* **Memory:** Delayed rewards mapped to historical context vectors via `moka`, a fast concurrent TTL cache.
* **Durability:** Asynchronous MPSC channel pipes interactions to a WAL (JSONL or binary MessagePack). Rotation and recovery are automatic.
* **Auth:** Constant-time API key comparison via bitwise accumulation — no branch on comparison result (timing-safe against timing oracle attacks).

## Contributing

BanditDB is open source. PRs are welcome!

Visit [banditdb.com](https://banditdb.com) to read the full documentation.
