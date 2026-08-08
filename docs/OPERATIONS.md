# BanditDB Operations Reference

Configuration, metrics, and alerting. Companion to [HA_RUNBOOK.md](HA_RUNBOOK.md)
(failure modes and recovery) and [PRODUCTION_STAGE1.md](PRODUCTION_STAGE1.md)
(measured limits and the SLA those limits support).

---

## Environment variables

Every variable the binary reads. Defaults are what you get with the variable unset.

### Core

| Variable | Default | Notes |
|---|---|---|
| `DATA_DIR` | `.` | WAL, checkpoints, `neural/`, `exports/`. An exclusive `flock` is taken on it — a second process refuses to start. |
| `PORT` | `8080` | |
| `LOG_FORMAT` | text | `json` for structured logging. |
| `RUST_LOG` | `info` | |

### Authentication and access

| Variable | Default | Notes |
|---|---|---|
| `BANDITDB_API_KEYS` | *(none)* | `key=role` or `key=role:tenant`, semicolon-separated. Roles: `admin`, `writer`, `reader`. |
| `BANDITDB_API_KEY` | *(none)* | Legacy single admin key. |
| `BANDITDB_REQUIRE_AUTH` | `false` | **Set `true` in production.** With no keys configured every caller is granted admin; this makes that fatal at startup instead of a log line. |
| `BANDITDB_TENANT_MODE` | `false` | Namespaces campaigns per tenant. Logical isolation, not a hard security boundary — see the SLA caveat. |
| `BANDITDB_CORS_ORIGINS` | *(deny all)* | Comma-separated allow-list. `*` permits any origin; avoid it if keys ever reach client-side code. |
| `BANDITDB_METRICS_PUBLIC` | `false` | `/metrics` names campaigns and arms, so it requires a key by default. |
| `BANDITDB_RATE_LIMIT_PER_SEC` | `1000` | Per-key limit. **Raise this for load tests and bulk ingest** — the default will silently fail a benchmark loop with 429s. |
| `BANDITDB_AUDIT_LOG` | *(disabled)* | Path to a JSONL audit file for write-path events. |

### Durability

| Variable | Default | Notes |
|---|---|---|
| `BANDITDB_FSYNC_INTERVAL_MS` | `200` | Group-commit window. **Measured to have almost no effect** across 0–1000 ms, because the writer also syncs whenever it goes idle. Not a useful tuning lever; leave it. |
| `BANDITDB_CHECKPOINT_INTERVAL` | *(disabled)* | Auto-checkpoint after N rewards. Controls WAL size and replay time — **not** durability. |
| `BANDITDB_MAX_WAL_SIZE_MB` | *(disabled)* | Auto-checkpoint when the WAL exceeds this. |
| `BANDITDB_WAL_FORMAT` | `json` | `msgpack` for a 3–5× smaller WAL. |
| `BANDITDB_REWARD_TTL_SECS` | `86400` | How long a prediction stays matchable. |
| `BANDITDB_EXPORT_RETAIN_SHARDS` | `50` | Parquet shards kept per campaign. `0` keeps everything — `exports/` then grows until the volume fills, which stops checkpointing. |
| `BANDITDB_ALLOW_CORRUPT_CHECKPOINT` | `false` | Escape hatch. Normally a corrupt checkpoint with no usable `checkpoint.prev` stops startup; `true` starts empty and **accepts the data loss**. |
| `BANDITDB_SKIP_DATA_DIR_LOCK` | `false` | Bypasses the single-writer lock. For read-only forensics on a copied data directory. **Never for serving** — two writers corrupt the WAL silently. |

### Limits

| Variable | Default | Notes |
|---|---|---|
| `BANDITDB_MAX_ARMS` | `1000` | Per campaign. |
| `BANDITDB_MAX_FEATURE_DIM` | `4096` | Also caps context length at predict time. |
| `BANDITDB_MAX_CONTEXT_MAGNITUDE` | `1e6` | Rejects values that would overflow the rank-one update. Finiteness is not enough: `1e155` and above squares to infinity, and the resulting NaN persists into the checkpoint. |
| `BANDITDB_MAX_PENDING_INTERACTIONS` | `100000` | Predictions awaiting a reward. ~1 KB each at `context_dim=64`, so the default is ≈70–130 MB. Eviction permanently breaks matching for that prediction. |

Campaign count is **not** capped — an admin key can create campaigns until memory runs out.

### Neural builds only

| Variable | Default | Notes |
|---|---|---|
| `BANDITDB_DEVICE` | auto | `cpu`, `cuda`, `cuda:N`, `metal`. Auto-detects CUDA → Metal → CPU. |
| `BANDITDB_RETRAIN_POLL_SECS` | `2` | Background retrain cadence. `0` falls back to retraining only at checkpoint. |
| `BANDITDB_NEURAL_BUFFER_CAP` | `50000` | Replay buffer entries per campaign. |
| `BANDITDB_NEURAL_BATCH_SIZE` | `4000` | Minibatch drawn per retrain, which decouples per-step cost from buffer size. |

---

## Metrics

`GET /metrics`, Prometheus text format. Requires a reader key unless
`BANDITDB_METRICS_PUBLIC=true`.

### Health and durability

| Metric | Type | Meaning |
|---|---|---|
| `banditdb_wal_healthy` | gauge | `1` healthy, `0` the WAL writer hit an unrecoverable I/O error. At `0` all writes are rejected and `/health` returns 503. |
| `banditdb_wal_channel_available` | gauge | Free slots in the WAL queue. Sustained near zero means the writer cannot keep up. |
| `banditdb_wal_fsync_total` | counter | Group-commit fsyncs. Compare against reward rate to see how effectively the commit window is batching. |
| `banditdb_wal_dropped_total` | counter | **Prediction records discarded** because the writer fell behind. Predictions are best-effort, so the request still succeeded — but a late reward for a dropped record cannot be matched. |

### Interaction cache

| Metric | Type | Meaning |
|---|---|---|
| `banditdb_interactions_pending` | gauge | Predictions awaiting a reward. Approaching `BANDITDB_MAX_PENDING_INTERACTIONS` means evictions are imminent. |
| `banditdb_interactions_evicted_total` | counter | Pending interactions dropped at the capacity limit. **Each one permanently breaks reward matching for that prediction.** |

### Campaigns and arms

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `banditdb_campaigns_active` | gauge | | Non-archived campaigns. |
| `banditdb_campaigns_archived` | gauge | | |
| `banditdb_arm_predictions_total` | counter | `campaign`, `arm` | Traffic per arm. Flat distribution means no learning; total collapse to one arm means either convergence or entropy collapse — `/campaign/:id/diagnostics` distinguishes them. |
| `banditdb_arm_rewards_total` | counter | `campaign`, `arm` | |
| `banditdb_tournament_traffic_bps` | gauge | `campaign` | Progressive challenger share, basis points (1000 = 10%). |

### HTTP

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `banditdb_http_requests_total` | counter | `endpoint`, `status` | `status` is `2xx`/`4xx`/`5xx`. |
| `banditdb_http_request_duration_seconds` | histogram | `endpoint` | Buckets 0.5 ms → 1 s. |

---

## Alerting

Ordered by how much a firing alert should worry you.

| Alert | Condition | Why it matters |
|---|---|---|
| **WAL writer down** | `banditdb_wal_healthy == 0` | All writes rejected. The database is read-only until restarted. |
| **Reward matching breaking** | `rate(banditdb_interactions_evicted_total[5m]) > 0` | Silent model degradation: predictions are being discarded before their rewards arrive, so the model stops learning from them. Nothing else surfaces this. |
| **Prediction logging degraded** | `rate(banditdb_wal_dropped_total[5m]) > 0` | The WAL writer is behind. Requests still succeed, so latency and error-rate dashboards look fine while log records are lost. |
| **Cache near capacity** | `banditdb_interactions_pending > 0.9 * max` | Evictions are about to start. Either reward lag has grown or the cap is too low. |
| **WAL queue saturating** | `banditdb_wal_channel_available < 1000` | Precursor to dropped predictions and rejected rewards. |
| **Entropy collapse** | `/health` reports `degraded` | One arm is taking all traffic without a convergence signal. Check `/campaign/:id/diagnostics` — `converged: true` means it is fine. |
| **Reward latency** | p99 of `banditdb_http_request_duration_seconds{endpoint="reward"}` | Rewards block on fsync. A rising p99 usually means disk latency, not application load. |

Two things deliberately have **no** alert:

- **fsync interval tuning.** Measured to be nearly irrelevant (see PRODUCTION_STAGE1 §6); the idle-sync path dominates.
- **Checkpoint frequency.** It affects WAL size and replay time, not durability. Acknowledged writes are already on disk.

---

## Operational tasks

```bash
# Backup, restore, and — the part that matters — verify
./scripts/backup_restore.sh drill /data          # backup then prove it restores

# Durability regression check (used in CI)
./scripts/crash_injection.sh 25 --strict

# Re-measure published limits after any write-path change
python3 benchmark/scale/limits.py
```

### Sizing

Measured on a 12-core host over loopback — treat as a ceiling, not a cloud figure.
Both paths peak at **concurrency 32**: ~10,000 predict/s (p99 4.6 ms) and
~4,400 reward/s (p99 8.0 ms). Throughput degrades past concurrency 64, so size the
client pool rather than assuming more is better. Full tables in PRODUCTION_STAGE1 §6.

### Reward latency is intentional

`POST /reward` does not return until the record is fsynced, so a 200 response means
it survives power loss. That costs about **3.4 ms at concurrency 1**. Group commit
amortises the fsync across concurrent callers, so throughput still reaches ~4,400/s
— but a strictly serial client pays the full latency on every call. Batch or
parallelise reward submission if throughput matters.
