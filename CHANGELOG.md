# Changelog

## v2.0.0 — Production hardening

Closes the durability, isolation, and operational gaps found by three independent
audits. The plan, the reconciliation of those audits, and the measured limits are in
[`docs/PRODUCTION_STAGE1.md`](docs/PRODUCTION_STAGE1.md).

**The headline change: an acknowledged write is now genuinely durable.** Before this
release, `POST /reward` returned 200 as soon as the event was queued in memory — a
crash a millisecond later lost it silently. It now returns only once the record is
fsynced to disk.

---

### Breaking changes

Read this section before upgrading. Several of these fail *silently* — the service
stays up and returns 200s while something you depend on stops working.

#### `/metrics` requires authentication

Previously public. It names campaigns and arms, which in tenant mode exposes your
customer list, so it now needs a reader key.

**A Prometheus scraper will start receiving 401 with no other symptom.** Either give
the scraper a key, or restore the old behaviour:

```bash
BANDITDB_METRICS_PUBLIC=true
```

#### `/health` no longer returns campaign data

The public probe returns `{status, version, features}` only. It sits outside the auth
layer, and the campaign IDs it used to expose carry the tenant prefix — enough for an
anonymous caller to enumerate tenants.

Per-campaign entropy moved to **`GET /health/detail`**, which requires a reader key
and is tenant-scoped. Anything parsing `campaigns` out of `/health` must move.

Kubernetes and load-balancer probes are unaffected.

#### CORS denies all origins by default

Browsers could previously call the API from anywhere, so a key in client-side code
was usable by any site. Set an explicit allow-list:

```bash
BANDITDB_CORS_ORIGINS=https://app.example.com,https://admin.example.com
# or "*" to restore the old behaviour
```

#### Rewards outside `[0, 1]` are rejected

The documented contract was always `[0, 1]`, but the engine accepted anything finite
and quietly corrupted the arm's confidence bounds. Out-of-range values now return
**400**. Callers relying on the old leniency must clamp or rescale.

#### The Rust library write API is async

`reward`, `interact`, `add_campaign`, `delete_campaign`, `archive_campaign`, and
`restore_campaign` return futures — they await the WAL fsync. Add `.await` at every
call site. Sync helpers that call them must become `async fn` too.

HTTP users are unaffected.

#### One process per data directory

`DATA_DIR` is now protected by an exclusive `flock`. A second process refuses to
start rather than interleaving WAL writes and corrupting both copies. If you ran
multiple instances against one volume, that was silently destroying data.

---

### Durability

- **Acknowledged rewards and campaign lifecycle changes are fsynced before the call
  returns.** RPO for acknowledged writes is zero.
- **Predictions are explicitly best-effort.** Under WAL backpressure a prediction
  *log record* is dropped rather than failing the request — watch
  `banditdb_wal_dropped_total`. Model state is never affected.
- **Crash-safe checkpointing.** The checkpoint is fsynced, the directory entry is
  fsynced, and the previous generation is retained as `checkpoint.prev`. A corrupt
  checkpoint falls back to it; if neither is readable the server **refuses to start**
  instead of coming up empty and overwriting the evidence.
- **Fixed: WAL rotation discarded committed data.** Rotation rewrites the WAL to
  begin at the checkpoint boundary, but the recorded offset was the pre-rotation
  absolute position. Recovery seeked to that byte in the *new* file and skipped
  everything before it. Reproduced deterministically: 41 fsynced, acknowledged
  rewards lost across one checkpoint and restart.

### Concurrency

- **Fixed: deadlock between prediction and neural retraining.** `predict` held
  `arms.read()` then wanted the neural mutex; retraining held the mutex then wanted
  `arms.read()`. With a writer queued, `parking_lot`'s task-fair `RwLock` completed a
  three-way cycle and the server stopped serving. Predictions now read an immutable
  weight snapshot and take no neural lock at all.
- Retraining no longer stalls predictions; it works on a private copy and swaps the
  published pointer when finished.
- Neural retraining is decoupled from checkpointing and runs on its own cadence
  (`BANDITDB_RETRAIN_POLL_SECS`).

### Isolation and access control

- Tenant ownership enforced on `/reward` (previously identified its target by
  interaction ID alone, so any tenant could write into another's model) and `/export`
  (which listed every tenant's shards).
- `BANDITDB_REQUIRE_AUTH=true` makes a missing key set fatal at startup instead of
  silently granting admin to every caller. Enabled by default in the Helm chart.

### Resource limits

- Pending-interaction cache is bounded (`BANDITDB_MAX_PENDING_INTERACTIONS`, default
  100,000). It previously had a TTL but no capacity limit — at 1,000 predictions/sec
  that is 86.4M records and an OOM kill.
- Parquet export retention (`BANDITDB_EXPORT_RETAIN_SHARDS`, default 50). `exports/`
  grew unbounded until the volume filled, at which point checkpointing fails.
- Context values are validated for finiteness *and* magnitude
  (`BANDITDB_MAX_CONTEXT_MAGNITUDE`, default 1e6). `1e155` is finite but squares to
  infinity in the rank-one update, and the resulting NaN persisted through the
  checkpoint and survived restart.
- Unmatched predictions travel in the checkpoint instead of being rewritten to the
  WAL on every checkpoint.

### Operations

- New: [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — every environment variable, every
  metric, and the alerts worth wiring up.
- New metrics: `banditdb_wal_dropped_total`, `banditdb_wal_fsync_total`,
  `banditdb_interactions_pending`, `banditdb_interactions_evicted_total`.
- New: `scripts/backup_restore.sh` with a `drill` mode that restores into a throwaway
  directory and boots it — a backup nobody has restored is a hypothesis.
- New: `scripts/crash_injection.sh`, SIGKILL at randomised points during
  checkpointing. Runs in CI.
- `cargo-deny` in CI. Two advisories fixed by upgrade (crossbeam-epoch, rand); one
  documented with reachability analysis (fast-float via polars, write-only path).
- `entrypoint.sh` no longer fails under Kubernetes: it dropped privileges
  unconditionally, which conflicted with the chart's own `runAsNonRoot`.

### Performance

Measured on a 12-core host over loopback. Both paths peak at concurrency 32.

| Path | Throughput | p99 |
|---|---|---|
| `/predict` | ~10,000/s | 4.6 ms |
| `/reward` | ~4,400/s | 8.0 ms |

Recovery replays 100,000 events (38 MB WAL) in **0.6 s**, so restart time is
dominated by scheduling rather than by BanditDB.

**Reward latency rose from ~0.3 ms to ~3.4 ms at concurrency 1** — that is the cost
of waiting for the fsync. Group commit amortises it across concurrent callers, so
parallel submission still reaches ~4,400/s while a strictly serial client pays the
full latency on every call. Batch or parallelise reward submission if throughput
matters.

### Toolchain

Requires **Rust 1.97+** to build from source. `ethnum` 1.5.2 (reached via polars)
fails to compile on recent rustc — `mem::transmute(())` into an 8-bit
`TryFromIntError`, which newer transmute size checking rejects. Pinned to 1.5.3,
which fixes it. Pre-built binaries and the Docker image are unaffected.

### Known gaps

Documented rather than hidden — see `docs/ROADMAP.md`:

- No cap on campaign count; Stage 1 assumes trusted admin credentials.
- Progressive promotion/rollback is not covered by a gating test (candle cannot seed
  the CPU RNG, so neural init is nondeterministic).
- Single-writer, single-node. No HA.
- Multi-tenancy is logical isolation, not a hard boundary between mutually hostile
  tenants.

---

## v1.0.4 and earlier

See the git history.
