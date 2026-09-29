# Changelog

## Unreleased — Runtime tenant provisioning

Keys were read from `BANDITDB_API_KEYS` once at startup, so adding a tenant meant
editing configuration and restarting. A hosted control plane cannot work that way.

- **`PUT /admin/tenants/:id`** creates or updates a tenant: key digests, quotas and
  status. Idempotent, so a control plane can retry until it succeeds.
  **`DELETE`** revokes a tenant's keys and deliberately leaves its campaign data
  intact. **`GET`** returns keys, quotas and per-key last-used times.
- **`GET /admin/tenants/:id/campaigns`**, plus `/report` and `/diagnostics` per
  campaign, let a console render dashboards without holding a usable tenant key.
- **`GET /limits`** reports a caller's quotas and current usage.
- Keys are stored as **SHA-256 digests**, never in the clear. A digest lookup is
  also O(1), replacing a constant-time scan over every configured key that cost
  more than scoring once a few hundred tenants existed.

Provisioning routes carry their own credential, **`BANDITDB_PROVISION_KEY`**, and
sit outside the API-key auth layer: no tenant key at any role can reach them. With
the variable unset the routes return 404.

Two behaviours worth knowing:

- A provisioned tenant is **always namespace-scoped**, regardless of
  `BANDITDB_TENANT_MODE`. Hosted tenants sharing a process without isolation would
  be a cross-tenant leak, and that must not be switchable by an environment variable.
- A tenant whose status is not `active` authenticates and is then refused with a
  message naming suspension, so a lapsed subscription reads as "suspended" rather
  than "bad key".

Three endpoints support a hosted console:

* **`DELETE /admin/tenants/:id/campaigns`** erases a tenant's campaigns. Kept
  separate from revoking credentials, because losing a key must never destroy
  models — erasure has to be asked for by name. This is what makes "delete my
  organization" mean it.
* **`GET /admin/tenants/:id/requests`** returns a bounded ring of that tenant's
  recent requests with status and latency, so a console can answer "what did my
  last few calls do?". In memory and capped at 50 per tenant: a debugging window,
  not an audit trail — `BANDITDB_AUDIT_LOG` remains the durable record.
* **`POST /admin/tenants/:id/campaigns/:campaign/predict`** and
  **`/admin/tenants/:id/reward`** let a console drive a campaign for an
  in-browser playground. Rewards are scoped to the tenant that owns the
  interaction, so the provisioning credential cannot be pointed at another
  tenant's id.

A tenant's footprint is bounded by **bytes**, not campaign count:
`max_campaign_bytes` is now enforced as a cumulative budget across all of a
tenant's campaigns, using the same `campaign_memory_estimate` the instance-wide
ceiling uses. Count alone is a poor proxy — the same number of campaigns can mean
kilobytes or gigabytes — and the budget also gates neural campaigns without a
separate rule, since a 256-dimensional neural campaign reserves ~105 MB for its
replay buffer and simply will not fit a small plan.

The figure is a *reservation*, not a measurement: a replay buffer is counted full
from day one, because admission control reserves the capacity a campaign grows
into rather than sampling what it occupies today.

An engine with `BANDITDB_PROVISION_KEY` set never falls back to open access. The
registry treats "no keys configured at all" as open mode for local development,
and a control-plane-managed engine matches that description between boot and its
first signup — which would have granted admin to anonymous callers during exactly
the window a freshly deployed node is most exposed.

Per-key usage is tracked in memory and written out when the store is persisted for
another reason — best effort by design, since recording it on disk per request
would put a write on the authentication path.

## Unreleased — Campaign admission control

Closes the "no cap on campaign count" gap from `docs/ROADMAP.md`.

- **`BANDITDB_MAX_CAMPAIGNS`** (default 10,000) bounds how many campaigns an instance
  will create. Previously unbounded: a retrying client or a test suite pointed at the
  wrong host could create campaigns until the process ran out of memory.
- **`BANDITDB_MAX_CAMPAIGN_BYTES`** (default 0 = unlimited) bounds one campaign's
  estimated steady-state memory. Count is a poor proxy for cost — two arms at d=4 is
  ~300 bytes, six arms over a 256-dimensional neural embedding is ~105 MB, nearly all
  replay buffer. Because arms, dimension and algorithm are fixed at creation, the
  footprint is computed exactly rather than estimated.
- Both return **403** with the limit and the current usage named, via a new
  `EngineError::LimitExceeded`.

Enforced on the create path only. Recovery and WAL replay bypass both checks by
design, so lowering a limit below what an instance already holds cannot make it
unrecoverable. Archived campaigns still count, because their state stays resident.

Defaults are chosen to change no existing behaviour: the size ceiling is off, and
10,000 campaigns is far above any current deployment.

## Unreleased — Dynamic arms

Arms are no longer fixed at campaign creation.

- **`POST /campaign/:id/arms`** (admin) adds an arm to a live campaign. Optional
  `warm_start` centres the new arm's ridge prior on the mean θ of the arms that
  already exist — the population, the arms in its `group`, or an explicit list —
  so it starts from a sensible estimate instead of from zero. `strength` is in
  pseudo-observations; at the default 1.0 the arm keeps a cold arm's uncertainty
  and is still explored.
- **`POST /campaign/:id/arms/:arm_id/status`** (admin) pauses, retires, or
  reactivates an arm. Exclusion is soft: matrices are untouched and the arm keeps
  learning from rewards for predictions already in flight. Pausing the last active
  arm is refused.
- **`eligible_arms` / `exclude_arms`** on `/predict` and `/batch_predict` narrow the
  candidate set for one request without changing any state. Filtering happens before
  propensities are computed, so logged propensities still describe the policy that
  ran and off-policy evaluation stays valid.

Behaviour changes worth noting on upgrade:

- `selection_entropy`, `converged`, and `leading_arm` now cover **active arms only**.
  Campaigns with no paused arms are unaffected.
- `/campaign/:id`, `/report`, and `/diagnostics` gained per-arm `status` and `group`;
  `/diagnostics` gained `active_arm_count`.
- Checkpoints written by older versions load unchanged — arms without a stored status
  load as active, which is what they were.
- Neural campaigns: a warm-start prior lives in the current embedding space, so the
  next retrain effectively discards it. Status and group survive retraining.

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
