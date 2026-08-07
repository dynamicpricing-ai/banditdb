# BanditDB — Stage 1 Production Readiness Plan

**Status:** P0/P1/P2 complete; gate reviewed 2026-08-07, pending SLA sign-off
**Target:** small public cloud database — single-writer, single-node, small shared multi-tenant
**Baseline audited:** v1.0.4 (`main`)

This plan consolidates three independent audits (one internal, two external) into a single
sequenced path from development software to a service that can take paying customers.
Line references are against `main` at the time of writing.

---

## 1. Scope: what Stage 1 is, and is not

Half the disagreement between the source audits came from unstated scope. Stage 1 is fixed here:

> **Single-writer, single-node, small shared multi-tenant. Explicitly not highly available.**

The SLA this plan makes honest to publish. Figures are measured, not estimated —
see §6 for the harness and the raw numbers.

| Property | Stage 1 commitment |
|---|---|
| Availability | 99.5% monthly, single-AZ, planned restart windows excluded |
| RPO (data loss) | **Zero for acknowledged rewards.** A 200 response means the record is on disk. |
| RTO (recovery) | ≤ 5 min, dominated by pod scheduling — WAL replay is **&lt;1 s for 100k events / 38 MB** |
| Durability | Rewards durable to disk before the caller is acknowledged. **Predictions are best-effort** and may be dropped under load (`banditdb_wal_dropped_total`). |
| Throughput | **~10,000 predict/s** (p99 4.6 ms) and **~4,400 reward/s** (p99 8.0 ms) at concurrency 32, 12-core host |
| Memory | ~1 KB per pending interaction; the 100k default cap is ≈ 70–130 MB |
| Multi-tenancy | Logical isolation by namespacing — **not** a hard boundary between mutually hostile tenants |

The multi-tenancy row is a commercial constraint, not just a technical one. Stage 1 can serve
many customers, but must not claim isolation sufficient for regulated or adversarial workloads.

The RPO row came out stronger than planned. The plan assumed rewards would be
acknowledged before reaching disk, making RPO equal to the fsync interval; P0.3b
made the caller wait for the fsync instead, so an acknowledged reward cannot be
lost at all. Unacknowledged in-flight requests are still lost on process death,
which is ordinary for any database.

**Explicitly deferred to Stage 2+:** multi-replica HA, warm standby, read replicas, campaign
sharding, Kafka-style external log, hard tenant isolation.

---

## 2. Findings reconciliation

### 2.1 Confirmed across audits

**Lock-order inversion between prediction and neural retraining.** Confirmed.
`checkpoint()` acquires `neural.lock()` at `src/engine.rs:706` and then `arms.read()` at
`src/engine.rs:721-724`. `predict()` acquires `arms.read()` at `src/engine.rs:1044` and then
`neural.lock()` via `Campaign::embed()`.

Two concurrent *readers* do not block each other, so the two-party description in the external
audit is incomplete. The real cycle needs three parties, and `parking_lot`'s task-fair `RwLock`
supplies it — a pending writer blocks *new* readers:

1. `predict` holds `arms.read`, waits on `neural.lock`
2. retrain holds `neural.lock`, waits on `arms.read` — queued behind the pending writer
3. `reward` waits on `arms.write` — queued behind `predict`'s read

The reward path takes `arms.write()` on every reward, so the trigger is ordinary traffic.

The in-code comment at `src/engine.rs:716` reasons only about `arms.write()` and misses the
`arms.read()` case, which is why this survived review.

> Note: the `neural-retrain-decoupling` branch moves retraining to a background worker. That
> branch does **not** introduce this bug, but it raises exposure from once per checkpoint to
> once per poll interval. P0.1 should land before or with that branch.

**Retraining blocks the prediction hot path.** The neural mutex is held across the entire
`retrain()` call — `retrain_steps` gradient steps (default 100) over a batch. Every `predict`
calling `embed()` blocks for that duration. On CPU this is hundreds of milliseconds to seconds.

**Unbounded interaction cache.** `src/engine.rs:493` builds the Moka cache with
`time_to_live` and no `max_capacity`. Default TTL 86,400s. At 1,000 predictions/sec that is up
to 86.4M live records, each holding a `Vec<f64>` context. OOM is a matter of time, not load spikes.

**No fsync on the WAL event path.** `src/engine.rs:369` calls `file.flush()`, a no-op for
`std::fs::File`. `sync_all()` runs only on Checkpoint (`:389`) and Rotate (`:399`). Everything
between checkpoints lives in the page cache: process crash survives, power loss or VM
preemption does not.

**Checkpoint plus rotation is not crash-safe.** `src/engine.rs:801-812` writes `checkpoint.tmp`
without `sync_all`, renames, then rotates the WAL — which discards every pre-checkpoint event
(`:419-422`). A crash after rotation but before the page cache flushes loses everything. Ordering
is correct; only the fsyncs are missing.

**Corrupt checkpoint fails silently open.** `src/engine.rs:516-518` collapses any parse failure
to `None`, so recovery starts from offset 0 with zero campaigns, replays a post-rotation WAL, and
reports `status: "ok"`. There is also only one checkpoint generation — no fallback.

**Validation bypass on `/campaign/:id/interact`.** `src/main.rs:361-379` validates IDs only, then
passes `context` and `reward` straight to `db.interact`. No length check, no finite check, no
`[0,1]` reward range — all of which `/predict` and `/reward` do enforce. It is a writer-role route.

**Missing tenant checks.** `/reward` takes only an `interaction_id` and never resolves tenant.
`handle_export` (`src/main.rs:748`) does not even accept `Extension<AuthContext>` and returns every
campaign's shard names across all tenants.

**Public `/health` leaks the campaign inventory.** Mounted outside the auth layer
(`src/main.rs:1086`); the response includes every campaign ID plus entropy and status. In tenant
mode those IDs carry the tenant prefix, so an unauthenticated caller can enumerate customers.
`/metrics` is public by default too.

**Open mode is a warning, not a failure.** With no `BANDITDB_API_KEYS`, every caller gets
`Role::Admin` behind a single `tracing::warn!` (`src/main.rs:948`).

**Counter mutated before WAL enqueue.** `predict` increments `prediction_count`
(`src/engine.rs:1076`) before `wal_try_send` (`:1122`). On `WalFull` the caller gets an error but
the counter already moved.

**Deployment defects.** `values.yaml` references a `podDisruptionBudget` with no template.
`entrypoint.sh` runs `chown -R` then `gosu`, requiring a root start, which conflicts with the
chart's `runAsNonRoot: true`. No `cargo-audit`/`cargo-deny`/image scanning in CI. `cargo fmt` is
non-blocking.

### 2.2 Corrections to the external audits

**"Exponential WAL expansion" is not exponential.** Each checkpoint re-emits the current unmatched
set exactly once. Amplification is `(checkpoints per TTL window) × unmatched_count` —
linear-multiplicative, bounded by `TTL / checkpoint_interval`. Real, but the correct fix is cache
bounding plus a compact pending-set, not backpressure.

**"Read path bound by disk throughput" is right in direction, wrong in mechanism — and this
matters for sequencing.** `predict` is *not* disk-bound today, precisely *because* the event path
never fsyncs. The current bottleneck is the bounded channel and memory.

> **The single most important scheduling consequence in this document:** adding fsync to the
> prediction path converts a data-loss bug into a throughput collapse. Durability and the
> prediction write path must be fixed together, or the first fix will be measured as a regression
> and reverted.

**Sherman-Morrison instability is partially mitigated.** Symmetry is enforced on every update
(`src/math.rs:71`) and variance is clamped to ≥ 0 in `score` (`src/math.rs:28`). The
negative-variance-to-NaN path described in the external audit is already guarded. Periodic
re-inversion remains worthwhile but is not acute.

**Moving prediction logging to Kafka is a Stage 3 answer.** It solves a problem Stage 1 can solve
by separating prediction logging from reward logging (P0.4). Do not take on a broker dependency at
this stage.

**`O(N)` tenant scan is real but not blocking.** It affects list endpoints, not the hot path.
Ship it as a documented limit.

### 2.3 What is already solid

Worth recording so the remediation list reads fairly:

- Zero `unsafe` in the codebase
- Constant-time API key comparison via `subtle` (`src/main.rs:113-130`), including the length case
- ID sanitisation (`src/main.rs:291`) blocks `/` injection, so tenant namespacing cannot be escaped
- Three-role RBAC layered per route group; tenant filtering on list/get via `owns`/`ns`/`strip_ns`
- Body size limits (1 MB / 1 KB); graceful shutdown with task cancellation and a bounded final checkpoint
- WAL writer retries transient I/O with backoff and sets `wal_healthy=false` on fatal error
- Atomic rename for both checkpoint and rotation — ordering correct, fsync missing
- Non-root container image, healthcheck, multi-stage build
- Helm chart with correct single-writer guards (`replicaCount: 1`, `Recreate`, `ReadWriteOnce`)
- CI with `clippy -D warnings` as a hard gate
- A written consistency model and HA runbook

---

## 3. Remediation plan

### P0 — Blocks any paying customer

**P0.1 — Eliminate the deadlock via double-buffered neural weights** *(highest leverage in the plan)*

Replace the `Mutex<NeuralLinUCBState>` on the read path with an immutable snapshot behind
`ArcSwap` (or `RwLock<Arc<Weights>>`). `embed()` loads the current `Arc` and holds no lock.
Retraining works on a private copy and atomically swaps the pointer on completion.

One change resolves three findings: the deadlock, the multi-second P99 spike during retraining,
and the ability of background work to block `predict` at all.

*Acceptance:* `predict` acquires zero neural locks. A stress harness running predict + reward +
retrain + checkpoint concurrently for 10 minutes shows no deadlock and no P99 above ~10 ms.

**P0.2 — Crash-safe checkpoint**

`fsync` the tmp file, `fsync` the parent directory, *then* rename; rotate the WAL only after both
are durable. Retain the previous checkpoint generation. On a corrupt checkpoint, refuse to start
with a clear error plus an explicit operator override — never start silently empty.

*Acceptance:* crash-injection harness `SIGKILL`s at randomised points across the
checkpoint/rotation sequence, 500+ iterations, zero state loss.

**P0.3 — Reward durability policy** *(sequence after P0.4)*

Group-commit fsync on the WAL with a configurable interval (start at 200 ms). Publish the
resulting RPO. Do not land before P0.4.

**P0.4 — Separate prediction logging from reward logging** *(architectural fork)*

Rewards carry state and must be durable. Predictions exist only to match a later reward, and the
interactions cache already holds them in memory. Make prediction WAL writes best-effort and
asynchronous; make reward writes durable with group commit.

This resolves the read-path-blocked-by-disk finding without a broker, keeps `predict` off the disk
path, and lets `predict` stop returning 503 on `WalFull`. Fix the
`prediction_count`-before-`wal_try_send` ordering in the same change.

**P0.5 — Bound the interactions cache**

Add `max_capacity` with TinyLFU eviction, sized from a documented memory budget. Emit an
eviction-rate metric: sustained eviction means customers are silently losing reward matching and
must be able to see it. Pair with a compact pending-set so unmatched predictions are not re-emitted
through the WAL.

**P0.6 — Validation on every write path**

Finite *and magnitude* checks on all `context` values, `alpha`, and neural configs; enforce the
dimension check before any matrix math. Magnitude matters independently: `1e200` is finite and
still overflows Sherman-Morrison into `NaN`, which is then persisted to the checkpoint and
survives restart.

Close the `/interact` bypass by routing it through the same validation as `/predict` + `/reward`.

### P1 — Blocks public multi-tenant

- **P1.1** Tenant checks on `/reward` (resolve `interaction_id` → campaign → tenant) and `/export`
  (filter shards by tenant).
- **P1.2** Fail closed — in production mode, absence of `BANDITDB_API_KEYS` is fatal. Ship the Helm
  chart with auth required.
- **P1.3** Split `/health` into public liveness (status only, no campaign IDs) and authenticated
  detail. Default `/metrics` to authenticated.
- **P1.4** Make CORS configurable, default deny (currently `allow_origin(Any)` +
  `allow_headers(Any)` at `src/main.rs:1080-1083`).

### P2 — Operational readiness

- **P2.1** Single-writer lockfile on `DATA_DIR` (PID + hostname, stale detection). Converts a
  silent-corruption class into a startup error.
- **P2.2** Backup and restore procedure with a **tested** restore drill. Untested backups are not backups.
- **P2.3** Export retention and compaction (already tracked in `ROADMAP.md`).
- **P2.4** `cargo-deny` + `cargo-audit` + image scanning in CI; make `cargo fmt` blocking.
- **P2.5** Helm: add the PDB template or drop it from values; resolve `entrypoint.sh`
  `chown`/`gosu` against `runAsNonRoot`; set real resource requests and limits.
- **P2.6** Reconcile docs with behaviour — `openapi.yaml` claims `/metrics` needs no auth while the
  runtime gates it; the HA runbook implies stronger durability than the code provides.

### P3 — Documented limits, not Stage 1 work

Publish as known ceilings rather than fixing now:

- Manual `O(d³)` Cholesky (`src/math.rs:7`) — acceptable below d≈256, degrades above
- No periodic re-inversion for very long-running arms
- `O(N)` tenant scan on list endpoints
- No read replicas, warm standby, or campaign sharding

---

## 4. Test workstream

None of the above is verifiable without this, and all three audits underweighted it.

1. **Concurrency stress harness** — predict + reward + retrain + checkpoint under sustained load,
   with a watchdog failing on lock-wait beyond a threshold. The only real proof for P0.1.
2. **Crash-injection suite** — `SIGKILL` at randomised points, verifying recovery invariants.
   Proof for P0.2 and P0.3.
3. **HTTP / RBAC / tenant integration tests** — currently absent entirely. Every P1 item needs one,
   especially cross-tenant negative cases.
4. **Fuzz / property tests on context values** — `NaN`, `Inf`, `1e300`, zero-length, dimension
   mismatch. Proof for P0.6.
5. **Load test establishing the published limits** — a scale ceiling that has not been measured
   cannot be documented.
6. ~~**Seed or remove the 3 ignored stochastic tournament tests**~~ — **not achievable as
   specified.** candle 0.10.2's CPU backend rejects seeding outright
   (`cpu_backend/mod.rs:3054`: `bail!("cannot seed the CPU rng with set_seed")`), so neural
   weight initialisation cannot be made deterministic through that API. The tests stay
   `#[ignore]`d and CI runs them non-blocking for signal.

   Promotion and rollback therefore remain uncovered by a gating test. The workable
   alternative is to commit a fixed-weight safetensors fixture and load it via
   `NeuralLinUCBState::load`, trading a binary test fixture for determinism. Deferred, and
   listed here so the gap is explicit rather than implied by three ignored tests.

Harnesses 1 and 2 should be built *alongside* P0.1 and P0.2, not afterwards.

---

## 5. Sequencing

1. **P0.1 alone, merged first.** It is an active deadlock and it unblocks clean measurement of
   everything after it.
2. **P0.2 → P0.4 → P0.3 → P0.5 → P0.6.** P0.4 must precede P0.3, or the durability change will be
   measured as a latency regression and misattributed.
3. **P1 and P2 in parallel** once P0 is complete.

### Stage 1 launch gate

Reviewed 2026-08-07 against `main`.

- [x] **All P0 items complete** — P0.1–P0.6 plus P0.3b, each merged with an acceptance
      test verified to fail against the pre-fix behaviour.
- [x] **All P1 items complete** — P1.1–P1.4, covered by `tests/http_rbac_tenant_tests.rs`
      driving a real server process.
- [x] **Test workstream items 1–5 green in CI** — items 1, 3, 4 run under `cargo test`;
      items 2 and 5 were shell/Python harnesses that nothing executed until the
      `durability` CI job was added. Reviewing the gate is what surfaced that.
- [x] **Published scale limits measured, not estimated** — §6, from `benchmark/scale/limits.py`.
- [x] **HA runbook rewritten to match actual durability semantics** — it had drifted in
      *both* directions: it claimed WAL journalling was durable when no fsync existed, and
      then (after P0.3b) still described a loss window bounded by the checkpoint interval
      that no longer exists.
- [ ] **SLA table reviewed and signed off** — requires a human owner. The figures are
      measured and the caveats are stated; the commitment itself is not mine to make.

**Verdict: technically ready for Stage 1, pending sign-off and the caveats below.**

### Known gaps at gate time

Accepted rather than fixed. Each is a deliberate Stage 1 boundary, not an oversight.

> Closed after the review: campaign lifecycle events are now durability-acked
> alongside rewards, so `add`/`delete`/`archive`/`restore` return only once the
> record is on disk. Covered by `campaign_lifecycle_is_durable_before_returning`
> and `created_campaign_survives_immediate_restart`.

| Gap | Impact | Why accepted |
|---|---|---|
| No cap on campaign count | An admin key can create campaigns until memory runs out | Stage 1 assumes trusted admin credentials |
| Promotion/rollback untested in CI | Tournament traffic shifts are covered only by ignored stochastic tests | candle 0.10.2 cannot seed the CPU RNG; needs a weights fixture |
| Helm templates unrendered | Chart changes are syntactically plausible but unverified | `helm` was unavailable in the working environment — run `helm template` before relying on them |
| Single-AZ, single-writer | No HA; node loss means downtime until reschedule | Explicit Stage 1 scope |

### Crash harness measurement caveat

`crash_injection.sh` counts "committed" from the server's in-memory report, which is
updated just before the durability ack. A kill inside that sub-millisecond window
inflates the expected count for a reward the client was never told succeeded. No
*acknowledged* write is lost, but the harness cannot distinguish the two — worth
tightening to track client-confirmed rewards before these numbers back a contractual
SLA.

---

## 6. Measured limits

`benchmark/scale/limits.py` produces these. Re-run after any change to the write
path — the reward figures in particular moved by an order of magnitude during P0.3b.

Host: 12 cores, local SSD, loopback. Treat as a ceiling, not a cloud figure.

### Throughput

| Concurrency | predict ops/s | p99 ms | reward ops/s | p99 ms |
|---|---|---|---|---|
| 1 | 5,422 | 0.30 | 244 | 6.36 |
| 8 | 9,729 | 1.16 | 1,254 | 11.10 |
| **32** | **10,057** | **4.62** | **4,364** | **7.99** |
| 64 | 10,056 | 8.20 | 3,506 | 15.24 |
| 128 | 9,372 | 19.24 | 1,880 | 30.24 |

Both paths peak at concurrency 32 and degrade past 64 — that is the operating
point to size against.

Reward throughput is ~2.3× lower than predict because each caller waits for the
fsync covering its record. Per-request latency is ~3.4 ms at concurrency 1, but
throughput scales to 4,364/s because group commit amortises one fsync across many
concurrent waiters. Serial clients pay the full latency; concurrent ones do not.

### The fsync interval barely matters

| `BANDITDB_FSYNC_INTERVAL_MS` | reward ops/s | p50 ms | p99 ms |
|---|---|---|---|
| 0 | 2,404 | 5.12 | 11.50 |
| 50 | 2,451 | 4.93 | 8.80 |
| 200 (default) | 2,651 | 4.52 | 8.69 |
| 1000 | 2,621 | 4.61 | 7.92 |

Changing the commit window across a 20× range moves nothing. The writer syncs
whenever it goes idle, and at realistic concurrency it goes idle constantly, so the
interval only binds under sustained saturation. Tuning it is not a useful lever;
leave it at the default.

### Recovery

| Events replayed | WAL size | Recovery |
|---|---|---|
| 5,000 | 1.9 MB | 0.4 s |
| 25,000 | 9.5 MB | 0.4 s |
| 100,000 | 38.0 MB | 0.6 s |

Replay is not the constraint. RTO is pod scheduling and volume attach; the database
itself is available in well under a second.

### Memory

≈ 700–1,300 bytes per pending interaction at `context_dim = 64` (RSS sampling is
noisy across runs). The 100,000 default cap is therefore roughly 70–130 MB. Size
`BANDITDB_MAX_PENDING_INTERACTIONS` from the reward-arrival lag: entries live until
their reward arrives or the TTL expires, and eviction permanently breaks matching
for that prediction.

### Enforced ceilings

| Limit | Default | Env |
|---|---|---|
| Arms per campaign | 1,000 | `BANDITDB_MAX_ARMS` |
| Feature dimension | 4,096 | `BANDITDB_MAX_FEATURE_DIM` |
| Context magnitude | 1e6 | `BANDITDB_MAX_CONTEXT_MAGNITUDE` |
| Pending interactions | 100,000 | `BANDITDB_MAX_PENDING_INTERACTIONS` |
| Export shards per campaign | 50 | `BANDITDB_EXPORT_RETAIN_SHARDS` |

Campaign count is **not** capped — an admin key can create campaigns until memory
runs out. Tracked as a known gap rather than fixed, since Stage 1 assumes trusted
admin credentials.
