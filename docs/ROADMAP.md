# BanditDB Roadmap

Deferred work and known gaps. Not committed dates — priority order within each section.

## Durability / storage

### ~~Export retention~~ — DONE (P2.3)

`BANDITDB_EXPORT_RETAIN_SHARDS` (default 50, per campaign) prunes the oldest shards at
checkpoint time. `0` restores the old keep-everything behaviour.

**Still open: compaction.** Retention bounds shard *count*, not the fragmentation that
comes from one small file per checkpoint. Merging them into larger periodic files would
help analytical read performance. Not urgent — the disk-fill risk, which was the actual
hazard, is closed.

### Upgrade polars (0.41 → 0.55)

Two advisories are ignored in `deny.toml`, and both are transitive through polars:
RUSTSEC-2025-0003 (fast-float segfault, unreachable — the Parquet path is write-only)
and RUSTSEC-2026-0249 (smartstring unmaintained, repo archived 2026-05-03). Neither
has a fix at polars 0.41; upgrading clears both. It spans 14 minor versions and will
break `write_campaign_parquet` against the current DataFrame/ParquetWriter API, so it
wants its own branch and its own testing. Both ignores are dated REVIEW BY 2026-12-31.

## Scale / availability (the #2 ceiling)

### Horizontal write scale — campaign sharding
Single-writer today: one WAL writer task, one node, RWO PVC. No horizontal write
scale. Campaigns are independent state → route `campaign_id → node` (consistent
hash), each campaign stays single-writer, cluster scales linearly. No consensus.
Pragmatic "scales horizontally" story. Build when a paying customer needs throughput.

### High availability — primary + warm standby
Ship WAL to a follower, leader election, failover (~seconds RTO). Buys an
availability SLA, not throughput. Build when a customer needs an SLA.

## Performance

### Offload neural retrain off the async worker
`checkpoint()` runs `neural.retrain()` synchronously on the async worker
(`src/engine.rs` ~724). A prior `block_in_place` attempt panicked on current-thread
runtimes and was reverted. Correct fix is `spawn_blocking`, which needs the `neural`
borrow to become owned/`Send` (small refactor). Do this only if retrain latency
bites under load.

## Already deferred (from project memory)
- DashMap sharded campaigns lock
- OpenTelemetry OTLP distributed tracing
- BLAS backend (ndarray-linalg needs LAPACK system dep)

> Binary WAL was on this list but is **implemented**: `BANDITDB_WAL_FORMAT=msgpack`
> with a `BDMP` magic header for backward-compat detection.

## Dynamic arms — what was deliberately left out

Shipped: arm add, soft exclusion (paused/retired plus per-request filters), and
warm-start priors from the population, a group, or a named list.

### Arm features and similarity-weighted priors
Warm start borrows from arms the caller names or groups. It cannot say "this new arm
resembles those two" on its own, because arms carry no feature vector. Adding
`features: Vec<f64>` to an arm would let the prior be a similarity-weighted mean over
the k nearest arms. Build it when a customer has real arm embeddings; explicit groups
cover the catalogue case without it.

### Continuously re-estimated population prior
The prior is resolved once, at add time. A true hierarchical model keeps shrinking
every arm toward a hyper-mean that is itself re-estimated. Mechanically feasible —
`b` is additive, so a checkpoint could apply `b += λ(μ_new − μ_old)` — but it needs
the prior term stored separately from the data term, and decay rescales both. Not
worth the state complexity until something demands it.

### Hybrid LinUCB (Li et al. 2010, Algorithm 2)
The textbook answer for sharing strength across arms: a shared coefficient block over
context × arm features. Rejected for now — the shared block is k×k with k = d·m, the
scoring path gains block-inverse terms, and it fits neither the neural embedding path
nor the tournament. The warm-start prior gets most of the cold-start benefit for
almost none of the surface area.

### Warm-start priors do not survive a neural retrain
`reaccumulate` rebuilds arm matrices in the new embedding space by replaying the
buffer from a cold start, so a prior injected in the old space is gone. Status and
group are preserved. Fixing it means re-deriving the prior from the other arms' θ in
the new space after reaccumulation — worth doing only if neural campaigns turn out to
add arms often.

## Known gaps after Stage 1 hardening

Carried forward from the gate review in PRODUCTION_STAGE1. Each is a deliberate
boundary, recorded here so it stays visible.

### No cap on campaign count
`max_arms` and `max_feature_dim` are enforced per campaign, but the number of
campaigns is unbounded — an admin key can create them until memory runs out. Stage 1
assumes trusted admin credentials. Fix is a `BANDITDB_MAX_CAMPAIGNS` check in
`add_campaign`.

### Promotion / rollback untested in CI
The three Progressive tournament tests are `#[ignore]`d because candle 0.10.2 cannot
seed the CPU RNG (`cpu_backend/mod.rs`: `bail!("cannot seed the CPU rng with
set_seed")`), so neural weight init is nondeterministic. CI runs them non-blocking for
signal only. The workable fix is to commit a fixed-weight safetensors fixture and load
it via `NeuralLinUCBState::load`, trading a binary test fixture for determinism.

### Helm templates unrendered
`auth.required` and `config.corsOrigins` were added to the chart but `helm` was
unavailable in the environment where they were written. `values.yaml` parses as YAML;
the template conditionals are unverified. Run `helm template` before relying on them.
