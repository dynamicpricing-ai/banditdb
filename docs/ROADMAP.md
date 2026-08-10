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
