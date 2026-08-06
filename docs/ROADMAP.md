# BanditDB Roadmap

Deferred work and known gaps. Not committed dates — priority order within each section.

## Durability / storage

### Export retention & compaction
**Problem:** `checkpoint()` writes one parquet shard per campaign per checkpoint
(`write_campaign_parquet`, `src/engine.rs`), named `{campaign_id}_{timestamp_us}.parquet`.
Nothing ever deletes or compacts them — `exports/` grows unbounded. Lowering
`BANDITDB_CHECKPOINT_INTERVAL` (e.g. pilot default 500) multiplies file count.
On a busy node this is a disk-fill risk.

Exports are offline-analysis artifacts only — recovery uses `checkpoint.json` + WAL
replay and never reads `exports/`, so shards are safe to prune.

**Options:**
- `BANDITDB_EXPORT_RETENTION_*` env (prune shards older than N, or keep last K per
  campaign) applied at checkpoint time.
- Compaction: merge small per-checkpoint shards into larger periodic files.
- Or skip per-checkpoint export entirely; export on demand via `GET /export`.

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
- Binary WAL (MessagePack length-framed) with backward-compat detection
- OpenTelemetry OTLP distributed tracing
- BLAS backend (ndarray-linalg needs LAPACK system dep)
