# BanditDB Distributed Architecture v2 — Router + Registry + Shards

**Status:** Proposal (replaces Phases 2–3 of `GCP_DISTRIBUTED_ARCHITECTURE.md`; Phase 1 of that document — moving `campaign_id` into the URL path — still stands and is a prerequisite for this design.)

---

## 1. The goal, in plain words

Today BanditDB runs as a single process. All campaigns live in one node's RAM, and one Write-Ahead Log (WAL) on one disk protects them. This is simple and fast, but it has two limits:

1. **Scale** — one machine's RAM and CPU cap how many campaigns and predictions/second we can serve.
2. **Availability** — if that machine dies, everything is down until it recovers.

We want to run **multiple BanditDB nodes**, each owning a *subset* of the campaigns, so that:

- Total capacity grows by adding nodes.
- One node dying takes down only its share of campaigns, briefly, and they recover automatically.

The key difficulty: BanditDB is **stateful**. Each campaign's learned matrices (`A_inv`, `b`, theta) live in the RAM of exactly one node and are updated on every reward. Two nodes updating the same campaign's matrices independently would produce two divergent "brains" that can never be merged. So the whole design boils down to one rule:

> **At any moment, exactly one node owns a given campaign. All requests for that campaign must reach that node — and only that node.**

Everything below exists to enforce that rule cheaply and safely.

---

## 2. Why not the previous idea (load-balancer consistent hashing)?

The earlier proposal asked the Google Cloud Load Balancer to hash the `campaign_id` and route each campaign to a fixed node. Three problems:

1. **The feature doesn't exist as described.** GCP load balancers can hash on a header, cookie, or client IP — but not on a piece of the URL path. The central mechanism can't be configured.
2. **Hashing moves campaigns without moving their state.** When a node dies, the hash function silently reassigns its campaigns to other nodes — but the learned matrices don't teleport along. When the dead node comes back, the hash reassigns them *again*. Result: two nodes have each updated the same campaign. State forks; learning is corrupted.
3. **Shared NFS storage (Filestore) has no fencing.** During a network hiccup, an "old" node that everyone thinks is dead can still be alive and writing to the shared WAL file at the same time as its replacement. Two writers on one WAL file means corruption. Nothing in that design prevents it.

The lesson: **routing must be explicit and deliberate, not an accident of a hash function.** A campaign should move between nodes only when *we* move it, together with its state.

---

## 3. The architecture

```
                        clients (SDKs, agents, curl)
                                    │
                                    ▼
                     ┌──────────────────────────┐
                     │   plain load balancer     │   (no special features needed)
                     └────────────┬─────────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
           ┌──────────┐    ┌──────────┐     ┌──────────┐
           │ router-a │    │ router-b │     │ router-c │   ← stateless, identical,
           └────┬─────┘    └────┬─────┘     └────┬─────┘     any number of replicas
                │               │                │
                │      reads campaign → shard map (cached)
                │               │                │
                │        ┌──────┴───────┐        │
                │        │   REGISTRY    │        │
                │        │ campaign_x →  │        │
                │        │   shard-0     │        │
                │        │ campaign_y →  │        │
                │        │   shard-2     │        │
                │        └──────────────┘        │
                │               │                │
      routes each request to the shard that owns the campaign
                │               │                │
         ┌──────┴───┐    ┌──────┴───┐     ┌──────┴───┐
         │ shard-0  │    │ shard-1  │     │ shard-2  │   ← BanditDB, unmodified core
         │ BanditDB │    │ BanditDB │     │ BanditDB │
         ├──────────┤    ├──────────┤     ├──────────┤
         │ own disk │    │ own disk │     │ own disk │   ← private WAL + checkpoints
         │ own WAL  │    │ own WAL  │     │ own WAL  │
         └──────────┘    └──────────┘     └──────────┘
```

Three kinds of pieces:

| Piece | Holds state? | How many | What it does |
|---|---|---|---|
| **Router** | No | 2+ (any number) | Looks up which shard owns the campaign, forwards the HTTP request |
| **Registry** | Yes, tiny (a map) | 1 logical (backed by GCS or Firestore) | The single source of truth: `campaign_id → shard` |
| **Shard** | Yes, the real state | 3 to start | A normal BanditDB process serving *its* campaigns |

### 3.1 The router

A small stateless HTTP proxy (a few hundred lines of Rust; lives in this repo as a `banditdb-router` crate). For every incoming request it:

1. Extracts `campaign_id` from the URL path (this is why Phase 1 matters).
2. Looks up the owning shard in its cached copy of the registry map.
3. Forwards the request to that shard and returns the response.

Because the router holds no state, you can run as many replicas as you like behind a completely ordinary load balancer — round-robin is fine. A router crashing costs nothing; the client retries and hits another one.

The router is also the natural home for **fan-out operations** — things that touch many campaigns:

- `GET /campaigns` (list) → ask every shard, merge results.
- `POST /batch_predict` with campaigns on different shards → split the batch, send each piece to its shard, merge responses. The existing partial-failure semantics carry over cleanly.

### 3.2 The registry

A single small document: the map of every campaign to its shard.

```json
{
  "version": 4127,
  "assignments": {
    "checkout_offer": "shard-0",
    "sleep":          "shard-1",
    "prompt_strategy": "shard-2"
  }
}
```

- **Stored in:** one object in a GCS bucket, or one Firestore document. Both are managed, replicated, and effectively never down. We are storing kilobytes, not gigabytes.
- **Read path:** routers cache the map in memory and refresh it every few seconds (and immediately on a "campaign not found" miss). Reads never hit the registry per-request.
- **Write path:** only two events write to it — *campaign created* and *campaign deliberately moved*. Both are rare.

**Concurrent creation is the one race to handle.** Two routers might try to create the same campaign at the same moment and assign it to different shards. Solved with a conditional write: GCS "generation match" (write succeeds only if the object hasn't changed since you read it) or a Firestore transaction. The loser of the race re-reads the map and uses the winner's assignment. This is a standard, well-trodden pattern — no consensus algorithm, no ZooKeeper.

**Assignment policy for new campaigns:** simplest that works — least-loaded shard (by campaign count or reported memory). Rendezvous hashing is an alternative, but explicit least-loaded keeps the mental model "the map is the truth" pure.

### 3.3 The shards

Each shard is a **completely ordinary BanditDB process** — the engine, WAL, checkpointing, and recovery code all run unmodified. The only difference from today is that each shard serves a subset of campaigns instead of all of them.

Deployment: a Kubernetes **StatefulSet**. For a reader newer to Kubernetes, the two properties that matter:

1. **Stable identity.** Pods are named `shard-0`, `shard-1`, `shard-2` — forever. When `shard-1` crashes, Kubernetes doesn't spin up a random new pod; it recreates *`shard-1`*, same name, same network address.
2. **Sticky disk.** Each pod has its own persistent disk (GCP Persistent Disk) that follows the identity. Recreated `shard-1` gets `shard-1`'s disk back, with its WAL and checkpoints intact.

This gives us **fencing for free**: a GCP Persistent Disk can only be mounted read-write by one VM at a time. Even in a weird network partition where an old pod is somehow still running, the replacement cannot mount the disk until the old one is truly gone — the infrastructure physically enforces the single-writer rule that shared NFS could not.

---

## 4. Failure scenarios, walked through

### 4.1 A shard pod crashes

1. Kubernetes notices within seconds and recreates `shard-1` with the same identity.
2. The new pod mounts the same disk, runs BanditDB's normal startup recovery: load last checkpoint, replay WAL tail. This is the *exact* code path that already exists and is already tested.
3. Routers were getting connection errors for `shard-1` campaigns during the gap; they return `503` to clients (clients retry). Once the pod is up, traffic flows again.
4. **Nothing about ownership changed.** The registry still says `sleep → shard-1`, and that's still true. No state moved, so no state could fork.

**Expected gap: roughly 30–90 seconds** (pod reschedule + disk reattach + WAL replay). During the gap, that shard's campaigns are unavailable; all other shards are unaffected.

### 4.2 A router pod crashes

Nothing happens. Other router replicas absorb the traffic. This is the beauty of keeping the routing layer stateless.

### 4.3 A whole VM / zone problem

Same as 4.1, just slower — Kubernetes reschedules the pod onto a healthy node and reattaches the disk there (GCP regional persistent disks make this work across zones). Ownership still never changes without us saying so.

### 4.4 What about rewards in flight during a crash?

A `/reward` whose interaction context was only in the dead pod's RAM cache (not yet WAL-durable) is lost, exactly as a single-node crash loses it today. The distributed design doesn't make this worse — and the checkpoint re-emit mechanism already minimizes it. If a stronger guarantee is ever needed, that's an engine-level durability decision, orthogonal to this architecture.

---

## 5. Scaling and rebalancing

### Adding shards (3 → 5)

1. Scale the StatefulSet; `shard-3` and `shard-4` come up empty.
2. New campaigns start landing on them (least-loaded policy does this automatically).
3. **Zero existing campaigns move.** Compare with consistent hashing, where scaling silently reassigns ~40% of campaigns to nodes that don't have their state. Here, nothing moves unless we move it.

### Deliberately moving a campaign (rebalancing)

A rare, explicit, admin-triggered operation:

1. Mark the campaign "moving" in the registry — routers briefly return `503 retry` for it.
2. Source shard: checkpoint the campaign, stop serving it, hand the snapshot to the target shard (via the checkpoint file).
3. Target shard loads it; registry is updated to the new owner; routers refresh and traffic flows.

Seconds of unavailability for one campaign, by choice, at a quiet hour. This needs one new engine capability: **export/import a single campaign's state** (today checkpointing is whole-database). That is the only engine change this architecture ever asks for beyond Phase 1.

### Faster failover (later, optional)

If the 30–90 s failover gap of §4.1 ever becomes unacceptable, add a **warm standby per shard** that continuously replays the primary's WAL and can be promoted quickly. This is already a roadmap item, and it slots into this design without touching the router or registry — the registry entry just flips to the standby.

---

## 6. What this design deliberately avoids

| Avoided | Why |
|---|---|
| Consistent hashing for placement | Moves campaigns without their state; forks learning |
| Shared NFS (Filestore) WAL | Multiple writers, no fencing, corruption risk; also ~$200/mo minimum for kilobytes of need |
| Special load-balancer features | Path-hashing doesn't exist on GCP; plain round-robin is all we need |
| Consensus systems (etcd/ZooKeeper) | A conditional write on one small document is enough for our one race |
| Engine rewrites | Shards run today's binary; recovery, WAL, checkpoints unchanged |

---

## 7. Implementation phases

**Phase 1 — API paths** *(from the original doc, unchanged)*
`campaign_id` moves into the URL: `POST /campaign/:id/predict`, `POST /campaign/:id/reward`. Valuable on its own; required for routing.

**Phase 2 — Router + registry**
New `banditdb-router` crate: path parsing, cached registry map, forwarding, fan-out for list/batch/diagnostics, conditional-write campaign creation. Testable locally against 3 BanditDB processes on different ports — no cloud needed.

**Phase 3 — GKE deployment**
StatefulSet (3 shards, per-pod PD), Deployment for routers (2 replicas), plain external LB, registry in GCS. Helm chart updates.

**Phase 4 — Verification**
- Kill a shard pod → confirm same-identity recovery from own disk, other shards unaffected, gap within budget.
- Kill a router pod → confirm zero impact.
- Concurrent campaign creation from two routers → confirm single consistent assignment.
- Batch predict spanning shards → confirm split/merge and partial-failure semantics.

**Phase 5 (optional, later) — Single-campaign export/import + rebalancing command; warm standby.**

---

## 8. One-paragraph summary

Keep BanditDB itself unchanged and single-writer. Put a tiny stateless router in front that consults an explicit map ("which shard owns which campaign") stored in a managed cloud document. Give every shard its own private disk so Kubernetes can resurrect it with its state intact, and let the cloud's disk-attachment rules enforce that no two nodes ever write the same state. Campaigns move between shards only when an operator says so — never as a side effect of a hash function or a crash. Boring, explicit, and every failure mode has a one-sentence answer.
