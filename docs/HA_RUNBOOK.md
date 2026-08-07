# BanditDB High-Availability Runbook

## Current Architecture

BanditDB is a **single-writer** service, enforced by an exclusive `flock` on `DATA_DIR`: a second process on the same volume refuses to start rather than corrupting it.

State lives in memory and is journalled to a WAL (`bandit_wal.jsonl`), with periodic checkpoints to `checkpoint.json` plus Parquet exports for offline analysis.

**Durability is split by event type, deliberately:**

| Event | Guarantee |
|---|---|
| Rewards | **Acknowledged only after fsync.** `POST /reward` blocks until the record is on disk, so a 200 response survives power loss. |
| Campaign lifecycle (create, delete, archive, restore) | **Acknowledged only after fsync**, same as rewards. A 200 from `POST /campaign` means the campaign survives restart. |
| Predictions | Best-effort. Dropped under WAL backlog rather than failing the request. |

Predictions are recoverable — the pending-interaction cache holds them, and the checkpoint carries them across restarts — so losing a prediction *record* costs only the ability to match a late reward, never model state.

```
┌─────────────────────────────┐
│  BanditDB Pod (single)      │
│  ┌───────────┐  ┌────────┐  │
│  │  In-mem   │→ │  WAL   │  │
│  │  state    │  │ .jsonl │  │
│  └───────────┘  └────────┘  │
│         │                   │
│         ↓                   │
│  ┌────────────────────────┐ │
│  │  checkpoint.json       │ │
│  │  exports/*.parquet     │ │
│  └────────────────────────┘ │
└──────────┬──────────────────┘
           │ PVC (ReadWriteOnce)
           └── StorageClass (cloud disk / NFS)
```

## What Happens on Pod Restart

1. Axum receives SIGTERM → graceful shutdown runs a **final checkpoint** (30 s timeout).
2. On restart, `BanditDB::recover()` loads `checkpoint.json`, falling back to the retained `checkpoint.prev` if the current generation is unreadable, then replays the WAL from the recorded offset.
3. **Maximum data loss = zero for acknowledged rewards.** `POST /reward` does not return until the record has been written *and* covered by an fsync, so a 200 response means it survives process death, power loss, and VM preemption alike. Only in-flight requests that never received a response are lost.

   This is not bounded by the checkpoint interval, and earlier revisions of this runbook were wrong to say so. Checkpointing controls WAL size and replay time, not durability.

   **Predictions are deliberately weaker.** They are best-effort: under a WAL backlog a prediction record is dropped rather than failing the request, which costs the ability to match a late reward for it. Watch `banditdb_wal_dropped_total`.
4. Recovery is automatic — no manual intervention for a clean restart. Measured replay: **0.6 s for 100k events / 38 MB WAL**, so restart time is dominated by pod scheduling.
5. If the checkpoint is corrupt *and* no usable `checkpoint.prev` exists, the server **refuses to start** rather than coming up empty. Restore from backup, or set `BANDITDB_ALLOW_CORRUPT_CHECKPOINT=true` to start empty and accept the loss.

## Backup Strategy

### What to back up

| File | Why |
|---|---|
| `checkpoint.json` | Fast-recovery starting point |
| `bandit_wal.jsonl` | Events since last checkpoint |
| `exports/*.parquet` | Historical arm interaction data |

### Kubernetes CronJob backup (GCS example)

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: banditdb-backup
spec:
  schedule: "0 * * * *"   # hourly
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: google/cloud-sdk:alpine
            command:
            - sh
            - -c
            - |
              TIMESTAMP=$(date +%Y%m%d-%H%M%S)
              gsutil -m cp /data/checkpoint.json gs://$BUCKET/banditdb/$TIMESTAMP/
              gsutil -m cp /data/bandit_wal.jsonl gs://$BUCKET/banditdb/$TIMESTAMP/
              gsutil -m rsync /data/exports/ gs://$BUCKET/banditdb/$TIMESTAMP/exports/
            env:
            - name: BUCKET
              value: my-banditdb-backups
            volumeMounts:
            - name: data
              mountPath: /data
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: banditdb-data
          restartPolicy: OnFailure
```

### Restore procedure

```bash
# 1. Stop the running pod
kubectl scale deployment banditdb --replicas=0

# 2. Copy backup files onto the PVC (via a temporary restore pod or
#    by mounting the PVC elsewhere)
kubectl run restore --rm -it --image=google/cloud-sdk:alpine \
  --overrides='{"spec":{"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"banditdb-data"}}],"containers":[{"name":"restore","image":"google/cloud-sdk:alpine","command":["sh"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}]}}'

# Inside the pod:
gsutil cp gs://$BUCKET/banditdb/$TIMESTAMP/checkpoint.json /data/
gsutil cp gs://$BUCKET/banditdb/$TIMESTAMP/bandit_wal.jsonl /data/
gsutil -m rsync gs://$BUCKET/banditdb/$TIMESTAMP/exports/ /data/exports/

# 3. Restart BanditDB
kubectl scale deployment banditdb --replicas=1

# 4. Verify recovery
kubectl logs -f deployment/banditdb | grep -E "recovered|checkpoint"
```

## Availability Characteristics

| Scenario | Behaviour |
|---|---|
| Pod OOM / crash | Kubernetes restarts pod; recovery is automatic |
| Node failure | Pod reschedules to another node (PVC must support cross-AZ or use regional disk) |
| Planned rolling update | Graceful shutdown triggers final checkpoint before termination |
| WAL writer failure | Health endpoint returns 503; new writes are rejected; existing state is safe |
| Storage full | WAL writes fail; health endpoint reflects degraded state |
| Second process on the same volume | Refuses to start — `flock` on `DATA_DIR` prevents the interleaved writes that would corrupt it |
| Corrupt checkpoint | Falls back to `checkpoint.prev`; refuses to start if neither is readable, rather than serving an empty database |
| Prediction backlog | Prediction log records are dropped, not requests. Serving continues; `banditdb_wal_dropped_total` rises and late rewards for dropped records will not match |
| Pending-interaction cache full | Oldest entries evicted; `banditdb_interactions_evicted_total` rises. Each eviction permanently breaks reward matching for that prediction — alert on it |

## Multi-Replica (Not Yet Supported)

BanditDB does not currently support multiple write replicas. The Helm chart enforces `replicaCount: 1`. Planned work (Sprint 4+):

- **Read replicas** — serve `/predict` from a warm in-memory snapshot replicated via Parquet on object storage.
- **Leader election** — via Kubernetes lease or etcd for transparent failover.

Until then, availability SLA is limited to single-pod restart time (~5–15 s including final checkpoint + recovery). For stricter SLAs, use a PVC backed by a regional/replicated storage class and configure `PodDisruptionBudget`.


## Backup and Restore

`scripts/backup_restore.sh` covers backup, restore, and — importantly — verification.

```bash
./scripts/backup_restore.sh backup  /data /backups     # create an archive
./scripts/backup_restore.sh restore <archive> /data    # restore into an empty dir
./scripts/backup_restore.sh verify  <archive>          # boot it in a temp dir
./scripts/backup_restore.sh drill   /data              # backup then verify, end to end
```

**What is captured, and why:**

| File | Why it matters |
|---|---|
| `checkpoint.json` | Model state as of the last checkpoint |
| `checkpoint.prev` | Retained previous generation — the fallback when the current one is unreadable |
| `bandit_wal.jsonl` | Events since that checkpoint; without it everything after the last checkpoint is lost |
| `neural/` | MLP weights. A neural campaign restored without these serves its random initialisation until the next retrain |

`exports/` is deliberately excluded: Parquet shards are for offline analysis, recovery never reads them, and they are the bulk of the volume. Restoring loses offline history, not model state.

**Run the drill on a schedule.** A backup nobody has restored is a hypothesis. `drill` restores into a throwaway directory, boots the server against it, and asserts the campaigns come back — it is the only step that distinguishes a backup from an untested tarball.

**Restore is refused into a non-empty data directory.** Recovering over a live database would merge two histories.
