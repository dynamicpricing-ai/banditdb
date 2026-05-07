# BanditDB High-Availability Runbook

## Current Architecture

BanditDB is a **single-writer** service. All state lives in memory, is durably journalled to a WAL (`bandit_wal.jsonl`), and is periodically checkpointed to Parquet + `checkpoint.json`.

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
2. On restart, `BanditDB::recover()` loads `checkpoint.json` first, then replays any WAL entries written after the last checkpoint.
3. **Maximum data loss** = events written to the WAL after the last successful checkpoint that did not make it into the final checkpoint. This window is bounded by `BANDITDB_CHECKPOINT_INTERVAL` (default: 5 000 rewards) and `BANDITDB_MAX_WAL_SIZE_MB` (default: 100 MB).
4. Recovery is automatic — no manual intervention required for a clean restart.

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

## Multi-Replica (Not Yet Supported)

BanditDB does not currently support multiple write replicas. The Helm chart enforces `replicaCount: 1`. Planned work (Sprint 4+):

- **Read replicas** — serve `/predict` from a warm in-memory snapshot replicated via Parquet on object storage.
- **Leader election** — via Kubernetes lease or etcd for transparent failover.

Until then, availability SLA is limited to single-pod restart time (~5–15 s including final checkpoint + recovery). For stricter SLAs, use a PVC backed by a regional/replicated storage class and configure `PodDisruptionBudget`.
