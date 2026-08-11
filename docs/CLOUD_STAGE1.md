# BanditDB Cloud — Stage 1: Manual Provisioning, Automated Observation

**Status:** Plan. Companion to [`PRODUCTION_STAGE1.md`](PRODUCTION_STAGE1.md), which
hardened the single node. This document takes that node and runs a fleet of them as a
paid service.

**Thesis:** the first ten customers can be provisioned by hand in ten minutes each. What
cannot be done by hand is *noticing* that customer #7's disk is filling or that their
rewards have stopped matching. So automate observation first and provisioning last —
the opposite of the usual instinct.

---

## 1. Scope

**In:** GKE cluster, one dedicated BanditDB instance per paying customer, host-based
routing, fleet metrics and alerting, automated backups with a proven restore drill,
manual provisioning by runbook.

**Out, deliberately:** self-serve signup, Stripe, the pooled free tier, the
router/registry sharding design in `DISTRIBUTED_ARCHITECTURE_V2.md`, cross-region HA,
SOC 2. Each has an entry criterion in §10.

**What the customer is buying:** a single-writer, single-region, restart-recovery
database with an honest SLA. Not a distributed system. Sell it as that.

---

## 2. Decisions to make before anything is built

| Decision | Recommendation | Why |
|---|---|---|
| Region | One, near your first customers | Cross-region is Stage 2; picking two now doubles the ops surface for zero revenue |
| Disk type | Start zonal PD-SSD; **measure** regional | See §3.3 — regional replication lands directly on the fsync path that every reward waits for |
| Node pool | 1 pool, `e2-standard-4`, autoscale 1→5 | Each customer pod requests 500m/512Mi, so one node holds ~6 comfortably |
| Tier at launch | Dedicated only, from $299/mo | A dedicated namespace has a ~$10–15/mo floor; the $39 SMB price point in `project_cloud_economics` only works pooled, and pooling is blocked on §9 |
| Ingress | GKE Gateway or ingress-nginx, host-based | Key-based routing puts a control-plane lookup in the hot path |

---

## 3. Phase 0 — Close the chart gaps (before any customer)

The Helm chart works but was written for self-hosting, not for a fleet. Four gaps, all
small, all blocking.

### 3.1 The chart cannot be scraped by Prometheus

**This breaks the entire observability-first plan on day one.** BanditDB 2.0.0 requires a
reader key on `/metrics` (it names campaigns and arms). The chart exposes no
`BANDITDB_METRICS_PUBLIC` value and no way to hand Prometheus a key, so a fleet scraper
gets **401 from every pod** and the dashboards stay empty.

Two options; take the second:

- `BANDITDB_METRICS_PUBLIC=true` — simple, but `/metrics` is then anonymous to anything
  that can reach the pod IP. Acceptable only because the Service is `ClusterIP`.
- Give Prometheus a dedicated **reader** key per release and scrape with an
  `Authorization`/`X-Api-Key` header. Keeps the metrics endpoint closed. Costs one extra
  key in the provisioning script.

Add to `values.yaml`:

```yaml
config:
  metricsPublic: false        # BANDITDB_METRICS_PUBLIC
metrics:
  serviceMonitor:
    enabled: false            # true on the cloud cluster
    interval: 30s
    readerKeySecret: ""       # Secret holding the scrape key
```

### 3.2 Nine supported settings are unreachable from the chart

The server reads these; the chart never sets them:

```
BANDITDB_MAX_PENDING_INTERACTIONS   BANDITDB_REWARD_TTL_SECS
BANDITDB_EXPORT_RETAIN_SHARDS       BANDITDB_METRICS_PUBLIC
BANDITDB_MAX_CONTEXT_MAGNITUDE      BANDITDB_FSYNC_INTERVAL_MS
BANDITDB_RETRAIN_POLL_SECS          BANDITDB_NEURAL_BUFFER_CAP
BANDITDB_NEURAL_BATCH_SIZE
```

The first two are the ones that bite: `MAX_PENDING_INTERACTIONS` is the real memory
bound (100k entries ≈ 100 MB), and `REWARD_TTL_SECS` is a per-customer product knob —
a customer whose conversions take three days needs it raised, and today that requires
editing the chart.

Deliberately **not** exposed: `BANDITDB_SKIP_DATA_DIR_LOCK` and
`BANDITDB_ALLOW_CORRUPT_CHECKPOINT`. Both disable a safety net; neither belongs in a
managed product's value surface.

### 3.3 Deployment → StatefulSet

The chart ships a `Deployment` with `strategy: Recreate` and a standalone PVC. Recreate
is correct — it fully terminates the old pod before starting the new one, so upgrades
cannot dual-attach. This is *workable*.

Move to a `StatefulSet` with `volumeClaimTemplates` anyway, for fleet reasons:

- Stable pod identity (`banditdb-0`) makes per-customer logs, alerts and runbooks legible.
- Explicit `podManagementPolicy` and ordered semantics, rather than relying on Recreate.
- It is what operators expect from a stateful managed service; it will come up in
  security review.

Not urgent enough to block customer #1, but do it before customer #5 — migrating a PVC
between chart shapes is more annoying with real data in it.

### 3.4 No PodDisruptionBudget by default

`podDisruptionBudget.enabled: false`. With one replica, a node drain (GKE autoupgrade,
scale-down) evicts the only pod. Recovery is fast — 100k events replay in ~0.6 s — so
this is seconds of downtime, not minutes, but it happens *unannounced* during
maintenance windows.

Set `minAvailable: 1` for the cloud values. Note the trade: a PDB of 1 on a
single-replica workload will **block** node drains rather than delay them, so autoupgrade
needs a maintenance window and an operator to shepherd it. That is the correct trade at
ten customers.

**Phase 0 sizing:** ~150 lines of chart changes, half a day, plus `helm template`
verification against each values file.

---

## 4. Phase 1 — Cluster bring-up

```bash
# One-time. Adjust region/project.
export PROJECT=banditdb-cloud REGION=europe-west1 ZONE=europe-west1-b

gcloud container clusters create banditdb-prod \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type=e2-standard-4 \
  --num-nodes=1 --enable-autoscaling --min-nodes=1 --max-nodes=5 \
  --enable-autoprovisioning-... \
  --disk-type=pd-ssd \
  --enable-ip-alias \
  --release-channel=regular \
  --maintenance-window-start="2026-01-01T02:00:00Z" \
  --maintenance-window-end="2026-01-01T06:00:00Z" \
  --maintenance-window-recurrence="FREQ=WEEKLY;BYDAY=SU"
```

The maintenance window is not optional — see §3.4. Pin it to hours when you are awake.

```bash
# Storage class with retain policy: a deleted PVC must not delete a customer's disk.
kubectl apply -f - <<'EOF'
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: banditdb-ssd-retain
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-ssd
reclaimPolicy: Retain          # the whole point
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true     # so a full disk is a resize, not a migration
EOF
```

`reclaimPolicy: Retain` is the single most important line in this document. With
`Delete`, one mistaken `helm uninstall` destroys a paying customer's learned state
permanently.

Then install the platform pieces: ingress controller, cert-manager,
kube-prometheus-stack.

---

## 5. Phase 2 — Provision a customer (the runbook)

Ten minutes, by hand, from a checklist. Do **not** script this until it has been run
about five times — the script should encode a process you already trust.

```bash
CUST=acme                      # [a-z0-9-], becomes the namespace and subdomain
PLAN_DISK=10Gi
PLAN_CPU=500m
PLAN_MEM=512Mi

# 1. Namespace
kubectl create namespace "cust-$CUST"

# 2. Keys. Three roles + one scrape key. Generated here, shown once.
ADMIN=$(openssl rand -hex 24)
WRITER=$(openssl rand -hex 24)
READER=$(openssl rand -hex 24)
SCRAPE=$(openssl rand -hex 24)

# 3. Values for this customer
cat > "/tmp/values-$CUST.yaml" <<EOF
image:
  repository: simeonlukov/banditdb
  tag: "2.0.0"                 # pin per customer; never :latest
auth:
  apiKeys: "$ADMIN=admin;$WRITER=writer;$READER=reader;$SCRAPE=reader"
  required: true
persistence:
  storageClass: banditdb-ssd-retain
  size: $PLAN_DISK
resources:
  requests: { cpu: $PLAN_CPU, memory: $PLAN_MEM }
  limits:   { cpu: 2000m, memory: 2Gi }
config:
  logFormat: json
  corsOrigins: ""              # tighten per customer if they call from a browser
  checkpointInterval: 5000
  maxWalSizeMb: 100
  maxPendingInteractions: 100000
  rewardTtlSecs: 86400
  exportRetainShards: 50
metrics:
  serviceMonitor:
    enabled: true
ingress:
  enabled: true
  className: nginx
  hosts:
    - host: $CUST.api.banditdb.com
      paths: [{ path: /, pathType: Prefix }]
  tls:
    - hosts: [$CUST.api.banditdb.com]
      secretName: $CUST-tls
podDisruptionBudget:
  enabled: true
  minAvailable: 1
EOF

# 4. Install
helm install "banditdb-$CUST" ./helm/banditdb \
  --namespace "cust-$CUST" -f "/tmp/values-$CUST.yaml" --wait --timeout 5m

# 5. Verify before handing over the key
kubectl -n "cust-$CUST" get pod,pvc
curl -sf "https://$CUST.api.banditdb.com/health" | jq .
curl -sf -H "X-Api-Key: $READER" "https://$CUST.api.banditdb.com/health/detail" | jq .

# 6. Prove durability on THIS instance, not in general
curl -sf -X POST "https://$CUST.api.banditdb.com/campaign" \
  -H "X-Api-Key: $ADMIN" -H 'Content-Type: application/json' \
  -d '{"campaign_id":"provision-check","arms":["a","b"],"feature_dim":2}'
IID=$(curl -sf -X POST "https://$CUST.api.banditdb.com/predict" \
  -H "X-Api-Key: $WRITER" -H 'Content-Type: application/json' \
  -d '{"campaign_id":"provision-check","context":[0.6,0.8]}' | jq -r .interaction_id)
curl -sf -X POST "https://$CUST.api.banditdb.com/reward" \
  -H "X-Api-Key: $WRITER" -H 'Content-Type: application/json' \
  -d "{\"interaction_id\":\"$IID\",\"reward\":1.0}"
kubectl -n "cust-$CUST" delete pod -l app.kubernetes.io/instance="banditdb-$CUST"   # NOT --force
kubectl -n "cust-$CUST" rollout status deploy/"banditdb-$CUST" --timeout=2m
curl -sf -H "X-Api-Key: $READER" \
  "https://$CUST.api.banditdb.com/campaign/provision-check" | jq '.total_rewards'
# Must print 1. If it prints 0, STOP — do not hand this instance to a customer.

# 7. Clean up the probe, record the keys in the password manager, delete /tmp values
curl -sf -X DELETE "https://$CUST.api.banditdb.com/campaign/provision-check" \
  -H "X-Api-Key: $ADMIN"
shred -u "/tmp/values-$CUST.yaml" 2>/dev/null || rm -f "/tmp/values-$CUST.yaml"
```

Step 6 is the part people skip. It proves *this specific PVC* survives a pod restart with
an acknowledged write intact. It is the only step that would catch a misconfigured
storage class, and it costs 90 seconds.

**Record per customer:** namespace, host, image tag, disk size, key fingerprints (not the
keys), provisioning date. A spreadsheet is fine at this scale and is the seed of the
registry the control plane will later own.

---

## 6. Phase 3 — Fleet observability (build this before customer #2)

### 6.1 Alerts that matter

Real metric names, from `src/main.rs`:

| Alert | Expression | Why it matters |
|---|---|---|
| **WAL unhealthy** | `banditdb_wal_healthy == 0` | Writes are failing. Page immediately. Checkpointing also fast-fails in this state. |
| **Rewards being dropped** | `rate(banditdb_wal_dropped_total[5m]) > 0` | Backpressure. Prediction log records are being shed. |
| **Predictions evicted unrewarded** | `increase(banditdb_interactions_evicted_total[1h]) > 0` | **The silent one.** The customer's rewards are 404ing and their model has quietly stopped learning. They cannot see this; you can. |
| **Pending cache near capacity** | `banditdb_interactions_pending > 90000` | Precedes the above. Raise `MAX_PENDING_INTERACTIONS` or the TTL. |
| **Disk filling** | `kubelet_volume_stats_available_bytes / capacity < 0.2` | A full PVC makes checkpointing fail — the failure mode that ends in data loss. Expand online. |
| **Pod restarting** | `increase(kube_pod_container_status_restarts_total[15m]) > 2` | OOM or crash loop. |
| **p99 latency** | `histogram_quantile(0.99, rate(banditdb_http_request_duration_seconds_bucket[5m])) > 0.05` | Baseline is ~4.6 ms predict, ~8 ms reward at concurrency 32. |
| **Entropy collapse** | `banditdb_campaigns_active > 0` + `/health` reporting `degraded` | Data-quality signal, not an outage. Weekly review, not a page. |

The eviction alert is the one that justifies this whole phase. Everything else fails
loudly; that one fails silently and costs you the customer.

### 6.2 Dashboard

One row per customer: request rate, p99, WAL health, pending depth, disk used, active
campaigns. If a customer emails "is something wrong?", the answer should take fifteen
seconds.

---

## 7. Phase 4 — Backups and the restore drill

### 7.1 Backups

Snapshots of the PVC are crash-consistent, and BanditDB is crash-safe by construction —
fsynced checkpoint, `checkpoint.prev` fallback, WAL replay — so a raw snapshot *is*
restorable. Improve it anyway by checkpointing first, which shrinks the replay to near
zero:

```bash
curl -sf -X POST "https://$CUST.api.banditdb.com/checkpoint" -H "X-Api-Key: $ADMIN"
gcloud compute disks snapshot "$PD_NAME" --snapshot-names="$CUST-$(date +%Y%m%d)" --zone="$ZONE"
```

Daily via CronJob, 30-day retention, plus `Backup for GKE` for the cluster objects
(namespaces, secrets, PVC bindings) so a rebuild does not depend on your laptop.

### 7.2 The drill — weekly, automated, non-negotiable

`scripts/backup_restore.sh` already implements this:

```
./scripts/backup_restore.sh backup  <data_dir> <backup_dir>
./scripts/backup_restore.sh restore <backup.tar.gz> <data_dir>
./scripts/backup_restore.sh verify  <backup.tar.gz>   # restores to a temp dir and boots it
./scripts/backup_restore.sh drill   <data_dir>        # backup -> verify, end to end
```

`verify` boots the restored directory, which is the only thing that distinguishes a
backup from a hopeful tarball. Run it weekly as a CronJob against a rotating customer,
into a throwaway namespace, and alert on failure. Note the script's own warning about the
`neural/` sidecars: a neural campaign restored without them comes back with random
weights — it serves, but it has forgotten everything.

**Restore target:** RTO 30 min, RPO = last snapshot (24 h) for a disk loss; RPO 0 for a
pod crash, since acknowledged rewards are fsynced.

---

## 8. Runbook: the incidents you will actually get

### Pod stuck `Terminating`, replacement stuck `Pending`

Node went NotReady. RWO will not attach the disk elsewhere until the old pod is confirmed
gone. **Do not run `kubectl delete pod --force --grace-period=0`.**

Force-delete removes the API object while the container may still be running and writing.
If the disk then attaches to a second node, two processes write one WAL and the
customer's learned state forks unrecoverably. The `flock` on `DATA_DIR` does **not** save
you here — it is kernel-local and advisory, so two pods on two nodes each acquire it
successfully. It protects only against two processes on the *same* node.

Correct action: wait for the node controller to evict (~5 min), or delete the Node object
if the machine is confirmed dead. Then the disk detaches cleanly and the replacement
schedules.

### Disk full

Checkpointing fails first; writes continue until the WAL cannot extend. Expand:

```bash
kubectl -n "cust-$CUST" patch pvc <claim> -p '{"spec":{"resources":{"requests":{"storage":"20Gi"}}}}'
```

`allowVolumeExpansion: true` makes this online. Then check `BANDITDB_EXPORT_RETAIN_SHARDS`
— unbounded exports were the historical cause.

### Customer reports "the model stopped improving"

Check `banditdb_interactions_evicted_total` and `banditdb_interactions_pending` first.
Nine times out of ten they are rewarding outside the TTL window or exceeding the pending
cache, not hitting a modelling problem.

### Upgrading a customer

```bash
helm upgrade "banditdb-$CUST" ./helm/banditdb -n "cust-$CUST" \
  -f "/tmp/values-$CUST.yaml" --set image.tag=2.1.0 --wait
```

`Recreate` means brief downtime: pod terminates, new pod pulls and starts, WAL replays
(~0.6 s per 100k events). Seconds, dominated by image pull. Roll one canary customer
first, wait 24 h, then the rest. Pin tags per customer so a bad release cannot reach
everyone at once.

---

## 9. What blocks the free/pooled tier

Two code-level holes make pooling *untrusted* tenants unsafe today. Both are small.

**No campaign-count cap.** `BANDITDB_MAX_ARMS` and `BANDITDB_MAX_FEATURE_DIM` bound one
campaign's size; nothing bounds how many exist. A pooled tenant looping `POST /campaign`
exhausts RAM and OOM-kills the shared pod, taking every co-tenant with it. `CHANGELOG.md`
already records this as "Stage 1 assumes trusted admin credentials" — pooling strangers
violates that assumption directly.

*Fix:* `BANDITDB_MAX_CAMPAIGNS` per process, and per-tenant when `TENANT_MODE` is on.
~40 lines plus tests.

**The pending-interaction cache is global.** One Moka cache for the whole process
(`engine.rs:804`). A noisy tenant fills it and evicts *other tenants'* predictions; their
rewards then 404 and their models silently stop learning.

*Fix:* per-tenant quota within the cache, or a cache per tenant. ~120 lines; the harder
of the two.

Until both land, every tenant on a shared instance must be someone you trust — i.e. your
own demos and design partners, not public signups.

---

## 10. Exit criteria — when to build the next thing

| Build this | When |
|---|---|
| Provisioning script | After 5 manual runs without checklist edits |
| Self-serve portal + Stripe | ≥ 10 paying customers, or provisioning is >2 h/week |
| Pooled free tier | Both §9 fixes shipped with tests |
| StatefulSet migration | Before customer #5 |
| Warm standby / 99.9% SLA | A customer contractually requires it — not before |
| Router + registry (`DISTRIBUTED_ARCHITECTURE_V2`) | One customer's campaigns exceed a single node's RAM |
| SOC 2 Type II | First enterprise deal is gated on it (budget $30–60k) |

Resist the router. It solves sharding campaigns within one logical database — a problem
none of the first fifty customers will have, since each has their own instance.

---

## 11. Cost sketch

| Item | Monthly |
|---|---|
| GKE control plane | ~$74 (free tier covers one zonal cluster) |
| 1× e2-standard-4 node | ~$100 on demand, ~$65 with 1-yr CUD |
| PD-SSD, 10 Gi × N customers | ~$1.70 per customer |
| LB + ingress IP | ~$18 |
| Snapshots | ~$0.03/GB/mo |

Roughly **$200/mo** to run the cluster empty, with per-customer marginal cost in single
dollars until a node fills (~6 customers per e2-standard-4). At $299/customer, break-even
is customer #1–2; margin approaches 90%+ from customer #5 as the fixed cluster cost
amortises.

This is the arithmetic that makes dedicated-per-customer viable — and the same arithmetic
that makes a $39 dedicated instance a loss leader.

---

## 12. Work items

| # | Item | Size | Blocks |
|---|---|---|---|
| 1 | Chart: metrics scraping (`metricsPublic` + ServiceMonitor + scrape key) | S | Everything in §6 |
| 2 | Chart: expose the 9 missing env vars | S | Per-customer tuning |
| 3 | Chart: PDB default + cloud values file | XS | Node drains |
| 4 | StorageClass with `reclaimPolicy: Retain` | XS | Customer #1 |
| 5 | Prometheus + the 8 alerts in §6.1 | M | Customer #2 |
| 6 | Snapshot CronJob + weekly restore drill CronJob | M | Customer #2 |
| 7 | Provisioning runbook, executed 5× by hand | M | — |
| 8 | Deployment → StatefulSet | M | Customer #5 |
| 9 | `BANDITDB_MAX_CAMPAIGNS` | S | Free tier |
| 10 | Per-tenant pending-cache quota | M | Free tier |

Items 1–7 are the launch set: roughly one to two focused weeks.
