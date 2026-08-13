# BanditDB Cloud — Stage 1: Manual Provisioning, Automated Observation

**Status:** Plan. Companion to [`PRODUCTION_STAGE1.md`](PRODUCTION_STAGE1.md), which
hardened the single node. This document takes that node and runs a fleet of them as a
paid service.

**Thesis:** the first ten customers can be provisioned by hand in ten minutes each. What
cannot be done by hand is *noticing* that customer #7's disk is filling or that their
rewards have stopped matching. So automate observation first and provisioning last —
the opposite of the usual instinct.

**Platform:** AWS Lightsail, one instance per customer, `eu-central-1` (co-located with
the existing sandbox). One provisioning script, no images, no Kubernetes. See
[§13](#13-why-not-kubernetes) for why the earlier GKE plan was abandoned.

---

## 1. Scope

**In:** one dedicated Lightsail instance per paying customer, provisioned by script,
host-based routing via DNS, fleet metrics and alerting, daily snapshots with a proven
restore drill.

**Out, deliberately:** self-serve signup, Stripe, the pooled free tier, Kubernetes, the
router/registry sharding design in `DISTRIBUTED_ARCHITECTURE_V2.md`, cross-region HA,
SOC 2. Each has an entry criterion in §10.

**What the customer is buying:** a single-writer, single-region, restart-recovery
database with an honest SLA. Not a distributed system. Sell it as that.

---

## 2. Decisions to make before anything is built

| Decision | Recommendation | Why |
|---|---|---|
| Region | `eu-central-1` | Where the sandbox already runs; one bill, one console, one latency profile |
| Bundle | `small_3_0` (2 GB) for Starter | `nano` cannot hold the pending cache plus matrices; `micro` works but leaves no headroom |
| Tier at launch | Dedicated only, from $299/mo | See §11 — a $10/mo instance makes $99 viable on infra, but your *time* is the binding cost |
| Provisioning | `deploy/lightsail/cloud-init.sh` | The thing in version control is the thing that runs |
| TLS | certbot per instance | ACM would need a load balancer per customer; certbot is free and renews itself |

---

## 3. Phase 0 — Prerequisites

Small, and mostly account setup.

- **IAM user with Lightsail permissions**, if you want `provision.sh` rather than the
  console. The console path needs nothing.
- **DNS zone for `api.banditdb.com`** with A records per customer. Note the apex is on
  OVH while the sandbox is on AWS — confirm which provider actually hosts the zone before
  hunting for records; that split has caused confusion before.
- **A fleet sheet.** Customer, instance name, domain, bundle, BanditDB version,
  provisioning date, key fingerprints (not keys). A spreadsheet is fine at this scale and
  becomes the registry a control plane would later own.

**Not needed:** Helm chart changes, a Kubernetes cluster, a golden image. The Helm chart
still exists in `helm/banditdb` and remains the **self-hosted / BYOC artifact** — the
highest-margin enterprise tier and the easiest path through a customer security review —
but it is not the cloud data plane.

---

## 4. Phase 1 — Provision a customer

Ten minutes, by hand, from a checklist. Do **not** script this until it has been run
about five times — a script should encode a process you already trust.

Full runbook: [`../deploy/lightsail/README.md`](../deploy/lightsail/README.md).

```bash
# Console: create instance → Ubuntu 24.04 LTS → small_3_0 → your SSH key
scp deploy/lightsail/cloud-init.sh ubuntu@<ip>:~/
ssh ubuntu@<ip>
sudo CUSTOMER_ID=acme SERVER_NAME=acme.api.banditdb.com bash cloud-init.sh
```

About three minutes, mostly `apt`. Or scripted, once you have AWS credentials:

```bash
./deploy/lightsail/provision.sh acme acme.api.banditdb.com --tier starter
```

Then, in order, none of it automated because each step wants a human to confirm the
result:

1. `curl http://<ip>/health` → `{"status":"ok","version":"2.0.0","features":["neural"]}`
2. DNS: `acme.api.banditdb.com A <ip>`, then `dig +short` until it resolves
3. **Open 443 in the Lightsail firewall** — `ufw` inside the instance allows it, but the
   Lightsail layer drops the packet first, and the symptom is certbot failing for no
   visible reason
4. `sudo certbot --nginx -d acme.api.banditdb.com --agree-tos -m ops@banditdb.com --redirect`
5. `sudo certbot renew --dry-run` — renewal fails silently otherwise
6. **The durability check** (§4.1). Do not skip it
7. Deliver `/etc/banditdb/credentials.txt` via a one-time secret link, never plain email,
   then `shred -u` it
8. Record everything in the fleet sheet

### 4.1 The durability check

The one step people skip. It proves *this instance's disk* persists an acknowledged
write, and it is the only check that would catch a bad volume before a customer depends
on it. Full script in the deploy README; the shape is:

```
create campaign → predict → reward → systemctl kill -s SIGKILL banditdb → total_rewards must be 1
```

SIGKILL rather than `restart`: a graceful stop runs a final checkpoint, which is the easy
case. SIGKILL forces WAL replay, which is what actually proves fsync-before-ack.

Also assert a reader key gets **403** on DELETE. That confirms the three roles parsed
distinctly rather than everyone silently getting admin — a real bug found during
development, caused by unquoted `;` separators in the env file.

---

## 5. Phase 2 — Fleet observability (build this before customer #2)

### 5.1 Alerts that matter

Real metric names, from `src/main.rs`:

| Alert | Expression | Why it matters |
|---|---|---|
| **WAL unhealthy** | `banditdb_wal_healthy == 0` | Writes are failing. Page immediately. Checkpointing also fast-fails in this state. |
| **Rewards being dropped** | `rate(banditdb_wal_dropped_total[5m]) > 0` | Backpressure. Prediction log records are being shed. |
| **Predictions evicted unrewarded** | `increase(banditdb_interactions_evicted_total[1h]) > 0` | **The silent one.** The customer's rewards are 404ing and their model has quietly stopped learning. They cannot see this; you can. |
| **Pending cache near capacity** | `banditdb_interactions_pending > 90000` | Precedes the above. Raise `MAX_PENDING_INTERACTIONS` or the TTL. |
| **Disk filling** | `node_filesystem_avail_bytes / node_filesystem_size_bytes < 0.2` | A full disk makes checkpointing fail — the failure mode that ends in data loss. |
| **Service restarting** | `changes(process_start_time_seconds[15m]) > 2` | OOM or crash loop. |
| **p99 latency** | `histogram_quantile(0.99, rate(banditdb_http_request_duration_seconds_bucket[5m])) > 0.05` | Baseline ~4.6 ms predict, ~8 ms reward at concurrency 32. |
| **Instance unreachable** | blackbox probe on `https://<domain>/health` | Covers instance loss, which nothing on the box can report. |
| **Certificate expiry** | blackbox `probe_ssl_earliest_cert_expiry` < 14 days | Renewal fails silently if the DNS record ever moves. |

The eviction alert is the one that justifies this whole phase. Everything else fails
loudly; that one fails silently and costs you the customer.

### 5.2 How to scrape

One small central Prometheus, static target list, scraping each customer instance over
the public internet with the reader key. `/metrics` requires it (the output names
campaigns and arms), and plain Prometheus can send arbitrary headers via `http_headers`
in `scrape_configs` — verify your version supports it.

This is one respect in which the Lightsail path is *better* than the Kubernetes one: a
`ServiceMonitor` cannot send a custom header, so the GKE design had to open `/metrics`
and fence it with a NetworkPolicy. Here it stays authenticated.

Add a **blackbox exporter** for the two probes above. On-box metrics cannot tell you the
box is gone.

### 5.3 Dashboard

One row per customer: request rate, p99, WAL health, pending depth, disk used, active
campaigns, cert expiry. If a customer emails "is something wrong?", the answer should
take fifteen seconds.

---

## 6. Phase 3 — Backups and the restore drill

### 6.1 Backups

`provision.sh` enables Lightsail automatic snapshots at 03:00 daily. Those are
crash-consistent, and BanditDB is crash-safe by construction — fsynced checkpoint,
`checkpoint.prev` fallback, WAL replay — so a raw snapshot *is* restorable. Improve it by
checkpointing first, which shrinks replay to near zero:

```bash
curl -sf -X POST https://<domain>/checkpoint -H "X-Api-Key: $ADMIN"
```

Lightsail keeps the last 7 automatic snapshots. For longer retention, take manual
snapshots monthly and keep them; they are billed separately (~$0.05/GB/month).

### 6.2 The drill — monthly, non-negotiable

A snapshot nobody has restored is a hypothesis.

1. Restore the newest snapshot of a rotating customer into a throwaway instance
2. Boot it, `curl http://<ip>/health`
3. Run the §4.1 durability check against it
4. Confirm campaign count and `total_rewards` match the source
5. Delete the throwaway instance

`scripts/backup_restore.sh drill <data_dir>` does the file-level equivalent on a single
box and is worth running too. Note its warning about the `neural/` sidecars: a neural
campaign restored without them comes back with random weights — it serves, but it has
forgotten everything.

**Targets:** RTO 30 min (restore + DNS repoint), RPO = last snapshot (24 h) for instance
loss; RPO 0 for a process crash, since acknowledged rewards are fsynced.

---

## 7. Runbook: the incidents you will actually get

### Instance unreachable

Lightsail instances do not migrate. Restore the newest snapshot into a new instance,
attach the static IP (which survives instance deletion), done — DNS does not need to
change if a static IP was allocated. This is why `provision.sh` allocates one: without it
the address changes on every stop/start and the customer's DNS silently breaks.

### Disk full

Checkpointing fails first; writes continue until the WAL cannot extend. Lightsail disks
cannot be resized in place — the path is snapshot → restore onto a larger bundle →
reattach static IP. Schedule it; the alert at 20% remaining should give days of warning.
Then check `BANDITDB_EXPORT_RETAIN_SHARDS`; unbounded exports were the historical cause.

### Customer reports "the model stopped improving"

Check `banditdb_interactions_evicted_total` and `banditdb_interactions_pending` first.
Nine times out of ten they are rewarding outside the TTL window or exceeding the pending
cache, not hitting a modelling problem.

Second most common: they are reading `converged` from `/campaign/:id/report` and
expecting it to become `true`. On a contextual campaign it never will — it tests whether
one arm is best for *everybody*, and where different arms win for different contexts the
confidence intervals overlap no matter how well the model routes. Point them at the
reward-rate trend and the traffic split instead.

### Upgrading a customer

```bash
V=v2.1.0
curl -fsSL "https://github.com/dynamicpricing-ai/banditdb/releases/download/$V/banditdb-$V-x86_64-unknown-linux-gnu.tar.gz" | tar xz
sudo install -m0755 banditdb /usr/local/bin/banditdb
sudo systemctl restart banditdb
```

SIGTERM triggers a final checkpoint, then the new process replays the WAL — 100k events
in ~0.6 s. Downtime is seconds, dominated by process start. Roll one canary customer,
wait 24 hours, then the rest. Pin versions per customer in the fleet sheet so a bad
release cannot reach everyone at once.

### TLS renewal failed

Certbot renews via an HTTP challenge on the customer's hostname. It fails silently if the
DNS record moved or 443 was closed. The cert-expiry alert in §5.1 is the only thing that
will tell you. Fix the record, then `sudo certbot renew --force-renewal`.

---

## 8. What blocks the free/pooled tier

Two code-level holes make pooling *untrusted* tenants unsafe today. Both are small.

**No campaign-count cap.** `BANDITDB_MAX_ARMS` and `BANDITDB_MAX_FEATURE_DIM` bound one
campaign's size; nothing bounds how many exist. A pooled tenant looping `POST /campaign`
exhausts RAM and OOM-kills the shared instance, taking every co-tenant with it.
`CHANGELOG.md` already records this as "Stage 1 assumes trusted admin credentials" —
pooling strangers violates that assumption directly.

*Fix:* `BANDITDB_MAX_CAMPAIGNS` per process, and per-tenant when `TENANT_MODE` is on.
~40 lines plus tests.

**The pending-interaction cache is global.** One Moka cache for the whole process
(`engine.rs:804`). A noisy tenant fills it and evicts *other tenants'* predictions; their
rewards then 404 and their models silently stop learning.

*Fix:* per-tenant quota within the cache, or a cache per tenant. ~120 lines; the harder
of the two.

Until both land, every tenant on a shared instance must be someone you trust — your own
demos and design partners, not public signups. **The existing sandbox already serves as
the free evaluation tier**, so there is no urgency here.

---

## 9. Pricing

Full reasoning in the pricing discussion; the summary:

| Plan | Price | What |
|---|---|---|
| Sandbox | $0 | `sandbox.banditdb.com`, shared, resets nightly. Evaluation only. |
| Starter | $99/mo | Dedicated 2 GB instance, 2M decisions included, $20 per additional 1M |
| Growth | $299/mo | Dedicated 4 GB, 10M decisions, $15/1M, priority support, config tuning |
| Enterprise / BYOC | Custom | Self-hosted licence via the Helm chart; highest margin, easiest security review |

Infrastructure is 3–10% of revenue at any of these prices, so it should not drive the
decision. The binding cost is **your time**: provisioning plus onboarding is about an
hour, and roughly an hour a month of support per customer. At $99 that is ~$170/hour of
founder time; at $39 it is ~$67 and cannot fund the white-glove onboarding that is
currently the main advantage over a self-serve competitor.

Put the decision meter in the contract from day one even if you never enforce it early.
Adding metering later is a repricing conversation with every existing customer; having it
and choosing not to charge is a favour you can grant.

---

## 10. Exit criteria — when to build the next thing

| Build this | When |
|---|---|
| `provision.sh` used routinely | After 5 manual runs without checklist edits |
| Self-serve portal + Stripe | ≥ 10 paying customers, or provisioning exceeds 2 h/week |
| Pooled free tier | Both §8 fixes shipped with tests |
| Warm standby / 99.9% SLA | A customer contractually requires it — not before |
| Kubernetes | Provisioning volume makes declarative reconciliation worth its overhead; realistically past ~10 customers |
| Router + registry (`DISTRIBUTED_ARCHITECTURE_V2`) | One customer's campaigns exceed a single node's RAM |
| SOC 2 Type II | First enterprise deal is gated on it (budget $30–60k) |

Resist the router. It solves sharding campaigns within one logical database — a problem
none of the first fifty customers will have, since each has their own instance.

---

## 11. Cost

Per customer, `eu-central-1`:

| Item | Monthly |
|---|---|
| `small_3_0` instance (2 GB, 60 GB SSD) | $10 |
| Static IP (attached) | $0 |
| Automatic snapshots (~7 × 60 GB) | ~$2 |
| **Per customer** | **~$12** |

Shared: one small instance for Prometheus and blackbox exporter, ~$10/mo.

**Idle cost is $0** — nothing runs until a customer exists. At $299 that is ~96% gross
margin; at $99, ~88%. Break-even is customer #1.

Compare to the abandoned GKE plan: ~$120/month standing burn before a single customer,
mostly the $73 control plane.

---

## 12. Work items

| # | Item | Size | Blocks |
|---|---|---|---|
| 1 | ~~Single-file provisioning (`cloud-init.sh`)~~ **done, proven end to end** | M | — |
| 2 | ~~`provision.sh` from stock blueprint + user-data~~ **done, dry-run only** | S | — |
| 3 | Prometheus + blackbox exporter + the 9 alerts in §5.1 | M | Customer #2 |
| 4 | Monthly restore drill, documented and calendared | S | Customer #2 |
| 5 | Fleet sheet | XS | Customer #1 |
| 6 | Run `provision.sh` for real (needs IAM credentials) | XS | Customer #1 |
| 7 | `BANDITDB_MAX_CAMPAIGNS` | S | Free tier |
| 8 | Per-tenant pending-cache quota | M | Free tier |

Items 3–6 are the launch set: a few focused days.

---

## 13. Why not Kubernetes

The first version of this plan ran the fleet on GKE, one namespace and StatefulSet per
customer. The Helm chart for it still exists and was extended for the purpose — fleet
metrics, resource bounds, a cloud values overlay (chart 2.1.0). It was abandoned for
Lightsail after two observations.

**Operator familiarity beats platform capability at this scale.** The sandbox already
runs on Lightsail as a systemd service. Debugging a Kubernetes stateful workload alone at
2 a.m. is a different proposition from debugging a VM you already understand.

**The complexity was not BanditDB's.** A golden-image build was attempted first and
abandoned: stripping SSH host keys made launched instances unreachable when regeneration
lost a race with socket-activated sshd; `machine-id` had to be cleared; generalisation
had to be the terminal step because it destroyed its own access path. None of that had
anything to do with the database. Provisioning from a script at boot removed all of it.

Kubernetes also costs ~$120/month before the first customer, versus $0 here.

The chart remains the **self-hosted / BYOC artifact** and the migration target when
provisioning volume justifies declarative reconciliation — see §10.
