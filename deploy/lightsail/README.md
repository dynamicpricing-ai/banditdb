# BanditDB on Lightsail

One script. Create a stock Ubuntu instance, run it, done — no golden image, no
snapshot lifecycle. Companion to [`../../docs/CLOUD_STAGE1.md`](../../docs/CLOUD_STAGE1.md).

| File | Purpose |
|---|---|
| `cloud-init.sh` | Provisions a complete instance. Upload and run, or paste as a launch script. |
| `provision.sh` | Creates the instance and feeds it `cloud-init.sh` as user-data. Needs the AWS CLI. |

## Provision a customer

**By hand** (no AWS CLI needed):

1. Lightsail → create instance → Ubuntu 24.04 LTS → **2 GB (`small_3_0`)** → your SSH key
2. Either paste `cloud-init.sh` into **Add launch script**, or upload it after boot:

```bash
scp deploy/lightsail/cloud-init.sh ubuntu@<ip>:~/
ssh ubuntu@<ip>
sudo CUSTOMER_ID=acme SERVER_NAME=acme.api.banditdb.com bash cloud-init.sh
```

**Or scripted:**

```bash
export AWS_REGION=eu-central-1
./provision.sh acme acme.api.banditdb.com --tier starter --dry-run   # inspect first
./provision.sh acme acme.api.banditdb.com --tier starter
```

Takes about three minutes, mostly `apt`. Then `curl http://<ip>/health`.

Bundles: `small_3_0` 2 GB (Starter) · `medium_3_0` 4 GB (Growth) · `large_3_0` 8 GB.
`nano_3_0` is too small — pending interactions alone are budgeted at ~100 MB
before matrices and the neural replay buffer.

## TLS

Two prerequisites, both of which fail confusingly if skipped:

- **Open 443 in the Lightsail firewall** (Networking tab). `ufw` inside the
  instance already allows it, but the Lightsail layer drops the packet first, and
  the symptom is certbot failing for no visible reason. `provision.sh` does this
  for you.
- **DNS must resolve first.** Certbot proves control by answering a challenge on
  that hostname, so the record has to exist before you run it. Repeated failures
  count against Let's Encrypt's limit of 5 per hostname per hour.

```bash
dig +short acme.api.banditdb.com        # must return the instance IP
sudo certbot --nginx -d acme.api.banditdb.com --agree-tos -m ops@banditdb.com --redirect
sudo certbot renew --dry-run            # confirm renewal works; it fails silently otherwise
```

## Verify before handing over

Not optional. It proves *this instance's disk* persists an acknowledged write,
and it takes about a minute. Reads the keys from the env file so no secrets get
copied around.

```bash
sudo bash <<'EOF'
KEYS=$(sed -n 's/^BANDITDB_API_KEYS="\(.*\)"$/\1/p' /etc/banditdb/banditdb.env)
ADMIN=${KEYS%%=admin*}
WRITER=$(echo "$KEYS" | sed 's/.*=admin;//;s/=writer.*//')
READER=$(echo "$KEYS" | sed 's/.*=writer;//;s/=reader.*//')
B=http://localhost

curl -sf -X POST $B/campaign -H "X-Api-Key: $ADMIN" -H 'Content-Type: application/json' \
  -d '{"campaign_id":"provision-check","arms":["a","b"],"feature_dim":2}' >/dev/null

IID=$(curl -sf -X POST $B/predict -H "X-Api-Key: $WRITER" -H 'Content-Type: application/json' \
  -d '{"campaign_id":"provision-check","context":[0.6,0.8]}' | jq -r .interaction_id)

curl -sf -X POST $B/reward -H "X-Api-Key: $WRITER" -H 'Content-Type: application/json' \
  -d "{\"interaction_id\":\"$IID\",\"reward\":1.0}" >/dev/null

# SIGKILL: no graceful shutdown, no final checkpoint. Recovery must replay the
# WAL. An acknowledged reward is fsynced, so it has to survive this.
systemctl kill -s SIGKILL banditdb
sleep 8

echo -n "total_rewards after SIGKILL (must be 1): "
curl -sf $B/campaign/provision-check -H "X-Api-Key: $READER" | jq '.total_rewards'
echo -n "reader DELETE (must be 403): "
curl -s -o /dev/null -w '%{http_code}\n' -X DELETE $B/campaign/provision-check -H "X-Api-Key: $READER"
curl -sf -X DELETE $B/campaign/provision-check -H "X-Api-Key: $ADMIN" >/dev/null
EOF
```

`total_rewards` must be `1`. If it is `0`, an acknowledged write was lost — stop
and investigate before anyone depends on this instance. The `403` confirms the
three roles parsed distinctly rather than everyone getting admin.

Then deliver `/etc/banditdb/credentials.txt` over a one-time secret link — never
plain email — and `shred -u` it.

## Operating

**Logs.** `journalctl -u banditdb -f` (JSON) and `/var/log/cloud-init-output.log`
for provisioning.

**Backups.** `provision.sh` enables daily Lightsail snapshots at 03:00. Those are
crash-consistent, and BanditDB is crash-safe — fsynced checkpoint,
`checkpoint.prev` fallback, WAL replay — so restoring one is sound. Checkpoint
first when you can, so replay is near zero:

```bash
curl -sf -X POST https://<domain>/checkpoint -H "X-Api-Key: $ADMIN"
```

A snapshot nobody has restored is a hypothesis. Restore one into a throwaway
instance monthly and run the check above against it.

**Upgrades.**

```bash
V=v2.1.0
curl -fsSL "https://github.com/dynamicpricing-ai/banditdb/releases/download/$V/banditdb-$V-x86_64-unknown-linux-gnu.tar.gz" | tar xz
sudo install -m0755 banditdb /usr/local/bin/banditdb
sudo systemctl restart banditdb     # SIGTERM checkpoints, then replays on start
```

Downtime is seconds — recovery replays 100k events in ~0.6 s. Roll one canary
customer, wait 24 hours, then the rest.

**Metrics.** `/metrics` needs the reader key. Plain Prometheus can send it via
`http_headers` in `scrape_configs` (check your version supports it), so unlike the
Kubernetes path there is no need to open the endpoint. Alerts worth wiring are in
`docs/CLOUD_STAGE1.md` §6.1 — the one that earns its keep is
`banditdb_interactions_evicted_total`: when it climbs, the customer's rewards are
404ing and their model has quietly stopped learning. They cannot see that.

## Deliberate limitations

- **No self-healing beyond the process.** `Restart=always` covers a crash; nothing
  covers instance loss. Recovery is restore-from-snapshot, ~15 minutes. Put that
  in the SLA rather than discovering it during an incident.
- **No live vertical scaling.** 2 GB → 4 GB is snapshot, restore to a bigger
  bundle, repoint DNS. Schedule it.
- **Single AZ.** Instance and disk live in one availability zone. An AZ outage is
  a restore, not a failover.

## Why there is no golden image

An earlier version of this directory built an appliance, snapshotted it, and
launched customers from the snapshot. It was removed. Nearly every problem it
caused was image hygiene rather than BanditDB: stripping SSH host keys made
launched instances unreachable when regeneration lost a race with socket-activated
sshd; `machine-id` had to be cleared; generalisation had to be the terminal step
because it destroyed its own access path; and the snapshot lifecycle needed its
own verification gate.

Provisioning from a script at boot removes all of it. The cost is roughly two
extra minutes per instance and a dependency on GitHub being reachable at
provision time. At this scale that is a good trade — and the thing under version
control is now the thing that actually runs, rather than a snapshot built from it
weeks ago.

The old scripts remain in git history if the trade ever reverses.
