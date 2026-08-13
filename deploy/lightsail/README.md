# BanditDB on Lightsail — golden appliance

One Ubuntu image, snapshotted once, launched per customer. Companion to
[`../../docs/CLOUD_STAGE1.md`](../../docs/CLOUD_STAGE1.md).

```
build box (never serves a customer)
    │  build-appliance.sh          binary pinned, unit installed, NO keys, NO data
    │  verify-appliance.sh         gate: refuses if customer state is present
    │  generalize.sh               strips identity; SSH stops accepting after this
    ▼
snapshot: banditdb-appliance-v2.0.0
    │  provision.sh acme acme.api.banditdb.com
    ▼
customer instance
    │  firstboot.sh                generates keys, renders vhost, starts service
    │  banditdb-enable-tls         once DNS resolves
    ▼
live
```

| File | Runs where | When |
|---|---|---|
| `build-appliance.sh` | build box, as root | once per BanditDB release |
| `verify-appliance.sh` | build box, as root | after build, before generalize |
| `generalize.sh` | build box, as root | **last command on that box** |
| `provision.sh` | your laptop | per customer |
| `firstboot.sh` | customer instance | automatically, first boot |
| `enable-tls.sh` | customer instance | after DNS resolves |
| `banditdb.service` | installed by build | — |
| `nginx-vhost.conf.template` | rendered by firstboot | — |

## Ubuntu version

The scripts depend only on `apt`, `systemd`, `nginx` and `ufw`, so any current
Ubuntu LTS works. `build-appliance.sh` prints the base image it ran on and
records it in `/etc/banditdb/appliance-manifest.json` rather than assuming.

Pick the newest LTS blueprint Lightsail offers in your region:

```bash
aws lightsail get-blueprints --region eu-central-1 \
  --query "blueprints[?contains(name,'Ubuntu')].[blueprintId,name]" --output table
```

Lightsail lags Canonical's release date, so the newest LTS may not be offered
for some months after it ships. Take the newest that is listed.

Verified build, 2026-08-13, eu-central-1: **Ubuntu 24.04.4 LTS**, x86_64,
BanditDB v2.0.0, artifact sha256 `16f7f2fb683e6d13cac20be34518a81f32f77bac5329fcfb155de6c16a362e43`.
The binary reports `banditdb 2.0.0 (neural)`, so the published release does
include the neural algorithms — no custom build needed for NeuralLinUCB or
NeuralTS campaigns.

## Build the appliance

```bash
# Launch a throwaway instance from the Ubuntu blueprint, then on it:
git clone https://github.com/dynamicpricing-ai/banditdb
cd banditdb/deploy/lightsail
sudo BANDITDB_VERSION=v2.0.0 bash build-appliance.sh
sudo bash verify-appliance.sh          # must pass
sudo bash generalize.sh                # LAST command on this box
```

Then stop the instance and snapshot it as `banditdb-appliance-v2.0.0`.

`generalize.sh` deletes the SSH host keys, and Ubuntu 22.10+ socket-activates
sshd — so from that moment every new SSH connection is refused with
`kex_exchange_identification: Connection reset by peer`. An already-open session
survives; a reconnect does not. This is why generalisation is the terminal step
rather than part of the build: anything you still need to do on the box has to
happen before it. If you must get back in, reboot — `regenerate-ssh-hostkeys`
recreates the keys — then generalise again before snapshotting.

**The rule that matters:** the build box must never have served a customer.
Snapshotting a working instance clones its API keys and its learned state onto
whoever launches next. `verify-appliance.sh` exists to catch exactly that — it
fails if `banditdb.env` exists, if `/var/lib/banditdb` is non-empty, if SSH host
keys are present, or if `machine-id` is populated.

Rebuild the appliance per release; never mutate a snapshot in place. Keep the
previous one until every customer has moved off it — it is your rollback.

## SSH access

Create the **build** instance with the SSH key you intend to use for the whole
fleet. `generalize.sh` preserves `~ubuntu/.ssh/authorized_keys`, so that key is
baked into the snapshot and works on every instance launched from it — no
dependency on cloud-init re-injecting a key at launch time.

Host keys are the opposite: removed by `generalize.sh` and regenerated per
instance by `firstboot.sh`, so instances cannot impersonate one another. Expect
your client to warn about an unknown host on each new instance.

Customers never get shell access, only the HTTPS API, so the operator key being
common across the fleet is the intended design rather than a compromise.

## Provision a customer

```bash
export AWS_REGION=eu-central-1
./provision.sh acme acme.api.banditdb.com --tier starter
```

Bundles: `nano_3_0` 512 MB · `small_3_0` 2 GB (Starter) · `medium_3_0` 4 GB
(Growth) · `large_3_0` 8 GB.

512 MB is too small for anything real — pending interactions alone are budgeted
at ~100 MB, before matrices and the neural replay buffer.

Then, in order: DNS record → `banditdb-enable-tls` → durability check →
deliver credentials → shred the credentials file.

## Verify before handing over

Do not skip this. It is the only step that proves *this instance's* disk
actually persists an acknowledged write, and it takes about ninety seconds.

```bash
DOMAIN=acme.api.banditdb.com
ADMIN=...; WRITER=...; READER=...

curl -sf https://$DOMAIN/health | jq .        # expect version 2.0.0

curl -sf -X POST https://$DOMAIN/campaign -H "X-Api-Key: $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{"campaign_id":"provision-check","arms":["a","b"],"feature_dim":2}'

IID=$(curl -sf -X POST https://$DOMAIN/predict -H "X-Api-Key: $WRITER" \
  -H 'Content-Type: application/json' \
  -d '{"campaign_id":"provision-check","context":[0.6,0.8]}' | jq -r .interaction_id)

curl -sf -X POST https://$DOMAIN/reward -H "X-Api-Key: $WRITER" \
  -H 'Content-Type: application/json' \
  -d "{\"interaction_id\":\"$IID\",\"reward\":1.0}"

# Kill it. An acknowledged reward is fsynced, so it must survive.
ssh ubuntu@$IP 'sudo systemctl restart banditdb'
sleep 5

curl -sf https://$DOMAIN/campaign/provision-check -H "X-Api-Key: $READER" \
  | jq '.total_rewards'      # MUST be 1

curl -sf -X DELETE https://$DOMAIN/campaign/provision-check -H "X-Api-Key: $ADMIN"
```

If `total_rewards` is `0`, stop and investigate. Do not hand the instance over.

Also confirm role separation, since a mis-parsed key string would silently grant
everyone admin:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X DELETE \
  https://$DOMAIN/campaign/provision-check -H "X-Api-Key: $READER"   # expect 403
```

## Operating

**Backups.** `provision.sh` enables Lightsail automatic snapshots at 03:00.
Those are crash-consistent, and BanditDB is crash-safe — fsynced checkpoint,
`checkpoint.prev` fallback, WAL replay — so restoring one is sound. Checkpoint
first when you can, so replay is near zero:

```bash
curl -sf -X POST https://$DOMAIN/checkpoint -H "X-Api-Key: $ADMIN"
```

A snapshot nobody has restored is a hypothesis. Run the drill weekly against a
rotating customer, into a throwaway instance:

```bash
sudo /opt/banditdb/scripts/backup_restore.sh drill /var/lib/banditdb
```

Note its warning about `neural/`: a neural campaign restored without those
sidecars comes back with random weights. It serves, but it has forgotten
everything.

**Upgrades.** Build a new appliance for the new release. For existing customers,
replace the binary in place:

```bash
curl -fsSL https://github.com/dynamicpricing-ai/banditdb/releases/download/v2.1.0/banditdb-v2.1.0-aarch64-unknown-linux-gnu.tar.gz | tar xz
sudo install -m0755 banditdb /usr/local/bin/banditdb
sudo systemctl restart banditdb    # SIGTERM checkpoints, then replay on start
```

Downtime is seconds. Roll one canary customer, wait 24 hours, then the rest.

**Logs.** `journalctl -u banditdb -f`. JSON, since `LOG_FORMAT=json`.

**Metrics.** `/metrics` needs the reader key. Unlike the Kubernetes path — where
ServiceMonitor cannot send a custom header — plain Prometheus can, if your
version supports `http_headers` in `scrape_configs`. Verify against your version;
if it works, keep `/metrics` closed and scrape with the reader key.

The alerts worth wiring are in `docs/CLOUD_STAGE1.md` §6.1. The one that earns
its keep is `banditdb_interactions_evicted_total`: when it climbs, the customer's
rewards are 404ing and their model has quietly stopped learning. They cannot see
that. You can.

## Deliberate limitations

- **No self-healing beyond the process.** `Restart=always` covers a crash; nothing
  covers instance loss. Recovery is restore-from-snapshot, roughly 15 minutes.
  Say so in the SLA rather than discovering it during an incident.
- **No live vertical scaling.** Growing 2 GB → 4 GB is snapshot, restore to a
  bigger bundle, repoint DNS. Schedule it.
- **Single AZ.** The instance and its disk live in one availability zone. An AZ
  outage is a restore, not a failover.
- **TLS is a second step.** Certbot must prove control of a name that resolves
  here, which cannot be true before the instance exists.
