# Deploying the engine

The engine runs as a systemd service on a single VM, with its data on a separate
disk mounted at `/data`. It is a stateful singleton — one process holds every
campaign's model in memory and takes an exclusive lock on its data directory —
so there is no rolling deploy and no second instance. A deploy is a restart, and
a restart replays the write-ahead log.

That is why deploys are manual. `Release` publishes artifacts when you tag;
`Deploy engine` puts one on the box, when you choose.

```
tag ──► Release ──► GitHub release assets
                         │
      you ──► Deploy engine ──► VM: fetch, install, restart, health-check
                                     └─ rolls back on a failed health check
```

## First-time setup on the VM

```bash
gcloud compute ssh <vm> --zone=<zone>
curl -fsSL https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/deploy/setup-vm.sh \
  | sudo bash
```

Creates the `banditdb` service account, formats and mounts the data disk, writes
`/etc/banditdb/banditdb.env`, and installs the unit. Idempotent — re-run it after
a failure rather than unpicking a partial run. It does not install a binary.

The disk must already be attached **with `--device-name=banditdb-data`**; the
script looks for `/dev/disk/by-id/google-banditdb-data` and stops if it is not
there. It formats only a device with no filesystem, so it cannot eat data.

Then put the provisioning key in place and install a release:

```bash
sudo sed -i 's/^BANDITDB_PROVISION_KEY=$/BANDITDB_PROVISION_KEY=<key>/' \
  /etc/banditdb/banditdb.env
sudo ./deploy/update.sh v2.0.0
curl -s localhost:8080/health
```

That key must equal the console's `BANDITDB_PROVISION_KEY`. They are compared
directly; a mismatch appears as every provisioning attempt parking in the
console's outbox with a 401, rather than as an error anyone sees at sign-up.

## Routine deploys

Run the **Deploy engine** workflow with a release tag. It connects over IAP,
runs `update.sh`, and then asserts that `/health` reports the version asked for
— without that check a successful rollback would look like a successful deploy.

Repository variables it needs:

| Variable | Value |
|---|---|
| `GCP_PROJECT` | `banditdb` |
| `GCP_WIF_PROVIDER` | the Workload Identity provider resource name |
| `GCP_DEPLOYER_SA` | deployer service account email |
| `ENGINE_VM` | instance name |
| `ENGINE_ZONE` | `us-central1-f` |

## Why the unit is written the way it is

**`RequiresMountsFor=/data`** is the line that matters. Started before the disk
is mounted, the engine would create a fresh empty database on the boot disk and
serve it, and the real data would reappear underneath the mount point — present
on disk, invisible to the process. systemd refusing to start is the only good
outcome there.

**`nofail` in `/etc/fstab`, `RequiresMountsFor` in the unit.** A missing disk
should not drop the VM into emergency mode, where you cannot SSH in to fix it.
It should stop the engine, and only the engine.

**`TimeoutStopSec=120`.** A checkpoint may be in flight. Losing it means
replaying the WAL from the previous one, which is startup measured in minutes
rather than seconds.

**Secrets in a root-owned `EnvironmentFile`, not the unit.** systemd reads it as
root before dropping to the `banditdb` user, so the service account never needs
access to its own credentials, and they stay out of `systemctl show`.

## Exposure

The engine binds `0.0.0.0:8080`, but nothing routes to it from outside: no
firewall rule opens 8080, and `default-allow-internal` admits only
`10.128.0.0/9`. That is how Cloud Run reaches it over Direct VPC egress, on the
private IP, with no public address on the path.
