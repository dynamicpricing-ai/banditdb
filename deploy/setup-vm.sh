#!/usr/bin/env bash
#
# One-time preparation of the engine VM. Run on the VM, as a user with sudo.
#
#   curl -fsSL https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/deploy/setup-vm.sh | sudo bash
#
# or, from a checkout: sudo ./deploy/setup-vm.sh
#
# Idempotent. Every step checks before acting, so re-running after a failure is
# the intended way to finish a partial run. It does NOT install the binary —
# deploy/update.sh does that, and does it the same way on the first deploy and
# the hundredth.

set -euo pipefail

DATA_DIR=/data
DEVICE_NAME=banditdb-data          # --device-name given at attach time
DEV="/dev/disk/by-id/google-${DEVICE_NAME}"
ENV_FILE=/etc/banditdb/banditdb.env
UNIT=/etc/systemd/system/banditdb.service

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

say() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }

say "Service account"
if ! id banditdb >/dev/null 2>&1; then
  # No login shell and no home: this account exists to own a directory and a
  # process, and should not be a way onto the box.
  useradd --system --no-create-home --shell /usr/sbin/nologin banditdb
  echo "  created"
else
  echo "  already present"
fi

say "Data disk"
if [ ! -e "$DEV" ]; then
  echo "  $DEV not found. The disk is not attached, or was attached without"
  echo "  --device-name=$DEVICE_NAME. Attach it and re-run:"
  echo
  echo "    gcloud compute instances attach-disk <vm> --zone=<zone> \\"
  echo "      --disk=banditdb-data --device-name=$DEVICE_NAME"
  exit 1
fi

# `blkid` says nothing for an unformatted disk, which is how a blank one is
# distinguished from one that already holds data. Formatting is the only
# destructive step in this script, so it happens exactly once and never to a
# device that already has a filesystem.
if ! blkid "$DEV" >/dev/null 2>&1; then
  echo "  no filesystem found; creating ext4"
  # -m 0 skips the 5% root reserve: this filesystem has one user, and on 200 GB
  # the reserve is 10 GB of nothing.
  mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$DEV"
else
  echo "  filesystem already present, leaving it alone"
fi

mkdir -p "$DATA_DIR"

UUID="$(blkid -s UUID -o value "$DEV")"
if ! grep -q "$UUID" /etc/fstab; then
  # By UUID, not by device path: device names are ordering-dependent and a
  # second disk would silently renumber them. `nofail` keeps a missing disk from
  # dropping the VM into emergency mode where you cannot SSH in to fix it —
  # the systemd unit's RequiresMountsFor is what stops the engine starting
  # without its data.
  echo "UUID=$UUID $DATA_DIR ext4 discard,defaults,nofail 0 2" >> /etc/fstab
  echo "  added to /etc/fstab"
else
  echo "  already in /etc/fstab"
fi

mountpoint -q "$DATA_DIR" || mount "$DATA_DIR"
chown banditdb:banditdb "$DATA_DIR"
chmod 750 "$DATA_DIR"
echo "  mounted: $(df -h "$DATA_DIR" | tail -1)"

say "Environment file"
mkdir -p "$(dirname "$ENV_FILE")"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
# Machine-to-machine credential for the control plane. Must equal the console's
# BANDITDB_PROVISION_KEY; they are compared directly, and a mismatch shows up as
# every provisioning attempt parking in the console's outbox with a 401.
BANDITDB_PROVISION_KEY=

# Optional. Leave unset in production: with a provision key configured, tenants
# and their keys come from the control plane, and a static key here would be a
# second, unmanaged way in.
# BANDITDB_API_KEYS=

# Auto-checkpoint thresholds. Both are safety nets over the periodic checkpoint;
# whichever trips first wins.
BANDITDB_CHECKPOINT_INTERVAL=1000
BANDITDB_MAX_WAL_SIZE_MB=512
EOF
  echo "  created $ENV_FILE — add BANDITDB_PROVISION_KEY before starting"
else
  echo "  already present, leaving it alone"
fi
# Root-owned and unreadable to everyone else. systemd reads it as root before
# dropping to the banditdb user, so the service account never needs access.
chown root:root "$ENV_FILE"
chmod 0600 "$ENV_FILE"

say "systemd unit"
if [ -n "${BANDITDB_SRC:-}" ] && [ -f "$BANDITDB_SRC/deploy/banditdb.service" ]; then
  install -m 0644 "$BANDITDB_SRC/deploy/banditdb.service" "$UNIT"
else
  curl -fsSL -o "$UNIT" \
    https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/deploy/banditdb.service
fi
systemctl daemon-reload
systemctl enable banditdb >/dev/null
echo "  installed and enabled (not started — no binary yet)"

say "Done"
cat <<EOF

Next:
  1. Put the provisioning key in $ENV_FILE
  2. Install a release:  sudo ./deploy/update.sh v2.0.0
  3. Check:              curl -s localhost:8080/health

The engine binds 0.0.0.0:8080 but nothing exposes it: no firewall rule opens
8080 externally, and default-allow-internal admits only 10.128.0.0/9 — which is
how Cloud Run reaches it over Direct VPC egress.
EOF
