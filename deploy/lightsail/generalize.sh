#!/usr/bin/env bash
# Strip machine identity from the appliance. THE LAST COMMAND YOU RUN ON THIS BOX.
#
#   sudo bash generalize.sh
#
# Run build-appliance.sh and verify-appliance.sh first. This is separate from
# both because of what it does to SSH: deleting the host keys means that on a
# socket-activated sshd (Ubuntu 22.10 and later) every NEW connection is refused
# immediately — "kex_exchange_identification: Connection reset by peer". An
# already-open session keeps working; a reconnect does not.
#
# That is the correct shape for an image build. Cleanup is terminal: generalise,
# stop, snapshot, never log in again. Folding this into build-appliance.sh meant
# the verifier that runs afterwards could not be reached.
#
# If you do need back in, reboot: regenerate-ssh-hostkeys.service recreates the
# keys on boot. You then have to generalise again before snapshotting.
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "must run as root" >&2; exit 1; }

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

# Refuse to generalise a box that was never built, or one carrying customer data.
[[ -x /usr/local/bin/banditdb ]] || { echo "banditdb not installed — run build-appliance.sh first" >&2; exit 1; }
if [[ -e /etc/banditdb/banditdb.env ]]; then
  echo "REFUSING: /etc/banditdb/banditdb.env exists — this box has been provisioned." >&2
  echo "Snapshotting it would clone its API keys onto every future customer." >&2
  exit 1
fi
shopt -s nullglob dotglob
data=(/var/lib/banditdb/*)
if (( ${#data[@]} )); then
  echo "REFUSING: /var/lib/banditdb is not empty — this box holds a customer's data." >&2
  exit 1
fi

cat <<'EOF'

  This deletes the SSH host keys. New SSH connections will be refused until
  the instance is rebooted. Do not run it until verify-appliance.sh passes.

EOF
read -rp "  Generalize now? [y/N] " reply
[[ "$reply" == "y" ]] || { echo "aborted"; exit 1; }

# ── Identity ─────────────────────────────────────────────────────────────────
log "Removing machine identity"

# Shared host keys would let anyone holding one instance impersonate every other
# instance to an SSH client.
rm -f /etc/ssh/ssh_host_*
echo "  ssh host keys removed (regenerate-ssh-hostkeys.service recreates on boot)"

# Duplicate machine-ids confuse journald, DHCP leases and anything keyed on it.
truncate -s 0 /etc/machine-id
rm -f /var/lib/dbus/machine-id
ln -sf /etc/machine-id /var/lib/dbus/machine-id
echo "  machine-id cleared"

# ── Traces ───────────────────────────────────────────────────────────────────
log "Clearing logs and history"
cloud-init clean --logs 2>/dev/null || true
apt-get clean
journalctl --rotate >/dev/null 2>&1 || true
journalctl --vacuum-time=1s >/dev/null 2>&1 || true
rm -rf /var/log/*.gz /var/log/*.[0-9] /tmp/* /var/tmp/*
rm -f /root/.bash_history /home/*/.bash_history
# The copied scripts themselves: harmless, but there is no reason to ship them.
rm -rf /home/ubuntu/lightsail

# ── Confirm our own work ─────────────────────────────────────────────────────
log "Post-checks"
fail=0
hostkeys=(/etc/ssh/ssh_host_*)
if (( ${#hostkeys[@]} )); then
  echo "  FAIL  host keys still present: ${hostkeys[*]}"; fail=1
else
  echo "  PASS  no ssh host keys"
fi
if [[ -s /etc/machine-id ]]; then
  echo "  FAIL  machine-id still populated"; fail=1
else
  echo "  PASS  machine-id empty"
fi
if systemctl is-enabled regenerate-ssh-hostkeys >/dev/null 2>&1; then
  echo "  PASS  regenerate-ssh-hostkeys enabled"
else
  echo "  FAIL  regenerate-ssh-hostkeys NOT enabled — launched instances will have no host keys"; fail=1
fi

if (( fail )); then
  printf '\n\033[0;31mGeneralisation incomplete. Do not snapshot.\033[0m\n'
  exit 1
fi

cat <<'EOF'

  Generalised. This box is now finished — SSH will refuse new connections.

  From your laptop:
    aws lightsail stop-instance --instance-name <build-box> --region <region>
    aws lightsail create-instance-snapshot --region <region> \
      --instance-name <build-box> --instance-snapshot-name banditdb-appliance-vX.Y.Z
    # wait for state "available", then:
    aws lightsail delete-instance --instance-name <build-box> --region <region>

EOF
