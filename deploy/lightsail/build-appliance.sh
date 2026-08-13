#!/usr/bin/env bash
# Build the BanditDB golden appliance on a fresh Ubuntu Lightsail instance.
#
# Run this ONCE on a throwaway instance that will never serve a customer, then
# snapshot that instance. Every customer machine is launched from the snapshot.
#
#   sudo BANDITDB_VERSION=v2.0.0 bash build-appliance.sh
#
# The result is deliberately INERT: the binary is installed and the unit is
# enabled, but there is no data, no API key, and no hostname. Those come from
# firstboot.sh when a customer instance is launched. Snapshotting a machine that
# has served anyone would clone that customer's keys and learned state onto the
# next customer's box — see verify-appliance.sh, which checks exactly that.
set -euo pipefail

# Namespaced deliberately. A plain VERSION would be clobbered by /etc/os-release
# below, which sets VERSION="24.04.4 LTS (Noble Numbat)" — the download URL then
# gets built from the Ubuntu version string and curl rejects it as malformed.
BDB_VERSION="${BANDITDB_VERSION:-v2.0.0}"
REPO="dynamicpricing-ai/banditdb"
DATA_DIR=/var/lib/banditdb
CONF_DIR=/etc/banditdb
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ $EUID -eq 0 ]] || { echo "must run as root" >&2; exit 1; }

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

# ── 0. Record what we are building on ────────────────────────────────────────
log "Base image"
# Subshells, not a bare `.` — /etc/os-release defines NAME, VERSION, ID and
# friends, and sourcing it into this shell silently overwrites any of ours that
# share a name.
os_pretty="$(. /etc/os-release && echo "${PRETTY_NAME:-unknown}")"
os_id="$(. /etc/os-release && echo "${ID:-unknown}")"
echo "  $os_pretty, kernel $(uname -r), arch $(uname -m)"
if [[ "$os_id" != "ubuntu" ]]; then
  echo "  WARNING: this script targets Ubuntu; $os_id may behave differently." >&2
fi

# Fail here rather than at a malformed URL 60 seconds later.
if [[ ! "$BDB_VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
  echo "ERROR: BANDITDB_VERSION='$BDB_VERSION' is not a release tag (expected vX.Y.Z)." >&2
  exit 1
fi

# ── 1. Base packages ─────────────────────────────────────────────────────────
log "Installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
  nginx certbot python3-certbot-nginx \
  curl ca-certificates jq unattended-upgrades ufw fail2ban chrony

# Security patches apply themselves. A fleet of instances nobody logs into is
# exactly the situation unattended-upgrades exists for.
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF

# ── 2. Service account and directories ───────────────────────────────────────
log "Creating banditdb user and directories"
if ! id -u banditdb >/dev/null 2>&1; then
  useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin banditdb
fi
install -d -o banditdb -g banditdb -m 0750 "$DATA_DIR"
install -d -o root     -g root     -m 0755 "$CONF_DIR"

# ── 3. BanditDB binary, pinned ───────────────────────────────────────────────
log "Installing BanditDB $BDB_VERSION"
case "$(uname -m)" in
  x86_64)        target=x86_64-unknown-linux-gnu  ;;
  aarch64|arm64) target=aarch64-unknown-linux-gnu ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
url="https://github.com/$REPO/releases/download/$BDB_VERSION/banditdb-$BDB_VERSION-$target.tar.gz"
echo "  $url"
curl -fsSL --retry 3 "$url" -o "$tmp/bdb.tar.gz"

# The release does not publish checksums, so record what we actually installed.
# A later rebuild that produces a different hash for the same tag means the
# artifact changed underneath you, which is worth knowing before it ships.
sha="$(sha256sum "$tmp/bdb.tar.gz" | cut -d' ' -f1)"
tar -xzf "$tmp/bdb.tar.gz" -C "$tmp"
install -o root -g root -m 0755 "$tmp/banditdb" /usr/local/bin/banditdb

/usr/local/bin/banditdb --version 2>/dev/null || true

cat > "$CONF_DIR/appliance-manifest.json" <<EOF
{
  "banditdb_version": "$BDB_VERSION",
  "target": "$target",
  "artifact_sha256": "$sha",
  "base_image": "$os_pretty",
  "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
chmod 0644 "$CONF_DIR/appliance-manifest.json"

# ── 4. systemd unit and templates ────────────────────────────────────────────
log "Installing unit and templates"
install -o root -g root -m 0644 "$SRC_DIR/banditdb.service"           /etc/systemd/system/banditdb.service
install -o root -g root -m 0644 "$SRC_DIR/nginx-vhost.conf.template"  "$CONF_DIR/nginx-vhost.conf.template"
install -o root -g root -m 0755 "$SRC_DIR/firstboot.sh"               /usr/local/sbin/banditdb-firstboot
install -o root -g root -m 0755 "$SRC_DIR/enable-tls.sh"              /usr/local/sbin/banditdb-enable-tls

# Runs once on the first boot of each launched instance, then disables itself.
cat > /etc/systemd/system/banditdb-firstboot.service <<'EOF'
[Unit]
Description=BanditDB first-boot provisioning
After=network-online.target
Wants=network-online.target
# Deliberately no ConditionPathExists on banditdb.env. The script exits early by
# itself once provisioned, and it also repairs missing SSH host keys — which has
# to keep working on every boot, not just the first one.
Before=ssh.socket ssh.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/banditdb-firstboot
RemainAfterExit=true
StandardOutput=journal+console
StandardError=journal+console

[Install]
WantedBy=multi-user.target
EOF

# Recreates SSH host keys on the first boot of every launched instance.
#
# generalize.sh deletes the keys so that N instances from one snapshot do not
# share a single SSH identity — anyone holding one box could otherwise
# impersonate every other box to an SSH client. This unit is what makes that
# safe to do: without it, launched instances come up with no host keys and no
# way to generate them, and are simply unreachable.
#
# It belongs here rather than in generalize.sh because it is part of the image.
# generalize.sh only removes state; it installs nothing.
cat > /etc/systemd/system/regenerate-ssh-hostkeys.service <<'EOF'
[Unit]
Description=Regenerate SSH host keys on first boot
Before=ssh.service ssh.socket
ConditionPathExists=!/etc/ssh/ssh_host_ed25519_key

[Service]
Type=oneshot
ExecStart=/usr/bin/ssh-keygen -A
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable nginx banditdb-firstboot regenerate-ssh-hostkeys

# banditdb itself is ENABLED but cannot start yet: the unit requires
# /etc/banditdb/banditdb.env, which firstboot writes. Enabling it here means a
# customer instance comes up serving immediately after firstboot, with no
# further action.
systemctl enable banditdb

rm -f /etc/nginx/sites-enabled/default

# ── 5. Firewall ──────────────────────────────────────────────────────────────
log "Configuring firewall"
ufw --force reset >/dev/null
ufw default deny incoming  >/dev/null
ufw default allow outgoing >/dev/null
ufw allow 22/tcp  >/dev/null   # keep SSH before enabling, or you lock yourself out
ufw allow 80/tcp  >/dev/null
ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null
ufw status verbose | sed 's/^/  /'

# Port 8080 is intentionally NOT opened. BanditDB binds locally and is reached
# only through nginx, so the API cannot be hit bypassing TLS and rate limits.

log "Build complete"
cat "$CONF_DIR/appliance-manifest.json"
cat <<'EOF'

Next, in this order:

  1. sudo bash verify-appliance.sh     # gate — must pass
  2. sudo bash generalize.sh           # LAST command you run on this box
  3. Stop the instance and snapshot it as banditdb-appliance-<version>

Step 2 deletes the SSH host keys. This host socket-activates sshd, so every
NEW connection is refused the moment they are gone — an already-open session
survives, a reconnect does not. That is intended: a generalised image is
finished, and anything you still need to do must happen before step 2.

Never snapshot an instance that has served a customer into this lineage.
EOF
