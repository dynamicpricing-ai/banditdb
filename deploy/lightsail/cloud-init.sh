#!/bin/bash
# BanditDB — single-file provisioning for a Lightsail instance.
#
# HOW TO USE — either way works, it is just a bash script run as root.
#
#   A. Upload and run (easier):
#        scp cloud-init.sh ubuntu@<ip>:~/
#        ssh ubuntu@<ip>
#        sudo CUSTOMER_ID=acme SERVER_NAME=acme.api.banditdb.com bash cloud-init.sh
#
#   B. Paste into Lightsail's "Add launch script" box at instance creation, with
#      the defaults below edited. Provisions unattended on first boot.
#
# Then: curl http://<ip>/health
#
# There is no golden image, no snapshot and no generalisation step. cloud-init
# runs this as root on first boot and the instance comes up serving. That means
# no shared machine identity to strip, and your SSH key is simply whichever one
# Lightsail attached — nothing to inject or swap.
#
# Progress and errors land in /var/log/cloud-init-output.log.

set -euo pipefail

# ─── CONFIGURE ───────────────────────────────────────────────────────────────
# Every value can be overridden from the environment, so uploading the file and
# passing variables on the command line avoids editing it:
#   sudo CUSTOMER_ID=acme SERVER_NAME=acme.api.banditdb.com bash cloud-init.sh
CUSTOMER_ID="${CUSTOMER_ID:-testco}"                  # [a-z0-9-]; labels the instance
SERVER_NAME="${SERVER_NAME:-_}"                       # hostname served; "_" = any (IP access)
BDB_VERSION="${BDB_VERSION:-v2.0.0}"                  # pinned; never "latest"

REWARD_TTL_SECS="${REWARD_TTL_SECS:-86400}"           # how long a prediction waits for its reward
MAX_PENDING="${MAX_PENDING:-100000}"                  # pending predictions; ~1 KB each
RATE_LIMIT="${RATE_LIMIT:-1000}"                      # per-key requests/sec
CORS_ORIGINS="${CORS_ORIGINS:-}"                      # empty = deny all cross-origin
# ─────────────────────────────────────────────────────────────────────────────

[[ "$CUSTOMER_ID" =~ ^[a-z0-9-]+$ ]] || { echo "CUSTOMER_ID must match [a-z0-9-]+" >&2; exit 1; }
[[ "$BDB_VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]] || { echo "BDB_VERSION must look like vX.Y.Z" >&2; exit 1; }

REPO="dynamicpricing-ai/banditdb"
DATA_DIR=/var/lib/banditdb
CONF_DIR=/etc/banditdb
ENV_FILE="$CONF_DIR/banditdb.env"

log()  { printf '\n[banditdb] %s\n' "$*"; }
fail() { printf '\n[banditdb] FAILED at line %s. See /var/log/cloud-init-output.log\n' "$1" >&2; }
trap 'fail $LINENO' ERR

if [[ -f "$ENV_FILE" ]]; then
  log "already provisioned; nothing to do"
  exit 0
fi

log "Provisioning $CUSTOMER_ID ($SERVER_NAME), BanditDB $BDB_VERSION"

# ── Packages ─────────────────────────────────────────────────────────────────
log "Installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nginx certbot python3-certbot-nginx \
  curl ca-certificates jq unattended-upgrades ufw fail2ban

# Security patches apply themselves; nobody logs into these boxes routinely.
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF

# ── Service account ──────────────────────────────────────────────────────────
id -u banditdb >/dev/null 2>&1 || \
  useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin banditdb
install -d -o banditdb -g banditdb -m 0750 "$DATA_DIR"
install -d -o root -g root -m 0755 "$CONF_DIR"

# ── Binary ───────────────────────────────────────────────────────────────────
log "Installing BanditDB $BDB_VERSION"
case "$(uname -m)" in
  x86_64)        target=x86_64-unknown-linux-gnu  ;;
  aarch64|arm64) target=aarch64-unknown-linux-gnu ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

tmp="$(mktemp -d)"
curl -fsSL --retry 3 \
  "https://github.com/$REPO/releases/download/$BDB_VERSION/banditdb-$BDB_VERSION-$target.tar.gz" \
  -o "$tmp/bdb.tar.gz"
sha="$(sha256sum "$tmp/bdb.tar.gz" | cut -d' ' -f1)"
tar -xzf "$tmp/bdb.tar.gz" -C "$tmp"
install -o root -g root -m 0755 "$tmp/banditdb" /usr/local/bin/banditdb
rm -rf "$tmp"

cat > "$CONF_DIR/manifest.json" <<EOF
{"customer":"$CUSTOMER_ID","banditdb_version":"$BDB_VERSION","target":"$target",
 "artifact_sha256":"$sha","provisioned":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

# ── API keys ─────────────────────────────────────────────────────────────────
ADMIN_KEY="bdb_admin_$(openssl rand -hex 24)"
WRITER_KEY="bdb_write_$(openssl rand -hex 24)"
READER_KEY="bdb_read_$(openssl rand -hex 24)"

umask 077
cat > "$ENV_FILE" <<EOF
# Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) for $CUSTOMER_ID.
DATA_DIR=$DATA_DIR
PORT=8080
LOG_FORMAT=json
RUST_LOG=info

# Quoted deliberately. systemd's EnvironmentFile treats ';' as an ordinary
# character, but anyone debugging with 'source /etc/banditdb/banditdb.env' gets
# shell parsing, where ';' ends the statement and the value truncates after the
# first key — leaving a database where only the admin key works and the other
# two return 401, with nothing in the logs to explain it.
BANDITDB_API_KEYS="$ADMIN_KEY=admin;$WRITER_KEY=writer;$READER_KEY=reader"
# Without keys every caller is granted admin, so a missing secret must stop the
# process rather than silently publish an open database.
BANDITDB_REQUIRE_AUTH=true

BANDITDB_TENANT_MODE=false
BANDITDB_CORS_ORIGINS=$CORS_ORIGINS

BANDITDB_CHECKPOINT_INTERVAL=5000
BANDITDB_MAX_WAL_SIZE_MB=100
BANDITDB_EXPORT_RETAIN_SHARDS=50

# The real memory bound. The TTL alone does not cap this: at 1,000 predictions
# per second a 24h window would accumulate 86 million records.
BANDITDB_MAX_PENDING_INTERACTIONS=$MAX_PENDING
BANDITDB_REWARD_TTL_SECS=$REWARD_TTL_SECS

BANDITDB_RATE_LIMIT_PER_SEC=$RATE_LIMIT

# /metrics names campaigns and arms, so it requires the reader key.
BANDITDB_METRICS_PUBLIC=false
EOF
chown root:root "$ENV_FILE"; chmod 0600 "$ENV_FILE"
umask 022

# ── systemd unit ─────────────────────────────────────────────────────────────
cat > /etc/systemd/system/banditdb.service <<'EOF'
[Unit]
Description=BanditDB — contextual bandit decision database
Documentation=https://banditdb.com/docs/
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=banditdb
Group=banditdb
EnvironmentFile=/etc/banditdb/banditdb.env
ExecStart=/usr/local/bin/banditdb
Restart=always
RestartSec=2s

# SIGTERM triggers a graceful shutdown with a final checkpoint. A campaign with
# large neural state takes a while to snapshot; killing it mid-checkpoint is
# survivable but throws away the work.
KillSignal=SIGTERM
TimeoutStopSec=120s

NoNewPrivileges=true
PrivateTmp=true
PrivateDevices=true
ProtectSystem=strict
ProtectHome=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
RestrictNamespaces=true
LockPersonality=true
RestrictSUIDSGID=true
RemoveIPC=true
# The only writable path: a compromised process cannot rewrite its own binary.
ReadWritePaths=/var/lib/banditdb
LimitNOFILE=65536

StandardOutput=journal
StandardError=journal
SyslogIdentifier=banditdb

[Install]
WantedBy=multi-user.target
EOF

# ── nginx ────────────────────────────────────────────────────────────────────
# Rate limiting here is a blunt safety net far above any legitimate load. A
# per-IP limit would be wrong: all of a customer's traffic arrives from one or
# two backend addresses, so a low per-IP cap would throttle the customer no
# matter what they are paying for. Real per-key limiting is the database's job.
cat > /etc/nginx/sites-available/banditdb <<EOF
limit_req_zone \$binary_remote_addr zone=safety:10m rate=2000r/s;

server {
    listen 80;
    listen [::]:80;
    server_name $SERVER_NAME;

    location /.well-known/acme-challenge/ { root /var/www/html; }

    location / {
        limit_req zone=safety burst=4000 nodelay;
        limit_req_status 429;

        proxy_pass         http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;

        # A reward blocks on an fsync before it answers.
        proxy_read_timeout    120s;
        proxy_send_timeout    120s;
        proxy_connect_timeout 5s;
        client_max_body_size  8m;
    }

    location = /health { proxy_pass http://127.0.0.1:8080; access_log off; }
    location = /metrics { proxy_pass http://127.0.0.1:8080; access_log off; }
}
EOF
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/banditdb /etc/nginx/sites-enabled/banditdb
install -d -o www-data -g www-data /var/www/html
nginx -t
systemctl reload nginx

# ── Firewall ─────────────────────────────────────────────────────────────────
# 8080 is deliberately closed: BanditDB is reached only through nginx, so the
# API cannot be hit bypassing TLS and rate limits.
ufw --force reset >/dev/null
ufw default deny incoming  >/dev/null
ufw default allow outgoing >/dev/null
ufw allow 22/tcp  >/dev/null
ufw allow 80/tcp  >/dev/null
ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null

# ── Start ────────────────────────────────────────────────────────────────────
log "Starting banditdb"
systemctl daemon-reload
systemctl enable --now banditdb

for _ in $(seq 1 30); do
  curl -sf --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break
  sleep 1
done

health="$(curl -sf --max-time 5 http://127.0.0.1:8080/health || echo '{}')"
version="$(echo "$health" | jq -r '.version // "unknown"')"
features="$(echo "$health" | jq -c '.features // []')"

if [[ "$version" == "unknown" ]]; then
  log "ERROR: BanditDB did not become healthy — journalctl -u banditdb"
  exit 1
fi

# ── Credentials ──────────────────────────────────────────────────────────────
cat > "$CONF_DIR/credentials.txt" <<EOF
BanditDB instance for: $CUSTOMER_ID
Provisioned:           $(date -u +%Y-%m-%dT%H:%M:%SZ)
Endpoint:              https://$SERVER_NAME
Version:               $version   features=$features

  admin   $ADMIN_KEY
  writer  $WRITER_KEY
  reader  $READER_KEY

Deliver over a one-time secret link, never plain email.
Then: shred -u $CONF_DIR/credentials.txt
EOF
chmod 0600 "$CONF_DIR/credentials.txt"

log "READY — $CUSTOMER_ID, BanditDB $version features=$features"
cat <<EOF

  Next:
    1. curl http://<this-ip>/health
    2. Point $SERVER_NAME at this instance, then:
         sudo certbot --nginx -d $SERVER_NAME --agree-tos -m you@example.com --redirect
       (add 443 to the Lightsail Networking tab first — ufw allows it, but the
        Lightsail firewall drops it before ufw ever sees the packet)
    3. Run the durability check in deploy/lightsail/README.md
    4. sudo cat $CONF_DIR/credentials.txt   → deliver → shred

EOF
