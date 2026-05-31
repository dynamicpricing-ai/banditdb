#!/usr/bin/env sh
# BanditDB installer
# Usage: curl -fsSL https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/scripts/install.sh | sh
set -e

REPO="dynamicpricing-ai/banditdb"
BIN_NAME="banditdb"
INSTALL_DIR="${BANDITDB_INSTALL_DIR:-/usr/local/bin}"

# ── Detect OS and architecture ────────────────────────────────────────────────
os="$(uname -s)"
arch="$(uname -m)"

case "$os" in
  Linux)
    case "$arch" in
      x86_64)  target="x86_64-unknown-linux-gnu" ;;
      aarch64|arm64) target="aarch64-unknown-linux-gnu" ;;
      *) echo "Unsupported Linux architecture: $arch" >&2; exit 1 ;;
    esac
    ;;
  Darwin)
    case "$arch" in
      x86_64)  target="x86_64-apple-darwin" ;;
      arm64)   target="aarch64-apple-darwin" ;;
      *) echo "Unsupported macOS architecture: $arch" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $os" >&2
    echo "For Windows, download the binary from: https://github.com/$REPO/releases/latest" >&2
    exit 1
    ;;
esac

# ── Resolve latest version ────────────────────────────────────────────────────
version="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
  | grep '"tag_name"' | sed 's/.*"tag_name": *"\(.*\)".*/\1/')"

if [ -z "$version" ]; then
  echo "Could not determine latest BanditDB version." >&2
  exit 1
fi

echo "Installing BanditDB $version for $target..."

# ── Download and extract ──────────────────────────────────────────────────────
url="https://github.com/$REPO/releases/download/$version/${BIN_NAME}-${version}-${target}.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsSL "$url" -o "$tmp/banditdb.tar.gz"
tar -xzf "$tmp/banditdb.tar.gz" -C "$tmp"

# ── Install binary ────────────────────────────────────────────────────────────
if [ -w "$INSTALL_DIR" ]; then
  mv "$tmp/$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"
  chmod +x "$INSTALL_DIR/$BIN_NAME"
else
  echo "Installing to $INSTALL_DIR (sudo required)..."
  sudo mv "$tmp/$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"
  sudo chmod +x "$INSTALL_DIR/$BIN_NAME"
fi

echo ""
echo "BanditDB $version installed to $INSTALL_DIR/$BIN_NAME"

# ── Setup prompt ──────────────────────────────────────────────────────────────
echo ""
printf "Configure BanditDB as a system service? [Y/n]: "
read -r setup_service </dev/tty
case "$setup_service" in
  [nN]*) echo "Skipping service setup. Run 'banditdb' manually to start."; exit 0 ;;
esac

# ── Collect settings ──────────────────────────────────────────────────────────
echo ""
echo "Press Enter to accept defaults shown in [brackets]."
echo ""

printf "Data directory (WAL, checkpoints, exports) [/var/lib/banditdb]: "
read -r cfg_data_dir </dev/tty
cfg_data_dir="${cfg_data_dir:-/var/lib/banditdb}"

printf "HTTP port [8080]: "
read -r cfg_port </dev/tty
cfg_port="${cfg_port:-8080}"

echo ""
echo "API keys — you will be prompted per role."
echo "  admin  = full access (predict, reward, manage campaigns)"
echo "  writer = predict + reward only"
echo "  reader = GET requests only"
echo "  Leave a key blank to skip that role."
echo ""

printf "Admin key (required for management access): "
read -r key_admin </dev/tty

printf "Writer key (optional, for app/service accounts): "
read -r key_writer </dev/tty

printf "Reader key (optional, for dashboards): "
read -r key_reader </dev/tty

cfg_api_keys=""
[ -n "$key_admin" ]  && cfg_api_keys="${key_admin}=admin"
[ -n "$key_writer" ] && cfg_api_keys="${cfg_api_keys:+$cfg_api_keys;}${key_writer}=writer"
[ -n "$key_reader" ] && cfg_api_keys="${cfg_api_keys:+$cfg_api_keys;}${key_reader}=reader"

if [ -z "$cfg_api_keys" ]; then
  echo "  Warning: no API keys set — BanditDB will run in open mode (no authentication)."
fi

printf "Auto-checkpoint after N rewarded events (leave empty to disable) [5000]: "
read -r cfg_checkpoint_interval </dev/tty
cfg_checkpoint_interval="${cfg_checkpoint_interval:-5000}"

printf "Auto-checkpoint when WAL exceeds N MB (leave empty to disable) [100]: "
read -r cfg_max_wal_mb </dev/tty
cfg_max_wal_mb="${cfg_max_wal_mb:-100}"

printf "Log format — 'json' for structured logs, 'text' for human-readable [json]: "
read -r cfg_log_format </dev/tty
cfg_log_format="${cfg_log_format:-json}"

# ── Create data and log directories ──────────────────────────────────────────
echo ""
echo "Creating directories..."
sudo mkdir -p "$cfg_data_dir"
sudo mkdir -p /var/log/banditdb

# ── Generate OS-native service config ────────────────────────────────────────
if [ "$os" = "Darwin" ]; then
  plist_path="/Library/LaunchDaemons/ai.dynamicpricing.banditdb.plist"
  echo "Generating launchd plist: $plist_path"

  sudo tee "$plist_path" > /dev/null << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>

  <key>Label</key>
  <string>ai.dynamicpricing.banditdb</string>

  <key>ProgramArguments</key>
  <array>
    <string>$INSTALL_DIR/$BIN_NAME</string>
  </array>

  <!-- Start at boot and restart automatically if the process exits -->
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>

  <key>EnvironmentVariables</key>
  <dict>

    <!-- Directory where WAL, checkpoints, and Parquet exports are stored -->
    <key>DATA_DIR</key>
    <string>$cfg_data_dir</string>

    <!-- HTTP port BanditDB listens on -->
    <key>PORT</key>
    <string>$cfg_port</string>

    <!-- Role-based API keys: key=role;key2=role2
         Roles: admin (full access), writer (predict/reward), reader (GET only)
         Remove or leave empty to run in open mode (no authentication) -->
    <key>BANDITDB_API_KEYS</key>
    <string>$cfg_api_keys</string>

    <!-- How long (seconds) to remember a predict token for reward matching.
         Default: 86400 (24h). Lower if memory is constrained. -->
    <key>BANDITDB_REWARD_TTL_SECS</key>
    <string>86400</string>

    <!-- Auto-checkpoint after this many rewarded events.
         Lower = more durable, more I/O. Higher = fewer checkpoints, faster writes. -->
    <key>BANDITDB_CHECKPOINT_INTERVAL</key>
    <string>$cfg_checkpoint_interval</string>

    <!-- Auto-checkpoint when WAL file exceeds this size in MB.
         Prevents unbounded WAL growth on high-traffic instances. -->
    <key>BANDITDB_MAX_WAL_SIZE_MB</key>
    <string>$cfg_max_wal_mb</string>

    <!-- Per-API-key request rate limit (requests/second).
         Default: 1000. Lower to protect against runaway clients. -->
    <key>BANDITDB_RATE_LIMIT_PER_SEC</key>
    <string>1000</string>

    <!-- Log format: "json" for structured logging (recommended for production),
         "text" for human-readable output -->
    <key>LOG_FORMAT</key>
    <string>$cfg_log_format</string>

    <!-- Uncomment to enable audit logging of all write-path events (JSONL format) -->
    <!-- <key>BANDITDB_AUDIT_LOG</key> -->
    <!-- <string>/var/log/banditdb/audit.jsonl</string> -->

  </dict>

  <key>StandardOutPath</key>
  <string>/var/log/banditdb/stdout.log</string>
  <key>StandardErrorPath</key>
  <string>/var/log/banditdb/stderr.log</string>

</dict>
</plist>
EOF

  sudo launchctl load "$plist_path"
  echo "Service registered and started."

elif [ "$os" = "Linux" ]; then
  service_path="/etc/systemd/system/banditdb.service"
  echo "Generating systemd unit: $service_path"

  sudo tee "$service_path" > /dev/null << EOF
[Unit]
Description=BanditDB — Contextual Bandit Database
After=network.target

[Service]
ExecStart=$INSTALL_DIR/$BIN_NAME
Restart=always
RestartSec=5

# ── Data and logging ──────────────────────────────────────────────────────────
# Directory where WAL, checkpoints, and Parquet exports are stored
Environment=DATA_DIR=$cfg_data_dir

# Log format: "json" for structured logging, "text" for human-readable
Environment=LOG_FORMAT=$cfg_log_format

# ── Network ───────────────────────────────────────────────────────────────────
# HTTP port BanditDB listens on
Environment=PORT=$cfg_port

# ── Authentication ────────────────────────────────────────────────────────────
# Role-based API keys: key=role;key2=role2
# Roles: admin (full access), writer (predict/reward), reader (GET only)
# Remove to run in open mode (no authentication)
Environment=BANDITDB_API_KEYS=$cfg_api_keys

# ── Durability ────────────────────────────────────────────────────────────────
# How long (seconds) to remember a predict token for reward matching (default: 86400 = 24h)
Environment=BANDITDB_REWARD_TTL_SECS=86400

# Auto-checkpoint after this many rewarded events
# Lower = more durable, more I/O. Higher = fewer checkpoints, faster writes.
Environment=BANDITDB_CHECKPOINT_INTERVAL=$cfg_checkpoint_interval

# Auto-checkpoint when WAL file exceeds this size in MB
Environment=BANDITDB_MAX_WAL_SIZE_MB=$cfg_max_wal_mb

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Per-API-key request rate limit (requests/second). Default: 1000.
Environment=BANDITDB_RATE_LIMIT_PER_SEC=1000

# ── Audit logging (optional) ──────────────────────────────────────────────────
# Uncomment to write all write-path events to a JSONL audit log
# Environment=BANDITDB_AUDIT_LOG=/var/log/banditdb/audit.jsonl

StandardOutput=append:/var/log/banditdb/stdout.log
StandardError=append:/var/log/banditdb/stderr.log

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable banditdb
  sudo systemctl start banditdb
  echo "Service registered and started."
fi

# ── Health check ──────────────────────────────────────────────────────────────
echo ""
echo "Waiting for BanditDB to start..."
sleep 2
if curl -fsSL "http://localhost:$cfg_port/health" > /dev/null 2>&1; then
  echo "BanditDB is running."
else
  echo "Health check failed — check logs at /var/log/banditdb/stderr.log"
fi

echo ""
echo "Health check:  curl http://localhost:$cfg_port/health"
if [ "$os" = "Darwin" ]; then
  echo "View logs:     tail -f /var/log/banditdb/stderr.log"
  echo "Stop service:  sudo launchctl unload $plist_path"
  echo "Config file:   $plist_path"
elif [ "$os" = "Linux" ]; then
  echo "View logs:     journalctl -u banditdb -f"
  echo "Stop service:  sudo systemctl stop banditdb"
  echo "Config file:   $service_path"
fi
