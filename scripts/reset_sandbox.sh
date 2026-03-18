#!/usr/bin/env sh
# BanditDB sandbox nightly reset
# Add to crontab: 0 3 * * * sh /opt/banditdb/scripts/reset_sandbox.sh
# Runs at 03:00 UTC — restarts BanditDB (clears WAL) and recreates demo campaigns.

set -e

BANDITDB_API_KEY="${BANDITDB_API_KEY:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$BANDITDB_API_KEY" ]; then
  echo "Error: BANDITDB_API_KEY is not set." >&2
  exit 1
fi

echo "[reset] Restarting BanditDB..."
systemctl restart banditdb
sleep 3

echo "[reset] Recreating demo campaigns..."
BANDITDB_API_KEY="$BANDITDB_API_KEY" sh "$SCRIPT_DIR/setup_sandbox.sh"

echo "[reset] Done."
