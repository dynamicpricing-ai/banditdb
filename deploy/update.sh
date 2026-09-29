#!/usr/bin/env bash
#
# Install a released engine binary and restart the service.
#
#   sudo ./deploy/update.sh v2.0.0
#
# Pulls the published Linux tarball from GitHub Releases — the repository is
# public, so this needs no credential on the VM. Idempotent: installing the
# version already running is a restart, not a failure.
#
# On a failed health check the previous binary is put back and the service
# restarted, so a bad release costs one restart rather than an outage.

set -euo pipefail

VERSION="${1:?usage: update.sh <version tag, e.g. v2.0.0>}"
REPO="${BANDITDB_REPO:-dynamicpricing-ai/banditdb}"
TARGET=x86_64-unknown-linux-gnu
BIN=/usr/local/bin/banditdb
PREVIOUS=/usr/local/bin/banditdb.previous
HEALTH_URL=http://127.0.0.1:8080/health

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

say() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }

say "Fetching $VERSION"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
URL="https://github.com/${REPO}/releases/download/${VERSION}/banditdb-${VERSION}-${TARGET}.tar.gz"
curl -fsSL "$URL" -o "$TMP/banditdb.tar.gz" || {
  echo "Could not download $URL"
  echo "Check the tag exists and its release has finished publishing assets."
  exit 1
}
tar -xzf "$TMP/banditdb.tar.gz" -C "$TMP"
chmod +x "$TMP/banditdb"

# Run the new binary before trusting it with the service. A tarball for the
# wrong architecture, or one built without the neural feature, is better found
# here than by a service that then refuses to come back up.
NEW_VERSION="$("$TMP/banditdb" --version)"
echo "  downloaded: $NEW_VERSION"

say "Installing"
[ -f "$BIN" ] && cp -p "$BIN" "$PREVIOUS" && echo "  kept previous at $PREVIOUS"
# install(1) replaces the file atomically, so a running process keeps its own
# open inode and is untouched until the restart below.
install -m 0755 "$TMP/banditdb" "$BIN"

say "Restarting"
systemctl restart banditdb

say "Health"
for i in $(seq 1 30); do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "  $(curl -fsS "$HEALTH_URL")"
    echo
    echo "$NEW_VERSION is live."
    exit 0
  fi
  sleep 2
done

say "FAILED — rolling back"
journalctl -u banditdb -n 40 --no-pager || true
if [ -f "$PREVIOUS" ]; then
  install -m 0755 "$PREVIOUS" "$BIN"
  systemctl restart banditdb
  echo "Restored the previous binary."
else
  echo "No previous binary to restore; the service is down."
fi
exit 1
