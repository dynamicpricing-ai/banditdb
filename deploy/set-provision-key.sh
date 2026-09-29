#!/usr/bin/env bash
#
# Generate the control plane's machine-to-machine credential, in place.
#
#   sudo ./deploy/set-provision-key.sh          # generate a new one
#   sudo ./deploy/set-provision-key.sh --show   # print the current one
#
# The key is generated on this machine and written straight into the environment
# file. It is never printed unless asked for, so it does not land in a shell
# history, a terminal scrollback, or the argument list of an ssh command — all
# of which happen if you generate it elsewhere and paste it in.
#
# The console needs the same value. Copy it there without printing it:
#
#   gcloud compute ssh <vm> --zone=<zone> \
#     --command='sudo /usr/local/sbin/banditdb-show-key' \
#     | gcloud secrets versions add BANDITDB_PROVISION_KEY --data-file=-
#
# Rotating is this script again followed by `systemctl restart banditdb` and a
# new secret version — in that order, since the engine reads the key at startup
# and the console retries a rejected provisioning call from its outbox.

set -euo pipefail

ENV_FILE=/etc/banditdb/banditdb.env

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
[ -f "$ENV_FILE" ] || { echo "$ENV_FILE not found — run setup-vm.sh first"; exit 1; }

current() { sed -n 's/^BANDITDB_PROVISION_KEY=//p' "$ENV_FILE"; }

if [ "${1:-}" = "--show" ]; then
  KEY="$(current)"
  [ -n "$KEY" ] || { echo "no key set" >&2; exit 1; }
  # No trailing newline: this is piped into `gcloud secrets versions add`, which
  # stores the bytes exactly as given, and a stray newline would make the two
  # sides compare unequal for a reason nothing reports.
  printf '%s' "$KEY"
  exit 0
fi

if [ -n "$(current)" ]; then
  echo "A key is already set. Rotating it breaks provisioning until the console"
  echo "has the new value too. To go ahead:"
  echo
  echo "    sudo sed -i 's/^BANDITDB_PROVISION_KEY=.*/BANDITDB_PROVISION_KEY=/' $ENV_FILE"
  echo "    sudo $0"
  exit 1
fi

# 48 hex characters: 192 bits, no shell-significant characters, nothing that
# needs quoting in an env file or a URL.
KEY="$(openssl rand -hex 24)"
sed -i "s|^BANDITDB_PROVISION_KEY=.*|BANDITDB_PROVISION_KEY=${KEY}|" "$ENV_FILE"

# A reader for the copy-out step above, so it does not need this whole script.
cat > /usr/local/sbin/banditdb-show-key <<'EOF'
#!/bin/sh
# No trailing newline: the output is piped straight into Secret Manager, and a
# stray one would make the two sides compare unequal with nothing reporting why.
sed -n 's/^BANDITDB_PROVISION_KEY=//p' /etc/banditdb/banditdb.env | tr -d '\n'
EOF
chmod 0700 /usr/local/sbin/banditdb-show-key

echo "Key generated and written to $ENV_FILE (not shown)."
echo "Copy it to the console's Secret Manager with:"
echo
echo "  gcloud compute ssh \$VM --zone=\$ZONE \\"
echo "    --command='sudo /usr/local/sbin/banditdb-show-key' \\"
echo "    | gcloud secrets versions add BANDITDB_PROVISION_KEY --data-file=-"
