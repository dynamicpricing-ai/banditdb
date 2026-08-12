#!/usr/bin/env bash
# Gate between "built an appliance" and "snapshotted it".
#
#   sudo bash verify-appliance.sh
#
# The failure this exists to prevent: snapshotting a machine that carries state
# — an API key, a campaign, an SSH host key — and cloning it onto every customer
# instance thereafter. That is silent, and by the time it is noticed the same
# credentials are on N boxes belonging to N different companies.
set -uo pipefail

pass=0 fail=0
ok()   { printf '  \033[0;32mPASS\033[0m  %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[0;31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); }
info() { printf '        %s\n' "$1"; }

echo "BanditDB appliance verification"
echo

# ── Must be empty ────────────────────────────────────────────────────────────
echo "State that must NOT be in the image:"

if [[ -e /etc/banditdb/banditdb.env ]]; then
  bad "/etc/banditdb/banditdb.env exists — this image carries API keys"
  info "firstboot writes this per instance; it must be absent from the snapshot"
else
  ok "no banditdb.env (keys are generated per instance)"
fi

shopt -s nullglob dotglob
data=(/var/lib/banditdb/*)
if (( ${#data[@]} )); then
  bad "/var/lib/banditdb is not empty — this image carries a customer's data"
  info "contains: ${data[*]}"
else
  ok "/var/lib/banditdb is empty"
fi
shopt -u nullglob dotglob

hostkeys=(/etc/ssh/ssh_host_*)
if (( ${#hostkeys[@]} )); then
  bad "SSH host keys present — every instance would share one identity"
else
  ok "SSH host keys removed (regenerated on boot)"
fi

if [[ -s /etc/machine-id ]]; then
  bad "/etc/machine-id is populated — instances would share a machine identity"
else
  ok "/etc/machine-id is empty"
fi

if [[ -s /root/.bash_history ]]; then
  bad "/root/.bash_history is non-empty"
else
  ok "no root shell history"
fi

enabled=(/etc/nginx/sites-enabled/*)
if (( ${#enabled[@]} )); then
  bad "an nginx vhost is already enabled: ${enabled[*]}"
  info "the vhost is rendered per instance from the template"
else
  ok "no nginx vhost enabled yet"
fi

echo
echo "Things that must BE in the image:"

[[ -x /usr/local/bin/banditdb ]] \
  && ok "binary at /usr/local/bin/banditdb" \
  || bad "binary missing"

[[ -f /etc/systemd/system/banditdb.service ]] \
  && ok "systemd unit installed" \
  || bad "systemd unit missing"

[[ -x /usr/local/sbin/banditdb-firstboot ]] \
  && ok "firstboot script installed" \
  || bad "firstboot script missing"

[[ -f /etc/banditdb/nginx-vhost.conf.template ]] \
  && ok "nginx vhost template installed" \
  || bad "nginx vhost template missing"

[[ -f /etc/banditdb/appliance-manifest.json ]] \
  && ok "manifest present" \
  || bad "manifest missing"

for unit in banditdb banditdb-firstboot nginx regenerate-ssh-hostkeys; do
  if systemctl is-enabled "$unit" >/dev/null 2>&1; then
    ok "$unit is enabled"
  else
    bad "$unit is NOT enabled — it will not start on a launched instance"
  fi
done

# banditdb must be enabled but NOT running: with no env file it cannot have
# started, and if it did it would be running without authentication.
if systemctl is-active banditdb >/dev/null 2>&1; then
  bad "banditdb is RUNNING on the build box — it should be inert until firstboot"
else
  ok "banditdb is not running (correct: no config yet)"
fi

if ufw status 2>/dev/null | grep -q "Status: active"; then
  ok "ufw active"
  if ufw status | grep -q "8080"; then
    bad "port 8080 is open — the API would be reachable bypassing nginx and TLS"
  else
    ok "port 8080 not exposed"
  fi
else
  bad "ufw is not active"
fi

echo
if (( fail )); then
  printf '\033[0;31m%d checks failed, %d passed. DO NOT SNAPSHOT THIS IMAGE.\033[0m\n' "$fail" "$pass"
  exit 1
fi
printf '\033[0;32mAll %d checks passed. Safe to snapshot.\033[0m\n' "$pass"
echo
jq . /etc/banditdb/appliance-manifest.json 2>/dev/null || cat /etc/banditdb/appliance-manifest.json
