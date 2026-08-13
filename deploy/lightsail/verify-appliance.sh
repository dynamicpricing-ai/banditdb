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

# nullglob for the whole script, never toggled off. Every check below tests
# "did this glob match anything", and without nullglob an unmatched pattern
# survives as a literal one-element array — which reads as "files present" and
# fails the check on a correctly-empty directory. dotglob so hidden files in the
# data directory are not missed.
shopt -s nullglob dotglob

pass=0 fail=0
ok()   { printf '  \033[0;32mPASS\033[0m  %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[0;31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); }
info() { printf '        %s\n' "$1"; }

echo "BanditDB appliance verification"
echo

# Runs BEFORE generalize.sh, so machine identity (ssh host keys, machine-id,
# shell history) is still present and is not checked here — generalize.sh
# removes it and confirms its own work. What matters at this point is that no
# CUSTOMER state exists, because that is what generalisation does not remove.
# ── Must be empty ────────────────────────────────────────────────────────────
echo "State that must NOT be in the image:"

if [[ -e /etc/banditdb/banditdb.env ]]; then
  bad "/etc/banditdb/banditdb.env exists — this image carries API keys"
  info "firstboot writes this per instance; it must be absent from the snapshot"
else
  ok "no banditdb.env (keys are generated per instance)"
fi

data=(/var/lib/banditdb/*)
if (( ${#data[@]} )); then
  bad "/var/lib/banditdb is not empty — this image carries a customer's data"
  info "contains: ${data[*]}"
else
  ok "/var/lib/banditdb is empty"
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

for unit in banditdb banditdb-firstboot nginx ssh; do
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

# The image must carry a login key. Without one, every instance launched from
# this snapshot is unreachable — and that is only discoverable after the fact.
authkeys=/home/ubuntu/.ssh/authorized_keys
if [[ -s "$authkeys" ]]; then
  ok "authorized_keys present ($(grep -c . "$authkeys") key(s)) — you can log in to launched instances"
else
  bad "no /home/ubuntu/.ssh/authorized_keys — instances from this snapshot would be UNREACHABLE"
  info "create the build instance with an SSH key you hold, then rebuild"
fi

if [[ -f /etc/systemd/system/ssh.service.d/10-hostkeys.conf ]]; then
  ok "sshd regenerates host keys via ExecStartPre (cannot race)"
else
  bad "missing ssh.service.d/10-hostkeys.conf — launched instances may have no host keys"
fi

if systemctl is-enabled ssh.socket >/dev/null 2>&1; then
  bad "ssh.socket is still enabled — socket activation reintroduces the startup race"
else
  ok "ssh.socket disabled (sshd runs as a plain service)"
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
