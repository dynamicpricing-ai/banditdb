#!/usr/bin/env bash
# Obtain and install a Let's Encrypt certificate.
#
#   sudo banditdb-enable-tls acme.api.banditdb.com ops@banditdb.com
#
# Separate from firstboot on purpose. Certbot proves control of the name by
# answering an HTTP challenge on it, so the DNS record must already resolve to
# this instance — which cannot be true at the moment the instance first boots.
# Folding this into firstboot would make provisioning fail on a race you cannot
# win, so it is a deliberate second step.
set -euo pipefail

DOMAIN="${1:-}"
EMAIL="${2:-}"

if [[ -z "$DOMAIN" || -z "$EMAIL" ]]; then
  echo "usage: banditdb-enable-tls <domain> <email>" >&2
  exit 1
fi
[[ $EUID -eq 0 ]] || { echo "must run as root" >&2; exit 1; }

# Fail early with a clear reason rather than inside certbot's output.
resolved="$(getent hosts "$DOMAIN" | awk '{print $1}' | head -1 || true)"
public="$(curl -sf --max-time 5 https://checkip.amazonaws.com || echo '')"
public="${public//[$'\r\n']/}"

if [[ -z "$resolved" ]]; then
  echo "ERROR: $DOMAIN does not resolve. Create the DNS record first." >&2
  exit 1
fi
if [[ -n "$public" && "$resolved" != "$public" ]]; then
  echo "ERROR: $DOMAIN resolves to $resolved but this instance is $public." >&2
  echo "       Wait for DNS to propagate, or fix the record. Certbot will fail" >&2
  echo "       the HTTP challenge otherwise, and repeated failures hit Let's" >&2
  echo "       Encrypt rate limits (5 per account per hostname per hour)." >&2
  exit 1
fi

echo "==> $DOMAIN -> $resolved (matches this instance)"

# Point the vhost at the real name before certbot edits it.
sed -i "s/server_name .*/server_name $DOMAIN;/" /etc/nginx/sites-available/banditdb
nginx -t && systemctl reload nginx

certbot --nginx \
  --non-interactive --agree-tos \
  --email "$EMAIL" \
  --domains "$DOMAIN" \
  --redirect

# certbot installs a renewal timer; confirm it rather than assuming.
systemctl list-timers 'certbot*' --no-pager | sed 's/^/  /'
certbot renew --dry-run

echo
echo "==> TLS active. Verifying end to end:"
curl -sf "https://$DOMAIN/health" | jq . || {
  echo "ERROR: HTTPS health check failed" >&2; exit 1; }
echo
echo "Certificate renews automatically. Renewal fails silently if the DNS"
echo "record is ever moved, so keep an uptime check on https://$DOMAIN/health."
