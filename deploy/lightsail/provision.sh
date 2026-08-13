#!/usr/bin/env bash
# Provision one customer instance from the golden snapshot. Runs on your
# machine, not on the instance.
#
#   ./provision.sh acme acme.api.banditdb.com
#   ./provision.sh acme acme.api.banditdb.com --bundle medium_3_0 --tier growth
#
# Needs the AWS CLI configured for the account holding the snapshot.
set -euo pipefail

CUSTOMER="${1:-}"
DOMAIN="${2:-}"
shift 2 || true

REGION="${AWS_REGION:-eu-central-1}"
AZ="${LIGHTSAIL_AZ:-${REGION}a}"
SNAPSHOT="${BANDITDB_SNAPSHOT:-}"
BUNDLE="small_3_0"     # 2 GB RAM / 2 vCPU — the Starter plan
TIER="starter"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)   BUNDLE="$2";   shift 2 ;;
    --tier)     TIER="$2";     shift 2 ;;
    --snapshot) SNAPSHOT="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1;     shift   ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$CUSTOMER" || -z "$DOMAIN" ]]; then
  cat >&2 <<'EOF'
usage: provision.sh <customer-id> <domain> [--bundle small_3_0] [--tier starter]

  customer-id   lowercase, [a-z0-9-]; becomes the instance name
  domain        the hostname this instance will serve

  bundles: nano_3_0 (512MB) small_3_0 (2GB) medium_3_0 (4GB) large_3_0 (8GB)
           the bundle disk must be >= the snapshot's, so a 40GB appliance
           snapshot cannot launch onto nano_3_0 (20GB)

  --dry-run  print the AWS calls and the generated user-data, change nothing
EOF
  exit 1
fi

[[ "$CUSTOMER" =~ ^[a-z0-9-]+$ ]] || { echo "customer-id must match [a-z0-9-]+" >&2; exit 1; }
command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 1; }

INSTANCE="banditdb-$CUSTOMER"

# Resolve the newest appliance snapshot unless one was named explicitly.
if [[ -z "$SNAPSHOT" ]]; then
  SNAPSHOT="$(aws lightsail get-instance-snapshots --region "$REGION" \
    --query "instanceSnapshots[?starts_with(name,'banditdb-appliance-')] | sort_by(@,&createdAt)[-1].name" \
    --output text)"
  [[ "$SNAPSHOT" != "None" && -n "$SNAPSHOT" ]] || {
    echo "no snapshot named banditdb-appliance-* found in $REGION" >&2; exit 1; }
fi

# Per-plan tuning. Memory is the binding resource: pending interactions are
# ~1 KB each, so the cap has to fit the bundle with room for matrices and, on
# neural campaigns, the replay buffer.
case "$TIER" in
  starter) MAX_PENDING=100000  ; RATE_LIMIT=1000 ;;
  growth)  MAX_PENDING=250000  ; RATE_LIMIT=2000 ;;
  *) echo "unknown tier: $TIER" >&2; exit 1 ;;
esac

cat <<EOF
Provisioning
  customer   $CUSTOMER
  domain     $DOMAIN
  instance   $INSTANCE
  snapshot   $SNAPSHOT
  bundle     $BUNDLE  (tier: $TIER)
  region/az  $REGION / $AZ
EOF
if (( DRY_RUN )); then
  echo
  echo "DRY RUN — no AWS calls will be made."
else
  read -rp "Proceed? [y/N] " reply
  [[ "$reply" == "y" ]] || exit 1
fi

# Every mutating call goes through this, so --dry-run cannot miss one.
aws_do() {
  if (( DRY_RUN )); then
    printf '  would run: aws %s\n' "$*"
    return 0
  fi
  aws "$@"
}

USER_DATA="$(cat <<EOF
#!/bin/bash
mkdir -p /etc/banditdb
cat > /etc/banditdb/customer.conf <<'CONF'
CUSTOMER_ID="$CUSTOMER"
SERVER_NAME="$DOMAIN"
PLAN_TIER="$TIER"
MAX_PENDING=$MAX_PENDING
RATE_LIMIT=$RATE_LIMIT
REWARD_TTL_SECS=86400
CORS_ORIGINS=""
CONF
chmod 0600 /etc/banditdb/customer.conf
EOF
)"

if (( DRY_RUN )); then
  echo
  echo "==> user-data that firstboot would consume:"
  printf '%s\n' "$USER_DATA" | sed 's/^/    /'
  echo
fi

echo "==> Creating instance"
aws_do lightsail create-instances-from-snapshot --region "$REGION" \
  --instance-snapshot-name "$SNAPSHOT" \
  --instance-names "$INSTANCE" \
  --availability-zone "$AZ" \
  --bundle-id "$BUNDLE" \
  --user-data "$USER_DATA" \
  --tags "key=customer,value=$CUSTOMER" "key=tier,value=$TIER" \
  --output text >/dev/null

echo "==> Waiting for running state"
if (( DRY_RUN )); then
  state=running
else
for _ in $(seq 1 60); do
  state="$(aws lightsail get-instance --region "$REGION" --instance-name "$INSTANCE" \
    --query 'instance.state.name' --output text 2>/dev/null || echo pending)"
  [[ "$state" == "running" ]] && break
  sleep 5
done
fi
[[ "$state" == "running" ]] || { echo "instance did not reach running state" >&2; exit 1; }

echo "==> Allocating static IP"
# Without a static IP the address changes on every stop/start, silently breaking
# the customer's DNS record.
aws_do lightsail allocate-static-ip --region "$REGION" --static-ip-name "$INSTANCE-ip" \
  --output text >/dev/null 2>&1 || true
aws_do lightsail attach-static-ip --region "$REGION" --static-ip-name "$INSTANCE-ip" \
  --instance-name "$INSTANCE" --output text >/dev/null

if (( DRY_RUN )); then
  IP="<static-ip>"
else
  IP="$(aws lightsail get-static-ip --region "$REGION" --static-ip-name "$INSTANCE-ip" \
    --query 'staticIp.ipAddress' --output text)"
fi

echo "==> Enabling automatic snapshots"
aws_do lightsail enable-add-on --region "$REGION" --resource-name "$INSTANCE" \
  --add-on-request 'addOnType=AutoSnapshot,autoSnapshotAddOnRequest={snapshotTimeOfDay=03:00}' \
  --output text >/dev/null

cat <<EOF

Instance up.

  $INSTANCE   $IP

Remaining steps — none of these are automated on purpose, because each one
wants a human to confirm the result:

  1. DNS:  $DOMAIN  A  $IP
  2. Wait for it to resolve:  dig +short $DOMAIN
  3. ssh ubuntu@$IP 'sudo banditdb-enable-tls $DOMAIN ops@banditdb.com'
  4. Durability check (README §"Verify before handing over") — DO NOT SKIP
  5. ssh ubuntu@$IP 'sudo cat /etc/banditdb/credentials.txt'
  6. Send credentials via one-time secret link
  7. ssh ubuntu@$IP 'sudo shred -u /etc/banditdb/credentials.txt'
  8. Record customer, instance, domain, bundle, version in the fleet sheet

EOF
