#!/usr/bin/env bash
# Provision one customer instance on Lightsail. Runs on your machine.
#
#   ./provision.sh acme acme.api.banditdb.com
#   ./provision.sh acme acme.api.banditdb.com --bundle medium_3_0 --tier growth
#   ./provision.sh acme acme.api.banditdb.com --dry-run
#
# Creates a stock Ubuntu instance and passes cloud-init.sh as user-data, so the
# machine provisions itself on first boot. There is no golden image and no
# snapshot: every instance is built from the script in this directory, which
# means the thing under version control is the thing that runs.
#
# Requires the AWS CLI configured for the account. Everything here can also be
# done by hand in the console — create an instance, paste cloud-init.sh into
# "Add launch script" — this just makes it repeatable.
set -euo pipefail

CUSTOMER="${1:-}"
DOMAIN="${2:-}"
shift 2 || true

REGION="${AWS_REGION:-eu-central-1}"
AZ="${LIGHTSAIL_AZ:-${REGION}a}"
BLUEPRINT="${LIGHTSAIL_BLUEPRINT:-ubuntu_24_04}"
BUNDLE="small_3_0"     # 2 GB RAM / 60 GB — the Starter plan
TIER="starter"
KEYPAIR="${LIGHTSAIL_KEYPAIR:-}"
DRY_RUN=0

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUD_INIT="$SRC_DIR/cloud-init.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)    BUNDLE="$2";    shift 2 ;;
    --tier)      TIER="$2";      shift 2 ;;
    --blueprint) BLUEPRINT="$2"; shift 2 ;;
    --keypair)   KEYPAIR="$2";   shift 2 ;;
    --dry-run)   DRY_RUN=1;      shift   ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$CUSTOMER" || -z "$DOMAIN" ]]; then
  cat >&2 <<'EOF'
usage: provision.sh <customer-id> <domain> [options]

  customer-id   lowercase [a-z0-9-]; names the instance
  domain        hostname this instance will serve

  --bundle      nano_3_0 (512MB) small_3_0 (2GB, default) medium_3_0 (4GB) large_3_0 (8GB)
                nano is too small for production: pending interactions alone are
                budgeted at ~100 MB before matrices and the neural replay buffer
  --tier        starter | growth   (sets pending-cache and rate limits)
  --blueprint   Lightsail OS blueprint (default ubuntu_24_04)
  --keypair     Lightsail key pair name; omit to use the region default
  --dry-run     print what would happen, change nothing
EOF
  exit 1
fi

[[ "$CUSTOMER" =~ ^[a-z0-9-]+$ ]] || { echo "customer-id must match [a-z0-9-]+" >&2; exit 1; }
command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 1; }
[[ -f "$CLOUD_INIT" ]] || { echo "cloud-init.sh not found beside this script" >&2; exit 1; }

INSTANCE="banditdb-$CUSTOMER"

# Per-plan tuning. Memory is the binding resource: pending interactions are ~1 KB
# each, so the cap has to fit the bundle alongside matrices and, for neural
# campaigns, the replay buffer.
case "$TIER" in
  starter) MAX_PENDING=100000 ; RATE_LIMIT=1000 ;;
  growth)  MAX_PENDING=250000 ; RATE_LIMIT=2000 ;;
  *) echo "unknown tier: $TIER" >&2; exit 1 ;;
esac

# cloud-init.sh takes its configuration from the environment, so prepend the
# per-customer values rather than editing the file.
USER_DATA="$(printf '%s\n' \
  "#!/bin/bash" \
  "export CUSTOMER_ID='$CUSTOMER'" \
  "export SERVER_NAME='$DOMAIN'" \
  "export MAX_PENDING=$MAX_PENDING" \
  "export RATE_LIMIT=$RATE_LIMIT" \
  "$(tail -n +2 "$CLOUD_INIT")")"

# Lightsail caps user-data at 16 KB. cloud-init.sh is ~12 KB, so there is room,
# but not unlimited room — fail here with a clear reason rather than letting AWS
# reject the call with something opaque.
UD_BYTES=$(printf '%s' "$USER_DATA" | wc -c | tr -d ' ')
if (( UD_BYTES > 16384 )); then
  echo "ERROR: user-data is $UD_BYTES bytes, over the 16384 limit." >&2
  echo "       Trim cloud-init.sh, or host it and fetch it from a short bootstrap." >&2
  exit 1
fi

cat <<EOF
Provisioning
  customer   $CUSTOMER
  domain     $DOMAIN
  instance   $INSTANCE
  blueprint  $BLUEPRINT
  bundle     $BUNDLE  (tier: $TIER)
  region/az  $REGION / $AZ
  keypair    ${KEYPAIR:-<region default>}
  user-data  $(wc -l <<<"$USER_DATA") lines, $UD_BYTES bytes (limit 16384)
EOF

if (( DRY_RUN )); then
  echo
  echo "DRY RUN — no AWS calls will be made."
  echo "First 8 lines of user-data:"
  head -8 <<<"$USER_DATA" | sed 's/^/    /'
else
  read -rp "Proceed? [y/N] " reply
  [[ "$reply" == "y" ]] || exit 1
fi

aws_do() {
  if (( DRY_RUN )); then printf '  would run: aws %s\n' "$*"; return 0; fi
  aws "$@"
}

echo "==> Creating instance"
create_args=(
  lightsail create-instances --region "$REGION"
  --instance-names "$INSTANCE"
  --availability-zone "$AZ"
  --blueprint-id "$BLUEPRINT"
  --bundle-id "$BUNDLE"
  --user-data "$USER_DATA"
  --tags "key=customer,value=$CUSTOMER" "key=tier,value=$TIER"
)
[[ -n "$KEYPAIR" ]] && create_args+=(--key-pair-name "$KEYPAIR")
aws_do "${create_args[@]}" --output text >/dev/null

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
# Without one the address changes on stop/start, silently breaking the DNS record.
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

echo "==> Opening HTTPS"
# ufw inside the instance allows 443, but the Lightsail firewall sits in front of
# it and drops the packet first. Without this, certbot fails in a way that looks
# like a certbot problem.
aws_do lightsail open-instance-public-ports --region "$REGION" --instance-name "$INSTANCE" \
  --port-info 'fromPort=443,toPort=443,protocol=TCP' --output text >/dev/null

echo "==> Enabling daily snapshots"
aws_do lightsail enable-add-on --region "$REGION" --resource-name "$INSTANCE" \
  --add-on-request 'addOnType=AutoSnapshot,autoSnapshotAddOnRequest={snapshotTimeOfDay=03:00}' \
  --output text >/dev/null

cat <<EOF

Instance created: $INSTANCE   $IP

cloud-init is provisioning it now — about 3 minutes. Then:

  1. curl http://$IP/health                       # {"status":"ok","version":"2.0.0",...}
  2. DNS:  $DOMAIN  A  $IP
     dig +short $DOMAIN                           # wait for it to resolve
  3. ssh ubuntu@$IP 'sudo certbot --nginx -d $DOMAIN --agree-tos -m ops@banditdb.com --redirect'
  4. Durability check — see README, DO NOT SKIP
  5. ssh ubuntu@$IP 'sudo cat /etc/banditdb/credentials.txt'
  6. Deliver credentials via a one-time secret link
  7. ssh ubuntu@$IP 'sudo shred -u /etc/banditdb/credentials.txt'
  8. Record customer, instance, domain, bundle, version in the fleet sheet

If /health does not answer after 5 minutes:
  ssh ubuntu@$IP 'sudo tail -50 /var/log/cloud-init-output.log'

EOF
