#!/usr/bin/env bash
# P0.2 crash-injection harness.
#
# Kills the server with SIGKILL at randomised points while it is checkpointing,
# then restarts it and asserts no committed state was lost. This exercises the
# real syscall ordering (fsync / rename / WAL rotation) that unit tests can only
# approximate.
#
# HARD invariants — these are what P0.2 guarantees, and a failure is a bug:
#   1. The server always restarts. A refusal means the checkpoint was left
#      unreadable with no usable fallback.
#   2. Campaigns survive every kill.
#
# MEASURED, not enforced until P0.3 + P0.4 land:
#   3. Reward loss. `predict`/`reward` currently ack the client after queueing to a
#      bounded channel, before the WAL writer has written anything (see the
#      consistency-model comment in src/engine.rs). SIGKILL discards that backlog,
#      so committed-but-unwritten rewards are lost. That is the ack-before-durability
#      gap P0.4 closes and P0.3 then makes durable; this harness reports the observed
#      loss as an RPO measurement. Pass --strict to enforce zero loss once both land.
#
# Usage:
#   ./scripts/crash_injection.sh [iterations] [--strict]   # default 50; plan target 500+
#
# Env:
#   BANDITDB_BIN   path to the binary (default ./target/release/banditdb)
#   PORT           default 8130

set -uo pipefail

ITERATIONS="${1:-50}"
STRICT=0
for a in "$@"; do [[ "$a" == "--strict" ]] && STRICT=1; done
BIN="${BANDITDB_BIN:-./target/release/banditdb}"
PORT="${PORT:-8130}"
KEY="crash-test-key"
WORK="$(mktemp -d)"
URL="http://127.0.0.1:${PORT}"

[[ -x "$BIN" ]] || { echo "binary not found: $BIN (cargo build --release --features neural)"; exit 1; }

cleanup() { pkill -9 -f "$BIN" 2>/dev/null; sleep 0.3; rm -rf "$WORK" 2>/dev/null; }
trap cleanup EXIT

api() { curl -sS --max-time 5 -H "X-Api-Key: $KEY" -H 'Content-Type: application/json' "$@"; }

start_server() {
  DATA_DIR="$WORK" PORT="$PORT" BANDITDB_API_KEY="$KEY" \
    BANDITDB_RATE_LIMIT_PER_SEC=100000 "$BIN" >>"$WORK/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    if api "$URL/health" >/dev/null 2>&1; then return 0; fi
    # Detect a refusal to start (corrupt checkpoint with no fallback).
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then return 1; fi
    sleep 0.1
  done
  return 1
}

drive_traffic() {
  local n="$1"
  for i in $(seq 1 "$n"); do
    local resp iid arm
    resp=$(api -X POST "$URL/predict" -d "{\"campaign_id\":\"crash\",\"context\":[0.$((i%9)),0.$((i%7))]}")
    iid=$(printf '%s' "$resp" | sed -n 's/.*"interaction_id":"\([^"]*\)".*/\1/p')
    arm=$(printf '%s' "$resp" | sed -n 's/.*"arm_id":"\([^"]*\)".*/\1/p')
    [[ -n "$iid" ]] || continue
    local r=0.0; [[ "$arm" == "A" ]] && r=1.0
    api -X POST "$URL/reward" -d "{\"interaction_id\":\"$iid\",\"reward\":$r}" >/dev/null
  done
}

total_rewards() {
  api "$URL/campaign/crash/report" | sed -n 's/.*"total_rewards":\([0-9]*\).*/\1/p'
}

echo "crash-injection: $ITERATIONS iterations, data dir $WORK"

start_server || { echo "FAIL: server did not start on a clean data dir"; exit 1; }
api -X POST "$URL/campaign" -d '{"campaign_id":"crash","arms":["A","B"],"feature_dim":2,"alpha":1.0}' >/dev/null
drive_traffic 30
api -X POST "$URL/checkpoint" -d '{}' >/dev/null
baseline=$(total_rewards)
echo "baseline rewards after first checkpoint: $baseline"

failures=0; total_lost=0; worst_lost=0
for iter in $(seq 1 "$ITERATIONS"); do
  drive_traffic 15
  committed=$(total_rewards)

  # Fire a checkpoint and SIGKILL mid-flight at a randomised offset.
  api -X POST "$URL/checkpoint" -d '{}' >/dev/null 2>&1 &
  delay=$(awk -v s="$RANDOM" 'BEGIN{srand(s); printf "%.4f", rand()*0.06}')
  sleep "$delay"
  kill -9 "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null

  if ! start_server; then
    echo "FAIL iter $iter: server refused to start after kill at ${delay}s"
    echo "  -> checkpoint left unreadable with no usable fallback"
    tail -5 "$WORK/server.log"
    failures=$((failures+1))
    rm -f "$WORK/checkpoint.json"   # try to continue the run from .prev
    start_server || break
    continue
  fi

  after=$(total_rewards)
  if [[ -z "$after" ]]; then
    echo "FAIL iter $iter: campaign missing after restart (kill at ${delay}s)"
    failures=$((failures+1)); continue
  fi

  # Reward loss: measured always, enforced only under --strict.
  if (( after < committed )); then
    lost=$(( committed - after ))
    total_lost=$(( total_lost + lost ))
    (( lost > worst_lost )) && worst_lost=$lost
    if (( STRICT == 1 )); then
      echo "FAIL iter $iter: lost $lost rewards (committed=$committed after=$after, kill at ${delay}s)"
      failures=$((failures+1)); continue
    fi
  fi
  baseline="$after"
  printf '.'
  (( iter % 50 == 0 )) && printf ' %d\n' "$iter"
done

echo
echo "--- reward loss (ack-before-durability, closed by P0.4 + P0.3)"
echo "    total lost across $ITERATIONS kills: $total_lost"
echo "    worst single kill:                   $worst_lost"
if (( failures == 0 )); then
  echo "PASS: $ITERATIONS kills — server always restarted, campaign always survived"
  (( STRICT == 0 )) && echo "      (reward loss measured, not enforced; rerun with --strict after P0.3/P0.4)"
  exit 0
fi
echo "FAIL: $failures/$ITERATIONS iterations violated a hard invariant"
exit 1
