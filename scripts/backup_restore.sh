#!/usr/bin/env bash
# BanditDB backup / restore / verify.
#
# What actually needs preserving, and why:
#
#   checkpoint.json   the full model state as of the last checkpoint
#   checkpoint.prev   the retained previous generation — the fallback when the
#                     current one is unreadable
#   bandit_wal.jsonl  events since that checkpoint; without it you lose everything
#                     written after the last checkpoint
#   neural/           MLP weights; a neural campaign restored without these serves
#                     its random initialisation until the next retrain
#
#   exports/          deliberately NOT backed up. Parquet shards are for offline
#                     analysis, recovery never reads them, and they are the bulk of
#                     the volume.
#
# A backup is only a backup once a restore has been proven, so `verify` exists and
# the drill is part of the runbook rather than an exercise for the reader.
#
# Usage:
#   ./scripts/backup_restore.sh backup  <data_dir> <backup_dir>
#   ./scripts/backup_restore.sh restore <backup.tar.gz> <data_dir>
#   ./scripts/backup_restore.sh verify  <backup.tar.gz>   # restore to a temp dir and boot it
#   ./scripts/backup_restore.sh drill   <data_dir>        # backup -> verify, end to end

set -euo pipefail

BIN="${BANDITDB_BIN:-./target/release/banditdb}"

die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------

do_backup() {
    local data_dir="$1" backup_dir="$2"
    [[ -d "$data_dir" ]] || die "data dir not found: $data_dir"
    mkdir -p "$backup_dir"

    local stamp archive
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    archive="$backup_dir/banditdb-$stamp.tar.gz"

    # Taking a copy while the server runs can catch a checkpoint mid-write. That is
    # survivable — checkpoint.prev is included, and recovery falls back to it — but
    # a backup taken just after a checkpoint is cleaner.
    local members=()
    for f in checkpoint.json checkpoint.prev bandit_wal.jsonl; do
        [[ -e "$data_dir/$f" ]] && members+=("$f")
    done
    [[ -d "$data_dir/neural" ]] && members+=("neural")

    [[ ${#members[@]} -gt 0 ]] || die "nothing to back up in $data_dir"

    tar -czf "$archive" -C "$data_dir" "${members[@]}"
    echo "$archive"
    echo "  contents: ${members[*]}"
    echo "  size:     $(du -h "$archive" | cut -f1)"
}

do_restore() {
    local archive="$1" data_dir="$2"
    [[ -f "$archive" ]] || die "archive not found: $archive"

    if [[ -e "$data_dir/bandit_wal.jsonl" || -e "$data_dir/checkpoint.json" ]]; then
        die "$data_dir already holds a database — restore into an empty directory"
    fi
    mkdir -p "$data_dir"
    tar -xzf "$archive" -C "$data_dir"
    echo "restored $archive -> $data_dir"
    echo "note: exports/ is not part of the backup; offline history starts fresh"
}

# Restore into a throwaway directory, boot the server against it, and confirm the
# campaigns actually come back. This is the step that distinguishes a backup from
# a tarball nobody has ever opened.
do_verify() {
    local archive="$1"
    [[ -x "$BIN" ]] || die "binary not found: $BIN (cargo build --release --features neural)"

    local tmp port
    tmp="$(mktemp -d)"
    port=18500
    trap 'pkill -f "$tmp" 2>/dev/null; rm -rf "$tmp"' RETURN

    tar -xzf "$archive" -C "$tmp"

    DATA_DIR="$tmp" PORT="$port" BANDITDB_API_KEY=verify \
        "$BIN" > "$tmp/verify.log" 2>&1 &
    local pid=$!

    local ok=0
    for _ in $(seq 1 60); do
        if curl -sS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ok=1; break; fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
    done

    if [[ "$ok" != "1" ]]; then
        echo "FAIL: server did not start from the restored backup" >&2
        tail -20 "$tmp/verify.log" >&2
        kill -9 "$pid" 2>/dev/null || true
        return 1
    fi

    local campaigns
    campaigns=$(curl -sS -H "X-Api-Key: verify" "http://127.0.0.1:$port/campaigns" 2>/dev/null)
    kill -9 "$pid" 2>/dev/null || true

    local count
    count=$(printf '%s' "$campaigns" | grep -o '"campaign_id"' | wc -l | tr -d ' ')
    echo "PASS: restored backup boots; $count campaign(s) recovered"
    [[ "$count" -gt 0 ]] || echo "  warning: zero campaigns — expected for a backup taken before any were created"
}

do_drill() {
    local data_dir="$1"
    local tmp out archive
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    # Capture whole output then take the first line. Piping to `head` closes the
    # pipe early, and under `set -o pipefail` the resulting SIGPIPE aborts the run.
    out=$(do_backup "$data_dir" "$tmp")
    archive=${out%%$'\n'*}
    echo "$out"
    echo "--- verifying $archive"
    do_verify "$archive"
}

# ---------------------------------------------------------------------------

case "${1:-}" in
    backup)  [[ $# -eq 3 ]] || die "usage: $0 backup <data_dir> <backup_dir>";  do_backup  "$2" "$3" ;;
    restore) [[ $# -eq 3 ]] || die "usage: $0 restore <archive> <data_dir>";    do_restore "$2" "$3" ;;
    verify)  [[ $# -eq 2 ]] || die "usage: $0 verify <archive>";                do_verify  "$2" ;;
    drill)   [[ $# -eq 2 ]] || die "usage: $0 drill <data_dir>";                do_drill   "$2" ;;
    *) sed -n '2,30p' "$0"; exit 1 ;;
esac
