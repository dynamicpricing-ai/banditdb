#!/bin/sh
# Container entrypoint.
#
# The chart sets securityContext.runAsNonRoot with runAsUser 1001, so under
# Kubernetes this script already runs as banditdb and cannot chown. Unconditionally
# running `chown -R` then `gosu` therefore failed the pod outright. Plain `docker
# run` still starts as root, where fixing ownership of a fresh volume is useful.
#
# So: adapt to whichever applies. Drop privileges only when we actually have them.
set -e

DATA_DIR="${DATA_DIR:-/data}"

if [ "$(id -u)" = "0" ]; then
    # Running as root (typical for `docker run`): take ownership of the volume,
    # then drop to the unprivileged user.
    chown -R banditdb:banditdb "$DATA_DIR" 2>/dev/null || true
    exec gosu banditdb "$@"
fi

# Already unprivileged (Kubernetes with runAsNonRoot, or `docker run --user`).
# The volume must already be writable — fsGroup handles that in the chart.
if [ ! -w "$DATA_DIR" ]; then
    echo "entrypoint: $DATA_DIR is not writable by uid $(id -u)." >&2
    echo "  Under Kubernetes set podSecurityContext.fsGroup to match runAsUser." >&2
    echo "  Under docker run, either start as root or pre-chown the volume." >&2
    exit 1
fi

exec "$@"
