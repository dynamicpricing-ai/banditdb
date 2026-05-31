#!/bin/sh
# Ensure /data is owned by the banditdb user regardless of how the volume was populated.
chown -R banditdb:banditdb /data
exec gosu banditdb "$@"
