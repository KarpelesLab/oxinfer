#!/bin/sh
# Run a command while holding the machine-wide benchmark lock (/tmp/oxinfer-bench.lock).
#
#   tools/bench/with-lock.sh cargo run --release --example dec_ab -- ...
#
# Waits for the lock (mkdir is atomic), records the holder's pid in it, and removes it
# afterwards only if this script created it, also on Ctrl-C / kill. Never remove the lock
# by hand while someone else holds it: their numbers would be taken on a loaded GPU.
set -u
LOCK=${OXINFER_BENCH_LOCK:-/tmp/oxinfer-bench.lock}
waited=0
until mkdir "$LOCK" 2>/dev/null; do
    if [ $((waited % 60)) -eq 0 ]; then
        echo "with-lock: waiting for $LOCK (held by $(cat "$LOCK/pid" 2>/dev/null || echo '?'))" >&2
    fi
    sleep 2
    waited=$((waited + 2))
done
echo $$ > "$LOCK/pid"
release() { rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null; }
trap 'release; exit 130' INT TERM HUP
"$@"
status=$?
release
exit $status
