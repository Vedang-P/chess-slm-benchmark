#!/bin/bash
# Detach a long-running benchmark command from the launching shell
# (macOS-safe: setsid does not exist on darwin; uses a double-fork +
# os.setsid). Logs to /tmp/bench_run.log. The launcher shell can die
# without killing the benchmark.
#
#   bash scripts/detach.sh python3 scripts/run_suite.py --output_dir results/chess ...
#
exec python3 - "$@" <<'PYEOF'
import os
import sys

argv = sys.argv[1:]
pid = os.fork()
if pid > 0:
    sys.exit(0)  # parent returns immediately; child is reparented
os.setsid()      # new session: immune to the launcher's process group

null = os.open(os.devnull, os.O_RDONLY)
os.dup2(null, 0)
log = os.open("/tmp/bench_run.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND)
os.dup2(log, 1)
os.dup2(log, 2)

os.execvp(argv[0], argv)
PYEOF
