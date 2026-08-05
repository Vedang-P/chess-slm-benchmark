#!/usr/bin/env python3
"""Poll a Kaggle kernel until it finishes; log + macOS notify on completion.

    python3 scripts/watch_kaggle_kernel.py [KERNEL_REF] [INTERVAL_S]

Logs to /tmp/kaggle_kernel_watch.log; on completion downloads the kernel
output (best effort) into the caller's results dir.
"""
import subprocess
import sys
import time
from pathlib import Path

KERNEL = sys.argv[1] if len(sys.argv) > 1 else "vedangpandeyyy/mate-gemma-e2b-100-positions"
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 60
OUT_DIR = Path(sys.argv[3] if len(sys.argv) > 3 else "results/kaggle_kernel_output")
LOG = Path("/tmp/kaggle_kernel_watch.log")
KAGGLE = "/Library/Frameworks/Python.framework/Versions/3.13/bin/kaggle"
RUNNING_STATES = ("RUNNING", "PENDING", "QUEUED", "UNKNOWN")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def status() -> str:
    out = subprocess.run([KAGGLE, "kernels", "status", KERNEL],
                         capture_output=True, text=True, timeout=60)
    text = (out.stdout or "") + (out.stderr or "")
    if "KernelWorkerStatus" in text:
        raw = text.split("has status ")[-1].strip().strip('"')
        return raw.split(".")[-1]
    return "UNKNOWN"


def notify(title: str, msg: str) -> None:
    subprocess.run(["osascript", "-e",
                    f'display notification "{msg}" with title "{title}"'],
                   capture_output=True)


def main() -> None:
    prev = None
    while True:
        try:
            s = status()
        except Exception as e:
            s = "UNKNOWN"
            log(f"status poll error: {e}")
        if s != prev:
            log(f"status -> {s}")
            prev = s
        if s not in RUNNING_STATES:
            log(f"DONE: {KERNEL} -> {s}")
            notify("Kaggle Kernel Watch",
                   f"{KERNEL.split('/')[-1]}: {s}")
            if s.startswith("COMPLETE"):
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                r = subprocess.run([KAGGLE, "kernels", "output", KERNEL,
                                    "-p", str(OUT_DIR)],
                                   capture_output=True, text=True, timeout=300)
                log(f"output download rc={r.returncode} -> {OUT_DIR}")
                if r.stderr.strip():
                    log("output download stderr: " + r.stderr.strip()[-300:])
            sys.exit(0)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
