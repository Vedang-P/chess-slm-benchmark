"""Live terminal monitor for the 5M-student pipeline.

Shows real-time progress bars for the current stage:
  - 270M teacher labeling (7 chunks x 262144 rows = 1,838,218)
  - student training (steps)
  - official 10K puzzle evals

Run in a separate terminal alongside the pipeline (Windows or WSL):
    python3 scripts/monitor_sl.py
Stdlib only; Ctrl+C to exit.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import time

SL_DATA = os.environ.get("SL_DATA", "C:/tmp/sl_data")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")

N_ROWS = 1838218
CHUNK = 262144
CHUNK_STARTS = [0, 262144, 524288, 786432, 1048576, 1310720, 1572864]


def gpu_util() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3)
        return out.stdout.strip().replace("\n", " ")
    except Exception:
        return "n/a"


def bar(label: str, frac: float, detail: str = "", width: int = 38) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(width * frac)
    pct = frac * 100
    return (f"{label:<16} [{'#' * filled}{'.' * (width - filled)}] "
            f"{pct:5.1f}%  {detail}")


def teacher_progress() -> tuple[float, str]:
    done_rows = 0
    active_rows = 0
    active_start = None
    active_log = None
    for s in CHUNK_STARTS:
        if os.path.exists(f"{SL_DATA}/teacher_chunk_{s}.npy.done"):
            done_rows += CHUNK
        elif active_start is None:
            active_start = s
            active_log = f"{SL_DATA}/teacher_chunk_{s}.npy.log"
    if active_log and os.path.exists(active_log):
        with open(active_log, errors="replace") as f:
            for line in f:
                m = re.search(r"(\d+)/262144", line)
                if m:
                    active_rows = int(m.group(1))
    total = done_rows + active_rows
    frac = total / N_ROWS
    detail = f"{total:,}/{N_ROWS:,}"
    if active_start is not None:
        detail += f"  (chunk {active_start}+, {active_rows:,} rows)"
    return frac, detail


def last_line(path: str) -> str:
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
        return lines[-1].strip() if lines else ""
    except Exception:
        return ""


def student_progress() -> tuple[float, str]:
    log = os.path.join(RESULTS, "student-train.log")
    line = last_line(log)
    m = re.search(r"step (\d+)/(\d+)", line)
    if not m:
        return 0.0, "waiting for training..."
    step, total = int(m.group(1)), int(m.group(2))
    m2 = re.search(r"loss=([\d.]+)", line)
    loss = f" loss={m2.group(1)}" if m2 else ""
    return step / total, f"{step}/{total}{loss}"


def eval_progress() -> tuple[float, str]:
    logs = sorted(glob.glob(os.path.join(RESULTS, "student-eval-*.log")))
    if not logs:
        return 0.0, "no student evals yet"
    line = last_line(logs[-1])
    m = re.search(r"(\d+)/10000", line)
    if not m:
        return 0.0, os.path.basename(logs[-1])
    n = int(m.group(1))
    m2 = re.search(r"acc=([\d.]+)", line)
    acc = f" acc={m2.group(1)}%" if m2 else ""
    return n / 10000, f"{n}/10000{acc}"


def stage() -> str:
    if not os.path.exists(f"{SL_DATA}/teacher_chunk_1572864.npy.done"):
        return "teacher-labeling"
    if not os.path.exists(os.path.join(RESULTS, "student-train.log")):
        return "idle"
    return "training/eval"


def main() -> None:
    print("SL pipeline monitor — Ctrl+C to exit\n", flush=True)
    t0 = time.time()
    while True:
        st = stage()
        lines = [f"stage: {st}   gpu: {gpu_util()}   "
                 f"uptime: {int(time.time()-t0)}s"]
        if st == "teacher-labeling":
            frac, det = teacher_progress()
            eta = ""
            if frac > 0.01:
                rate = frac * N_ROWS / max(1, time.time() - t0)
                eta = f" eta {int((1-frac)*N_ROWS/max(rate,1))}s"
            lines.append(bar("teacher labels", frac, det + eta))
        elif st in ("training/eval", "idle"):
            frac, det = student_progress()
            lines.append(bar("student train", frac, det))
            frac2, det2 = eval_progress()
            lines.append(bar("10K eval", frac2, det2))
        sys.stdout.write("\x1b[2K" + "\n\x1b[2K".join(lines) + "\r")
        sys.stdout.flush()
        time.sleep(1.5)


if __name__ == "__main__":
    main()
