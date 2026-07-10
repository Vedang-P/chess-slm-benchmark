#!/usr/bin/env python3
"""Monitor benchmark progress and GPU stats in real-time."""
import time, os, json, subprocess
from pathlib import Path

LOG = Path(__file__).parent / "benchmark_run.log"
RESULTS = Path(__file__).parent / "data" / "results" / "gemma4-e2b_q3_k_s"

def get_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader"], timeout=5, text=True
        ).strip()
        return out
    except:
        return "N/A"

def get_progress():
    if not LOG.exists():
        return "No log file"
    with open(LOG) as f:
        lines = f.readlines()
    progress_lines = [l for l in lines if "Progress:" in l]
    result_files = list(RESULTS.glob("*.json")) if RESULTS.exists() else []
    return progress_lines, result_files

if __name__ == "__main__":
    last_line_count = 0
    last_result_count = 0
    while True:
        progress_lines, result_files = get_progress()
        if progress_lines:
            latest = progress_lines[-1].strip()
        else:
            latest = "Starting..."
        gpu = get_gpu()

        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"⏱  {time.strftime('%H:%M:%S')}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Latest: {latest}")
        print(f"Log lines: {len(open(LOG).readlines()) if LOG.exists() else 0}")
        print(f"Result files: {[f.name for f in result_files]}")
        print(f"GPU: {gpu}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        proc = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=3)
        for line in proc.stdout.split('\n'):
            if 'run_bench' in line or 'python3' in line and 'bench' in line:
                print(f"  Process: {line.strip()}")
        time.sleep(10)
