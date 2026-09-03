#!/usr/bin/env python3
"""Wait for Kaggle GPU quota reset, then relaunch the 6 training kernels.

Polls `kaggle quota` on all 3 accounts every 10 min. When every account has
remaining GPU > 0, pushes all 6 kernels via scripts/launch_trainers.py (which
resumes each run from its latest HF checkpoint) and exits 0.

Progress is logged to logs/quota_relaunch.log. Designed to run detached with
nohup; safe against double-launch (checks kernel RUNNING status first).

Usage:
  nohup python3 scripts/quota_relaunch.py >/dev/null 2>&1 &
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from launch_trainers import env_for_account, TRAINER_CONFIGS  # noqa: E402

ACCOUNTS = ["vedanggggg", "vedangpandeyyy", "softmaxsimp"]
LOG = ROOT / "logs" / "quota_relaunch.log"
POLL_S = 600


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def gpu_remaining(account: str) -> float | None:
    r = subprocess.run([sys.executable, "-m", "kaggle", "quota"],
                       env=env_for_account(account),
                       capture_output=True, text=True, timeout=60)
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "GPU":
            try:
                return float(parts[2].rstrip("h"))
            except ValueError:
                return None
    return None


def kernel_status(ref: str, account: str) -> str:
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", ref],
                       env=env_for_account(account),
                       capture_output=True, text=True, timeout=60)
    return (r.stdout or r.stderr).strip()


def main() -> None:
    log("quota_relaunch armed: waiting for GPU quota reset (per-account relaunch)")
    launched: set = set()  # slugs already relaunched this process lifetime
    while True:
        remaining = {a: gpu_remaining(a) for a in ACCOUNTS}
        log(f"quota remaining: {remaining}")
        for owner, slug, *_ in TRAINER_CONFIGS:
            if slug in launched:
                continue
            quota = remaining.get(owner)
            if quota is None or quota <= 0:
                continue
            st = kernel_status(f"{owner}/{slug}", owner)
            if "RUNNING" in st:
                log(f"{owner}/{slug} already RUNNING; marking launched")
                launched.add(slug)
                continue
            log(f"quota on {owner} ({quota}h) — relaunching {owner}/{slug}")
            import launch_trainers
            try:
                # push just this kernel via the launcher's own machinery
                import tempfile, json, shutil
                cfg = next(c for c in TRAINER_CONFIGS if c[1] == slug)
                _, _, tmpl, desc, repls = cfg
                template = ROOT / f"notebooks/{tmpl}_kaggle_{'baseline_5m' if tmpl == '01' else 'train_gavn'}.ipynb"
                push_dir = Path(tempfile.mkdtemp(prefix=f"kaggle_push_{slug}_"))
                txt = template.read_text()
                for o, n in repls.items():
                    assert o in txt, f"{slug}: replacement source missing: {o!r}"
                    txt = txt.replace(o, n)
                (push_dir / f"{slug}.ipynb").write_text(txt)
                meta = {
                    "id": f"{owner}/{slug}",
                    "title": slug,
                    "code_file": f"{slug}.ipynb",
                    "language": "python",
                    "kernel_type": "notebook",
                    "is_private": True,
                    "enable_gpu": True,
                    "enable_tpu": False,
                    "enable_internet": True,
                    "machine_shape": "NvidiaTeslaT4",
                    "dataset_sources": [f"{owner}/chess-creds"],
                    "competition_sources": [],
                    "kernel_sources": [],
                }
                (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
                r = subprocess.run(
                    [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(push_dir)],
                    capture_output=True, text=True, env=env_for_account(owner), timeout=180)
                out = (r.stdout + r.stderr).strip()
                if r.returncode == 0:
                    log(f"[push] OK {owner}/{slug}")
                    launched.add(slug)
                else:
                    log(f"[push] FAIL {owner}/{slug}: {out[:400]} (will retry)")
                shutil.rmtree(push_dir, ignore_errors=True)
            except Exception as e:
                log(f"[push] EXC {owner}/{slug}: {e} (will retry)")
        if len(launched) == len(TRAINER_CONFIGS):
            log("all 6 kernels relaunched. exiting.")
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
