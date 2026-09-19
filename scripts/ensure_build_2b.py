"""Ensure the four 2B-build labeling kernels stay alive until their slices are
complete. Called by .github/workflows/watch-build-2b.yml every 30 minutes.

Per account: read its slice assignment (kernels/build-2b/slices.json), check
HF for every assigned shard's train_set.npz + teacher_logp.npy. If incomplete
and the kernel is not RUNNING/QUEUED, push it. Idempotent and safe to run
locally with the right credentials.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "vedangfake/chess-slm-benchmark"
PREFIX = "chessbench-full-build"
SLICES = json.loads((ROOT / "kernels" / "build-2b" / "slices.json").read_text())
ALL_SLICES = json.loads((ROOT / "kernels" / "build-2b" / "slices.json").read_text())


def hf_files() -> set[str]:
    from huggingface_hub import HfApi
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import hf_token
    api = HfApi(token=hf_token(ROOT))
    return set(api.list_repo_files(HF_REPO, repo_type="dataset"))


def main() -> None:
    slices = [list(map(str, s)) for s in
              json.loads((ROOT / "kernels" / "build-2b" / "shard_slices.json").read_text())]
    assign = json.loads((ROOT / "kernels" / "build-2b" / "slices.json").read_text())
    if len(slices) < 8:
        print("[build-2b] shard_slices.json missing or short; nothing to do")
        return
    files = hf_files()
    sys.path.insert(0, str(ROOT / "scripts"))
    from launch_trainers import env_for_account
    # 1B-first stop rule (user decision 2026-09-19): once enough shards are
    # labeled for ~1B new rows, stop supervising so the fleet is not re-pushed.
    rows_map = json.loads((ROOT / "kernels" / "build-2b" / "shard_rows.json").read_text())
    done_new_rows = sum(int(rows_map[s]) for s in rows_map
                        if f"{PREFIX}/shard-{s}/teacher_logp.npy" in files)
    TARGET_NEW_ROWS = 920_000_000
    if done_new_rows >= TARGET_NEW_ROWS:
        print(f"[build-2b] 1B target reached ({done_new_rows/1e6:.0f}M new rows labeled); "
              f"not pushing further")
        return
    print(f"[build-2b] progress: {done_new_rows/1e6:.0f}M / {TARGET_NEW_ROWS/1e6:.0f}M new rows")
    for account, idxs in assign.items():
        want = []
        for idx in idxs:
            want.extend(slices[idx])
        missing = [s for s in want
                   if f"{PREFIX}/shard-{s}/train_set.npz" not in files
                   or f"{PREFIX}/shard-{s}/teacher_logp.npy" not in files]
        if not missing:
            print(f"[build-2b] {account}: slice complete ({len(want)} shards)")
            continue
        env = env_for_account(account)
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status",
                            f"{account}/build-2b-slice"], capture_output=True,
                           text=True, timeout=90, env=env)
        st = (r.stdout or r.stderr).strip()
        if "RUNNING" in st or "QUEUED" in st or "PENDING" in st:
            print(f"[build-2b] {account}: active, {len(missing)} shards left ({st[-40:]})")
            continue
        print(f"[build-2b] {account}: not active ({st[-40:]}), pushing "
              f"({len(missing)} shards left)")
        time.sleep(120)
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status",
                            f"{account}/build-2b-slice"], capture_output=True,
                           text=True, timeout=90, env=env)
        st = (r.stdout or r.stderr).strip()
        if "RUNNING" in st or "QUEUED" in st or "PENDING" in st:
            print(f"[build-2b] {account}: became active, skipping push")
            continue
        push = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                               "-p", str(ROOT / "kernels" / "build-2b" / account)],
                              capture_output=True, text=True, timeout=300, env=env)
        print(f"[build-2b] {account}: push rc={push.returncode} "
              f"{(push.stdout + push.stderr).strip()[-160:]}")


if __name__ == "__main__":
    main()
