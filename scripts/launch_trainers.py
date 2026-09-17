#!/usr/bin/env python3
"""Push training kernels to Kaggle.

2026-09-09 two-model decision: the old arms (baseline-5m, gavn-3m seed0/seed1,
gavn-5m-seed0, gavn-5m-loss) are RETIRED and removed from this launcher —
their results are final (see results/frozen-evals-2026-09-09/) and their
stranded runs will not be resumed. Registered trainers are the two models of
the endgame: the trimmed fixed-bias geometry arm (notebook 08) and CC-GAVN
seed 0 (notebook 07).

Usage:
  python3 scripts/launch_trainers.py --status   # just print kernel statuses
  python3 scripts/launch_trainers.py            # push all registered
  python3 scripts/launch_trainers.py --only gavn-5m-geometry-slim
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TRAINER_CONFIGS = [
    # (owner, slug, template, description, {old: new} replacements)
    # Old arms retired 2026-09-09; the two endgame models remain.
    ("softmaxsimp", "gavn-5m-geometry-slim", "08", "trimmed fixed-bias geometry (3.46M)", {}),
    ("vedangpandeyyy", "ccgavn-5m-seed0", "07", "candidate-conditioned geometry (4.76M)", {}),
    ("shoumikmitra", "ccgavn-5m-seed1", "07", "candidate-conditioned geometry, seed 1 (4.76M)",
     {"RUN_ID = 'ccgavn-5m-seed0'": "RUN_ID = 'ccgavn-5m-seed1'",
      "SEED, STEPS = 0, 160_000": "SEED, STEPS = 1, 160_000",
      "'smoke-ccgavn-disposable'": "'smoke-ccgavn-seed1-disposable'"}),
]

# Notebook template filenames that do not follow the {NN}_kaggle_train_gavn
# convention.
TEMPLATE_OVERRIDES = {
    "01": "01_kaggle_baseline_5m.ipynb",
    "07": "07_kaggle_train_ccgavn.ipynb",
    "08": "08_kaggle_train_gavn_slim.ipynb",
}


def env_for_account(account: str) -> dict:
    env = os.environ.copy()
    env.pop("KAGGLE_CONFIG_DIR", None)
    env.pop("KAGGLE_API_TOKEN", None)
    if account == "vedanggggg":
        pass
    elif account == "vedangpandeyyy":
        fake_home = Path(tempfile.gettempdir()) / "kaggle_home_vedangpandeyyy"
        kdir = fake_home / ".kaggle"
        kdir.mkdir(parents=True, exist_ok=True)
        src = Path.home() / ".kaggle/kaggle.json"
        if src.exists():
            shutil.copy(str(src), str(kdir / "kaggle.json"))
            os.chmod(str(kdir / "kaggle.json"), 0o600)
        (kdir / "access_token").unlink(missing_ok=True)
        env["HOME"] = str(fake_home)
        env["KAGGLE_CONFIG_DIR"] = str(kdir)
    elif account in ("softmaxsimp", "samaltmannnn", "shoumikmitra"):
        token = (Path.home() / f".kaggle/profiles/{account}/access_token").read_text().strip()
        env["KAGGLE_API_TOKEN"] = token
    else:
        raise ValueError(account)
    return env


def kaggle(args: list[str], account: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "kaggle", *args],
                          capture_output=True, text=True, env=env_for_account(account),
                          timeout=timeout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    configs = [c for c in TRAINER_CONFIGS if not args.only or c[1] == args.only]
    if args.status:
        for owner, slug, _, desc, _ in configs:
            r = kaggle(["kernels", "status", f"{owner}/{slug}"], owner)
            print(f"{owner}/{slug:22s} {r.stdout.strip() or r.stderr.strip()}")
        return

    failures = []
    for owner, slug, tmpl, desc, repls in configs:
        if tmpl in TEMPLATE_OVERRIDES:
            template = ROOT / "notebooks" / TEMPLATE_OVERRIDES[tmpl]
        else:
            template = ROOT / f"notebooks/{tmpl}_kaggle_{'baseline_5m' if tmpl == '01' else 'train_gavn'}.ipynb"
        push_dir = Path(tempfile.mkdtemp(prefix=f"kaggle_push_{slug}_"))
        txt = template.read_text()
        for old, new in repls.items():
            assert old in txt, f"{slug}: replacement source missing: {old!r}"
            txt = txt.replace(old, new)
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
            # Kaggle's default GPU is P100; its cu128 torch build cannot run
            # Pascal (sm_60). T4 (sm_75) works and gives 2 GPUs -> DataParallel.
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{owner}/chess-creds"],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"[push] {owner}/{slug} ({desc}) ...", flush=True)
        r = kaggle(["kernels", "push", "-p", str(push_dir)], owner)
        out = (r.stdout + r.stderr).strip()
        if r.returncode == 0:
            print(f"[push] OK  {owner}/{slug}: {out.splitlines()[-1] if out else ''}")
        else:
            print(f"[push] FAIL {owner}/{slug}: {out[:600]}")
            failures.append(slug)
        shutil.rmtree(push_dir, ignore_errors=True)
        time.sleep(15)
    if failures:
        sys.exit(f"push failures: {failures}")
    print("all kernels pushed")


if __name__ == "__main__":
    main()
