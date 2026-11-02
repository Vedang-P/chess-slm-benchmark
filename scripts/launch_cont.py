"""Push one cont1 learner session to a Kaggle account (smoke test or sprint arm).

The kernel checks out the pinned commit from GitHub, so the commit must already
be pushed. The sequential main run is launched by scripts/watch_cont.py instead.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cont_config import (ACCOUNTS, ARM_RUN_PREFIX, ARM_STEPS, ARMS, COMMON_ARGS,  # noqa: E402
                                 HF_REPO, SEARCHLESS_COMMIT, STATUS_PREFIX)
from scripts.kaggle_pool import gpu_quota, push_kernel  # noqa: E402

TEMPLATE = ROOT / "kernels" / "cont" / "worker.py"
ARM_CONTROL = ["--max-steps", str(ARM_STEPS), "--val-every", "5000", "--ckpt-every", "5000",
               "--milestone-every", "0", "--guard-after", "99999999"]
SMOKE_ARGS = ["--max-steps", "60", "--val-every", "30", "--ckpt-every", "30",
              "--milestone-every", "0", "--guard-after", "99999999", "--broad-rows", "256",
              "--tactical-groups", "4", "--lambda-list", "1.0", "--block-steps", "40",
              "--dev-batch", "2048"]


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def pushed_head() -> str:
    head = git("rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin"], check=True)
    if not git("branch", "-r", "--contains", head):
        raise SystemExit(f"commit {head[:12]} is not on GitHub; push it first")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("working tree has uncommitted changes; the kernel would not see them")
    return head


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--account", required=True, choices=ACCOUNTS)
    p.add_argument("--arm", choices=sorted(ARMS), help="sprint arm preset")
    p.add_argument("--smoke", action="store_true", help="60-step end-to-end check")
    p.add_argument("--run", help="HF run prefix (default derived from --arm/--smoke)")
    p.add_argument("--slug", help="kernel slug (default derived)")
    p.add_argument("--session-seconds", type=float)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("extra", nargs="*", help="extra trainer flags (after --)")
    args = p.parse_args()
    if bool(args.arm) == bool(args.smoke):
        raise SystemExit("choose exactly one of --arm or --smoke")

    if args.smoke:
        name, trainer_args = "smoke", COMMON_ARGS + SMOKE_ARGS
        phase, seconds = "smoke", args.session_seconds or 2400
    else:
        name = args.arm
        trainer_args = COMMON_ARGS + ARMS[args.arm]["args"] + ARM_CONTROL
        phase, seconds = "arm", args.session_seconds or 6.5 * 3600
    run = args.run or f"{ARM_RUN_PREFIX}{name}"
    slug = args.slug or f"ccgavn-cont-{name}"
    # later flags win in argparse, so presets can be overridden from the command line
    trainer_args = trainer_args + args.extra
    quota = gpu_quota(args.account)
    seconds = min(seconds, max(0.0, quota - 0.4) * 3600)
    if seconds < 900:
        raise SystemExit(f"{args.account} has {quota:.2f} GPU hours left; not enough for a session")
    commit = pushed_head()
    config = {"phase": phase, "run": run, "hf_repo": HF_REPO, "status_prefix": STATUS_PREFIX,
              "source_commit": commit, "searchless_commit": SEARCHLESS_COMMIT,
              "session_seconds": int(seconds), "trainer_args": trainer_args,
              "account_index": ACCOUNTS.index(args.account)}
    print(f"[launch] {args.account}/{slug} run={run} commit={commit[:12]} "
          f"session={seconds / 3600:.2f}h quota={quota:.2f}h", flush=True)
    print("[launch] trainer flags: " + " ".join(trainer_args), flush=True)
    if not args.dry_run:
        push_kernel(args.account, slug, TEMPLATE, "__CONT_CONFIG__", config, gpu=True)


if __name__ == "__main__":
    main()
