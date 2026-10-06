"""Push the preview-eval kernel whenever a corrected-v2 milestone checkpoint
exists on HF without a COMPLETE eval summary.

Development diagnostics only: these evals are repeated monitoring runs on the
frozen MATE/puzzle sets and are never a holdout. Only `*-final` output is the
one-shot protocol result, and its claims must disclose prior exposure.

Fix (2026-10-06): the milestone list and run prefix live here and are embedded
into the pushed kernel code, so the watcher and the kernel can never disagree
(the legacy pair drifted: watcher through 3,896,658, kernel stopped at
1,620,000). Completion means an eval-summary.json with complete=true,
returncode=0, and exact expected totals; INCOMPLETE/partial summaries are
re-scored on the next push from the persisted per-example JSONL.

The eval kernel is pushed to the account with the most free GPU quota (the
owner account's quota can be exhausted by the training session), so both the
kernel status checks and the push use that account's credentials. Rewriting
the kernel id per account mirrors scripts/watch_1b.py.

Never holds a GPU session: the kernel is pushed only when work is ready.
Called by .github/workflows/pipeline-tick.yml; safe locally
(`python3 scripts/ensure_eval_preview.py --check-only` never pushes).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from watch_2b import FINAL_STEP, RUN  # noqa: E402  single source of truth

HF_REPO = "vedangfake/chess-slm-benchmark"
KERNEL_DIR = ROOT / "kernels" / "eval-preview"
ACCOUNTS = ["vedanggggg", "vedangpandeyyy", "softmaxsimp", "samaltmannnn", "shoumikmitra"]
CRED_DS = {"vedanggggg": "vedanggggg/chess-creds",
           "vedangpandeyyy": "vedangpandeyyy/chess-creds",
           "softmaxsimp": "softmaxsimp/chess-creds",
           "samaltmannnn": "samaltmannnn/hf-creds",
           "shoumikmitra": "shoumikmitra/hf-creds"}
ACTIVE = ("RUNNING", "QUEUED", "PENDING")
CONFIG_MARKER = "__EVALPREVIEW_CONFIG__"
EXPECT_MATE = 4000
EXPECT_PUZ = 10000
# 100k-step monitoring milestones from the corrected stage onward, plus the
# one-shot final. Embedded into the kernel; never maintained twice.
TARGETS = [step for step in range(1_800_000, FINAL_STEP, 100_000)] + [FINAL_STEP]


def out_prefix(step: int) -> str:
    kind = "final" if step == FINAL_STEP else "preview"
    return f"eval-results/{RUN}-{step // 1000}k-{kind}"


def hf_token() -> str:
    from kaggle_checkpoint import hf_token as token
    return token(ROOT)


def hf_files() -> set[str]:
    from huggingface_hub import HfApi
    return set(HfApi(token=hf_token()).list_repo_files(HF_REPO, repo_type="dataset"))


def source_commit(files: set[str]) -> str:
    """Pin the eval kernel to the code that produced the checkpoints when
    available; otherwise to the CI checkout."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import hf_token as token, latest_remote_checkpoint
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token(ROOT))
    name = latest_remote_checkpoint(api, HF_REPO, RUN)
    if name is not None:
        cached = hf_hub_download(HF_REPO, f"{RUN}/{name}/config.json",
                                 repo_type="dataset", token=api.token)
        commit = json.loads(Path(cached).read_text(encoding="utf-8")).get("source_commit")
        if commit:
            return str(commit)
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   text=True).strip()


def summary_complete(prefix: str) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        cached = hf_hub_download(HF_REPO, f"{prefix}/eval-summary.json",
                                 repo_type="dataset", token=hf_token())
        summary = json.loads(Path(cached).read_text(encoding="utf-8"))
    except Exception:
        return False
    if not summary.get("complete") or summary.get("returncode") != 0:
        return False
    mate = summary.get("mate") or {}
    puz = summary.get("puzzles") or {}
    return mate.get("total") == EXPECT_MATE and puz.get("total") == EXPECT_PUZ


def env_for(account: str) -> dict:
    from launch_trainers import env_for_account
    return env_for_account(account)


def kernel_status(account: str) -> str:
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status",
                            f"{account}/eval-ccgavn-preview"], env=env_for(account),
                           capture_output=True, text=True, timeout=90)
        return (r.stdout or r.stderr).strip()
    except Exception as exc:
        return f"status error: {exc}"


def gpu_quota(account: str) -> float:
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "quota"], env=env_for(account),
                           capture_output=True, text=True, timeout=90)
        for line in (r.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "GPU":
                return float(parts[2].rstrip("h"))
    except Exception:
        pass
    return -1.0


def render_kernel(cfg: dict) -> str:
    src = (KERNEL_DIR / "eval_preview.py").read_text(encoding="utf-8")
    placeholder = f'"""{CONFIG_MARKER}"""'
    if placeholder not in src:
        raise RuntimeError(f"{KERNEL_DIR / 'eval_preview.py'} lost its {placeholder} placeholder")
    return src.replace(placeholder, '"""' + json.dumps(cfg) + '"""')


def push(account: str, cfg: dict) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix=f"evalprev_{account}_") as tmp:
        tmpd = Path(tmp)
        (tmpd / "eval_preview.py").write_text(render_kernel(cfg), encoding="utf-8")
        (tmpd / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{account}/eval-ccgavn-preview", "title": "eval-ccgavn-preview",
            "code_file": "eval_preview.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [CRED_DS[account]],
            "competition_sources": [], "kernel_sources": []}, indent=1))
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                            "-p", str(tmpd)], env=env_for(account),
                           capture_output=True, text=True, timeout=300)
        out = (r.stdout + r.stderr).strip()
        print(f"[preview] push {account}: rc={r.returncode} {out[-200:]}", flush=True)
        return r.returncode, out


def main(check_only: bool = False) -> None:
    files = hf_files()
    steps = [int(m.group(1)) for f in files
             if (m := re.match(rf"{RUN}/checkpoint-(\d+)/state\.pt$", f))]
    if not steps:
        print(f"[preview] no {RUN} checkpoints on HF; nothing to do")
        return
    latest = max(steps)
    pending = [t for t in TARGETS if t <= latest and not summary_complete(out_prefix(t))]
    if not pending:
        print(f"[preview] latest checkpoint {latest}; all reached milestones complete")
        return
    print(f"[preview] latest {latest}; pending: {pending}")
    for acct in ACCOUNTS:
        st = kernel_status(acct)
        if any(a in st for a in ACTIVE):
            print(f"[preview] eval kernel active on {acct}: {st[-60:]}")
            return
    if check_only:
        commit = source_commit(files)
        print(f"[preview] check-only: no active kernel; a real run would push "
              f"commit {commit[:12]} with {len(pending)} pending")
        return
    quotas = {a: gpu_quota(a) for a in ACCOUNTS}
    usable = {a: q for a, q in quotas.items() if q != 0}
    if not usable:
        print(f"[preview] quotas={quotas}; every account reports 0 GPU hours; "
              "waiting for the weekly refresh")
        return
    best = max(usable, key=usable.get)
    print(f"[preview] no active kernel; quotas={quotas}; pushing on {best}")
    time.sleep(120)
    for acct in ACCOUNTS:
        st = kernel_status(acct)
        if any(a in st for a in ACTIVE):
            print(f"[preview] {acct} kernel became active; skipping push")
            return
    cfg = {"hf_repo": HF_REPO, "run": RUN, "targets": TARGETS,
           "final_step": FINAL_STEP, "source_commit": source_commit(files),
           "expect_mate_rows": EXPECT_MATE, "expect_puzzles": EXPECT_PUZ}
    for acct in sorted(usable, key=usable.get, reverse=True):
        rc, out = push(acct, cfg)
        if "successfully pushed" in out.lower():
            print(f"[preview] pushed on {acct}")
            return
        print(f"[preview] {acct} push failed; trying next account")


if __name__ == "__main__":
    main(check_only="--check-only" in sys.argv[1:])
