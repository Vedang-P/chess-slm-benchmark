"""Supervisor for five CPU verifier jobs and one rotating Kaggle T4 learner.

New runs are explicit: only this watcher launches them.  It fails closed on
unknown Kaggle state, verifier failure, changed labels, or failed GPU jobs.
GitHub Actions invokes it on the existing pipeline tick.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_accounts import env_for_account
from scripts.kaggle_checkpoint import api as make_api, latest_remote_checkpoint
from scripts.rl_labels import ACCOUNTS

HF_REPO = "vedangfake/chess-slm-benchmark"
LABEL_PREFIX = "rl-root-v1-labels"
RUN = "ccgavn-rl-root-v1-seed0"
SOURCE_RUN = "ccgavn-5m-seed0-v2"
SOURCE_CHECKPOINT = "checkpoint-3896658"
DEPTH = 14
ROOTS_PER_ACCOUNT = 2000
TOTAL_STEPS = 20_000
LABEL_SLUG = "ccgavn-rl-labels"
TRAIN_SLUG = "ccgavn-rl-train"
TEMPLATE = ROOT / "kernels/rl/run.py"
TAGS = ROOT / "configs/ccgavn-2b-shard-tags.json"
CREDS = {"vedanggggg": "vedanggggg/chess-creds",
         "vedangpandeyyy": "vedangpandeyyy/chess-creds",
         "softmaxsimp": "softmaxsimp/chess-creds",
         "samaltmannnn": "samaltmannnn/hf-creds",
         "shoumikmitra": "shoumikmitra/hf-creds"}
ACTIVE = ("RUNNING", "QUEUED", "PENDING")


def run_cli(account: str, *args: str, timeout: int = 90):
    return subprocess.run([sys.executable, "-m", "kaggle", *args],
                          env=env_for_account(account), capture_output=True,
                          text=True, timeout=timeout)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()


def available_slugs(account: str) -> set[str]:
    result = run_cli(account, "kernels", "list", "--mine", "--page-size", "200", "--csv")
    if result.returncode:
        raise RuntimeError(f"{account}: kernels list failed: {result.stderr[-300:]}")
    lines = result.stdout.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith("ref,title,")), None)
    if start is None:
        raise RuntimeError(f"{account}: Kaggle kernel list schema changed")
    return {row["ref"] for row in csv.DictReader(io.StringIO("\n".join(lines[start:])))
            if row.get("ref", "").startswith(account + "/")}


def kernel_status(account: str, slug: str, known: set[str]) -> str:
    if f"{account}/{slug}" not in known:
        return "ABSENT"
    result = run_cli(account, "kernels", "status", f"{account}/{slug}")
    if result.returncode:
        raise RuntimeError(f"{account}/{slug}: status failed: {(result.stdout + result.stderr)[-400:]}")
    raw = result.stdout + result.stderr
    for status in (*ACTIVE, "COMPLETE", "CANCEL_ACKNOWLEDGED", "ERROR", "FAILED"):
        if status in raw:
            return status
    raise RuntimeError(f"{account}/{slug}: unknown Kaggle status: {raw[-400:]}")


def quota(account: str) -> float:
    result = run_cli(account, "quota")
    if result.returncode:
        raise RuntimeError(f"{account}: quota failed: {result.stderr[-300:]}")
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "GPU":
            return float(parts[2].rstrip("h"))
    raise RuntimeError(f"{account}: GPU quota row missing")


def remote_status(api, prefix: str, files: set[str]) -> str:
    name = f"{prefix}/run-status.txt"
    if name not in files:
        return ""
    source = hf_hub_download(HF_REPO, name, repo_type="dataset", token=api.token)
    return Path(source).read_text(encoding="utf-8")


def label_state(api, account: str, files: set[str]):
    name = f"{LABEL_PREFIX}/{account}/state.json"
    if name not in files:
        return None
    source = hf_hub_download(HF_REPO, name, repo_type="dataset", token=api.token)
    state = json.loads(Path(source).read_text(encoding="utf-8"))
    identity = state["identity"]
    if (identity["account"] != account or identity["depth"] != DEPTH
            or identity["target"] != ROOTS_PER_ACCOUNT):
        raise ValueError(f"{account}: verifier identity changed")
    return state


def render(config: dict) -> str:
    source = TEMPLATE.read_text(encoding="utf-8")
    marker = '"""__CCGAVN_RL_CONFIG__"""'
    if marker not in source:
        raise ValueError("RL kernel config marker missing")
    # A Python string literal must preserve the escapes in nested JSON values
    # (notably the frozen label manifest). Triple quotes consumed those escapes.
    rendered = source.replace(marker, repr(json.dumps(config, sort_keys=True)))
    tree = ast.parse(rendered, filename=str(TEMPLATE))
    embedded = next((ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "CONFIG_JSON"
                             for target in node.targets)), None)
    if embedded is None or json.loads(embedded) != config:
        raise ValueError("rendered Kaggle configuration failed round-trip validation")
    return rendered


def push(account: str, slug: str, config: dict, gpu: bool):
    with tempfile.TemporaryDirectory(prefix="ccgavn_rl_") as temp:
        root = Path(temp)
        (root / "run.py").write_text(render(config), encoding="utf-8")
        metadata = {"id": f"{account}/{slug}", "title": slug,
                    "code_file": "run.py", "language": "python", "kernel_type": "script",
                    "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
                    "enable_internet": True, "dataset_sources": [CREDS[account]],
                    "competition_sources": [], "kernel_sources": []}
        if gpu:
            metadata["machine_shape"] = "NvidiaTeslaT4"
        (root / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2),
                                                   encoding="utf-8")
        result = run_cli(account, "kernels", "push", "-p", str(root), timeout=300)
        message = (result.stdout + result.stderr).strip()
        if result.returncode or "successfully pushed" not in message.lower():
            raise RuntimeError(f"{account}/{slug}: push failed: {message[-500:]}")
        print(f"[rl-watch] pushed {account}/{slug}", flush=True)


def manifest_from_states(states: dict) -> dict:
    digests = {state["identity"]["exclusion_digest"] for state in states.values()}
    if len(digests) != 1:
        raise ValueError("label jobs used different frozen benchmark exclusions")
    commits = {state["identity"]["source_commit"] for state in states.values()}
    if len(commits) != 1:
        raise ValueError("label jobs used different pinned source commits")
    return {"label_prefix": LABEL_PREFIX, "depth": DEPTH,
            "source_commit": commits.pop(),
            "exclusion_digest": digests.pop(),
            "sources": {account: {"count": ROOTS_PER_ACCOUNT,
                                   "sha256": state["sha256"]}
                        for account, state in sorted(states.items())}}


def main(check_only: bool = False):
    api = make_api(ROOT)
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    for name in (f"{SOURCE_RUN}/{SOURCE_CHECKPOINT}/state.pt",
                 f"{SOURCE_RUN}/{SOURCE_CHECKPOINT}/config.json"):
        if name not in files:
            raise RuntimeError(f"missing preserved baseline source {name}")
    tags = json.loads(TAGS.read_text(encoding="utf-8"))["tags"]
    if len(tags) < len(ACCOUNTS):
        raise ValueError("frozen corpus has too few independent shards")
    current_commit = git_head()
    label_states = {account: label_state(api, account, files) for account in ACCOUNTS}
    pinned = {state["identity"]["source_commit"] for state in label_states.values()
              if state is not None}
    if len(pinned) > 1:
        raise ValueError("verifier jobs have different pinned source commits")
    commit = pinned.pop() if pinned else current_commit
    states = {}
    listed = {account: available_slugs(account) for account in ACCOUNTS}
    for index, account in enumerate(ACCOUNTS):
        state = label_states[account]
        status_text = remote_status(api, f"{LABEL_PREFIX}/{account}", files)
        if "FAILED" in status_text:
            raise RuntimeError(f"{account}: verifier job failed: {status_text[-300:]}")
        if state and state["identity"]["shard_tag"] != tags[index]:
            raise ValueError(f"{account}: source shard assignment changed")
        count = int(state["count"]) if state else 0
        if count >= ROOTS_PER_ACCOUNT:
            states[account] = state
            print(f"[rl-watch] {account} labels complete: {count}", flush=True)
            continue
        status = kernel_status(account, LABEL_SLUG, listed[account])
        print(f"[rl-watch] {account} labels {count}/{ROOTS_PER_ACCOUNT}; kernel={status}",
              flush=True)
        if status in ACTIVE:
            continue
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"{account}: Kaggle verifier kernel {status}")
        if status == "COMPLETE":
            raise RuntimeError(f"{account}: verifier kernel completed with only "
                               f"{count}/{ROOTS_PER_ACCOUNT} persisted labels")
        if not check_only:
            push(account, LABEL_SLUG, {"mode": "labels", "account": account,
                                      "shard_tag": tags[index], "depth": DEPTH,
                                      "target": ROOTS_PER_ACCOUNT,
                                      "source_commit": commit}, gpu=False)
    if len(states) != len(ACCOUNTS):
        print(f"[rl-watch] verifier collection {len(states)}/{len(ACCOUNTS)} complete; "
              "learner remains gated", flush=True)
        return
    checkpoint = latest_remote_checkpoint(api, HF_REPO, RUN)
    if checkpoint:
        name = f"{RUN}/{checkpoint}/config.json"
        cfg = json.loads(Path(hf_hub_download(HF_REPO, name, repo_type="dataset",
                                              token=api.token)).read_text(encoding="utf-8"))
        if cfg.get("source_commit") != commit:
            raise ValueError("RL checkpoint source commit differs from frozen labels")
    if checkpoint and int(checkpoint.rsplit("-", 1)[-1]) >= TOTAL_STEPS:
        print(f"[rl-watch] learner complete: {checkpoint}; checking frozen evaluation",
              flush=True)
        from scripts import ensure_eval_preview as evaluator
        evaluator.RUN = RUN
        evaluator.FINAL_STEP = TOTAL_STEPS
        evaluator.TARGETS = [TOTAL_STEPS]
        evaluator.KERNEL_SLUG = "eval-ccgavn-rl"
        eval_status = remote_status(api, evaluator.out_prefix(TOTAL_STEPS), files)
        if "INCOMPLETE" in eval_status or "FAILED" in eval_status:
            raise RuntimeError(f"frozen evaluation failed: {eval_status[-300:]}")
        evaluator.main(check_only=check_only)
        return
    status_text = remote_status(api, RUN, files)
    if "FAILED" in status_text:
        raise RuntimeError(f"learner failed: {status_text[-300:]}")
    active = []
    for account in ACCOUNTS:
        status = kernel_status(account, TRAIN_SLUG, listed[account])
        if status in ACTIVE:
            active.append((account, status))
        elif status in ("ERROR", "FAILED"):
            raise RuntimeError(f"{account}: Kaggle learner kernel {status}")
        elif status == "COMPLETE":
            raise RuntimeError(f"{account}: learner kernel completed before "
                               f"the {TOTAL_STEPS}-step checkpoint was persisted")
    if active:
        print(f"[rl-watch] learner active: {active}", flush=True)
        return
    quotas = {account: quota(account) for account in ACCOUNTS}
    eligible = [account for account in ACCOUNTS if quotas[account] > 0]
    if not eligible:
        raise RuntimeError(f"no verified T4 quota across five accounts: {quotas}")
    selected = max(eligible, key=lambda account: quotas[account])
    manifest = manifest_from_states(states)
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    print(f"[rl-watch] learner ready; checkpoint={checkpoint}; quotas={quotas}; "
          f"selected={selected}", flush=True)
    if not check_only:
        push(selected, TRAIN_SLUG, {"mode": "train", "run": RUN,
                                    "steps": TOTAL_STEPS, "manifest_json": payload,
                                    "source_commit": commit}, gpu=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    main(args.check_only)
