"""GitHub Actions supervisor for the frozen two-million-step puzzle stage.

Stages advance only after remote artifacts pass their declared gates:
small end-to-end demo -> all Stockfish rows -> all 9M teacher distributions ->
2M-step learner, with checkpoint-scoped evaluation every 50k steps.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_accounts import env_for_account  # noqa: E402
from scripts.kaggle_checkpoint import api as make_hf_api  # noqa: E402
from scripts.tactical_config import (  # noqa: E402
    ACCOUNTS, BASE_RUN, BASE_STEP, CREDENTIAL_DATASETS, DATA_PREFIX, DEMO_RUN,
    FINAL_STEP, HF_REPO, INCLUDED_ROOTS, MILESTONE_EVERY, ROOTS_PER_SHARD,
    FUNCTIONAL_RUN, MODEL_PARAMETERS, RUN, SEARCHLESS_COMMIT, SHARD_COUNT,
    SOURCE_PREFIX, SOURCE_SHA256,
    STOCKFISH16_COMMIT, milestones,
)

DEMO_PREFIX = "puzzle-pipeline-demo-v1"
DEMO_DATA_PREFIX = "puzzle-action-values-demo-v1"
STATUS_PREFIX = "puzzle-pipeline-v1/status"
WORKER_SLUG = "ccgavn-puzzle-stage"
EVAL_SLUG = "ccgavn-puzzle-eval"
ACTIVE = ("RUNNING", "QUEUED", "PENDING")
FAILED = ("ERROR", "FAILED", "CANCELLED", "CANCELED")
WORKER_TEMPLATE = ROOT / "kernels" / "tactical" / "worker.py"
EVAL_TEMPLATE = ROOT / "kernels" / "eval-preview" / "eval_preview.py"


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   text=True).strip()


def remote_json(client, path: str) -> dict:
    local = hf_hub_download(HF_REPO, path, repo_type="dataset", token=client.token)
    return json.loads(Path(local).read_text(encoding="utf-8"))


def remote_text(client, files: set[str], path: str) -> str:
    if path not in files:
        return ""
    local = hf_hub_download(HF_REPO, path, repo_type="dataset", token=client.token)
    return Path(local).read_text(encoding="utf-8")


def complete_shards(files: set[str], teacher: bool = False) -> set[int]:
    result = set()
    for tag in range(SHARD_COUNT):
        prefix = f"{DATA_PREFIX}/shard-{tag:05d}"
        if (f"{prefix}/train_set.npz" in files
                and f"{prefix}/meta.json" in files
                and (not teacher or f"{prefix}/teacher_logp.npy" in files)):
            result.add(tag)
    return result


def demo_complete(client, files: set[str]) -> tuple[bool, str | None]:
    checkpoint = f"{DEMO_RUN}/checkpoint-{BASE_STEP + 2}"
    if (f"{DEMO_PREFIX}/eval-summary.json" not in files
            or f"{checkpoint}/state.pt" not in files
            or f"{checkpoint}/config.json" not in files):
        return False, None
    summary = remote_json(client, f"{DEMO_PREFIX}/eval-summary.json")
    if (summary.get("mate", {}).get("total") != 10
            or summary.get("puzzles", {}).get("total") != 20
            or summary.get("checkpoint") != checkpoint):
        raise ValueError("demo evaluation artifacts do not match the declared smoke contract")
    config = remote_json(client, f"{checkpoint}/config.json")
    if (config.get("stage_start_step") != BASE_STEP
            or config.get("model_version") != "v2"
            or config.get("parameter_count") != MODEL_PARAMETERS):
        raise ValueError("demo learner checkpoint identity/parameter count mismatch")
    return True, str(config["source_commit"])


def frozen_rows(client, files: set[str]) -> dict[str, int]:
    """Validate all completed raw shards before the learner sees any of them."""
    if len(complete_shards(files, teacher=True)) != SHARD_COUNT:
        raise ValueError("teacher corpus is not fully complete")

    def fetch(tag: int):
        meta = remote_json(client, f"{DATA_PREFIX}/shard-{tag:05d}/meta.json")
        identity = meta.get("identity") or {}
        expected_roots = min(ROOTS_PER_SHARD,
                             INCLUDED_ROOTS - tag * ROOTS_PER_SHARD)
        if (not meta.get("complete") or meta.get("roots") != expected_roots
                or meta.get("target") != expected_roots
                or identity.get("source_sha256") != SOURCE_SHA256
                or identity.get("engine") != "Stockfish 16"
                or identity.get("seconds_per_action") != 0.05
                or identity.get("max_roots") != 0
                or identity.get("shard") != tag):
            raise ValueError(f"shard {tag:05d} failed frozen identity gate")
        rows = int(meta.get("rows", 0))
        if rows < expected_roots:
            raise ValueError(f"shard {tag:05d} has too few legal actions")
        return f"{tag:05d}", rows, expected_roots

    with ThreadPoolExecutor(max_workers=16) as pool:
        result = list(pool.map(fetch, range(SHARD_COUNT)))
    if sum(item[2] for item in result) != INCLUDED_ROOTS:
        raise ValueError("frozen corpus root total changed")
    return {tag: rows for tag, rows, _ in result}


def account_slugs(account: str) -> set[str]:
    result = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "list",
                             "--mine", "--page-size", "200", "--csv"],
                            env=env_for_account(account), capture_output=True,
                            text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"{account}: Kaggle listing failed: {result.stderr[-300:]}")
    lines = result.stdout.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("ref,title,")), None)
    if start is None:
        raise RuntimeError(f"{account}: Kaggle list schema changed")
    return {row["ref"] for row in csv.DictReader(io.StringIO("\n".join(lines[start:])))
            if row.get("ref")}


def kernel_status(account: str, slug: str, slugs: set[str]) -> str:
    ref = f"{account}/{slug}"
    if ref not in slugs:
        return "ABSENT"
    result = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", ref],
                            env=env_for_account(account), capture_output=True,
                            text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"{ref}: status failed: {(result.stdout + result.stderr)[-300:]}")
    return (result.stdout + result.stderr).strip()


def gpu_quota(account: str) -> float:
    result = subprocess.run([sys.executable, "-m", "kaggle", "quota"],
                            env=env_for_account(account), capture_output=True,
                            text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"{account}: quota failed: {result.stderr[-300:]}")
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "GPU":
            return float(fields[2].rstrip("h"))
    raise RuntimeError(f"{account}: GPU quota missing from response")


def push(account: str, slug: str, template: Path, marker: str,
         config: dict, gpu: bool) -> None:
    source = template.read_text(encoding="utf-8")
    placeholder = f'"""{marker}"""'
    if source.count(placeholder) != 1:
        raise ValueError(f"kernel template {template} lost its unique config marker")
    rendered = source.replace(placeholder, '"""' + json.dumps(config, sort_keys=True) + '"""')
    with tempfile.TemporaryDirectory(prefix=f"tactical_{account}_") as tmp:
        folder = Path(tmp)
        file = folder / template.name
        file.write_text(rendered, encoding="utf-8")
        metadata = {"id": f"{account}/{slug}", "title": slug,
                    "code_file": file.name, "language": "python",
                    "kernel_type": "script", "is_private": True,
                    "enable_gpu": gpu, "enable_tpu": False,
                    "enable_internet": True,
                    "dataset_sources": [CREDENTIAL_DATASETS[account]],
                    "competition_sources": [], "kernel_sources": []}
        if gpu:
            metadata["machine_shape"] = "NvidiaTeslaT4"
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        result = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                                 "-p", str(folder)], env=env_for_account(account),
                                capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise RuntimeError(f"{account}/{slug} push failed: "
                               f"{(result.stdout + result.stderr)[-500:]}")
        print(f"[tactical] pushed {slug} on {account}", flush=True)


def worker_config(phase: str, index: int, commit: str, rows=None) -> dict:
    return {"phase": phase, "account_index": index,
            "source_commit": commit, "searchless_commit": SEARCHLESS_COMMIT,
            "stockfish_commit": STOCKFISH16_COMMIT,
            "source_sha256": SOURCE_SHA256, "source_prefix": SOURCE_PREFIX,
            "data_prefix": DATA_PREFIX, "demo_data_prefix": DEMO_DATA_PREFIX,
            "demo_run": DEMO_RUN, "demo_prefix": DEMO_PREFIX,
            "status_prefix": STATUS_PREFIX, "hf_repo": HF_REPO,
            "base_run": BASE_RUN, "base_step": BASE_STEP,
            "functional_run": FUNCTIONAL_RUN,
            "model_version": "v2", "parameter_count": MODEL_PARAMETERS,
            "run": RUN, "final_step": FINAL_STEP,
            "shard_count": SHARD_COUNT, "rows": rows or {}}


def blocked_failure(client, files: set[str], phase: str, index: int, commit: str) -> bool:
    status = remote_text(client, files, f"{STATUS_PREFIX}/{phase}-acct-{index}.txt")
    return status.startswith("FAILED") and f"source_commit={commit}" in status


def record_launch(client, phase: str, index: int, commit: str) -> None:
    client.upload_file(path_or_fileobj=f"LAUNCHED {phase} source_commit={commit}".encode(),
                       path_in_repo=f"{STATUS_PREFIX}/{phase}-acct-{index}.txt",
                       repo_id=HF_REPO, repo_type="dataset",
                       commit_message=f"Launch tactical {phase} worker {index}")


def current_worker_failures(client, files: set[str], phase: str,
                            statuses: dict, commit: str,
                            worker_kind: str = "worker",
                            indices: range = range(len(ACCOUNTS))) -> list[str]:
    """Block only on errors belonging to this phase and source revision.

    Kaggle keeps the last kernel status after a failed run. Once code has been
    corrected, that old ERROR must not prevent the new revision from launching.
    A missing marker remains a hard stop because the failed version is unknown.
    """
    failed = []
    for index in indices:
        account = ACCOUNTS[index]
        status = statuses[account][worker_kind]
        if not any(word in status for word in FAILED):
            continue
        marker = remote_text(client, files,
                             f"{STATUS_PREFIX}/{phase}-acct-{index}.txt")
        if not marker:
            # This account's failed kernel belongs to an earlier phase; only
            # a phase-specific launch marker makes the failure actionable here.
            continue
        recorded = re.search(r"source_commit=([0-9a-f]{40})", marker)
        if not recorded or recorded.group(1) == commit:
            failed.append(account)
    return failed


def eval_prefix(step: int) -> str:
    suffix = "final" if step == FINAL_STEP else "preview"
    return f"eval-results/{RUN}-{step // 1000}k-{suffix}"


def eval_complete(client, files: set[str], step: int) -> bool:
    name = f"{eval_prefix(step)}/eval-summary.json"
    if name not in files:
        return False
    summary = remote_json(client, name)
    return (summary.get("complete") is True and summary.get("returncode") == 0
            and summary.get("checkpoint") == f"{RUN}/checkpoint-{step}"
            and summary.get("mate", {}).get("total") == 4000
            and summary.get("puzzles", {}).get("total") == 10000)


def main(check_only: bool = False) -> dict:
    client = make_hf_api(ROOT)
    files = set(client.list_repo_files(HF_REPO, repo_type="dataset"))
    required = {f"{SOURCE_PREFIX}/source.csv.zst",
                f"{SOURCE_PREFIX}/manifest.json",
                f"{BASE_RUN}/checkpoint-{BASE_STEP}/state.pt",
                f"{BASE_RUN}/checkpoint-{BASE_STEP}/config.json"}
    missing = sorted(required - files)
    if missing:
        return {"phase": "waiting", "missing": missing}
    demo_ok, demo_commit = demo_complete(client, files)
    statuses = {}
    slugs = {}
    for index, account in enumerate(ACCOUNTS):
        slugs[account] = account_slugs(account)
        statuses[account] = {
            "worker": kernel_status(account, WORKER_SLUG, slugs[account]),
            "eval": kernel_status(account, EVAL_SLUG, slugs[account]),
        }
    active_worker = {a for a in ACCOUNTS
                     if any(word in statuses[a]["worker"] for word in ACTIVE)}
    active_eval = {a for a in ACCOUNTS
                   if any(word in statuses[a]["eval"] for word in ACTIVE)}
    commit = demo_commit or git_head()
    if not demo_ok:
        if any(blocked_failure(client, files, "demo", i, commit)
               for i in range(len(ACCOUNTS))):
            return {"phase": "blocked", "reason": "demo worker failed", "statuses": statuses}
        phase_errors = current_worker_failures(client, files, "demo", statuses,
                                               commit)
        if phase_errors:
            return {"phase": "blocked", "reason": "demo Kaggle worker failed",
                    "accounts": sorted(phase_errors), "statuses": statuses}
        if active_worker:
            return {"phase": "demo", "active": sorted(active_worker)}
        quotas = {a: gpu_quota(a) for a in ACCOUNTS}
        usable = [a for a in ACCOUNTS if quotas[a] > 1.0]
        if not usable:
            return {"phase": "waiting", "reason": "no positive GPU quota", "quotas": quotas}
        account = max(usable, key=lambda a: quotas[a])
        if not check_only:
            index = ACCOUNTS.index(account)
            record_launch(client, "demo", index, commit)
            push(account, WORKER_SLUG, WORKER_TEMPLATE, "__TACTICAL_CONFIG__",
                 worker_config("demo", index, commit), gpu=True)
        return {"phase": "demo", "target_account": account, "check_only": check_only}

    raw = complete_shards(files)
    if len(raw) < SHARD_COUNT:
        phase_errors = current_worker_failures(client, files, "labels", statuses, commit)
        if phase_errors:
            return {"phase": "blocked", "reason": "Stockfish label worker failed",
                    "accounts": sorted(phase_errors), "statuses": statuses}
        pushed = []
        blocked = []
        for index, account in enumerate(ACCOUNTS):
            assigned = set(range(index, SHARD_COUNT, len(ACCOUNTS)))
            if assigned <= raw or account in active_worker:
                continue
            if blocked_failure(client, files, "labels", index, commit):
                blocked.append(account)
                continue
            if not check_only:
                record_launch(client, "labels", index, commit)
                push(account, WORKER_SLUG, WORKER_TEMPLATE, "__TACTICAL_CONFIG__",
                     worker_config("labels", index, commit), gpu=False)
            pushed.append(account)
        return {"phase": "labels", "ready": len(raw), "target": SHARD_COUNT,
                "active": sorted(active_worker), "pushed": pushed, "blocked": blocked,
                "check_only": check_only}

    taught = complete_shards(files, teacher=True)
    if len(taught) < SHARD_COUNT:
        phase_errors = current_worker_failures(client, files, "teacher", statuses, commit)
        if phase_errors:
            return {"phase": "blocked", "reason": "9M teacher worker failed",
                    "accounts": sorted(phase_errors), "statuses": statuses}
        quotas = {a: gpu_quota(a) for a in ACCOUNTS}
        pushed = []
        blocked = []
        for index, account in enumerate(ACCOUNTS):
            assigned = set(range(index, SHARD_COUNT, len(ACCOUNTS)))
            if assigned <= taught or account in active_worker or quotas[account] <= 1.0:
                continue
            if blocked_failure(client, files, "teacher", index, commit):
                blocked.append(account)
                continue
            if not check_only:
                record_launch(client, "teacher", index, commit)
                push(account, WORKER_SLUG, WORKER_TEMPLATE, "__TACTICAL_CONFIG__",
                     worker_config("teacher", index, commit), gpu=True)
            pushed.append(account)
        return {"phase": "teacher", "ready": len(taught), "target": SHARD_COUNT,
                "active": sorted(active_worker), "pushed": pushed, "blocked": blocked,
                "quotas": quotas, "check_only": check_only}

    steps = sorted({int(match.group(1)) for name in files
                    if (match := re.fullmatch(rf"{re.escape(RUN)}/checkpoint-(\d+)/state\.pt", name))
                    and f"{RUN}/checkpoint-{match.group(1)}/config.json" in files})
    latest = steps[-1] if steps else BASE_STEP
    reached = [step for step in milestones() if step in steps]
    pending = [step for step in reached if not eval_complete(client, files, step)]
    launched_eval = None
    if pending and not active_eval:
        eval_errors = current_worker_failures(client, files, "eval", statuses,
                                              commit, worker_kind="eval")
        if eval_errors:
            return {"phase": "blocked", "reason": "evaluation worker failed",
                    "accounts": sorted(eval_errors), "step": latest, "pending_evals": pending}
        quotas = {a: gpu_quota(a) for a in ACCOUNTS}
        usable = [a for a in ACCOUNTS if a not in active_worker and quotas[a] > 1.0]
        if usable:
            account = max(usable, key=lambda a: quotas[a])
            config = {"hf_repo": HF_REPO, "run": RUN,
                      "targets": milestones(), "final_step": FINAL_STEP,
                      "source_commit": commit,
                      "searchless_commit": SEARCHLESS_COMMIT,
                      "expect_mate_rows": 4000, "expect_puzzles": 10000}
            if not check_only:
                index = ACCOUNTS.index(account)
                record_launch(client, "eval", index, commit)
                push(account, EVAL_SLUG, EVAL_TEMPLATE,
                     "__EVALPREVIEW_CONFIG__", config, gpu=True)
            launched_eval = account
    if latest >= FINAL_STEP:
        return {"phase": "complete" if eval_complete(client, files, FINAL_STEP) else "evaluation",
                "step": latest, "pending_evals": pending, "eval_account": launched_eval}
    if active_worker:
        return {"phase": "training", "step": latest,
                "active": sorted(active_worker), "pending_evals": pending,
                "eval_account": launched_eval}
    if any(blocked_failure(client, files, "train", i, commit)
           for i in range(len(ACCOUNTS))):
        return {"phase": "blocked", "reason": "learner worker failed", "step": latest}
    phase_errors = current_worker_failures(client, files, "train", statuses,
                                           commit)
    if phase_errors:
        return {"phase": "blocked", "reason": "learner Kaggle worker failed",
                "accounts": sorted(phase_errors), "step": latest}
    quotas = {a: gpu_quota(a) for a in ACCOUNTS}
    usable = [a for a in ACCOUNTS if a not in active_eval and quotas[a] > 1.0]
    if not usable:
        return {"phase": "waiting", "reason": "no free GPU account", "quotas": quotas}
    rows = frozen_rows(client, files)
    account = max(usable, key=lambda a: quotas[a])
    if not check_only:
        index = ACCOUNTS.index(account)
        record_launch(client, "train", index, commit)
        push(account, WORKER_SLUG, WORKER_TEMPLATE, "__TACTICAL_CONFIG__",
             worker_config("train", index, commit, rows), gpu=True)
    return {"phase": "training", "step": latest, "target": FINAL_STEP,
            "account": account, "pending_evals": pending,
            "eval_account": launched_eval, "check_only": check_only}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(main(args.check_only), sort_keys=True), flush=True)
