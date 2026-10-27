"""Coordinate frozen-cohort training alongside full-corpus CPU labeling.

The first 200 verified Stockfish shards are frozen as a 1M-root cohort. The
smoke demo and 9M teacher labels prepare in parallel with the remaining CPU
label work; the 2M-step learner starts once the cohort and demo gates pass.
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
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_accounts import env_for_account  # noqa: E402
from scripts.kaggle_checkpoint import api as make_hf_api  # noqa: E402
from scripts.hf_upload import upload_file as upload_hf_file  # noqa: E402
from scripts.tactical_config import (  # noqa: E402
    ACCOUNTS, BASE_RUN, BASE_STEP, COHORT_MANIFEST, COHORT_ROOTS,
    COHORT_SHARD_COUNT, COHORT_TAGS, CREDENTIAL_DATASETS, DATA_PREFIX, DEMO_RUN,
    CPU_BATCH_SESSIONS_PER_ACCOUNT, FINAL_STEP, HF_REPO, INCLUDED_ROOTS,
    MILESTONE_EVERY, ROOTS_PER_SHARD,
    FUNCTIONAL_RUN, MODEL_PARAMETERS, RUN, SEARCHLESS_COMMIT, SHARD_COUNT,
    SOURCE_PREFIX, SOURCE_SHA256,
    STOCKFISH16_COMMIT, milestones,
)

DEMO_PREFIX = "puzzle-pipeline-demo-v1"
DEMO_DATA_PREFIX = "puzzle-action-values-demo-v1"
STATUS_PREFIX = "puzzle-pipeline-v1/status"
WORKER_SLUG = "ccgavn-puzzle-stage"
LABEL_WORKER_SLUG = "ccgavn-label-w"
LABEL_WORKER_COUNT = len(ACCOUNTS) * CPU_BATCH_SESSIONS_PER_ACCOUNT
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
    expected_smoke_error = ["partial run (--max-rows) is not a complete evaluation"]
    reported_checkpoint = Path(str(summary.get("checkpoint", "")))
    if (summary.get("mate", {}).get("total") != 10
            or summary.get("mate", {}).get("expected") != 10
            or summary.get("puzzles", {}).get("total") != 20
            or summary.get("puzzles", {}).get("expected") != 20
            or not summary.get("partial")
            or summary.get("complete")
            or summary.get("returncode") != 1
            or summary.get("errors") != expected_smoke_error
            or reported_checkpoint.name != f"checkpoint-{BASE_STEP + 2}"
            or reported_checkpoint.parent.name != DEMO_RUN):
        raise ValueError("demo evaluation artifacts do not match the declared smoke contract")
    config = remote_json(client, f"{checkpoint}/config.json")
    if (config.get("stage_start_step") != BASE_STEP
            or config.get("model_version") != "v2"
            or config.get("parameter_count") != MODEL_PARAMETERS):
        raise ValueError("demo learner checkpoint identity/parameter count mismatch")
    return True, str(config["source_commit"])


def frozen_rows(client, files: set[str], tags: tuple[int, ...] | list[int],
                require_teacher: bool = True) -> dict[str, int]:
    """Validate one explicit, immutable shard set before training."""
    raw = complete_shards(files)
    taught = complete_shards(files, teacher=True)
    tags = tuple(int(tag) for tag in tags)
    if not tags or len(set(tags)) != len(tags):
        raise ValueError("training cohort shard tags are empty or duplicated")
    missing = set(tags) - raw
    if missing:
        raise ValueError(f"training cohort lacks raw shards: {sorted(missing)[:8]}")
    if require_teacher and set(tags) - taught:
        raise ValueError(f"training cohort lacks teacher shards: {sorted(set(tags) - taught)[:8]}")

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
        result = list(pool.map(fetch, tags))
    expected_total = sum(min(ROOTS_PER_SHARD,
                             INCLUDED_ROOTS - tag * ROOTS_PER_SHARD)
                         for tag in tags)
    if sum(item[2] for item in result) != expected_total:
        raise ValueError("frozen cohort root total changed")
    return {tag: rows for tag, rows, _ in result}


def frozen_shard_digests(client, tags: tuple[int, ...] | list[int]) -> dict[str, str]:
    def fetch(tag: int):
        meta = remote_json(client, f"{DATA_PREFIX}/shard-{tag:05d}/meta.json")
        digest = str(meta.get("train_set_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"shard {tag:05d} has no valid label SHA256")
        return f"{tag:05d}", digest
    with ThreadPoolExecutor(max_workers=16) as pool:
        return dict(pool.map(fetch, tags))


def frozen_cohort(client, files: set[str], raw: set[int],
                  persist: bool = True) -> dict | None:
    """Create or read the first 1M-root cohort; never extend it after training."""
    if COHORT_MANIFEST in files:
        cohort = remote_json(client, COHORT_MANIFEST)
        if (cohort.get("cohort_id") != "first-1m-v1"
                or cohort.get("source_sha256") != SOURCE_SHA256
                or tuple(cohort.get("tags", ())) != COHORT_TAGS
                or int(cohort.get("roots", 0)) != COHORT_ROOTS
                or set(cohort.get("rows", {})) != {f"{tag:05d}" for tag in COHORT_TAGS}
                or set(cohort.get("train_set_sha256", {})) !=
                   {f"{tag:05d}" for tag in COHORT_TAGS}):
            raise ValueError("frozen training cohort manifest identity mismatch")
        return cohort
    if not set(COHORT_TAGS) <= raw:
        return None
    rows = frozen_rows(client, files, COHORT_TAGS, require_teacher=False)
    digests = frozen_shard_digests(client, COHORT_TAGS)
    cohort = {
        "cohort_id": "first-1m-v1", "data_prefix": DATA_PREFIX,
        "source_sha256": SOURCE_SHA256, "engine": "Stockfish 16",
        "seconds_per_action": 0.05, "roots": COHORT_ROOTS,
        "tags": list(COHORT_TAGS), "rows": rows, "train_set_sha256": digests,
        "training_run": RUN, "base_run": BASE_RUN, "base_step": BASE_STEP,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if persist:
        upload_hf_file(client, HF_REPO,
                       (json.dumps(cohort, sort_keys=True,
                                   separators=(",", ":")) + "\n").encode(),
                       COHORT_MANIFEST,
                       commit_message="Freeze first 1M-root training cohort")
    return cohort


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
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines()
             if line.strip()]
    return lines[-1] if lines else ""


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


def label_worker_slug(worker_index: int) -> str:
    return f"{LABEL_WORKER_SLUG}{worker_index:02d}"


def worker_config(phase: str, index: int, commit: str, rows=None,
                  label_worker_index: int | None = None,
                  teacher_tags: list[int] | None = None,
                  cohort_sha256: dict[str, str] | None = None) -> dict:
    config = {"phase": phase, "account_index": index,
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
    if label_worker_index is not None:
        config["label_worker_index"] = label_worker_index
        config["label_worker_count"] = LABEL_WORKER_COUNT
    if teacher_tags is not None:
        config["teacher_tags"] = teacher_tags
    if cohort_sha256 is not None:
        config["cohort_sha256"] = cohort_sha256
    return config


def blocked_failure(client, files: set[str], phase: str, index: int,
                    commit: str, worker_index: int | None = None) -> bool:
    suffix = (f"worker-{worker_index:02d}" if worker_index is not None
              else f"acct-{index}")
    status = remote_text(client, files, f"{STATUS_PREFIX}/{phase}-{suffix}.txt")
    if not status.startswith("FAILED"):
        return False
    recorded = re.search(r"source_commit=([0-9a-f]{40})", status)
    # A failure from an older worker revision must not block the corrected
    # commit. If the marker predates source-commit stamping, fail closed.
    return recorded is None or recorded.group(1) == commit


def record_launch(client, phase: str, index: int, commit: str,
                  worker_index: int | None = None) -> None:
    worker_suffix = (f"worker-{worker_index:02d}" if worker_index is not None
                     else f"acct-{index}")
    upload_hf_file(client, HF_REPO,
                   f"LAUNCHED {phase} source_commit={commit}".encode(),
                   f"{STATUS_PREFIX}/{phase}-{worker_suffix}.txt",
                   commit_message=f"Launch tactical {phase} worker {index}",
                   attempts=4, max_elapsed_s=300.0)


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
    slugs = {}
    with ThreadPoolExecutor(max_workers=len(ACCOUNTS)) as pool:
        for account, refs in zip(ACCOUNTS, pool.map(account_slugs, ACCOUNTS)):
            slugs[account] = refs
    status_tasks = {}
    for index, account in enumerate(ACCOUNTS):
        status_tasks[(account, "worker")] = (account, WORKER_SLUG)
        status_tasks[(account, "eval")] = (account, EVAL_SLUG)
        local_labels = {}
        for local_slot in range(CPU_BATCH_SESSIONS_PER_ACCOUNT):
            worker_index = index * CPU_BATCH_SESSIONS_PER_ACCOUNT + local_slot
            status_tasks[(account, f"label-{worker_index}")] = (
                account, label_worker_slug(worker_index))
    def query_status(task):
        account, slug = task
        return kernel_status(account, slug, slugs[account])
    with ThreadPoolExecutor(max_workers=LABEL_WORKER_COUNT + 2 * len(ACCOUNTS)) as pool:
        queried = list(pool.map(query_status, status_tasks.values()))
    statuses = {account: {"labels": {}} for account in ACCOUNTS}
    for (account, kind), status in zip(status_tasks, queried):
        if kind.startswith("label-"):
            worker_index = int(kind.split("-", 1)[1])
            statuses[account]["labels"][worker_index] = status
        else:
            statuses[account][kind] = status
    active_worker = {a for a in ACCOUNTS
                     if any(word in statuses[a]["worker"] for word in ACTIVE)}
    active_eval = {a for a in ACCOUNTS
                   if any(word in statuses[a]["eval"] for word in ACTIVE)}
    commit = demo_commit or git_head()
    raw = complete_shards(files)
    failed_workers = []
    active_label_workers = []
    for worker_index in range(LABEL_WORKER_COUNT):
        account_index = worker_index // CPU_BATCH_SESSIONS_PER_ACCOUNT
        account = ACCOUNTS[account_index]
        status = statuses[account]["labels"][worker_index]
        if any(word in status for word in FAILED):
            marker_path = f"{STATUS_PREFIX}/labels-worker-{worker_index:02d}.txt"
            marker = remote_text(client, files, marker_path)
            recorded = re.search(r"source_commit=([0-9a-f]{40})", marker)
            if not marker or not recorded or recorded.group(1) == commit:
                failed_workers.append({"worker": worker_index, "account": account})
        elif any(word in status for word in ACTIVE):
            active_label_workers.append(worker_index)
    if failed_workers:
        return {"phase": "blocked", "reason": "Stockfish label worker failed",
                "failed_workers": failed_workers, "statuses": statuses}

    # Keep the CPU label pool running while GPU stages work on a frozen,
    # auditable one-million-root snapshot. New shards are deliberately excluded
    # from this run and remain available for a later full-corpus stage.
    if len(raw) < SHARD_COUNT:
        pushed = []
        blocked = []
        for worker_index in range(LABEL_WORKER_COUNT):
            index = worker_index // CPU_BATCH_SESSIONS_PER_ACCOUNT
            account = ACCOUNTS[index]
            assigned = set(range(worker_index, SHARD_COUNT, LABEL_WORKER_COUNT))
            if (assigned <= raw
                    or worker_index in active_label_workers):
                continue
            if blocked_failure(client, files, "labels", index, commit,
                               worker_index=worker_index):
                blocked.append({"worker": worker_index, "account": account})
                continue
            if not check_only:
                record_launch(client, "labels", index, commit,
                              worker_index=worker_index)
                push(account, label_worker_slug(worker_index), WORKER_TEMPLATE,
                     "__TACTICAL_CONFIG__",
                     worker_config("labels", index, commit,
                                   label_worker_index=worker_index), gpu=False)
            pushed.append({"worker": worker_index, "account": account})
        label_state = {"ready": len(raw), "target": SHARD_COUNT,
                       "active": sorted(active_label_workers), "pushed": pushed,
                       "blocked": blocked, "workers_per_account": CPU_BATCH_SESSIONS_PER_ACCOUNT,
                       "worker_target": LABEL_WORKER_COUNT}
    else:
        label_state = {"ready": len(raw), "target": SHARD_COUNT,
                       "active": [], "pushed": [], "blocked": []}

    cohort = frozen_cohort(client, files, raw, persist=not check_only)
    if cohort is None:
        return {"phase": "labels", "labels": label_state,
                "ready": len(raw), "target": SHARD_COUNT,
                "cohort_ready": False, "demo": "waiting for frozen cohort",
                "statuses": statuses, "check_only": check_only}
    cohort_tags = [int(tag) for tag in cohort["tags"]]
    cohort_taught = complete_shards(files, teacher=True) & set(cohort_tags)
    with ThreadPoolExecutor(max_workers=len(ACCOUNTS)) as pool:
        quotas = dict(zip(ACCOUNTS, pool.map(gpu_quota, ACCOUNTS)))

    # The 2-step smoke run remains a gate for the learner, but does not block
    # parallel 9M teacher preparation for the frozen cohort.
    demo_pushed = None
    if not demo_ok:
        phase_errors = current_worker_failures(client, files, "demo", statuses,
                                               commit)
        if phase_errors:
            return {"phase": "blocked", "reason": "end-to-end demo failed",
                    "accounts": sorted(phase_errors), "labels": label_state,
                    "cohort": {"id": cohort["cohort_id"], "roots": cohort["roots"]},
                    "statuses": statuses}
        if not active_worker:
            usable = [a for a in ACCOUNTS if quotas[a] > 1.0]
            if usable:
                account = max(usable, key=lambda a: quotas[a])
                index = ACCOUNTS.index(account)
                if not check_only:
                    record_launch(client, "demo", index, commit)
                    push(account, WORKER_SLUG, WORKER_TEMPLATE,
                         "__TACTICAL_CONFIG__", worker_config("demo", index, commit),
                         gpu=True)
                demo_pushed = account
                active_worker.add(account)

    teacher_pushed = []
    teacher_blocked = []
    teacher_errors = current_worker_failures(client, files, "teacher", statuses, commit)
    if teacher_errors:
        return {"phase": "blocked", "reason": "9M cohort teacher worker failed",
                "accounts": sorted(teacher_errors), "labels": label_state,
                "cohort": {"id": cohort["cohort_id"], "roots": cohort["roots"]},
                "statuses": statuses}
    for index, account in enumerate(ACCOUNTS):
        assigned = cohort_tags[index::len(ACCOUNTS)]
        if not assigned or set(assigned) <= cohort_taught or account in active_worker:
            continue
        if quotas[account] <= 1.0:
            teacher_blocked.append(account)
            continue
        if blocked_failure(client, files, "teacher", index, commit):
            teacher_blocked.append(account)
            continue
        if not check_only:
            record_launch(client, "teacher", index, commit)
            push(account, WORKER_SLUG, WORKER_TEMPLATE, "__TACTICAL_CONFIG__",
                 worker_config("teacher", index, commit, teacher_tags=assigned),
                 gpu=True)
        teacher_pushed.append(account)
        active_worker.add(account)

    cohort_info = {"id": cohort["cohort_id"], "roots": cohort["roots"],
                   "shards": len(cohort_tags), "teacher_ready": len(cohort_taught),
                   "teacher_target": len(cohort_tags)}
    if len(cohort_taught) < len(cohort_tags):
        phase = "labels+demo+teacher" if label_state["active"] else "demo+teacher"
        return {"phase": phase, "labels": label_state, "cohort": cohort_info,
                "demo": "complete" if demo_ok else "running or queued",
                "demo_pushed": demo_pushed, "teacher_pushed": teacher_pushed,
                "teacher_blocked": teacher_blocked, "active": sorted(active_worker),
                "statuses": statuses, "check_only": check_only}
    if not demo_ok:
        return {"phase": "labels+demo" if label_state["active"] else "demo",
                "labels": label_state, "cohort": cohort_info,
                "demo": "running or queued", "active": sorted(active_worker),
                "statuses": statuses, "check_only": check_only}

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
                    "accounts": sorted(eval_errors), "step": latest,
                    "pending_evals": pending, "statuses": statuses}
        with ThreadPoolExecutor(max_workers=len(ACCOUNTS)) as pool:
            quotas = dict(zip(ACCOUNTS, pool.map(gpu_quota, ACCOUNTS)))
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
            active_eval.add(account)
    if latest >= FINAL_STEP:
        phase = "complete" if eval_complete(client, files, FINAL_STEP) else "evaluation"
        if label_state["active"] and phase == "complete":
            phase = "complete+labels"
        return {"phase": phase, "step": latest, "pending_evals": pending,
                "eval_account": launched_eval, "labels": label_state,
                "cohort": cohort_info, "statuses": statuses}
    if active_worker:
        phase = "training+labels" if label_state["active"] else "training"
        return {"phase": phase, "step": latest,
                "active": sorted(active_worker), "pending_evals": pending,
                "eval_account": launched_eval, "labels": label_state,
                "cohort": cohort_info, "statuses": statuses}
    if any(blocked_failure(client, files, "train", i, commit)
           for i in range(len(ACCOUNTS))):
        return {"phase": "blocked", "reason": "learner worker failed", "step": latest,
                "statuses": statuses}
    phase_errors = current_worker_failures(client, files, "train", statuses,
                                           commit)
    if phase_errors:
        return {"phase": "blocked", "reason": "learner Kaggle worker failed",
                "accounts": sorted(phase_errors), "step": latest,
                "statuses": statuses}
    with ThreadPoolExecutor(max_workers=len(ACCOUNTS)) as pool:
        quotas = dict(zip(ACCOUNTS, pool.map(gpu_quota, ACCOUNTS)))
    usable = [a for a in ACCOUNTS if a not in active_eval and quotas[a] > 1.0]
    if not usable:
        return {"phase": "waiting", "reason": "no free GPU account", "quotas": quotas,
                "labels": label_state, "cohort": cohort_info,
                "statuses": statuses}
    rows = frozen_rows(client, files, cohort_tags)
    if frozen_shard_digests(client, cohort_tags) != cohort.get("train_set_sha256"):
        raise ValueError("labeled shard content changed after cohort freeze")
    account = max(usable, key=lambda a: quotas[a])
    if not check_only:
        index = ACCOUNTS.index(account)
        record_launch(client, "train", index, commit)
        push(account, WORKER_SLUG, WORKER_TEMPLATE, "__TACTICAL_CONFIG__",
             worker_config("train", index, commit, rows,
                           cohort_sha256=cohort["train_set_sha256"]), gpu=True)
    phase = "training+labels" if label_state["active"] else "training"
    return {"phase": phase, "step": latest, "target": FINAL_STEP,
            "account": account, "pending_evals": pending,
            "eval_account": launched_eval, "check_only": check_only,
            "labels": label_state, "cohort": cohort_info,
            "statuses": statuses}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(main(args.check_only), sort_keys=True), flush=True)
