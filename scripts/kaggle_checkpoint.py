"""Small, dependency-light checkpoint helpers for Kaggle jobs.

The Kaggle filesystem is disposable. These helpers deliberately upload a
complete local checkpoint directory (weights, optimizer, RNG, config, and
metrics) to a Hugging Face *dataset* repository, matching the persistence
contract in AGENTS.md.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def hf_token(root: Path | None = None) -> str:
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token and root is not None:
        env_file = root / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith(("HF_WRITE_TOKEN=", "HF_TOKEN=")):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        raise RuntimeError("HF_WRITE_TOKEN or HF_TOKEN is required for persistence")
    return token


def api(root: Path | None = None):
    from huggingface_hub import HfApi
    return HfApi(token=hf_token(root))


def upload_checkpoint(api_client, repo_id: str, local_dir: Path,
                      remote_prefix: str, checkpoint_name: str,
                      attempts: int = 3) -> None:
    checkpoint = local_dir / checkpoint_name
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    files = [p for p in checkpoint.rglob("*") if p.is_file()]
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            for path in files:
                remote = f"{remote_prefix.strip('/')}/{checkpoint_name}/{path.relative_to(checkpoint)}"
                api_client.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=remote,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            print(f"[hf] uploaded {checkpoint_name} ({len(files)} files)", flush=True)
            return
        except Exception as exc:  # Kaggle network flaps; retry the whole dir
            last_exc = exc
            print(f"[hf] upload attempt {attempt}/{attempts} failed: {exc}", flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError(f"upload failed after {attempts} attempts") from last_exc


def latest_local_checkpoint(root: Path) -> Path | None:
    candidates = []
    for path in root.glob("checkpoint-*"):
        try:
            candidates.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            continue
    return max(candidates, default=(0, None), key=lambda x: x[0])[1]


def latest_remote_checkpoint(api_client, repo_id: str, remote_prefix: str) -> str | None:
    files = api_client.list_repo_files(repo_id=repo_id, repo_type="dataset")
    prefix = remote_prefix.strip("/") + "/checkpoint-"
    names = set()
    for filename in files:
        if filename.startswith(prefix):
            part = filename[len(remote_prefix.strip("/")) + 1:].split("/", 1)[0]
            if part.startswith("checkpoint-"):
                names.add(part)
    def key(name: str) -> int:
        try:
            return int(name.split("-")[-1])
        except ValueError:
            return -1
    return max(names, key=key) if names else None


def download_latest(api_client, repo_id: str, remote_prefix: str,
                    local_root: Path) -> Path | None:
    from huggingface_hub import hf_hub_download
    name = latest_remote_checkpoint(api_client, repo_id, remote_prefix)
    if name is None:
        print(f"[hf] no remote checkpoint under {remote_prefix}", flush=True)
        return None
    destination = local_root / remote_prefix.strip("/") / name
    destination.mkdir(parents=True, exist_ok=True)
    files = api_client.list_repo_files(repo_id=repo_id, repo_type="dataset")
    prefix = f"{remote_prefix.strip('/')}/{name}/"
    for filename in files:
        if filename.startswith(prefix):
            hf_hub_download(repo_id=repo_id, filename=filename,
                            repo_type="dataset", local_dir=str(local_root),
                            token=api_client.token)
    print(f"[hf] downloaded {name} -> {destination}", flush=True)
    return destination


def write_status(api_client, repo_id: str, remote_prefix: str, status: str) -> None:
    api_client.upload_file(
        path_or_fileobj=status.encode("utf-8"),
        path_in_repo=f"{remote_prefix.strip('/')}/run-status.txt",
        repo_id=repo_id,
        repo_type="dataset",
    )


class UploadTimer:
    def __init__(self, interval_s: float):
        self.interval_s = float(interval_s)
        self.last = 0.0

    def due(self) -> bool:
        return time.time() - self.last >= self.interval_s

    def mark(self) -> None:
        self.last = time.time()


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
