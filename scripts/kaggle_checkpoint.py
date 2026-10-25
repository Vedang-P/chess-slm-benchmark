"""Small, dependency-light checkpoint helpers for Kaggle jobs.

The Kaggle filesystem is disposable. These helpers deliberately upload a
complete local checkpoint directory (weights, optimizer, RNG, config, and
metrics) to a Hugging Face *dataset* repository, matching the persistence
contract in AGENTS.md.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


def parallel_download(url: str, path: Path, progress=None, stage: str = "download",
                      connections: int = 4) -> None:
    """Download byte ranges concurrently; writes into one preallocated file."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        size = path.stat().st_size
        if progress:
            progress.update(stage, force=True, bytes_done=size, bytes_total=size, pct=100)
        return
    headers = {"User-Agent": "chess-slm-benchmark/1.0"}
    request = urllib.request.Request(url, method="HEAD", headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
    if total <= 0:
        raise RuntimeError(f"missing Content-Length for {url}")
    workers = max(1, min(connections, total // (1 << 20) or 1))
    ranges = [(i * total // workers, (i + 1) * total // workers - 1)
              for i in range(workers)]
    part = Path(str(path) + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    part.unlink(missing_ok=True)
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, total)
    done = 0
    lock = threading.Lock()

    def fetch(start: int, end: int) -> None:
        nonlocal done
        local = 0
        last_exc = None
        for attempt in range(1, 5):
            try:
                req = urllib.request.Request(url, headers={**headers,
                                                           "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as src:
                    if src.status != 206:
                        raise RuntimeError(f"range request returned HTTP {src.status}")
                    pos = start
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        os.pwrite(fd, chunk, pos)
                        pos += len(chunk)
                        local += len(chunk)
                        with lock:
                            done += len(chunk)
                if local != end - start + 1:
                    raise RuntimeError(f"short range {start}-{end}: {local} bytes")
                return
            except Exception as exc:
                last_exc = exc
                with lock:
                    done -= local
                local = 0
                print(f"[download] range {start}-{end} attempt {attempt}/4 failed: {exc}",
                      flush=True)
                time.sleep(10 * attempt)
        raise RuntimeError(f"range {start}-{end} failed") from last_exc

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, start, end) for start, end in ranges]
            last = time.monotonic()
            while not all(future.done() for future in futures):
                time.sleep(1)
                if progress and time.monotonic() - last >= 10:
                    progress.update(stage, bytes_done=done, bytes_total=total,
                                    pct=100 * done / total)
                    last = time.monotonic()
            for future in futures:
                future.result()
        os.close(fd)
        fd = None
        if done != total:
            raise RuntimeError(f"download incomplete: {done}/{total} bytes")
        part.replace(path)
    except BaseException:
        if fd is not None:
            os.close(fd)
        part.unlink(missing_ok=True)
        raise
    if progress:
        progress.update(stage, force=True, bytes_done=total, bytes_total=total, pct=100)


def upload_checkpoint(api_client, repo_id: str, local_dir: Path,
                      remote_prefix: str, checkpoint_name: str,
                      attempts: int = 3) -> None:
    checkpoint = local_dir / checkpoint_name
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    files = [p for p in checkpoint.rglob("*") if p.is_file()]
    # config.json is the commit marker: it goes up last so an interrupted
    # upload never leaves a checkpoint dir that looks resumable but is not
    # (latest_remote_checkpoint requires both config.json and state.pt).
    files.sort(key=lambda p: p.name == "config.json")
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
    # Only complete checkpoints (training state + config) are resumable; a
    # session killed mid-upload must not become the permanent "latest" that
    # crashes every future resume.
    root = remote_prefix.strip("/")
    complete = {n for n in names
                if f"{root}/{n}/state.pt" in files and f"{root}/{n}/config.json" in files}
    return max(complete, key=key) if complete else None


def download_latest(api_client, repo_id: str, remote_prefix: str,
                    local_root: Path, attempts: int = 3) -> Path | None:
    from huggingface_hub import hf_hub_download
    name = latest_remote_checkpoint(api_client, repo_id, remote_prefix)
    if name is None:
        print(f"[hf] no remote checkpoint under {remote_prefix}", flush=True)
        return None
    destination = local_root / remote_prefix.strip("/") / name
    destination.mkdir(parents=True, exist_ok=True)
    files = api_client.list_repo_files(repo_id=repo_id, repo_type="dataset")
    prefix = f"{remote_prefix.strip('/')}/{name}/"
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            for filename in files:
                if filename.startswith(prefix):
                    hf_hub_download(repo_id=repo_id, filename=filename,
                                    repo_type="dataset", local_dir=str(local_root),
                                    token=api_client.token)
            print(f"[hf] downloaded {name} -> {destination}", flush=True)
            return destination
        except Exception as exc:  # network flap at resume must not kill the run
            last_exc = exc
            print(f"[hf] download attempt {attempt}/{attempts} failed: {exc}", flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError(f"download failed after {attempts} attempts") from last_exc


def write_status(api_client, repo_id: str, remote_prefix: str, status: str) -> None:
    # Timestamp failure statuses: identical text would otherwise be skipped as a
    # no-op upload by HF, hiding repeated crash cycles from the watchers that
    # key off the commit history. DONE / IN PROGRESS markers stay untouched so
    # their startswith checks keep working.
    if not status.startswith(("DONE", "IN PROGRESS")):
        import datetime
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        status = f"# {stamp}\n{status}"
    try:
        api_client.upload_file(
            path_or_fileobj=status.encode("utf-8"),
            path_in_repo=f"{remote_prefix.strip('/')}/run-status.txt",
            repo_id=repo_id,
            repo_type="dataset",
        )
    except Exception as exc:  # rate limits or network flaps must not kill a run
        print(f"[status] upload failed: {exc}", flush=True)


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
