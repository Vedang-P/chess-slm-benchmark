"""Rate-aware Hugging Face writes shared by Kaggle workers and the supervisor."""
from __future__ import annotations

import datetime as dt
import email.utils
import random
import time
from pathlib import Path
from typing import Callable, Iterable


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            when = email.utils.parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=dt.timezone.utc)
            return max(0.0, (when - dt.datetime.now(dt.timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _retryable(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        return code in (408, 425, 429) or (isinstance(code, int) and code >= 500)
    module = type(exc).__module__
    return (isinstance(exc, (TimeoutError, ConnectionError, OSError))
            or module.startswith(("httpx", "requests")))


def retry_hf_call(operation: Callable[[], object], label: str,
                   attempts: int = 12, max_elapsed_s: float = 5_400.0,
                   max_delay_s: float = 900.0) -> object:
    """Retry transient Hub failures, honoring Retry-After and bounding the wait."""
    started = time.monotonic()
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts or not _retryable(exc):
                raise
            elapsed = time.monotonic() - started
            retry_after = _retry_after_seconds(exc)
            backoff = min(max_delay_s, 20.0 * (2 ** min(attempt - 1, 10)))
            delay = retry_after if retry_after is not None else random.uniform(backoff * .8, backoff * 1.2)
            remaining = max_elapsed_s - elapsed
            if delay > remaining:
                raise TimeoutError(
                    f"{label}: Hugging Face retry window exhausted after {elapsed:.0f}s"
                ) from exc
            print(f"[hf] {label} attempt {attempt}/{attempts} failed; "
                  f"retrying in {delay:.0f}s: {exc}", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def upload_file(api_client, repo_id: str, path_or_fileobj, path_in_repo: str,
                repo_type: str = "dataset", commit_message: str | None = None,
                attempts: int = 12, max_elapsed_s: float = 5_400.0):
    kwargs = {"path_or_fileobj": path_or_fileobj, "path_in_repo": path_in_repo,
              "repo_id": repo_id, "repo_type": repo_type}
    if commit_message is not None:
        kwargs["commit_message"] = commit_message
    return retry_hf_call(lambda: api_client.upload_file(**kwargs),
                         f"upload {path_in_repo}", attempts, max_elapsed_s)


def upload_files_batch(api_client, repo_id: str,
                       files: Iterable[tuple[str | Path, str]],
                       commit_message: str, repo_type: str = "dataset",
                       attempts: int = 12,
                       max_elapsed_s: float = 5_400.0):
    """Add multiple already-computed files in one atomic Hub commit."""
    from huggingface_hub import CommitOperationAdd

    items = [(str(path), remote) for path, remote in files]
    if not items:
        return None

    def commit():
        operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
                      for local, remote in items]
        return api_client.create_commit(repo_id=repo_id, repo_type=repo_type,
                                        operations=operations,
                                        commit_message=commit_message)

    return retry_hf_call(commit, f"batch upload ({len(items)} files)",
                         attempts, max_elapsed_s)
