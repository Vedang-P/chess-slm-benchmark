"""Strict cleanup policy for resumable evaluation artifacts.

The 2026-10-06 follow-up review found a remaining recovery hole: if deleting
the old saved examples failed silently, the kernel still rescored, uploaded a
replacement identity.json, and (when the replacement row uploads failed) left
old rows sitting under the new identity. The next resume then accepted those
rows. This module makes cleanup strict and ordered:

  - identity.json is deleted first, so a partial cleanup can never leave a
    fresh marker attached to stale rows,
  - missing files (404) are fine; any other error aborts,
  - the rescore callback only runs after cleanup fully succeeded; on failure
    the caller must mark INCOMPLETE and publish nothing.
"""
from __future__ import annotations

import shutil
from pathlib import Path

EXAMPLE_FILES = ("identity.json", "mate.jsonl", "puzzles.jsonl")


def is_missing_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 404:
        return True
    text = str(exc).lower()
    return "404" in text or "not found" in text or "entrynotfound" in text


def delete_remote_examples(api, repo_id: str, prefix: str,
                           names: tuple[str, ...] = EXAMPLE_FILES) -> None:
    """Delete saved eval artifacts; raise on any unexpected failure."""
    for name in names:
        path = f"{prefix}/examples/{name}"
        try:
            api.delete_file(path, repo_id=repo_id, repo_type="dataset")
        except Exception as exc:
            if is_missing_error(exc):
                continue
            raise RuntimeError(f"failed to delete {path}: {exc}") from exc


def recover_from_unusable(api, repo_id: str, prefix: str, examples_dir: Path,
                          rescore, log=print) -> tuple[bool, object]:
    """Cleanup-before-rescore. Returns (recovered, rescore_result).

    On cleanup failure returns (False, None) WITHOUT running rescore: the
    caller must publish INCOMPLETE rather than a replacement identity over
    possibly-stale rows.
    """
    try:
        delete_remote_examples(api, repo_id, prefix)
    except Exception as exc:
        log(f"[eval] remote examples cleanup failed; refusing to rescore: {exc}")
        return False, None
    shutil.rmtree(examples_dir, ignore_errors=True)
    return True, rescore()
