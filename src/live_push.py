"""Single-file uploads to the public live repo (GitHub contents API).

Used by run_suite (state/results batch) and run_chess (live.json per sample).
Token: GITHUB_TOKEN / GH_TOKEN env. Never raises on network errors — callers
log and continue; live telemetry must never break the sweep.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

PUBLIC_LIVE_REPO = "Vedang-P/chess-bench-live"
PUBLIC_LIVE_BRANCH = "main"


def resolve_token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    # local runs: read the gitignored .env (never committed)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(("GITHUB_TOKEN=", "GH_TOKEN=")):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _get_sha(url: str, token: str) -> str | None:
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "User-Agent": "chess-monitor"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["sha"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch_file(token: str, remote: str) -> bytes | None:
    """GET one file's content from the public repo. None if it doesn't
    exist (e.g. a worker that hasn't published its first state yet) --
    that is a normal, expected state for a caller polling several workers'
    files, not an error."""
    url = f"https://api.github.com/repos/{PUBLIC_LIVE_REPO}/contents/{remote}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "User-Agent": "chess-monitor"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.load(r)
        return base64.b64decode(body["content"])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def upload_file(token: str, remote: str, data: bytes,
                message: str | None = None) -> str | None:
    """PUT one file to the public repo. Returns None on success, or a
    diagnostic string (HTTP code + body snippet) on failure — so the caller
    can print/record exactly why a push failed."""
    url = f"https://api.github.com/repos/{PUBLIC_LIVE_REPO}/contents/{remote}"
    sha = _get_sha(url, token)
    payload = {
        "message": message or f"monitor {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "content": base64.b64encode(data).decode(),
        "branch": PUBLIC_LIVE_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    def _put() -> str | None:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {token}", "User-Agent": "chess-monitor",
                     "Content-Type": "application/json"},
            method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                json.load(r)
            return None
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:200]
            except Exception:
                pass
            return f"HTTP {e.code} {body}"
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    err = _put()
    if err is None:
        return None
    if err.startswith("HTTP 409") and sha:  # stale sha race — retry once
        payload["sha"] = _get_sha(url, token)
        return _put()
    return err
