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


def upload_file(token: str, remote: str, data: bytes,
                message: str | None = None) -> bool:
    """PUT one file to the public repo. Returns True on success."""
    url = f"https://api.github.com/repos/{PUBLIC_LIVE_REPO}/contents/{remote}"
    sha = _get_sha(url, token)
    payload = {
        "message": message or f"monitor {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "content": base64.b64encode(data).decode(),
        "branch": PUBLIC_LIVE_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "User-Agent": "chess-monitor",
                 "Content-Type": "application/json"},
        method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            json.load(r)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 409 and sha:  # stale sha race — retry once with a fresh sha
            payload["sha"] = _get_sha(url, token)
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {token}", "User-Agent": "chess-monitor",
                         "Content-Type": "application/json"},
                method="PUT")
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    json.load(r)
                return True
            except Exception:
                return False
        return False
    except Exception:
        return False
