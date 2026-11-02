"""Kaggle helpers shared by the cont1 supervisor and launcher.

Status queries return HTTP 429 when many run at once, so every call backs off.
Stopping a session is `delete_kernel`: the CLI has no cancel command, and the
kernels are throwaway (they are rebuilt from the pinned commit on each push).
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cont_config import CREDENTIAL_DATASETS  # noqa: E402
from scripts.kaggle_accounts import env_for_account  # noqa: E402

ACTIVE = ("RUNNING", "QUEUED", "PENDING", "STARTING")
FAILED = ("ERROR", "FAILED", "CANCELLED", "CANCELED")


def kaggle(account: str, *args: str, timeout: int = 120, attempts: int = 6) -> tuple[int, str]:
    out = (1, "")
    for attempt in range(attempts):
        result = subprocess.run([sys.executable, "-W", "ignore", "-m", "kaggle", *args],
                                env=env_for_account(account), capture_output=True,
                                text=True, timeout=timeout)
        out = (result.returncode, (result.stdout + result.stderr).strip())
        if "429" not in out[1]:
            return out
        time.sleep(5 * (attempt + 1))
    return out


def account_slugs(account: str) -> set[str]:
    code, out = kaggle(account, "kernels", "list", "--mine", "--page-size", "200", "--csv")
    lines = out.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("ref,title,")), None)
    if code or start is None:
        raise RuntimeError(f"{account}: Kaggle listing failed: {out[-300:]}")
    return {row["ref"] for row in csv.DictReader(io.StringIO("\n".join(lines[start:])))
            if row.get("ref")}


def kernel_status(account: str, slug: str, slugs: set[str]) -> str:
    ref = f"{account}/{slug}"
    if ref not in slugs:
        return "ABSENT"
    code, out = kaggle(account, "kernels", "status", ref, timeout=90)
    if code:
        raise RuntimeError(f"{ref}: status failed: {out[-300:]}")
    return out.splitlines()[-1] if out else ""


def is_active(status: str) -> bool:
    return any(word in status for word in ACTIVE)


def is_failed(status: str) -> bool:
    return any(word in status for word in FAILED)


def gpu_quota(account: str) -> float:
    code, out = kaggle(account, "quota", timeout=90)
    if code:
        raise RuntimeError(f"{account}: quota failed: {out[-300:]}")
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "GPU":
            return float(fields[2].rstrip("h"))
    raise RuntimeError(f"{account}: GPU quota missing from response")


def push_kernel(account: str, slug: str, template: Path, marker: str, config: dict,
                gpu: bool = True, strict: bool = False) -> None:
    """Render the template with `config` and push it as a Kaggle script kernel.

    The CLI exits 0 on some refusals ("Kernel push error: Maximum batch CPU session count of
    5 reached."), so a zero exit code does not prove a session started. `strict` turns that
    message into an error; it is off by default so the learner's launch path is unchanged.
    """
    source = template.read_text(encoding="utf-8")
    placeholder = f'"""{marker}"""'
    if source.count(placeholder) != 1:
        raise ValueError(f"kernel template {template} lost its unique config marker")
    rendered = source.replace(placeholder, '"""' + json.dumps(config, sort_keys=True) + '"""')
    with tempfile.TemporaryDirectory(prefix=f"cont_{account}_") as tmp:
        folder = Path(tmp)
        (folder / template.name).write_text(rendered, encoding="utf-8")
        metadata = {"id": f"{account}/{slug}", "title": slug, "code_file": template.name,
                    "language": "python", "kernel_type": "script", "is_private": True,
                    "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
                    "dataset_sources": [CREDENTIAL_DATASETS[account]],
                    "competition_sources": [], "kernel_sources": []}
        if gpu:
            metadata["machine_shape"] = "NvidiaTeslaT4"
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        code, out = kaggle(account, "kernels", "push", "-p", str(folder), timeout=300)
        if code or (strict and "push error" in out.lower()):
            raise RuntimeError(f"{account}/{slug} push failed: {out[-500:]}")
        print(f"[pool] pushed {slug} on {account}", flush=True)


def delete_kernel(account: str, slug: str) -> str:
    """Stop a session by deleting its (throwaway) kernel; returns the CLI message."""
    code, out = kaggle(account, "kernels", "delete", "-y", f"{account}/{slug}")
    if code:
        raise RuntimeError(f"{account}/{slug} delete failed: {out[-300:]}")
    return out.splitlines()[-1] if out else ""
