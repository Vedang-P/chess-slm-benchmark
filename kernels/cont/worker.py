"""Pinned Kaggle entry point for the cont1 continuation learner.

scripts/launch_cont.py and scripts/watch_cont.py embed a JSON config before each
push. Kaggle only uploads this one file, so nothing else is assumed to exist.
The session budget is measured from the moment this script starts, which is also
when Kaggle's own 12 h clock starts.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

CONFIG_JSON = """__CONT_CONFIG__"""
if CONFIG_JSON.startswith("__CONT_CONFIG__"):
    raise SystemExit("missing embedded cont config")
CFG = json.loads(CONFIG_JSON)
SESSION_STARTED = time.time()
WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
SL = WORK / "searchless_chess"

hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
TOKEN = Path(hits[0]).read_text().strip() if hits else os.environ.get("HF_WRITE_TOKEN", "")
if not TOKEN:
    raise SystemExit("HF token missing; cannot persist Kaggle progress")
os.environ["HF_WRITE_TOKEN"] = TOKEN
os.environ["SHARD_CACHE"] = "/kaggle/tmp/shards"
Path(os.environ["SHARD_CACHE"]).mkdir(parents=True, exist_ok=True)


def run(command: list[str], **kwargs) -> None:
    print("[cont] " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, **kwargs)


def pinned_git(url: str, dest: Path, commit: str) -> None:
    run(["git", "init", "-q", str(dest)])
    run(["git", "-C", str(dest), "remote", "add", "origin", url])
    run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", commit])
    run(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"])
    actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError(f"pinned checkout mismatch: {actual} != {commit}")


def official_puzzles() -> Path:
    dest = SL / "data" / "puzzles.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        url = "https://storage.googleapis.com/searchless_chess/data/puzzles.csv"
        with urllib.request.urlopen(url, timeout=120) as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return dest


def persist_status(text: str) -> None:
    sys.path.insert(0, str(REPO))
    from huggingface_hub import HfApi
    from scripts.hf_upload import upload_file
    suffix = f"{CFG['phase']}-{CFG['run']}"
    upload_file(HfApi(token=TOKEN), CFG["hf_repo"], text.encode(),
                f"{CFG['status_prefix']}/{suffix}.txt",
                commit_message=f"cont {suffix}: {text.split(chr(10))[0][:60]}",
                attempts=4, max_elapsed_s=300.0)


def main() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4", "python-chess",
         "pandas==2.2.3", "zstandard", "huggingface_hub"])
    pinned_git("https://github.com/Vedang-P/chess-slm-benchmark.git", REPO, CFG["source_commit"])
    pinned_git("https://github.com/google-deepmind/searchless_chess.git", SL, CFG["searchless_commit"])
    sys.path.insert(0, str(REPO))
    run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'no GPU'"])
    puzzles = official_puzzles()
    run([sys.executable, str(REPO / "scripts" / "train_cont.py"),
         "--hf-repo", CFG["hf_repo"], "--hf-run", CFG["run"], "--outdir", str(WORK / CFG["run"]),
         "--sl-repo", str(SL), "--searchless-commit", CFG["searchless_commit"],
         "--exclusion-puzzles", str(puzzles),
         "--session-seconds", str(CFG["session_seconds"]),
         "--kernel-started-epoch", str(SESSION_STARTED), "--resume-from-hf",
         *CFG["trainer_args"]], cwd=REPO)


try:
    main()
    persist_status(f"OK {CFG['phase']} source_commit={CFG['source_commit']} "
                   f"hours={(time.time() - SESSION_STARTED) / 3600:.2f}")
except Exception as exc:
    status = (f"FAILED {CFG['phase']} source_commit={CFG['source_commit']} "
              f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3500:]}")
    try:
        persist_status(status)
    except Exception as upload_exc:
        print(f"[cont] could not persist failure: {upload_exc}", flush=True)
    raise
