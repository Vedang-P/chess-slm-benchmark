"""Pinned Kaggle entry point for CPU verifier labels or T4 root-policy RL."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG_JSON = """__CCGAVN_RL_CONFIG__"""
if CONFIG_JSON.startswith("__CCGAVN_RL_CONFIG"):
    raise SystemExit("RL job config must be embedded by scripts/watch_rl.py")
CFG = json.loads(CONFIG_JSON)
WORK = Path("/kaggle/working")

hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
token = Path(hits[0]).read_text(encoding="utf-8").strip() if hits else os.environ.get("HF_WRITE_TOKEN", "")
if not token:
    raise RuntimeError("HF_WRITE_TOKEN credential dataset is missing")
os.environ["HF_WRITE_TOKEN"] = token

repo = WORK / "chess-slm-benchmark"
sl = WORK / "searchless_chess"
subprocess.run(["git", "init", "-q", str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "remote", "add", "origin",
                "https://github.com/Vedang-P/chess-slm-benchmark.git"], check=True)
subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", "origin",
                CFG["source_commit"]], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "-q", "FETCH_HEAD"], check=True)
actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                 text=True).strip()
if actual != CFG["source_commit"]:
    raise RuntimeError(f"pinned source mismatch: {actual}")
subprocess.run(["git", "init", "-q", str(sl)], check=True)
subprocess.run(["git", "-C", str(sl), "remote", "add", "origin",
                "https://github.com/google-deepmind/searchless_chess.git"], check=True)
subprocess.run(["git", "-C", str(sl), "fetch", "-q", "--depth", "1", "origin",
                CFG["searchless_commit"]], check=True)
subprocess.run(["git", "-C", str(sl), "checkout", "-q", "FETCH_HEAD"], check=True)
sl_commit = subprocess.check_output(["git", "-C", str(sl), "rev-parse", "HEAD"],
                                    text=True).strip()
if sl_commit != CFG["searchless_commit"]:
    raise RuntimeError(f"pinned Searchless commit mismatch: {sl_commit}")
puzzles = sl / "data" / "puzzles.csv"
if not puzzles.exists():
    subprocess.run(["curl", "-sSL", "--fail", "-o", str(puzzles),
                    "https://storage.googleapis.com/searchless_chess/data/puzzles.csv"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-chess",
                "pandas", "huggingface_hub"], check=True)
os.environ["PYTHONPATH"] = os.pathsep.join((str(WORK), str(repo)))

if CFG["mode"] == "labels":
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "stockfish"], check=True)
    cmd = [sys.executable, "-m", "scripts.rl_labels",
           "--account", CFG["account"], "--shard-tag", CFG["shard_tag"],
           "--depth", str(CFG["depth"]), "--target", str(CFG["target"]),
           "--source-commit", CFG["source_commit"],
           "--searchless-commit", CFG["searchless_commit"],
           "--prefix", CFG["label_prefix"],
           "--outdir", str(WORK / "rl-labels"), "--puzzles", str(puzzles)]
elif CFG["mode"] == "train":
    manifest = WORK / "rl-manifest.json"
    manifest.write_text(CFG["manifest_json"], encoding="utf-8")
    cmd = [sys.executable, "-m", "scripts.train_rl",
           "--manifest", str(manifest), "--outdir", str(WORK / "ccgavn-rl"),
           "--sl-repo", str(sl), "--puzzles", str(puzzles),
           "--steps", str(CFG["steps"]), "--hf-run", CFG["run"]]
    cmd += ["--searchless-commit", CFG["searchless_commit"]]
else:
    raise ValueError(f"unknown RL mode {CFG['mode']}")
print(f"[rl-kernel] mode={CFG['mode']} commit={actual[:12]} cmd={' '.join(cmd)}", flush=True)
subprocess.run(cmd, check=True, cwd=str(repo))
