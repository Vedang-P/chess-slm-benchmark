"""Wait for ccgavn-5m-seed0/checkpoint-320000 on HF, then run the frozen protocol.

Polls HF every 5 minutes (up to ~9.5h), then evaluates all four MATE sets and
the official 10K puzzles with the distribution score, and uploads logs, a
summary, and a wake-up summary to HF. CPU kernel; no GPU quota.
"""
import glob
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RUN = "ccgavn-5m-seed0"
STEP = 320000
CKPT = f"{RUN}/checkpoint-{STEP}"
HF_REPO = "vedangfake/chess-slm-benchmark"
PREFIX = f"eval-results/{RUN}-320k-frozen-2026-09-19"
MAX_WAIT_S = 900

WORK = Path("/kaggle/working")
hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
TOKEN = Path(hits[0]).read_text().strip() if hits else os.environ.get("HF_WRITE_TOKEN", "")
assert TOKEN, "no HF token"

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402
api = HfApi(token=TOKEN)


def ckpt_ready() -> bool:
    files = api.list_repo_files(HF_REPO, repo_type="dataset")
    return f"{CKPT}/config.json" in files and f"{CKPT}/state.pt" in files


def upload(local: Path, remote: str) -> None:
    try:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=HF_REPO, repo_type="dataset")
    except Exception as exc:
        print(f"[eval] upload failed {remote}: {exc}", flush=True)


print("[eval] waiting for checkpoint ...", flush=True)
t0 = time.time()
while not ckpt_ready():
    if time.time() - t0 > MAX_WAIT_S:
        api.upload_file(
            path_or_fileobj=("TIMEOUT: checkpoint not found within %.0fh\n" % (MAX_WAIT_S / 3600)).encode(),
            path_in_repo=f"{PREFIX}/run-status.txt", repo_id=HF_REPO, repo_type="dataset")
        raise SystemExit("checkpoint never appeared")
    time.sleep(300)
    print(f"[eval] still waiting ({(time.time()-t0)/60:.0f} min)", flush=True)

print("[eval] checkpoint found; preparing environment", flush=True)
REPO = WORK / "chess-slm-benchmark"
SL = WORK / "searchless_chess"
if not REPO.exists():
    for _ in range(4):
        if subprocess.run(["git", "clone", "--depth", "1",
                           "https://github.com/Vedang-P/chess-slm-benchmark.git", str(REPO)]).returncode == 0:
            break
        time.sleep(20)
if not SL.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/google-deepmind/searchless_chess.git", str(SL)], check=True)
pz = SL / "data" / "puzzles.csv"
if not pz.exists():
    subprocess.run(["curl", "-sL", "--fail", "-o", str(pz),
                    "https://storage.googleapis.com/searchless_chess/data/puzzles.csv"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-chess", "pandas"], check=True)

ck = WORK / "ckpt"
snapshot_download(repo_id=HF_REPO, repo_type="dataset", token=TOKEN, local_dir=str(ck),
                  allow_patterns=[f"{CKPT}/*"])

MATE = ",".join(str(REPO / "data/positions" / f) for f in
                ["mate-selection-test-noexplain.json", "mate-selection-test-both.json",
                 "mate-selection-test-tactic.json", "mate-selection-test.json"])
out = WORK / "eval-full.log"
cmd = [sys.executable, str(REPO / "scripts" / "eval_gavn.py"),
       "--checkpoint", str(ck / CKPT), "--sl-repo", str(SL), "--eval", MATE,
       "--puzzles", str(pz), "--num-puzzles", "10000", "--score", "auto"]
print("[eval] " + " ".join(cmd), flush=True)
last = time.time()
lines = []
with open(out, "w") as fh:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        fh.write(line); fh.flush(); print(line.rstrip(), flush=True)
        lines.append(line.rstrip())
        if time.time() - last > 300:
            upload(out, f"{PREFIX}/eval-full.partial.log")
            last = time.time()
    proc.wait()

mate = re.findall(r"\[gavn\] MATE: (\d+)/(\d+) = ([\d.]+)%", "\n".join(lines))
puz = re.findall(r"\[gavn\] puzzles: (\d+)/(\d+) = ([\d.]+)%", "\n".join(lines))
summary = {"checkpoint": CKPT, "returncode": proc.returncode,
           "mate": [{"correct": int(a), "total": int(b), "pct": float(c)} for a, b, c in mate],
           "puzzles": [{"correct": int(a), "total": int(b), "pct": float(c)} for a, b, c in puz]}
(WORK / "eval-summary.json").write_text(json.dumps(summary, indent=2))
upload(out, f"{PREFIX}/eval-full.log")
upload(WORK / "eval-summary.json", f"{PREFIX}/eval-summary.json")
wake = (f"# CC-GAVN 320k frozen eval\n\n"
        f"- MATE (4,000): {mate[0][0]}/{mate[0][1]} = {mate[0][2]}%\n" if mate else "MATE missing\n")
if mate and puz:
    wake = (f"# CC-GAVN 320k frozen eval\n\n"
            f"- MATE (4,000): {mate[0][0]}/{mate[0][1]} = {mate[0][2]}%\n"
            f"- Puzzles (10,000): {puz[0][0]}/{puz[0][1]} = {puz[0][2]}%\n\n"
            f"Baselines — CC-GAVN@160k: 87.38% / 51.86%; 9M teacher: 98.72% / 86.13%\n")
(WORK / "wake-up-summary.md").write_text(wake)
upload(WORK / "wake-up-summary.md", f"{PREFIX}/wake-up-summary.md")
api.upload_file(path_or_fileobj=("DONE\n" + json.dumps(summary)).encode(),
                path_in_repo=f"{PREFIX}/run-status.txt", repo_id=HF_REPO, repo_type="dataset")
print("[eval] DONE", json.dumps(summary))
