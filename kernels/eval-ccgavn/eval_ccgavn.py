"""HISTORICAL one-shot eval for ccgavn-5m-seed0/checkpoint-320000 (archived).

Polls HF every 5 minutes (up to ~9.5h), then evaluates all four MATE sets and
the official 10K puzzles with the distribution score. DONE is written only
when eval_gavn.py exits 0 with exact totals (4,000 MATE rows / 10,000 puzzles);
otherwise INCOMPLETE plus the partial artifacts are uploaded. CPU kernel.
"""
import glob
import io
import json
import os
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
summary_path = WORK / "eval-summary.json"
cmd = [sys.executable, str(REPO / "scripts" / "eval_gavn.py"),
       "--checkpoint", str(ck / CKPT), "--sl-repo", str(SL), "--eval", MATE,
       "--puzzles", str(pz), "--num-puzzles", "10000", "--score", "auto",
       "--examples-out", str(WORK / "examples"), "--summary-out", str(summary_path)]
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

summary = {}
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
mate = summary.get("mate") or {}
puz = summary.get("puzzles") or {}
complete = (proc.returncode == 0 and summary.get("complete")
            and mate.get("total") == 4000 and puz.get("total") == 10000)
summary.update({"checkpoint": CKPT, "returncode": proc.returncode,
                "complete": bool(complete)})
summary_path.write_text(json.dumps(summary, indent=2))
upload(out, f"{PREFIX}/eval-full.log")
if complete:
    upload(summary_path, f"{PREFIX}/eval-summary.json")
    wake = (f"# CC-GAVN 320k frozen eval\n\n"
            f"- MATE (4,000): {mate['correct']}/{mate['total']}\n"
            f"- Puzzles (10,000): {puz['solved']}/{puz['total']}\n\n"
            f"Baselines — CC-GAVN@160k: 87.38% / 51.86%; 9M teacher: 98.72% / 86.13%\n")
    (WORK / "wake-up-summary.md").write_text(wake)
    upload(WORK / "wake-up-summary.md", f"{PREFIX}/wake-up-summary.md")
state = "DONE " if complete else "INCOMPLETE "
api.upload_file(path_or_fileobj=(state + json.dumps(summary)).encode(),
                path_in_repo=f"{PREFIX}/run-status.txt", repo_id=HF_REPO, repo_type="dataset")
print(f"[eval] {state}{json.dumps(summary)}")
