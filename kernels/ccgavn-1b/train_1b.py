"""Wait briefly for checkpoint-320000 + the 1B corpus, then continue CC-GAVN.

Continuation of `ccgavn-5m-seed0` from step 320,000 to 1,620,000 (+1.3M steps)
on the expanded shard corpus (8 original ChessBench shards + 2 puzzle shards +
the newly labeled 1B shards), identical recipe. The CI (scripts/watch_1b.py)
gates the push on both prerequisites, so this kernel only waits ~15 min for a
race, then exits (a GPU session must never idle on the quota clock).
Persistence, resume, and re-push supervision are the usual ones.

Pushed per account by scripts/watch_1b.py; kernel id is <account>/ccgavn-1b.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN = "ccgavn-5m-seed0"
START_STEP = 320000
TOTAL_STEPS = 1_620_000
TARGET_NEW_ROWS = 920_000_000
HF_REPO = "vedangfake/chess-slm-benchmark"
PREFIX = "chessbench-full-build"
MAX_WAIT_S = 900

WORK = Path("/kaggle/working")
hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
TOKEN = Path(hits[0]).read_text().strip() if hits else os.environ.get("HF_WRITE_TOKEN", "")
assert TOKEN, "no HF token"

from huggingface_hub import HfApi  # noqa: E402
api = HfApi(token=TOKEN)


def ready() -> tuple[bool, str]:
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    if f"{RUN}/checkpoint-{START_STEP}/config.json" not in files or \
            f"{RUN}/checkpoint-{START_STEP}/state.pt" not in files:
        return False, "waiting for checkpoint-320000"
    rows_map = json.loads(Path(__file__).with_name("shard_rows.json").read_text()) \
        if Path(__file__).with_name("shard_rows.json").exists() else {}
    done = sum(int(v) for k, v in rows_map.items()
               if f"{PREFIX}/shard-{k}/teacher_logp.npy" in files)
    if done < TARGET_NEW_ROWS:
        return False, f"waiting for corpus ({done/1e6:.0f}M / {TARGET_NEW_ROWS/1e6:.0f}M new rows)"
    return True, f"corpus ready ({done/1e6:.0f}M new rows)"


t0 = time.time()
while True:
    ok, msg = ready()
    print(f"[1b] {msg}", flush=True)
    if ok:
        break
    if time.time() - t0 > MAX_WAIT_S:
        raise SystemExit("prerequisites not ready within the wait window; re-push later")
    time.sleep(300)

print("[1b] prerequisites ready; preparing environment", flush=True)
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-chess", "pandas"], check=True)

out = WORK / RUN
cmd = [sys.executable, str(REPO / "scripts" / "train_ccgavn.py"),
       "--hf-shards", PREFIX, "--outdir", str(out), "--sl-repo", str(SL),
       "--dim", "208", "--layers", "8", "--heads", "8", "--batch", "2048",
       "--steps", str(TOTAL_STEPS), "--lr", "0.0005", "--warmup", "2000",
       "--temperature", "1.0", "--w-dist", "1.0", "--w-ce", "0.25",
       "--reflect-prob", "0.5", "--dev-mod", "100", "--dev-fold", "0",
       "--dev-batch", "8192", "--seed", "0", "--ckpt-every", "5000",
       "--hf-repo", HF_REPO, "--hf-run", RUN, "--hf-upload-every", "1800",
       "--resume-from-hf"]
print("[1b] " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=str(REPO))
print("[1b] training returned", flush=True)
