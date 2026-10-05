"""Wait briefly for the 102-tag stage-2B corpus + checkpoint-1,620,000, then continue CC-GAVN.

Stage 2B-A (user decision 2026-10-05): warm-start `ccgavn-5m-seed0` from
step 1,620,000 and train to step 3,896,658 (a new cosine schedule sized to
~2.67 passes over the 723.0M rows added by the 41 remaining built shards;
schedule proportional over the 102-tag union, 1.746B rows). The last 16
planned shards (~268.7M rows) are being labeled and may be appended in a
later segment.

The exact tag list is embedded into this file at push time by
`scripts/watch_2b.py`, because `kaggle kernels push` uploads only the code
file -- a sibling JSON can never reach the kernel (the previous revision of
this pattern silently counted 0 rows and exited after 15 minutes on every
push).

The CI (scripts/watch_2b.py) gates the push on the prerequisites, so this
kernel only waits ~15 min for a race, then exits (a GPU session must never
idle on the quota clock). Persistence, resume, and re-push supervision are
the usual ones.

Pushed per account by scripts/watch_2b.py; kernel id is <account>/ccgavn-2b.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN = "ccgavn-5m-seed0"
START_STEP = 1_620_000
TOTAL_STEPS = 3_896_658
HF_REPO = "vedangfake/chess-slm-benchmark"
PREFIX = "chessbench-full-build"
MAX_WAIT_S = 900

TRAIN_TAGS_JSON = """__CCGAVN2B_TRAIN_TAGS__"""
if TRAIN_TAGS_JSON.startswith("__CCGAVN"):
    raise SystemExit("shard tag list was not embedded at push time; "
                     "re-push via scripts/watch_2b.py")
TRAIN_TAGS = json.loads(TRAIN_TAGS_JSON)
assert TRAIN_TAGS, "empty stage-2B shard tag list"

WORK = Path("/kaggle/working")
hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
TOKEN = Path(hits[0]).read_text().strip() if hits else os.environ.get("HF_WRITE_TOKEN", "")
assert TOKEN, "no HF token"
# The trainer (and its failure-status upload) read the token from the
# environment; without this export every run dies at make_hf_api.
os.environ["HF_WRITE_TOKEN"] = TOKEN

from huggingface_hub import HfApi  # noqa: E402
api = HfApi(token=TOKEN)


def ready() -> tuple[bool, str]:
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    missing = [t for t in TRAIN_TAGS
               if f"{PREFIX}/shard-{t}/train_set.npz" not in files
               or f"{PREFIX}/shard-{t}/teacher_logp.npy" not in files]
    if missing:
        shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        return False, (f"waiting for corpus ({len(TRAIN_TAGS) - len(missing)}/"
                       f"{len(TRAIN_TAGS)} stage-2B shards ready; missing {shown})")
    if f"{RUN}/checkpoint-{START_STEP}/config.json" not in files or \
            f"{RUN}/checkpoint-{START_STEP}/state.pt" not in files:
        return False, f"waiting for checkpoint-{START_STEP}"
    return True, f"corpus ready ({len(TRAIN_TAGS)} shards) + checkpoint-{START_STEP}"


t0 = time.time()
while True:
    ok, msg = ready()
    print(f"[2b] {msg}", flush=True)
    if ok:
        break
    if time.time() - t0 > MAX_WAIT_S:
        raise SystemExit("prerequisites not ready within the wait window; re-push later")
    time.sleep(300)

print("[2b] prerequisites ready; preparing environment", flush=True)
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
       "--shard-tags-file", str(REPO / "configs" / "ccgavn-2b-shard-tags.json"),
       "--allow-shard-superset",
       "--dim", "208", "--layers", "8", "--heads", "8", "--batch", "2048",
       "--steps", str(TOTAL_STEPS), "--lr", "0.0005", "--warmup", "2000",
       "--temperature", "1.0", "--w-dist", "1.0", "--w-ce", "0.25",
       "--reflect-prob", "0.5", "--dev-mod", "100", "--dev-fold", "0",
       "--dev-batch", "8192", "--seed", "0", "--ckpt-every", "5000",
       "--hf-repo", HF_REPO, "--hf-run", RUN, "--hf-upload-every", "1800",
       "--resume-from-hf"]
print("[2b] " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=str(REPO))
print("[2b] training returned", flush=True)
