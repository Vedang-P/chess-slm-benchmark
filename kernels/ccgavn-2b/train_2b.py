"""Corrected-v2 stage-2B continuation of CC-GAVN (new HF run prefix).

Stage 2B-A (user decision 2026-10-05; corrected 2026-10-06): warm-start the
legacy `ccgavn-5m-seed0` continuation at its latest complete checkpoint and
train to step 3,896,658 under the corrected-v2 recipe:

  - the step schedule is stage-relative over the full 102-tag frozen union
    (the legacy launcher built the full-budget schedule and indexed it by the
    global step, silently skipping whole shard blocks -- 41/102 tags were
    never sampled),
  - MATE positions and official puzzle positions are excluded from training,
  - all optimizer groups follow the cosine schedule, not just group 0,
  - development sampling uses a checkpoint-independent RNG.

Checkpoints are written under a NEW run prefix (RUN below), never to the
legacy prefix, so the historical training trajectory stays intact and
disclosed. `INIT_RUN/checkpoint-<START_STEP>` is the frozen warm-start point.

The exact run/stage/tag values are embedded into this file at push time by
`scripts/watch_2b.py`, because `kaggle kernels push` uploads only the code file
-- a sibling JSON can never reach the kernel (the previous revision of this
pattern silently counted 0 rows and exited after 15 minutes on every push).

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

RUN = "ccgavn-5m-seed0-v2"
INIT_RUN = "ccgavn-5m-seed0"
START_STEP = 1_715_000
TOTAL_STEPS = 3_896_658
HF_REPO = "vedangfake/chess-slm-benchmark"
PREFIX = "chessbench-full-build"
MAX_WAIT_S = 900

CONFIG_JSON = """__CCGAVN2B_CONFIG__"""
if CONFIG_JSON.startswith("__CCGAVN"):
    raise SystemExit("stage config was not embedded at push time; "
                     "re-push via scripts/watch_2b.py")
CFG = json.loads(CONFIG_JSON)
RUN = CFG["run"]
INIT_RUN = CFG["init_run"]
START_STEP = int(CFG["start_step"])
TOTAL_STEPS = int(CFG["total_steps"])
SOURCE_COMMIT = CFG["source_commit"]
TRAIN_TAGS = CFG["tags"]
assert TRAIN_TAGS, "empty stage-2B shard tag list"
assert START_STEP < TOTAL_STEPS, "stage start must precede the total step budget"

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
    if f"{INIT_RUN}/checkpoint-{START_STEP}/config.json" not in files or \
            f"{INIT_RUN}/checkpoint-{START_STEP}/state.pt" not in files:
        return False, f"waiting for warm-start {INIT_RUN}/checkpoint-{START_STEP}"
    return True, f"corpus ready ({len(TRAIN_TAGS)} shards) + {INIT_RUN}/checkpoint-{START_STEP}"


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
subprocess.run(["git", "init", "-q", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "remote", "add", "origin",
                "https://github.com/Vedang-P/chess-slm-benchmark.git"], check=True)
for attempt in range(4):
    if subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "--depth", "1",
                       "origin", SOURCE_COMMIT]).returncode == 0:
        break
    time.sleep(20)
subprocess.run(["git", "-C", str(REPO), "checkout", "-q", "FETCH_HEAD"], check=True)
actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                 text=True).strip()
assert actual == SOURCE_COMMIT, f"pinned checkout mismatch: {actual} != {SOURCE_COMMIT}"
print(f"[2b] repo pinned at {SOURCE_COMMIT}", flush=True)
if not SL.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/google-deepmind/searchless_chess.git", str(SL)], check=True)
pz = SL / "data" / "puzzles.csv"
if not pz.exists():
    subprocess.run(["curl", "-sL", "--fail", "-o", str(pz),
                    "https://storage.googleapis.com/searchless_chess/data/puzzles.csv"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-chess", "pandas"], check=True)

out = WORK / RUN
cmd = [sys.executable, str(REPO / "scripts" / "train_ccgavn.py"),
       "--hf-shards", PREFIX, "--outdir", str(out), "--sl-repo", str(SL),
       "--shard-tags-file", str(REPO / "configs" / "ccgavn-2b-shard-tags.json"),
       "--dim", "208", "--layers", "8", "--heads", "8", "--batch", "2048",
       "--steps", str(TOTAL_STEPS), "--lr", "0.0005", "--warmup", "2000",
       "--temperature", "1.0", "--w-dist", "1.0", "--w-ce", "0.25",
       "--reflect-prob", "0.5", "--dev-mod", "100", "--dev-fold", "0",
       "--dev-batch", "8192", "--seed", "0", "--ckpt-every", "5000",
       "--hf-repo", HF_REPO, "--hf-run", RUN, "--hf-upload-every", "1800",
       "--stage-start-step", str(START_STEP),
       "--exclusion-puzzles", str(pz),
       "--init-from-hf-run", INIT_RUN,
       "--init-checkpoint", f"checkpoint-{START_STEP}",
       "--resume-from-hf"]
print("[2b] " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=str(REPO))
print("[2b] training returned", flush=True)
