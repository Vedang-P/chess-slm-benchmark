"""Generate kaggle_mate1000_w{N}.ipynb for N in 1..5 -- five parallel CPU
workers covering the remaining 900 MATE positions (100-999; the first 100
are already complete and archived at HF run 2026-08-04T16:09:37Z and are
never touched).

    python notebooks/build_mate1000_worker_notebook.py

5 workers, not 6: Kaggle caps CPU kernel sessions at 5 concurrent per
account (measured directly -- a 6th push failed with "Maximum batch CPU
session count of 5 reached"; user decision 2026-08-04: design around 5,
don't queue a 6th).

Each worker gets a disjoint 180-position slice and its OWN API key, so the
gateway's documented per-key serialization (one in-flight generation per
key) doesn't bottleneck a single key across 900 slow thinking-mode calls:
    w1: positions[100:280]   w4: positions[640:820]
    w2: positions[280:460]   w5: positions[820:1000]
    w3: positions[460:640]

Same config as the verified 100-position run (thinking on, no
--thinking-budget cap, --force-answer-prompt, 131072 max_new_tokens) --
this is meant to be the SAME methodology at ~10x scale, not a different one.

Worker 1 reuses the existing kernel
(vedangpandeyyy/mate-thinking-100-resume-14-truncated), which already has
GITHUB_TOKEN/HF_TOKEN/OPENCODE_API_KEY working via hardcoded env vars from
earlier session runs. That cell is left AS IS here rather than switched to
Kaggle Secrets -- not touching a known-good credential path right before a
long unattended run.

Workers 2-5 hardcode credentials too (user decision 2026-08-04: "just use
the same tokens dude" -- same GITHUB_TOKEN/HF_TOKEN as worker 1 for all of
them, only the OPENCODE_API_KEY differs per worker, taken straight from
the local .env's OPENCODE_API_KEY_2 through _5). An earlier draft of this
generator required the user to manually create+attach Kaggle Secrets per
kernel before anything could run -- that's needless friction for a
private notebook using the same low-risk pattern the existing kernel
already uses successfully; dropped in favor of hardcoding, matching
worker 1.

Every worker also recovers its own progress from HF before running
--resume: /kaggle/working is wiped on every "Restart & Run All" (the
clone step re-creates it), so without this, --resume would have nothing
local to resume FROM after a Kaggle session dies and gets relaunched --
it would just silently restart the whole 180-position slice from zero.

The clone step targets `-b main` explicitly: this repo has other feature
branches in active use (e.g. mate-e2b-kaggle) and a bare `git clone` picks
the remote's default branch, which happens to be main today -- but being
explicit here means a future default-branch change on GitHub can't
silently break which code Kaggle actually runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent

MAX_NEW_TOKENS = 131072
SLICE_SIZE = 180
BASE_OFFSET = 100  # the first 100 positions are done; workers start here

# shared across every worker (user decision 2026-08-04: same tokens for all)
GITHUB_TOKEN = "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"
HF_TOKEN = "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"

WORKERS = [
    {"n": 1, "offset": BASE_OFFSET + 0 * SLICE_SIZE,
     "kernel_id": "vedangpandeyyy/mate-thinking-100-resume-14-truncated",
     "kernel_title": "MATE Thinking 100 -- resume 14 truncated",
     "reuse_existing_kernel": True,
     "opencode_key": "sk-u7Ygubnm46wXqkIlbIGlhB8ApJ71YBNTSLkQaBPPHmjNt5bTPY1Dcf14yutM7hyn"},
    {"n": 2, "offset": BASE_OFFSET + 1 * SLICE_SIZE,
     "kernel_id": "vedangpandeyyy/mate-1000-worker-2-of-5",
     "kernel_title": "MATE 1000 -- worker 2 of 5", "reuse_existing_kernel": False,
     "opencode_key": "sk-pEX7KXuKmtkPgT6zXiBGU8skp52GTLcf8KJ9NhSxXzenML1RGusqtAN7ANOx3TWW"},
    {"n": 3, "offset": BASE_OFFSET + 2 * SLICE_SIZE,
     "kernel_id": "vedangpandeyyy/mate-1000-worker-3-of-5",
     "kernel_title": "MATE 1000 -- worker 3 of 5", "reuse_existing_kernel": False,
     "opencode_key": "sk-0idWGQ5O8kYuuxzcZKslB1jB6L5K0GifxbH5DaQr5PCvqTOI5T0QpTBAUYH5lAZi"},
    {"n": 4, "offset": BASE_OFFSET + 3 * SLICE_SIZE,
     "kernel_id": "vedangpandeyyy/mate-1000-worker-4-of-5",
     "kernel_title": "MATE 1000 -- worker 4 of 5", "reuse_existing_kernel": False,
     "opencode_key": "sk-2zswwTiXrbgWcKnVLdUR4UknFl9GcNx0PWd2QMdPIP8Utwb5oDx7ISigmk4sVyjG"},
    {"n": 5, "offset": BASE_OFFSET + 4 * SLICE_SIZE,
     "kernel_id": "vedangpandeyyy/mate-1000-worker-5-of-5",
     "kernel_title": "MATE 1000 -- worker 5 of 5", "reuse_existing_kernel": False,
     "opencode_key": "sk-qzsQELq8YiqAd9jcDD9dKuPHVR0rbKy1bTCTiNqFDf4Vsk7D9KGLcnK745xQPJBm"},
]
assert WORKERS[-1]["offset"] + SLICE_SIZE == 1000, "worker slices must cover through position 1000"


def secrets_cell(worker: dict) -> dict:
    """Hardcoded env vars, same pattern as the already-working kernel for
    every worker -- no Kaggle Secrets UI step required, nothing to attach
    before a push actually runs. Each worker gets its own OPENCODE_API_KEY
    so the gateway's per-key serialization doesn't bottleneck them; the
    other two tokens are shared (user decision 2026-08-04)."""
    return _code(f'''
import os
from pathlib import Path
os.environ["GITHUB_TOKEN"] = {GITHUB_TOKEN!r}
os.environ["HF_TOKEN"] = {HF_TOKEN!r}
os.environ["OPENCODE_API_KEY"] = {worker["opencode_key"]!r}
print("env secrets set:", bool(os.environ["GITHUB_TOKEN"]), bool(os.environ["HF_TOKEN"]), bool(os.environ["OPENCODE_API_KEY"]))
'''.strip())


CLONE_CELL = r'''
import shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
if REPO.exists():
    shutil.rmtree(REPO)

url = "https://github.com/Vedang-P/chess-slm-benchmark.git"
url = url.replace("https://", f"https://x-access-token:{os.environ['GITHUB_TOKEN']}@")
res = subprocess.run(["git", "clone", "--quiet", "-b", "main", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed (bad/missing GITHUB_TOKEN?): " + res.stderr[-300:])
os.chdir(REPO)
print("cwd:", Path.cwd())
'''.strip()

DEPS_CELL = r'''
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U", "-r", "requirements.txt"], check=True)
print("deps installed")
'''.strip()

GATE_CELL = r'''
status = subprocess.run([sys.executable, "scripts/test_engine.py"], capture_output=True, text=True)
print(status.stdout[-3000:])
if status.returncode != 0:
    print(status.stderr[-2000:])
    raise RuntimeError("test_engine failed -- see output above")
print("ALL TESTS PASSED")
'''.strip()


def recover_cell(worker: dict, out_dir: str, run_name: str, bench_run_id: str) -> dict:
    return _code(f'''
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

OUT_DIR = Path({out_dir!r})
OUT_DIR.mkdir(parents=True, exist_ok=True)
BENCH_RUN_ID = {bench_run_id!r}
RUN_NAME = {run_name!r}

# /kaggle/working is wiped on every "Restart & Run All" (the clone cell
# re-creates it), so if THIS kernel died mid-slice and is being relaunched,
# there is nothing local to --resume from unless we pull this worker's own
# progress back down first. Each worker uploads to its OWN run_id path
# (see run_mate_eval.py --hf-upload-every), so this can never collide with
# another worker's or the original 100's data.
try:
    src = hf_hub_download(
        repo_id="vedangfake/chess-bench-results", repo_type="dataset",
        filename=f"runs/{{BENCH_RUN_ID}}/{{RUN_NAME}}.samples.jsonl",
        token=os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN"))
    dest = OUT_DIR / f"{{RUN_NAME}}.samples.jsonl"
    dest.write_bytes(Path(src).read_bytes())
    n = sum(1 for line in dest.read_text().splitlines() if line.strip())
    print(f"recovered {{n}} previously-scored positions for this worker from HF -- --resume will skip them")
except Exception as e:
    print(f"nothing to recover (first launch of this worker, or no checkpoint yet): {{e}}")
'''.strip())


def run_cell(worker: dict, out_dir: str, worker_tag: str, bench_run_id: str) -> dict:
    return _code(f'''
import time
from pathlib import Path

os.environ["BENCH_RUN_ID"] = {bench_run_id!r}
out = Path({out_dir!r})
out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "deepseek-v4-flash",
       "--offset", "{worker['offset']}",
       "--n", "{SLICE_SIZE}",
       "--max_new_tokens", "{MAX_NEW_TOKENS}",
       "--force-answer-prompt",
       "--worker-id", {worker_tag!r},
       "--output_dir", {out_dir!r},
       "--live-push",
       "--resume",
       "--verbose"]
print("running:", " ".join(cmd))
t0 = time.time()
# unbuffered + merged stderr so a crash's real traceback lands in THIS
# cell's own output instead of vanishing into the subprocess's own stderr
# pipe (which showed up nowhere -- an earlier version of this cell just
# printed a return code and let the notebook finish "successfully" even
# when the actual run crashed instantly).
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
elapsed_h = (time.time() - t0) / 3600
print(f"run exited rc={{res.returncode}} after {{elapsed_h:.2f}}h")
if res.returncode != 0:
    raise RuntimeError(
        f"run_mate_eval.py exited rc={{res.returncode}} after only "
        f"{{elapsed_h:.2f}}h -- this is a real failure, not a completed run. "
        "See the subprocess output printed above this cell for the actual "
        "traceback."
    )
'''.strip())


def summary_cell(out_dir: str, run_name: str) -> dict:
    return _code(f'''
import json, glob, collections

rows = []
for f in glob.glob(str(Path({out_dir!r}) / "*.samples.jsonl")):
    for line in open(f):
        if line.strip(): rows.append(json.loads(line))
n = len(rows)
by_status = collections.Counter(r["status"] for r in rows)
correct = sum(bool(r["compliance"]) for r in rows if r["status"] != "api_error")
scored = n - by_status.get("api_error", 0)
print(f"this worker: {{n}} / {SLICE_SIZE} positions attempted")
print("status breakdown:", dict(by_status))
if scored:
    print(f"accuracy (of {{scored}} scored): {{correct}}/{{scored}} = {{correct/scored:.3f}}")
'''.strip())


def build_worker_notebook(worker: dict) -> list:
    n = worker["n"]
    offset = worker["offset"]
    worker_tag = f"w{n}"
    bench_run_id = f"mate1000-w{n}"
    out_dir = f"results/mate-1000-w{n}"
    run_name = "deepseek-v4-flash_mate-selection-test_strategy"

    cells = [
        _md(f"# MATE 1000: Worker {n} of 5 (positions [{offset}:{offset + SLICE_SIZE}])\n\n"
            "Part of a 5-way parallel extension of the verified 100-position thinking-mode "
            "run to the full 1000-position mate-selection-test.json (5, not 6: Kaggle caps "
            "concurrent CPU kernel sessions at 5 per account). Each worker owns a disjoint "
            "180-position slice and its own API key (the gateway serializes per-key, so 5 "
            "keys = real 5x parallelism, not 5 queued requests on one key).\n\n"
            "Same methodology as the archived 100 (HF run `2026-08-04T16:09:37Z`, 94/100): "
            "thinking ON, no `--thinking-budget` cap, `--force-answer-prompt`, "
            f"`--max_new_tokens {MAX_NEW_TOKENS}`.\n\n"
            + "Credentials are hardcoded in the first cell (same GITHUB_TOKEN/HF_TOKEN as "
              "every other kernel; this worker's own OPENCODE_API_KEY) -- no manual secret "
              "attachment needed, this kernel runs as soon as it's pushed."),
        _md("## 1. Secrets"),
        secrets_cell(worker),
        _md("## 2. Get the repo (main branch, explicitly)"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies (CPU only -- no torch/gpu install needed for the gateway arm)"),
        _code(DEPS_CELL),
        _md("## 4. Engine/dataset gate"),
        _code(GATE_CELL),
        _md("## 5. Recover this worker's own progress (if this is a relaunch after a "
            "died/timed-out session)"),
        recover_cell(worker, out_dir, run_name, bench_run_id),
        _md(f"## 6. Run positions [{offset}:{offset + SLICE_SIZE}]\n\n"
            "Uploads to HF every 25 positions (not just at the end) and publishes live "
            f"state to `monitor/workers/{worker_tag}.state.json` on the public monitor "
            "repo -- scripts/aggregate_live_state.py combines all 5 workers + the "
            "completed 100 into the canonical dashboard the website reads."),
        run_cell(worker, out_dir, worker_tag, bench_run_id),
        _md("## 7. This worker's summary"),
        summary_cell(out_dir, run_name),
        _md("## Notes\n"
            f"- Scope: ONLY positions [{offset}:{offset + SLICE_SIZE}) of "
            "mate-selection-test.json. Never touches the first 100 or another "
            "worker's slice.\n"
            "- If this session dies/times out, just push this same notebook again "
            "(or Restart & Run All): step 5 recovers this worker's own progress from "
            "HF, and --resume skips it.\n"
            f"- Results upload to HF under run_id `{bench_run_id}` and the live "
            "dashboard at chess-bench-live.pages.dev."),
    ]
    return cells


def kernel_metadata(worker: dict) -> dict:
    return {
        "id": worker["kernel_id"],
        "title": worker["kernel_title"],
        "code_file": f"kaggle_mate1000_w{worker['n']}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def main() -> None:
    for worker in WORKERS:
        n = worker["n"]
        nb_path = NB_DIR / f"kaggle_mate1000_w{n}.ipynb"
        nb_path.write_text(json.dumps(_notebook(build_worker_notebook(worker)), indent=1))
        print(f"wrote {nb_path}")
        meta_dir = NB_DIR / f"push_w{n}"
        meta_dir.mkdir(exist_ok=True)
        (meta_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(worker), indent=1))
        (meta_dir / f"kaggle_mate1000_w{n}.ipynb").write_text(nb_path.read_text())
        print(f"wrote {meta_dir}/kernel-metadata.json (id={worker['kernel_id']})")


if __name__ == "__main__":
    main()
