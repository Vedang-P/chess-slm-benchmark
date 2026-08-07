"""Generate kaggle_mate{N}_{VARIANT}_w{K}.ipynb -- five parallel CPU
workers per MATE subset (noexplain / tactic / both), mirroring the
strategy-subset 1000-position deepseek thinking run exactly.

    python notebooks/build_mate1000_variants_notebook.py

The strategy run (HF run_ids mate1000-w1..w5, 91.0% headline) used 5
workers x 180 positions with its own API key each (the gateway serializes
per-key) at --max_new_tokens 131072 --force-answer-prompt with thinking
ON and no --thinking-budget cap. We replicate that recipe here per
subset, one kernel id per (subset, worker), with the eval file selected
via the new --task-file flag in run_mate_eval.py.

Slices: 5 workers x 200 positions over the 1000-position subset.
Tags/BENCH_RUN_IDs are per-subset so live-push and HF archives never
collide between variants:
    noexplain: mate-noexplain-w1..w5, tags noexplain-w1..w5
    tactic:    mate-tactic-w1..w5,    tags tactic-w1..w5
    both:      mate-both-w1..w5,      tags both-w1..w5

Kaggle caps concurrent CPU kernel sessions at 5 per account (measured
directly -- a 6th push failed with "Maximum batch CPU session count of
5 reached"), so push ONE subset at a time (5 kernels), wait for all five
to COMPLETE, then push the next.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent

MAX_NEW_TOKENS = 131072
SLICE_SIZE = 200

GITHUB_TOKEN = "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"
HF_TOKEN = "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"

# the five per-worker gateway keys (opencode keys; deepseek-v4-flash via
# the opencode gateway) -- only one subset runs at a time, so no key is
# shared by two live kernels
API_KEYS = [
    "sk-c9Ss9jld4cCohNh55KCwQSuU3Ql4yJEmBiFP1NSetyvHGX0RDwdvU1DKdEysgqg7",
    "sk-GcDtOakeOqVNk04SYjKNPoA692hX0lijMfi2RjpHJYT1jKmlIxRdD0gbvg6X8iU6",
    "sk-KMBJo5xbOVI4M5CfWpLyAlTPlayjVsfpJgh0bDvthgfJKMpTqvcLqk1xd7ai6V2A",
    "sk-K4Et7DRL4AdR4L4ERo8dQLvOrggPj2TBFGpP4eBiZMsEqCJvTUjL9xiSAtxOuMBy",
    "sk-cB6D8wNqSUHA4D03qGITFf22FhMlTtcZWepWbvLcIXbFk1QurFjKIsn8oWOIwifo",
]

VARIANTS = ["noexplain", "tactic", "both"]

TASK_FILES = {
    "noexplain": "mate-selection-test-noexplain.json",
    "tactic": "mate-selection-test-tactic.json",
    "both": "mate-selection-test-both.json",
}


def secrets_cell(key: str) -> dict:
    return _code(f'''
import os
from pathlib import Path
os.environ["GITHUB_TOKEN"] = {GITHUB_TOKEN!r}
os.environ["HF_TOKEN"] = {HF_TOKEN!r}
os.environ["OPENCODE_API_KEY"] = {key!r}
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


def recover_cell(out_dir: str, run_name: str, bench_run_id: str) -> dict:
    return _code(f'''
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

OUT_DIR = Path({out_dir!r})
OUT_DIR.mkdir(parents=True, exist_ok=True)
BENCH_RUN_ID = {bench_run_id!r}
RUN_NAME = {run_name!r}

# /kaggle/working is wiped on every "Restart & Run All", so a relaunched
# kernel must pull this worker's own HF checkpoint back down first --
# otherwise --resume has nothing local to resume from and the whole slice
# silently restarts from zero.
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


def run_cell(out_dir: str, worker_tag: str, bench_run_id: str,
             offset: int, task_file: str) -> dict:
    return _code(f'''
import time
from pathlib import Path

os.environ["BENCH_RUN_ID"] = {bench_run_id!r}
out = Path({out_dir!r})
out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "deepseek-v4-flash",
       "--task-file", {task_file!r},
       "--offset", "{offset}",
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


def summary_cell(out_dir: str, slice_size: int) -> dict:
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
print(f"this worker: {{n}} / {slice_size} positions attempted")
print("status breakdown:", dict(by_status))
if scored:
    print(f"accuracy (of {{scored}} scored): {{correct}}/{{scored}} = {{correct/scored:.3f}}")
'''.strip())


def build_worker_notebook(variant: str, n: int) -> list:
    offset = (n - 1) * SLICE_SIZE
    worker_tag = f"{variant}-w{n}"
    bench_run_id = f"mate-{variant}-w{n}"
    out_dir = f"results/mate-{variant}-w{n}"
    run_name = f"deepseek-v4-flash_mate-selection-test_{variant}"
    task_file = TASK_FILES[variant]

    cells = [
        _md(f"# MATE {variant}: Worker {n} of 5 (positions [{offset}:{offset + SLICE_SIZE}])\n\n"
            f"5-way parallel deepseek thinking-mode run over the MATE **{variant}** "
            "subset (1000 positions, `data/positions/" + task_file + "`). "
            "Same methodology as the strategy-subset 1000-position run (HF "
            "run_ids mate1000-w1..w5, 91.0% headline): thinking ON, no "
            f"`--thinking-budget` cap, `--force-answer-prompt`, `--max_new_tokens {MAX_NEW_TOKENS}`.\n\n"
            "Each worker owns a disjoint 200-position slice and its own API "
            "key (the gateway serializes per-key, so 5 keys = real 5x "
            "parallelism)."),
        secrets_cell(API_KEYS[n - 1]),
        _md("## Get the repo (main branch)"),
        _code(CLONE_CELL),
        _md("## Dependencies"),
        _code(DEPS_CELL),
        _md("## Engine/dataset gate"),
        _code(GATE_CELL),
        _md("## Recover progress (resume support)"),
        recover_cell(out_dir, run_name, bench_run_id),
        _md(f"## Run positions [{offset}:{offset + SLICE_SIZE}] of {variant}"),
        run_cell(out_dir, worker_tag, bench_run_id, offset, task_file),
        _md("## This worker's summary"),
        summary_cell(out_dir, SLICE_SIZE),
    ]
    return cells


def kernel_metadata(variant: str, n: int) -> dict:
    return {
        "id": f"vedangpandeyyy/mate-{variant}-worker-{n}-of-5",
        "title": f"MATE {variant} -- worker {n} of 5",
        "code_file": f"kaggle_mate_{variant}_w{n}.ipynb",
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
    for variant in VARIANTS:
        for n in range(1, 6):
            meta_dir = NB_DIR / f"push_mate_{variant}_w{n}"
            meta_dir.mkdir(exist_ok=True)
            nb_path = meta_dir / f"kaggle_mate_{variant}_w{n}.ipynb"
            nb_path.write_text(json.dumps(_notebook(build_worker_notebook(variant, n)), indent=1))
            (meta_dir / "kernel-metadata.json").write_text(
                json.dumps(kernel_metadata(variant, n), indent=1))
            print(f"wrote {meta_dir}/kernel-metadata.json (id={kernel_metadata(variant, n)['id']})")


if __name__ == "__main__":
    main()
