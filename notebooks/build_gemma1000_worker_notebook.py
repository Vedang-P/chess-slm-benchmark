"""Generate the Gemma 4 E2B 1000-position campaign notebooks:
kaggle_gemma1000_w1.ipynb + kaggle_gemma1000_w2.ipynb (GPU workers,
500 positions each) and kaggle_gemma1000_agg.ipynb (CPU aggregator).

    python notebooks/build_gemma1000_worker_notebook.py

The SAME 1000 positions the DeepSeek direct-mode run scored
(data/positions/mate-selection-test.json, 48.6% strict at n=1000), now with
local Gemma 4 E2B thinking ON -- the exact methodology validated on the
check10 kernel (10 positions, 50%, all answered, thinking split verified).

Kaggle's free tier allows TWO concurrent GPU kernels, so the 1000 is split
into two disjoint 500-position slices:
    w1: positions [0:500)   w2: positions [500:1000)

Each worker: --worker-id wN --live-namespace gemma (streams to the existing
gemma dashboard page), --hf-upload-every 25 (backup every 25 positions),
--resume, and a recovery cell that pulls the worker's own checkpoint back
from HF before running -- /kaggle/working is wiped on every session restart,
so a died session relaunches and continues instead of restarting from zero.

The aggregator kernel (CPU -- does not count against the 2-GPU limit) loops
scripts/aggregate_live_state.py --namespace gemma --run-id gemma-1000-campaign
--workers w1,w2, combining both workers into monitor/gemma/state.json +
live.json for the live dashboard.

Timing (measured check10: 78.7s/position on a P100): 500 positions ~ 10.9h
per worker -- inside Kaggle's 12h session limit; T4 is ~2x faster if assigned.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent

MAX_NEW_TOKENS = 32768
SLICE_SIZE = 500

WORKERS = [
    {"n": 1, "offset": 0},
    {"n": 2, "offset": SLICE_SIZE},
]
assert WORKERS[-1]["offset"] + SLICE_SIZE == 1000

BRANCH = "mate-e2b-kaggle"
RUN_NAME = "gemma4-e2b_mate-selection-test_strategy"

CLONE_CELL = r'''
import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
if REPO.exists():
    shutil.rmtree(REPO)

def find_token():
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        return None

token = find_token()
url = "https://github.com/Vedang-P/chess-slm-benchmark.git"
if token:
    url = url.replace("https://", f"https://x-access-token:{token}@")
# --local-thinking, --live-namespace, scored-state live publishing and the
# mid-stream thinking split live on the mate-e2b-kaggle branch; clone the
# branch explicitly so the kernel always runs the intended code.
res = subprocess.run(["git", "clone", "--quiet", "--branch", "mate-e2b-kaggle",
                      url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed (token not attached?): " + res.stderr[-300:])
os.chdir(REPO)
print("cwd:", Path.cwd())
'''.strip()

DEPS_CELL = r'''
import subprocess, sys
# Kaggle's free tier hands out a P100 (sm_60) OR a T4 (sm_75). Recent torch
# wheels dropped sm_60, so pin the last CUDA-12.1 build with both archs
# BEFORE anything else installs torch; torchvision/torchaudio must match the
# pinned torch (Kaggle ships torchvision built for the latest torch -- the
# ABI mismatch crashes transformers' AutoProcessor import). bitsandbytes is
# pinned to the matching multi-CUDA build, and the requirements install
# afterwards must NOT clobber these pins (no -U: it still upgrades
# transformers to >=5.13 on its own).
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "torch==2.5.1", "--index-url",
                "https://download.pytorch.org/whl/cu121"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "torchvision==0.20.1", "torchaudio==2.5.1", "--index-url",
                "https://download.pytorch.org/whl/cu121"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "bitsandbytes==0.44.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "-r", "requirements.txt"], check=True)
import torch, transformers
if int(transformers.__version__.split(".")[0]) < 5:
    raise RuntimeError(f"transformers {transformers.__version__} too old "
                       "for Gemma 4 (needs >= 5.13)")
print("transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0),
          "| cap", torch.cuda.get_device_capability(0),
          "| vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
'''.strip()

GATE_CELL = r'''
status = subprocess.run([sys.executable, "scripts/test_engine.py", "--quick"],
                        capture_output=True, text=True)
if status.returncode != 0:
    print(status.stdout[-2000:]); print(status.stderr[-2000:])
    raise RuntimeError("test_engine failed")
print("ALL TESTS PASSED")
'''.strip()


def recover_cell(worker: dict, out_dir: str, bench_run_id: str) -> dict:
    return _code(f'''
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

OUT_DIR = Path({out_dir!r})
OUT_DIR.mkdir(parents=True, exist_ok=True)
BENCH_RUN_ID = {bench_run_id!r}
RUN_NAME = {RUN_NAME!r}

# /kaggle/working is wiped on every session restart, so if this kernel died
# mid-slice and is being relaunched, --resume has nothing local to resume
# FROM unless we pull this worker's own checkpoint back down first. Each
# worker uploads to its OWN run_id path every 25 positions, so this can
# never collide with the other worker's data.
try:
    src = hf_hub_download(
        repo_id="vedangfake/chess-bench-results", repo_type="dataset",
        filename=f"runs/{{BENCH_RUN_ID}}/{{RUN_NAME}}.samples.jsonl",
        token=os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN"))
    dest = OUT_DIR / f"{{RUN_NAME}}.samples.jsonl"
    dest.write_bytes(Path(src).read_bytes())
    n = sum(1 for line in dest.read_text().splitlines() if line.strip())
    print(f"recovered {{n}} previously-scored positions from HF -- --resume will skip them")
except Exception as e:
    print(f"nothing to recover (first launch, or no checkpoint yet): {{e}}")
'''.strip())


def demo_cell(worker: dict, out_dir: str, worker_tag: str, bench_run_id: str) -> dict:
    return _code(f'''
import os, subprocess, sys, time
from pathlib import Path

# 2-position demo through the REAL pipeline: model load, thinking channel,
# extraction, live push, HF end-upload -- everything the full run does, on
# whichever GPU Kaggle assigned (T4 or P100; both are supported by the
# pinned torch build). Shares the run id + output dir with the full run, so
# the archive ends up with ONE cell and the full run resumes past these 2.
os.environ["BENCH_RUN_ID"] = {bench_run_id!r}
out = Path({out_dir!r})
out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "gemma4-e2b",
       "--offset", "{worker['offset']}",
       "--n", "2",
       "--max_new_tokens", "{MAX_NEW_TOKENS}",
       "--force-answer-prompt",
       "--local-thinking",
       "--worker-id", {worker_tag!r},
       "--live-namespace", "gemma",
       "--output_dir", {out_dir!r},
       "--live-push",
       "--resume",
       "--verbose"]
print("demo:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
print(f"demo exit rc={{res.returncode}} after {{(time.time()-t0)/60:.1f}}min")
if res.returncode != 0:
    raise RuntimeError(f"demo failed with rc={{res.returncode}} -- fix before the full run")
'''.strip())


def inspect_cell(out_dir: str) -> dict:
    return _code(f'''
import json, glob
from pathlib import Path

files = glob.glob(str(Path({out_dir!r}) / "*.samples.jsonl"))
if not files:
    raise RuntimeError(f"no samples file under {out_dir!r} -- the demo did not write results")
rows = [json.loads(l) for l in open(files[0]) if l.strip()]
print(f"demo samples on disk: {{len(rows)}}")
for s in rows:
    tu = s.get("token_usage") or {{}}
    print(f"{{s['position_id']}}: {{s['status']:9s}} move={{s.get('move')}} "
          f"correct={{s.get('compliance')}} reasoning_chars={{s.get('reasoning_chars')}} "
          f"reason_tokens={{tu.get('reasoning_tokens')}}")
parsed = [r for r in rows if r["status"] in ("correct", "wrong")]
unclean = [r for r in rows if not (r.get("reasoning") or "").strip()]
if not parsed:
    raise RuntimeError("0 parsed positions -- extraction/thinking split is broken on this GPU")
if unclean:
    raise RuntimeError(f"{{len(unclean)}} sample(s) have no reasoning text -- the thinking "
                       "channel split is broken; check the samples above")
print(f"\\nDEMO OK: {{len(parsed)}}/{{len(rows)}} parsed, thinking split verified -- safe to run the full slice")
'''.strip())


def run_cell(worker: dict, out_dir: str, worker_tag: str, bench_run_id: str) -> dict:
    return _code(f'''
import os, subprocess, sys, time
from pathlib import Path

os.environ["BENCH_RUN_ID"] = {bench_run_id!r}
out = Path({out_dir!r})
out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "gemma4-e2b",
       "--offset", "{worker['offset']}",
       "--n", "{SLICE_SIZE}",
       "--max_new_tokens", "{MAX_NEW_TOKENS}",
       "--force-answer-prompt",
       "--local-thinking",
       "--worker-id", {worker_tag!r},
       "--live-namespace", "gemma",
       "--output_dir", {out_dir!r},
       "--live-push",
       "--hf-upload-every", "25",
       "--resume",
       "--verbose"]
print("running:", " ".join(cmd))
t0 = time.time()
# unbuffered + merged stderr so a crash's real traceback lands in THIS
# cell's own output instead of vanishing into the subprocess's stderr pipe.
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
elapsed_h = (time.time() - t0) / 3600
print(f"run exited rc={{res.returncode}} after {{elapsed_h:.2f}}h")
if res.returncode != 0:
    raise RuntimeError(
        f"run_mate_eval.py exited rc={{res.returncode}} after only "
        f"{{elapsed_h:.2f}}h -- a real failure, not a completed run. See the "
        "subprocess output printed above for the traceback."
    )
'''.strip())


def summary_cell(out_dir: str) -> dict:
    return _code(f'''
import json, glob, collections
from pathlib import Path

rows = []
for f in glob.glob(str(Path({out_dir!r}) / "*.samples.jsonl")):
    for line in open(f):
        if line.strip():
            rows.append(json.loads(line))
n = len(rows)
by_status = collections.Counter(r["status"] for r in rows)
correct = sum(bool(r["compliance"]) for r in rows if r["status"] != "api_error")
scored = n - by_status.get("api_error", 0)
print(f"this worker: {{n}} / {SLICE_SIZE} positions attempted")
print("status breakdown:", dict(by_status))
if scored:
    print(f"accuracy (of {{scored}} scored): {{correct}}/{{scored}} = {{correct/scored:.3f}}")
print()
print("DeepSeek reference (direct mode, 1000 positions): 48.6% strict")
'''.strip())


def build_worker_notebook(worker: dict) -> list:
    n = worker["n"]
    offset = worker["offset"]
    worker_tag = f"w{n}"
    bench_run_id = f"gemma1000-w{n}"
    out_dir = f"results/gemma-1000-w{n}"

    cells = [
        _md(f"# Gemma 4 E2B 1000-position campaign: Worker {n} of 2 (positions [{offset}:{offset + SLICE_SIZE}])\n\n"
            "Local Gemma 4 E2B, thinking ON, on the SAME 1000 positions the "
            "DeepSeek direct-mode run scored (48.6% strict). Two GPU workers "
            "cover 500 positions each (Kaggle's 2-concurrent-GPU limit). "
            "Methodology identical to the validated check10 kernel: "
            f"`--local-thinking`, `--force-answer-prompt`, `--max_new_tokens {MAX_NEW_TOKENS}`.\n\n"
            "Backups: HF upload every 25 positions; the recovery cell pulls "
            "this worker's checkpoint back before `--resume`, so a died "
            "session relaunches and continues. Live: streams to "
            "monitor/gemma/workers/w{n}.* -- the aggregator kernel combines "
            "both workers into the gemma dashboard page."),
        _md("## 1. Secrets (hardcoded env vars)"),
        _code("import os\n"
              'print("secrets are injected at build time; this cell is a placeholder")'),
        _md("## 2. Get the repo (mate-e2b-kaggle branch)"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies (torch pins for P100+T4)"),
        _code(DEPS_CELL),
        _md("## 4. Engine/dataset gate"),
        _code(GATE_CELL),
        _md("## 5. Demo: 2 positions through the REAL pipeline\n\n"
            "Verifies model load, the thinking channel split, extraction, "
            "live push and the HF upload all work on the GPU Kaggle assigned "
            "(T4 or P100 -- the pinned torch 2.5.1 build supports both). "
            "The demo shares the run id and output dir with the full run, so "
            "the full run's `--resume` skips these 2 positions and the HF "
            "archive ends up with one complete cell."),
        demo_cell(worker, out_dir, worker_tag, bench_run_id),
        _md("## 6. Inspect the demo before the full run\n\n"
            "Raises if anything is broken: no samples written, 0 parsed "
            "answers, or missing reasoning text (thinking split failed). "
            "On a relaunch after a died session this cell validates the "
            "recovered samples instead."),
        inspect_cell(out_dir),
        _md(f"## 7. Run positions [{offset}:{offset + SLICE_SIZE}]\n\n"
            "HF backup every 25 positions; live state to "
            f"monitor/gemma/workers/{worker_tag}.* (the aggregator kernel "
            "combines both workers into the dashboard)."),
        run_cell(worker, out_dir, worker_tag, bench_run_id),
        _md("## 8. This worker's summary"),
        summary_cell(out_dir),
        _md("## Notes\n"
            f"- Scope: ONLY positions [{offset}:{offset + SLICE_SIZE}) of "
            "mate-selection-test.json. The other worker owns the rest.\n"
            "- If this session dies/times out, push the same notebook again "
            "(or Restart & Run All): the demo/inspect cells are no-ops once "
            "the slice is done, step 5 recovers progress from HF, --resume "
            "skips it.\n"
            "- GPU: Kaggle assigns T4 or P100 -- both supported; T4 is ~2x "
            "faster. ~78s/position measured on P100 (check10), so 500 "
            "positions ~ 11h, within the 12h session limit.\n"
            f"- Results: HF runs/{bench_run_id}/; live: chess-bench-live.pages.dev/gemma.html"),
    ]
    return cells


def build_agg_notebook() -> list:
    return [
        _md("# Gemma 4 E2B 1000 campaign: Aggregator (CPU)\n\n"
            "Combines the two GPU workers' live state "
            "(monitor/gemma/workers/w1.*, w2.*) into the canonical "
            "monitor/gemma/state.json + history.jsonl + live.json the gemma "
            "dashboard page reads. Runs every 45s for the whole campaign. "
            "CPU-only -- does not count against Kaggle's 2-concurrent-GPU "
            "limit. If this session dies, relaunch it; nothing is lost."),
        _md("## 1. Secrets (hardcoded env vars)"),
        _code("import os\n"
              'print("secrets are injected at build time; this cell is a placeholder")'),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Run the aggregator (forever)"),
        _code('''
import subprocess, sys
cmd = [sys.executable, "scripts/aggregate_live_state.py",
       "--namespace", "gemma",
       "--run-id", "gemma-1000-campaign",
       "--workers", "w1,w2",
       "--interval", "45"]
print("running:", " ".join(cmd))
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
if res.returncode != 0:
    raise RuntimeError(f"aggregator exited rc={res.returncode} -- see output above")
'''.strip()),
        _md("## Notes\n"
            "- If a worker stops publishing, the aggregator keeps going with "
            "the remaining worker and shows a stale worker in its report.\n"
            "- The dashboard: chess-bench-live.pages.dev/gemma.html"),
    ]


def kernel_metadata(kernel_id: str, title: str, code_file: str, gpu: bool) -> dict:
    return {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def main() -> None:
    import os

    env = {}
    for line in (NB_DIR.parent / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")

    for worker in WORKERS:
        n = worker["n"]
        clean = _notebook(build_worker_notebook(worker))
        nb_path = NB_DIR / f"kaggle_gemma1000_w{n}.ipynb"
        nb_path.write_text(json.dumps(clean, indent=1))
        print(f"wrote {nb_path} (clean, placeholder secrets)")
        nb = json.loads(json.dumps(clean))
        inject_secrets(nb, env, ["GITHUB_TOKEN", "HF_WRITE_TOKEN"])
        push_dir = NB_DIR / f"push_gemma1000/w{n}"
        push_dir.mkdir(parents=True, exist_ok=True)
        (push_dir / f"kaggle_gemma1000_w{n}.ipynb").write_text(json.dumps(nb, indent=1))
        (push_dir / "kernel-metadata.json").write_text(json.dumps(
            kernel_metadata(f"vedangpandeyyy/gemma-1000-worker-{n}-of-2",
                            f"Gemma 1000 -- worker {n} of 2", f"kaggle_gemma1000_w{n}.ipynb",
                            gpu=True), indent=1))
        print(f"wrote {push_dir}/kernel-metadata.json (GPU, secrets injected)")

    clean_agg = _notebook(build_agg_notebook())
    agg_path = NB_DIR / "kaggle_gemma1000_agg.ipynb"
    agg_path.write_text(json.dumps(clean_agg, indent=1))
    print(f"wrote {agg_path} (clean, placeholder secrets)")
    agg = json.loads(json.dumps(clean_agg))
    inject_secrets(agg, env, ["GITHUB_TOKEN"])
    push_dir = NB_DIR / "push_gemma1000/agg"
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "kaggle_gemma1000_agg.ipynb").write_text(json.dumps(agg, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps(
        kernel_metadata("vedangpandeyyy/gemma-1000-aggregator",
                        "Gemma 1000 -- aggregator (CPU)", "kaggle_gemma1000_agg.ipynb",
                        gpu=False), indent=1))
    print(f"wrote {push_dir}/kernel-metadata.json (CPU, secrets injected)")

    print()
    print("PUSH DIRS READY (secrets injected, never committed):")
    for d in sorted((NB_DIR / "push_gemma1000").iterdir()):
        print("  notebooks/push_gemma1000/" + d.name)


def inject_secrets(nb: dict, env: dict, names: list[str]) -> None:
    """Replace the placeholder secrets cell with the hardcoded values from
    the local .env (the user's explicit choice for unattended kernels; the
    generated files live in the gitignored push_gemma1000/ dir)."""
    lines = ["import os\n"]
    for name in names:
        lines.append(f'os.environ[{name!r}] = {env[name]!r}\n')
    lines.append(f"print('secrets set:', {', '.join(f'bool(os.environ.get({n!r}))' for n in names)})\n")
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source") or [])
        if "secrets are injected at build time" in src:
            nb["cells"][i] = cell
            return
    raise RuntimeError("placeholder secrets cell not found in notebook")


if __name__ == "__main__":
    main()
