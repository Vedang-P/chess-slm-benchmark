"""Generate gemma 4 E2B thinking-run kernels for the remaining MATE subsets:
noexplain / tactic / both. Same 2-worker GPU recipe as the strategy campaign
(build_gemma1000_worker_notebook.py), parameterized per subset.

    python notebooks/build_gemma1000_variants_notebook.py [noexplain|tactic|both]

Each subset = 2 GPU workers x 500 positions (Kaggle's 2-concurrent-GPU
limit), --task-file selects the eval file, --local-thinking + --live-namespace
gemma, --resume with per-worker HF checkpoints, HF upload every 25.

Worker tags / run ids / kernel slugs are per-subset so live-push and HF
archives never collide between variants:
    noexplain: mate-gemma-noexplain-w{1,2}
    tactic:    mate-gemma-tactic-w{1,2}
    both:      mate-gemma-both-w{1,2}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent

MAX_NEW_TOKENS = 32768
SLICE_SIZE = 500
BRANCH = "mate-e2b-kaggle"

TASK_FILES = {
    "noexplain": "mate-selection-test-noexplain.json",
    "tactic": "mate-selection-test-tactic.json",
    "both": "mate-selection-test-both.json",
}

CLONE_CELL = r'''
import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
if REPO.exists():
    shutil.rmtree(REPO)

def find_token():
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        v = os.environ.get(name)
        if v:
            return v
    raise SystemExit("GITHUB_TOKEN missing")

url = "https://github.com/Vedang-P/chess-slm-benchmark.git"
url = url.replace("https://", f"https://x-access-token:{find_token()}@")
res = subprocess.run(["git", "clone", "--quiet", "-b", "mate-e2b-kaggle", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed (bad/missing GITHUB_TOKEN?): " + res.stderr[-300:])
os.chdir(REPO)
print("cwd:", Path.cwd())
'''.strip()

DEPS_CELL = r'''
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "torch==2.5.1", "torchvision", "torchaudio",
                "transformers>=4.40,<5", "accelerate", "bitsandbytes",
                "peft>=0.10.0", "numpy", "huggingface_hub", "tqdm",
                "pyyaml", "python-chess", "zstandard"], check=True)
import torch, transformers
print("deps:", transformers.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("no GPU in this session")
'''.strip()

GATE_CELL = r'''
import subprocess, sys
status = subprocess.run([sys.executable, "scripts/test_engine.py", "--quick"],
                        capture_output=True, text=True)
print(status.stdout[-2000:])
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


def run_cell(worker: dict, out_dir: str, worker_tag: str, bench_run_id: str,
             task_file: str) -> dict:
    return _code(f'''
import os, subprocess, sys, time
from pathlib import Path

os.environ["BENCH_RUN_ID"] = {bench_run_id!r}
out = Path({out_dir!r})
out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "gemma4-e2b",
       "--task-file", {task_file!r},
       "--offset", "{worker['offset']}",
       "--n", "{SLICE_SIZE}",
       "--max_new_tokens", "{MAX_NEW_TOKENS}",
       "--force-answer-prompt",
       "--local-thinking",
       "--worker-id", {worker_tag!r},
       "--live-namespace", "gemma",
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
        "See the subprocess output printed above this cell."
    )
'''.strip())


def summary_cell(out_dir: str, task_name: str) -> dict:
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
print(f"this worker: {{n}} / 500 positions attempted ({task_name})")
print("status breakdown:", dict(by_status))
if scored:
    print(f"accuracy (of {{scored}} scored): {{correct}}/{{scored}} = {{correct/scored:.3f}}")
'''.strip())


def build_worker_notebook(variant: str, n: int) -> list:
    offset = (n - 1) * SLICE_SIZE
    worker_tag = f"{variant}-w{n}"
    bench_run_id = f"mate-gemma-{variant}-w{n}"
    out_dir = f"results/gemma-{variant}-w{n}"
    run_name = f"gemma4-e2b_mate-selection-test_{variant}"
    task_file = TASK_FILES[variant]

    cells = [
        _md(f"# Gemma 4 E2B {variant}: Worker {n} of 2 (positions [{offset}:{offset + SLICE_SIZE}])\n\n"
            f"Local Gemma 4 E2B, thinking ON, on the MATE **{variant}** subset "
            f"(`data/positions/{task_file}`). Two GPU workers cover 500 positions "
            "each (Kaggle's 2-concurrent-GPU limit). Methodology identical to the "
            "strategy campaign: `--local-thinking`, `--force-answer-prompt`, "
            f"`--max_new_tokens {MAX_NEW_TOKENS}`, `--task-file {task_file}`.\n\n"
            "Backups: HF upload every 25 positions; the recovery cell pulls "
            "this worker's checkpoint back before `--resume`, so a died "
            "session relaunches and continues."),
        _md("## 1. Secrets (hardcoded env vars)"),
        _code("import os\n"
              'print("secrets are injected at build time; this cell is a placeholder")'),
        _md("## 2. Get the repo (mate-e2b-kaggle branch)"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies (torch pins for P100+T4)"),
        _code(DEPS_CELL),
        _md("## 4. Engine/dataset gate"),
        _code(GATE_CELL),
        _md("## 5. Run positions\n\n"
            "HF backup every 25 positions; live state to "
            f"monitor/gemma/workers/{worker_tag}.*"),
        run_cell({"offset": offset}, out_dir, worker_tag, bench_run_id, task_file),
        _md("## 6. This worker's summary"),
        summary_cell(out_dir, variant),
        _md("## Notes\n"
            f"- Scope: ONLY positions [{offset}:{offset + SLICE_SIZE}) of {task_file}.\n"
            "- If this session dies/times out, push the same notebook again "
            "(or Restart & Run All): step 5 recovers progress from HF, "
            "--resume skips it.\n"
            "- GPU: Kaggle assigns T4 or P100 -- both supported; T4 is ~2x "
            "faster. ~78s/position measured on P100, so 500 positions ~ 11h, "
            "within the 12h session limit.\n"
            f"- Results: HF runs/{bench_run_id}/"),
    ]
    return cells


def kernel_metadata(kernel_id: str, title: str, code_file: str) -> dict:
    return {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def inject_secrets(nb: dict, env: dict, names: list[str]) -> None:
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


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else None
    variants = [variant] if variant else list(TASK_FILES)
    if variant and variant not in TASK_FILES:
        raise SystemExit(f"unknown variant {variant!r}; choose from {sorted(TASK_FILES)}")

    env = {}
    for line in (NB_DIR.parent / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")

    for v in variants:
        for n in (1, 2):
            clean = _notebook(build_worker_notebook(v, n))
            nb = json.loads(json.dumps(clean))
            inject_secrets(nb, env, ["GITHUB_TOKEN", "HF_WRITE_TOKEN"])
            push_dir = NB_DIR / f"push_gemma_{v}/w{n}"
            push_dir.mkdir(parents=True, exist_ok=True)
            code_file = f"kaggle_gemma_{v}_w{n}.ipynb"
            (push_dir / code_file).write_text(json.dumps(nb, indent=1))
            (push_dir / "kernel-metadata.json").write_text(json.dumps(
                kernel_metadata(f"vedangpandeyyy/gemma-{v}-worker-{n}-of-2",
                                f"Gemma {v} -- worker {n} of 2", code_file), indent=1))
            print(f"wrote {push_dir}/kernel-metadata.json (GPU, secrets injected)")

    print()
    print("PUSH DIRS READY (secrets injected, never committed):")
    for v in variants:
        for n in (1, 2):
            print(f"  notebooks/push_gemma_{v}/w{n}")


if __name__ == "__main__":
    main()
