"""Periodic preview evals for ccgavn-5m-seed0 milestone checkpoints.

Evaluates every reached milestone in TARGETS (MATE 4k + official 10K puzzles)
that has no eval-summary yet, then exits. Pushed by
scripts/ensure_eval_preview.py from CI only when a target checkpoint is
already on HF, so the GPU session never idles while waiting for training.

Output layout per target:
  eval-results/ccgavn-5m-seed0-<K>k-preview/eval-summary.json
  (the 1,620,000 target lands in ...-1620k-frozen/)
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RUN = "ccgavn-5m-seed0"
HF_REPO = "vedangfake/chess-slm-benchmark"
TARGETS = [400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000, 1_600_000, 1_620_000]
BUDGET_S = 10.5 * 3600

WORK = Path("/kaggle/working")
hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
TOKEN = Path(hits[0]).read_text().strip() if hits else os.environ.get("HF_WRITE_TOKEN", "")
assert TOKEN, "no HF token"

from huggingface_hub import HfApi, snapshot_download  # noqa: E402
api = HfApi(token=TOKEN)
T0 = time.time()


def out_prefix(step: int) -> str:
    k = step // 1000
    return (f"eval-results/{RUN}-{k}k-frozen" if step == 1_620_000
            else f"eval-results/{RUN}-{k}k-preview")


def upload(local: Path, remote: str) -> None:
    try:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=HF_REPO, repo_type="dataset")
    except Exception as exc:
        print(f"[preview] upload failed {remote}: {exc}", flush=True)


def hf_files() -> set[str]:
    return set(api.list_repo_files(HF_REPO, repo_type="dataset"))


print(f"[preview] session start; targets {TARGETS}", flush=True)

print("[preview] preparing environment", flush=True)
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

MATE = ",".join(str(REPO / "data/positions" / f) for f in
                ["mate-selection-test-noexplain.json", "mate-selection-test-both.json",
                 "mate-selection-test-tactic.json", "mate-selection-test.json"])

evaluated = 0
for step in TARGETS:
    ckpt = f"{RUN}/checkpoint-{step}"
    prefix = out_prefix(step)
    files = hf_files()
    if f"{prefix}/eval-summary.json" in files:
        print(f"[preview] step {step}: already evaluated; skip", flush=True)
        continue
    if f"{ckpt}/config.json" not in files or f"{ckpt}/state.pt" not in files:
        print(f"[preview] step {step}: checkpoint not on HF; exiting for CI re-push", flush=True)
        break
    if time.time() - T0 > BUDGET_S:
        print(f"[preview] budget exhausted before step {step}; exiting for CI re-push", flush=True)
        break

    print(f"[preview] step {step}: evaluating", flush=True)
    ck = WORK / "ckpt"
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", token=TOKEN, local_dir=str(ck),
                      allow_patterns=[f"{ckpt}/*"])
    out = WORK / "eval-full.log"
    cmd = [sys.executable, str(REPO / "scripts" / "eval_gavn.py"),
           "--checkpoint", str(ck / ckpt), "--sl-repo", str(SL), "--eval", MATE,
           "--puzzles", str(pz), "--num-puzzles", "10000", "--score", "auto"]
    print("[preview] " + " ".join(cmd), flush=True)
    last = time.time()
    lines = []
    with open(out, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            fh.write(line); fh.flush(); print(line.rstrip(), flush=True)
            lines.append(line.rstrip())
            if time.time() - last > 300:
                upload(out, f"{prefix}/eval-full.partial.log")
                last = time.time()
        proc.wait()

    mate = re.findall(r"\[gavn\] MATE: (\d+)/(\d+) = ([\d.]+)%", "\n".join(lines))
    puz = re.findall(r"\[gavn\] puzzles: (\d+)/(\d+) = ([\d.]+)%", "\n".join(lines))
    summary = {"checkpoint": ckpt, "returncode": proc.returncode,
               "mate": [{"correct": int(a), "total": int(b), "pct": float(c)} for a, b, c in mate],
               "puzzles": [{"correct": int(a), "total": int(b), "pct": float(c)} for a, b, c in puz]}
    (WORK / "eval-summary.json").write_text(json.dumps(summary, indent=2))
    upload(out, f"{prefix}/eval-full.log")
    upload(WORK / "eval-summary.json", f"{prefix}/eval-summary.json")
    state = "DONE " if (mate and puz) else "INCOMPLETE "
    api.upload_file(path_or_fileobj=(state + json.dumps(summary)).encode(),
                    path_in_repo=f"{prefix}/run-status.txt", repo_id=HF_REPO, repo_type="dataset")
    print(f"[preview] step {step}: {state}{json.dumps(summary)}", flush=True)
    evaluated += 1

print(f"[preview] session done; evaluated {evaluated} checkpoints", flush=True)
