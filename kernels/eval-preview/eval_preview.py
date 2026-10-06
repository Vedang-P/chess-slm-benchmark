"""Periodic monitoring evals for corrected-v2 milestone checkpoints.

Development diagnostics, not a holdout protocol: these checkpoints have been
selected by dev loss, and every MATE/puzzle score here has been seen during
training. The single one-shot final protocol result is stored separately (the
`*-final` prefix); claims must disclose this exposure.

Guarantees (2026-10-06 review fixes):
  - the repository is checked out at the exact source commit embedded at push
    time (no silent default-branch drift),
  - per-example JSONL is downloaded before scoring and uploaded periodically,
    so a killed session resumes instead of rescoring,
  - a target is only marked DONE when eval_gavn.py exits 0 with exact expected
    totals (4,000 MATE rows / 10,000 puzzles); otherwise INCOMPLETE is written
    and CI re-pushes.

Pushed by scripts/ensure_eval_preview.py from CI only when a target checkpoint
is already on HF, so the GPU session never idles while waiting for training.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG_JSON = """__EVALPREVIEW_CONFIG__"""
if CONFIG_JSON.startswith("__EVALPREVIEW"):
    raise SystemExit("eval config was not embedded at push time; "
                     "re-push via scripts/ensure_eval_preview.py")
CFG = json.loads(CONFIG_JSON)
HF_REPO = CFG.get("hf_repo", "vedangfake/chess-slm-benchmark")
RUN = CFG["run"]
TARGETS = [int(x) for x in CFG["targets"]]
FINAL_STEP = int(CFG["final_step"])
SOURCE_COMMIT = CFG["source_commit"]
EXPECT_MATE = int(CFG.get("expect_mate_rows", 4000))
EXPECT_PUZ = int(CFG.get("expect_puzzles", 10000))
BUDGET_S = 10.5 * 3600

WORK = Path("/kaggle/working")
hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
TOKEN = Path(hits[0]).read_text().strip() if hits else os.environ.get("HF_WRITE_TOKEN", "")
assert TOKEN, "no HF token"

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402
api = HfApi(token=TOKEN)
T0 = time.time()


def out_prefix(step: int) -> str:
    kind = "final" if step == FINAL_STEP else "preview"
    return f"eval-results/{RUN}-{step // 1000}k-{kind}"


def upload(local: Path, remote: str) -> None:
    try:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=HF_REPO, repo_type="dataset")
    except Exception as exc:
        print(f"[preview] upload failed {remote}: {exc}", flush=True)


def upload_examples(examples: Path, prefix: str) -> None:
    for name in ("mate.jsonl", "puzzles.jsonl"):
        path = examples / name
        if path.exists():
            upload(path, f"{prefix}/examples/{name}")


def download_examples(prefix: str, examples: Path) -> None:
    examples.mkdir(parents=True, exist_ok=True)
    for name in ("mate.jsonl", "puzzles.jsonl"):
        remote = f"{prefix}/examples/{name}"
        try:
            cached = hf_hub_download(HF_REPO, remote, repo_type="dataset", token=TOKEN)
            shutil.copy(cached, examples / name)
            print(f"[preview] resumed {remote}", flush=True)
        except Exception:
            pass


def summary_complete(summary: dict) -> bool:
    if not summary or not summary.get("complete") or summary.get("returncode") != 0:
        return False
    mate = summary.get("mate") or {}
    puz = summary.get("puzzles") or {}
    return (mate.get("total") == EXPECT_MATE and mate.get("correct") is not None
            and puz.get("total") == EXPECT_PUZ and puz.get("solved") is not None)


def target_complete(step: int) -> bool:
    prefix = out_prefix(step)
    try:
        cached = hf_hub_download(HF_REPO, f"{prefix}/eval-summary.json",
                                 repo_type="dataset", token=TOKEN)
        return summary_complete(json.loads(Path(cached).read_text(encoding="utf-8")))
    except Exception:
        return False


def fetch_pinned_repo(url: str, dest: Path, sha: str) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", url], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", sha],
                   check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != sha:
        raise RuntimeError(f"pinned checkout mismatch: {actual} != {sha}")
    print(f"[preview] repo pinned at {sha}", flush=True)


print(f"[preview] session start; targets {TARGETS} commit {SOURCE_COMMIT}", flush=True)
print("[preview] preparing environment", flush=True)
REPO = WORK / "chess-slm-benchmark"
SL = WORK / "searchless_chess"
fetch_pinned_repo("https://github.com/Vedang-P/chess-slm-benchmark.git", REPO, SOURCE_COMMIT)
if not SL.exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/google-deepmind/searchless_chess.git", str(SL)], check=True)
pz = SL / "data" / "puzzles.csv"
if not pz.exists():
    subprocess.run(["curl", "-sL", "--fail", "-o", str(pz),
                    "https://storage.googleapis.com/searchless_chess/data/puzzles.csv"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-chess", "pandas"], check=True)
sl_commit = subprocess.check_output(["git", "-C", str(SL), "rev-parse", "HEAD"],
                                    text=True).strip()

MATE = ",".join(str(REPO / "data/positions" / f) for f in
                ["mate-selection-test-noexplain.json", "mate-selection-test-both.json",
                 "mate-selection-test-tactic.json", "mate-selection-test.json"])

evaluated = incomplete = 0
for step in TARGETS:
    ckpt = f"{RUN}/checkpoint-{step}"
    prefix = out_prefix(step)
    if target_complete(step):
        print(f"[preview] step {step}: complete already; skip", flush=True)
        continue
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
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
    examples = WORK / "examples"
    download_examples(prefix, examples)
    out = WORK / "eval-full.log"
    summary_path = WORK / "eval-summary.json"
    if summary_path.exists():
        summary_path.unlink()
    cmd = [sys.executable, str(REPO / "scripts" / "eval_gavn.py"),
           "--checkpoint", str(ck / ckpt), "--sl-repo", str(SL), "--eval", MATE,
           "--puzzles", str(pz), "--num-puzzles", str(EXPECT_PUZ), "--score", "auto",
           "--examples-out", str(examples), "--summary-out", str(summary_path)]
    print("[preview] " + " ".join(cmd), flush=True)
    last = time.time()
    with open(out, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            fh.write(line); fh.flush(); print(line.rstrip(), flush=True)
            if time.time() - last > 300:
                upload(out, f"{prefix}/eval-full.partial.log")
                upload_examples(examples, prefix)
                last = time.time()
        proc.wait()

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    complete = proc.returncode == 0 and summary_complete(summary)
    summary["checkpoint"] = ckpt
    summary["returncode"] = proc.returncode
    summary["complete"] = bool(complete)
    summary["source_commit"] = SOURCE_COMMIT
    summary["searchless_commit"] = sl_commit
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    upload(out, f"{prefix}/eval-full.log")
    upload_examples(examples, prefix)
    if complete:
        upload(summary_path, f"{prefix}/eval-summary.json")
        evaluated += 1
    else:
        incomplete += 1
    state = "DONE " if complete else "INCOMPLETE "
    api.upload_file(path_or_fileobj=(state + json.dumps(summary)).encode(),
                    path_in_repo=f"{prefix}/run-status.txt", repo_id=HF_REPO,
                    repo_type="dataset")
    print(f"[preview] step {step}: {state}{json.dumps(summary)}", flush=True)
    if not complete:
        break

print(f"[preview] session done; evaluated {evaluated}, incomplete {incomplete}", flush=True)
if incomplete:
    raise SystemExit(1)
