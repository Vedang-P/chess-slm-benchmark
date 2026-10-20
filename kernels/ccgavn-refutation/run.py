"""Resumable CC-GAVN-R pilot: source prep, oracle labels, then planner training."""
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
SL = WORK / "searchless_chess"
DATA = WORK / "ccgavn-refutation-r0"
HF_REPO = "vedangfake/chess-slm-benchmark"
DATA_RUN = "refutation-data-r0"
TRAIN_RUN = "ccgavn-refutation-r0-seed0"
BASE = WORK / "base-checkpoint"
PHASE_RUN = "refutation-r0"
EVAL_RUN = "eval-results/ccgavn-refutation-r0"
STEPS = 12500


def run(cmd, **kwargs):
    print("[job]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kwargs)


def download(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return
    run(["curl", "-fL", "--retry", "5", "--retry-all-errors", "-o", path, url])


def main():
    hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
    if not hits:
        raise RuntimeError("attached private HF credential dataset lacks hf_token.txt")
    os.environ["HF_WRITE_TOKEN"] = Path(hits[0]).read_text().strip()
    if not os.environ["HF_WRITE_TOKEN"]:
        raise RuntimeError("empty HF write token")

    if not (REPO / "scripts" / "refutation_data.py").exists():
        run(["git", "clone", "--depth", "1", "https://github.com/Vedang-P/chess-slm-benchmark.git", REPO])
    if not SL.exists():
        run(["git", "clone", "https://github.com/google-deepmind/searchless_chess.git", SL])
        run(["git", "-C", SL, "checkout", "90ae0e6b121673fc3079aaeffa047580bb600c0a"])
    os.chdir(REPO)
    run([sys.executable, "-m", "pip", "install", "--quiet",
         "python-chess==1.999", "zstandard==0.23.0", "pandas==2.2.3", "huggingface_hub==0.34.4"])

    from huggingface_hub import HfApi, hf_hub_download
    for name in ("config.json", "state.pt"):
        src = hf_hub_download(HF_REPO, f"ccgavn-5m-seed0-v2/checkpoint-3896658/{name}",
                              repo_type="dataset", token=False)
        target = BASE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    inputs = WORK / "inputs"
    inputs.mkdir(exist_ok=True)
    source = WORK / "parents-r0.jsonl"
    source_remote = f"{DATA_RUN}/input/parents-r0.jsonl"
    official = inputs / "official-puzzles.csv"
    api = HfApi()
    from scripts.kaggle_checkpoint import write_status
    write_status(api, HF_REPO, PHASE_RUN, "IN PROGRESS: boot / restore")
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    if source_remote in files:
        cached = hf_hub_download(HF_REPO, source_remote, repo_type="dataset", token=os.environ["HF_WRITE_TOKEN"])
        shutil.copy2(cached, source)
        print("[source] restored deterministic pilot input from HF", flush=True)
    else:
        write_status(api, HF_REPO, PHASE_RUN, "IN PROGRESS: source preparation")
        games = Path("/kaggle/temp") / "lichess-standard-2025-04.pgn.zst"
        Path("/kaggle/temp").mkdir(exist_ok=True)
        tactics = inputs / "lichess-puzzles.csv.zst"
        disagreements = inputs / "rlvr-train-5k.jsonl"
        download("https://database.lichess.org/standard/lichess_db_standard_rated_2025-04.pgn.zst", games)
        download("https://database.lichess.org/lichess_db_puzzle.csv.zst", tactics)
        download("https://huggingface.co/datasets/vedangfake/chess-slm-benchmark/resolve/main/rlvr-pool/train-5k.jsonl", disagreements)
        download("https://storage.googleapis.com/searchless_chess/data/puzzles.csv", official)
        run([sys.executable, "scripts/prepare_refutation_sources.py",
             "--games", games, "--tactics", tactics, "--disagreements", disagreements,
             "--checkpoint", BASE, "--sl-repo", SL, "--out", source, "--count", "1944"])
        api.upload_file(path_or_fileobj=str(source), path_in_repo=source_remote,
                        repo_id=HF_REPO, repo_type="dataset")
        print(f"[source] uploaded deterministic input: {source_remote}", flush=True)
    source_count = sum(1 for _ in source.open())
    print(f"[source] rows={source_count}", flush=True)
    download("https://storage.googleapis.com/searchless_chess/data/puzzles.csv", official)

    archive = inputs / "stockfish-16.tar"
    download("https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2.tar", archive)
    sf_dir = WORK / "stockfish16"
    sf_dir.mkdir(exist_ok=True)
    run(["tar", "-xf", archive, "-C", sf_dir])
    engines = [p for p in sf_dir.rglob("*") if p.is_file() and os.access(p, os.X_OK)]
    if len(engines) != 1:
        raise RuntimeError(f"expected one Stockfish executable in release archive, found {len(engines)}")
    engine = engines[0]
    engine_hash = hashlib.sha256(engine.read_bytes()).hexdigest()
    print(f"[engine] path={engine} sha256={engine_hash}", flush=True)

    write_status(api, HF_REPO, PHASE_RUN, "IN PROGRESS: oracle labeling")
    labels = DATA / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    data_cmd = [sys.executable, "scripts/refutation_data.py", "--input", source,
                "--outdir", labels, "--stockfish", engine, "--engine-sha256", engine_hash,
                "--nodes", "50000", "--horizon", "4", "--width", "2",
                "--student-checkpoint", BASE, "--sl-repo", SL,
                "--exclusion-puzzles", official, "--mate-dir", REPO / "data/positions",
                "--split-unit", "position", "--hf-repo", HF_REPO, "--hf-run", DATA_RUN,
                "--hf-upload-every", "900", "--session-hours", "10.5"]
    if f"{DATA_RUN}/data/groups.jsonl" in files:
        data_cmd.append("--resume-from-hf")
    run(data_cmd)

    journal = labels / "groups.jsonl"
    accepted = sum(1 for line in journal.open() if '"branches"' in line)
    rejected = sum(1 for line in journal.open() if '"rejected"' in line)
    if accepted + rejected < source_count:
        print(f"[job] oracle pilot paused: accepted={accepted} rejected={rejected} of {source_count}; continue with --resume-from-hf on next pooled account", flush=True)
        return
    if accepted == 0:
        raise RuntimeError("all source positions rejected; no training data")
    print(f"[job] labels complete: accepted={accepted} rejected={rejected}; starting planner optimizer", flush=True)
    write_status(api, HF_REPO, PHASE_RUN, "IN PROGRESS: planner training")

    train_out = WORK / "planner-r0"
    train_cmd = [sys.executable, "scripts/posttrain.py", "train", "--init-checkpoint", BASE,
                 "--data", journal, "--outdir", train_out, "--hf-repo", HF_REPO,
                 "--hf-run", TRAIN_RUN, "--mode", "planner", "--horizon", "4", "--width", "2",
                 "--steps", "12500", "--batch-groups", "2", "--accumulate", "16",
                 "--warmup-steps", "500", "--freeze-steps", "1000", "--hf-upload-every", "900",
                 "--session-hours", "10.5"]
    if any(name.startswith(f"{TRAIN_RUN}/checkpoint-") and name.endswith("/config.json") for name in files):
        train_cmd.append("--resume-from-hf")
    run(train_cmd)

    final = train_out / f"checkpoint-{STEPS}"
    if not final.exists():
        final = train_out / "resume" / TRAIN_RUN / f"checkpoint-{STEPS}"
    if not final.exists():
        print("[job] training paused; final evaluation deferred", flush=True)
        return
    write_status(api, HF_REPO, PHASE_RUN, "IN PROGRESS: final evaluation")
    eval_out = WORK / "eval-r0"
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    eval_cmd = [sys.executable, "scripts/posttrain.py", "eval", "--checkpoint", final,
                "--puzzles", official, "--sl-repo", SL, "--outdir", eval_out,
                "--hf-repo", HF_REPO, "--hf-run", EVAL_RUN,
                "--hf-upload-every", "900", "--session-hours", "10.5"]
    if any(name.startswith(f"{EVAL_RUN}/eval-") for name in files):
        eval_cmd.append("--resume-from-hf")
    run(eval_cmd)
    summary = eval_out / "summary.json"
    if not summary.exists() or not json.loads(summary.read_text()).get("complete"):
        write_status(api, HF_REPO, PHASE_RUN, "PAUSED: evaluation resume required")
        return
    write_status(api, HF_REPO, PHASE_RUN, "DONE: CC-GAVN-R pilot complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        try:
            from huggingface_hub import HfApi
            token = os.environ.get("HF_WRITE_TOKEN")
            if token:
                p = WORK / "run-status.txt"
                p.write_text(f"FAILED: {type(exc).__name__}: {exc}\n")
                HfApi(token=token).upload_file(path_or_fileobj=str(p), path_in_repo="refutation-r0/run-status.txt",
                                               repo_id="vedangfake/chess-slm-benchmark", repo_type="dataset")
        except Exception as status_exc:
            print(f"[job] status upload also failed: {status_exc}", flush=True)
        raise
