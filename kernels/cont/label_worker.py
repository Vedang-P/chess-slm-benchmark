"""Pinned Kaggle CPU entry point for the later-ply puzzle labeling fleet.

scripts/watch_ply_labels.py embeds a JSON config before each push. Kaggle only
uploads this one file, so nothing else is assumed to exist. A worker owns a fixed
slice of the job list (scripts/ply_data.py), builds Stockfish 16, and labels one shard
at a time with several engine processes, resuming partial shards from Hugging Face.
Phase "smoke" labels a few roots into a scratch prefix and times the engines instead.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

CONFIG_JSON = """__LABEL_CONFIG__"""
if CONFIG_JSON.startswith("__LABEL_CONFIG__"):
    raise SystemExit("missing embedded label config")
CFG = json.loads(CONFIG_JSON)
PHASE = CFG["phase"]
WORKER = int(CFG["worker_index"])
WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
SL = WORK / "searchless_chess"
STOCKFISH = WORK / "Stockfish16"
INPUTS = WORK / "inputs"
SESSION_STARTED = time.time()
SESSION_BUDGET = 10.5 * 3600

hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
TOKEN = Path(hits[0]).read_text().strip() if hits else os.environ.get("HF_WRITE_TOKEN", "")
if not TOKEN:
    raise SystemExit("HF token missing; cannot persist Kaggle progress")
os.environ["HF_WRITE_TOKEN"] = TOKEN

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

api = HfApi(token=TOKEN)
HF_REPO = CFG["hf_repo"]


def run(command: list[str], **kwargs) -> None:
    print("[ply-label] " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, **kwargs)


def pinned_git(url: str, dest: Path, commit: str) -> None:
    run(["git", "init", "-q", str(dest)])
    run(["git", "-C", str(dest), "remote", "add", "origin", url])
    run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", commit])
    run(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"])
    actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != commit:
        raise ValueError(f"pinned checkout mismatch: {actual} != {commit}")


def prepare_searchless_tokenizer() -> None:
    """Make upstream's type-only annotation safe without importing JAX (labels need none)."""
    path = SL / "src" / "tokenizer.py"
    source = path.read_text(encoding="utf-8")
    expected = ('"""Implements tokenization of FEN strings."""\n\n'
                'import jaxtyping as jtp\n')
    replacement = ('"""Implements tokenization of FEN strings."""\n\n'
                   'from __future__ import annotations\n')
    if source.count(expected) == 1:
        path.write_text(source.replace(expected, replacement, 1), encoding="utf-8")
    elif "from __future__ import annotations" not in source or "import jaxtyping as jtp" in source:
        raise ValueError("unexpected pinned Searchless tokenizer header")


def stockfish16() -> Path:
    pinned_git("https://github.com/official-stockfish/Stockfish.git", STOCKFISH,
               CFG["stockfish_commit"])
    net_name = "nn-5af11540bbfe.nnue"
    net = STOCKFISH / "src" / net_name
    url = f"https://raw.githubusercontent.com/official-stockfish/networks/master/{net_name}"
    with urllib.request.urlopen(url, timeout=120) as src, net.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    if hashlib.sha256(net.read_bytes()).hexdigest()[:12] != "5af11540bbfe":
        raise ValueError("Stockfish 16 NNUE network SHA256 mismatch")
    run(["make", "-C", str(STOCKFISH / "src"), "-j4", "build", "ARCH=x86-64-avx2"])
    binary = STOCKFISH / "src" / "stockfish"
    if not binary.is_file():
        raise ValueError("Stockfish 16 build has no executable")
    return binary


def fetch(path: str) -> Path:
    return Path(hf_hub_download(HF_REPO, path, repo_type="dataset", local_dir=str(INPUTS),
                                token=TOKEN))


def fetch_inputs() -> dict:
    """Source corpus, splits, exclusion set and plan, each verified against the plan."""
    from scripts.freeze_puzzle_corpus import file_sha256

    prefix = CFG["plan_prefix"]
    plan_path = fetch(f"{prefix}/plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    files = {"source": fetch(f"{CFG['source_prefix']}/source.csv.zst"),
             "manifest": fetch(f"{CFG['source_prefix']}/manifest.json"),
             "val": fetch("cont-v1/val.csv"), "test": fetch("cont-v1/test.csv"),
             "hashes": fetch(f"{prefix}/exclusion-hashes.npy")}
    expected = {"source": plan["source_sha256"], "val": plan["val_csv_sha256"],
                "test": plan["test_csv_sha256"], "hashes": plan["exclusion_sha256"]}
    for key, digest in expected.items():
        if file_sha256(files[key]) != digest:
            raise ValueError(f"{key} digest does not match the labeling plan")
    if plan["source_sha256"] != CFG["source_sha256"]:
        raise ValueError("labeling plan was built for a different source")
    return {"plan": plan, **files}


def shard_command(inputs: dict, ply: int, shard: int, folder: Path, prefix: str,
                  stockfish: Path, extra: list[str]) -> list[str]:
    return [sys.executable, str(REPO / "scripts" / "label_ply_shard.py"),
            "--source", str(inputs["source"]), "--manifest", str(inputs["manifest"]),
            "--val-csv", str(inputs["val"]), "--test-csv", str(inputs["test"]),
            "--exclusion-hashes", str(inputs["hashes"]), "--ply", str(ply),
            "--shard", str(shard), "--stockfish", str(stockfish), "--sl-repo", str(SL),
            "--out", str(folder / "train_set.npz"), "--partial", str(folder / "partial.npz"),
            "--engines", str(CFG["engines"]), "--hf-repo", HF_REPO, "--hf-prefix", prefix,
            *extra]


def label_phase(inputs: dict, binary: Path) -> None:
    from scripts.ply_data import complete_shards, plan_jobs, shard_dir, worker_jobs

    prefix = CFG["data_prefix"]
    jobs = worker_jobs(plan_jobs(inputs["plan"]), WORKER, int(CFG["worker_count"]))
    print(f"[ply-label] worker {WORKER}: {len(jobs)} jobs, cpus={os.cpu_count()}, "
          f"engines={CFG['engines']}", flush=True)
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    done = complete_shards(files, prefix)
    for ply, shard in jobs:
        if shard in done.get(ply, ()):
            continue
        remaining = SESSION_BUDGET - (time.time() - SESSION_STARTED)
        if remaining < 900:
            print("[ply-label] session budget reached", flush=True)
            return
        folder = WORK / f"ply{ply}-{shard:05d}"
        folder.mkdir(exist_ok=True)
        extra = ["--stop-after-seconds", str(max(1, remaining - 600)),
                 "--save-every-seconds", str(CFG["save_every_seconds"])]
        for attempt in (1, 2):
            try:
                run(shard_command(inputs, ply, shard, folder, prefix, binary, extra), cwd=REPO)
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                print("[ply-label] shard job failed; retrying once", flush=True)
                time.sleep(30)
        if not (folder / "train_set.npz").exists():
            # Either the session ran out of time (the partial is on the Hub) or someone else
            # finished the shard first; only the Hub listing can tell which.
            listing = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
            if shard not in complete_shards(listing, prefix).get(ply, ()):
                print(f"[ply-label] ply{ply} shard {shard:05d} stopped partial; next session resumes",
                      flush=True)
                return
        shutil.rmtree(folder, ignore_errors=True)
        done.setdefault(ply, set()).add(shard)
        print(f"[ply-label] {shard_dir(prefix, ply, shard)} complete", flush=True)


def smoke_phase(inputs: dict, binary: Path) -> None:
    """Time the engines alone and under load, then label a few roots into the scratch prefix."""
    from scripts.hf_upload import upload_file

    roots = int(CFG.get("smoke_roots", 40))
    bench = WORK / "bench-report.json"
    run([sys.executable, str(REPO / "scripts" / "ply_label_bench.py"),
         "--stockfish", str(binary), "--sl-repo", str(SL), "--source", str(inputs["source"]),
         "--manifest", str(inputs["manifest"]), "--val-csv", str(inputs["val"]),
         "--test-csv", str(inputs["test"]), "--roots", str(int(CFG.get("bench_roots", 60))),
         "--engines", str(CFG["engines"]), "--out", str(bench)], cwd=REPO)
    report = json.loads(bench.read_text(encoding="utf-8"))
    report.update(cpus=os.cpu_count(), worker=WORKER)
    for ply in (3, 5):
        folder = WORK / f"smoke-ply{ply}"
        folder.mkdir(exist_ok=True)
        started = time.time()
        run(shard_command(inputs, ply, 0, folder, CFG["data_prefix"], binary,
                          ["--max-roots", str(roots)]), cwd=REPO)
        report[f"ply{ply}_seconds_per_root"] = (time.time() - started) / roots
    upload_file(api, HF_REPO, json.dumps(report, indent=2, sort_keys=True).encode(),
                f"{CFG['data_prefix']}/smoke-report.json",
                commit_message="Later-ply labeling smoke report", attempts=4, max_elapsed_s=300.0)
    print(f"[ply-label] smoke report {report}", flush=True)


def persist_status(text: str) -> None:
    sys.path.insert(0, str(REPO))
    from scripts.hf_upload import upload_file
    # A smoke run must not write a fleet worker's file: a failure there would block it.
    name = "smoke" if PHASE == "smoke" else f"w{WORKER:02d}"
    upload_file(api, HF_REPO, text.encode(), f"{CFG['status_prefix']}/ply-label-{name}.txt",
                commit_message=f"ply-label {name}: {text.splitlines()[0][:60]}",
                attempts=4, max_elapsed_s=300.0)


def main() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4", "python-chess",
         "zstandard", "huggingface_hub"])
    pinned_git("https://github.com/Vedang-P/chess-slm-benchmark.git", REPO, CFG["source_commit"])
    pinned_git("https://github.com/google-deepmind/searchless_chess.git", SL,
               CFG["searchless_commit"])
    prepare_searchless_tokenizer()
    sys.path.insert(0, str(REPO))
    binary = stockfish16()
    inputs = fetch_inputs()
    if PHASE == "smoke":
        smoke_phase(inputs, binary)
    elif PHASE == "labels":
        label_phase(inputs, binary)
    else:
        raise ValueError(f"unknown phase {PHASE}")


try:
    main()
    persist_status(f"OK {PHASE} source_commit={CFG['source_commit']} "
                   f"hours={(time.time() - SESSION_STARTED) / 3600:.2f}")
except Exception as exc:
    status = (f"FAILED {PHASE} source_commit={CFG['source_commit']} "
              f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3500:]}")
    try:
        persist_status(status)
    except Exception as upload_exc:
        print(f"[ply-label] could not persist failure: {upload_exc}", flush=True)
    raise
