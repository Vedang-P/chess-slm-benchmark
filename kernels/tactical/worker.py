"""Pinned Kaggle entry point for frozen puzzle labels, teacher, demo, or learner.

scripts/watch_tactical.py embeds a JSON config before each push. Kaggle only
uploads this code file, so sibling configuration files are never assumed.
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
import urllib.request
import zipfile
from pathlib import Path

CONFIG_JSON = """__TACTICAL_CONFIG__"""
if CONFIG_JSON.startswith("__TACTICAL_CONFIG__"):
    raise SystemExit("missing embedded tactical run config")
CFG = json.loads(CONFIG_JSON)
PHASE = CFG["phase"]
ACCOUNT_INDEX = int(CFG["account_index"])
WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
SL = WORK / "searchless_chess"
STOCKFISH = WORK / "Stockfish16"
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
    print("[tactical] " + " ".join(map(str, command)), flush=True)
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
    """Make upstream's type-only annotation safe without importing JAX.

    The tokenizer uses jaxtyping only in its return annotation. Kaggle's global
    Python has preloaded CUDA JAX plugins, so replacing JAX with a CPU build can
    break unrelated packages. Postponing annotations and removing that unused
    import leaves tokenizer behavior unchanged and keeps labeling lightweight.
    """
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
    print("[tactical] Searchless tokenizer annotations deferred; JAX not needed for labels",
          flush=True)


def install_base() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4",
         "python-chess", "pandas==2.2.3", "zstandard", "huggingface_hub"])


def install_teacher() -> Path:
    env = WORK / "teachvenv"
    shutil.rmtree(env, ignore_errors=True)
    py311 = shutil.which("python3.11")
    if py311:
        run([py311, "-m", "venv", str(env)])
    else:
        uv = shutil.which("uv")
        if not uv:
            run([sys.executable, "-m", "pip", "install", "--quiet", "uv==0.8.17"])
            uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        run([uv, "python", "install", "3.11"])
        run([uv, "venv", "--python", "3.11", str(env)])
    pip = str(env / "bin" / "pip")
    run([pip, "install", "--quiet", "-f",
         "https://storage.googleapis.com/jax-releases/jax_releases.html",
         "jax==0.4.35", "jaxlib==0.4.35", "jax_cuda12_pjrt==0.4.35",
         "jax_cuda12_plugin==0.4.35", "orbax-checkpoint==0.5.5",
         "dm-haiku==0.0.15", "numpy==1.26.4", "pandas==2.2.3",
         "jaxtyping", "typing-extensions", "python-chess", "zstandard",
         "apache-beam", "grain", "wrapt", "huggingface_hub"])
    python = env / "bin" / "python"
    run([str(python), "-c",
         "import jax; assert any(d.platform == 'gpu' for d in jax.devices()), jax.devices()"])
    return python


def teacher_checkpoint() -> Path:
    ck = WORK / "teacher"
    ck.mkdir(exist_ok=True)
    target = ck / "9M" / "6400000" / "params_ema"
    if target.is_dir():
        return target
    archive = ck / "9M.zip"
    if not archive.exists():
        url = "https://storage.googleapis.com/searchless_chess/checkpoints/9M.zip"
        with urllib.request.urlopen(url, timeout=120) as src, archive.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(ck)
    if not target.is_dir():
        raise ValueError("9M teacher checkpoint missing params_ema after extraction")
    return target


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
    run(["make", "-C", str(STOCKFISH / "src"), "-j2", "build", "ARCH=x86-64-avx2"])
    binary = STOCKFISH / "src" / "stockfish"
    if not binary.is_file():
        raise ValueError("Stockfish 16 build has no executable")
    return binary


def source_files() -> tuple[Path, Path]:
    prefix = CFG["source_prefix"]
    archive = Path(hf_hub_download(HF_REPO, f"{prefix}/source.csv.zst",
                                   repo_type="dataset", local_dir=str(WORK), token=TOKEN))
    manifest = Path(hf_hub_download(HF_REPO, f"{prefix}/manifest.json",
                                    repo_type="dataset", local_dir=str(WORK), token=TOKEN))
    from scripts.freeze_puzzle_corpus import file_sha256
    if file_sha256(archive) != CFG["source_sha256"]:
        raise ValueError("frozen puzzle source SHA256 mismatch")
    return archive, manifest


def official_puzzles() -> Path:
    dest = SL / "data" / "puzzles.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(
            "https://storage.googleapis.com/searchless_chess/data/puzzles.csv", timeout=120
        ) as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return dest


def shard_tags():
    return range(ACCOUNT_INDEX, int(CFG["shard_count"]), 5)


def label_phase(source: Path, manifest: Path, binary: Path,
                prefix: str, max_roots: int = 0) -> None:
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    for tag in ([0] if max_roots else shard_tags()):
        remote = f"{prefix}/shard-{tag:05d}"
        if f"{remote}/train_set.npz" in files and f"{remote}/meta.json" in files:
            continue
        remaining = SESSION_BUDGET - (time.time() - SESSION_STARTED)
        if remaining < 900:
            print("[tactical] label session budget reached", flush=True)
            return
        folder = WORK / f"label-{tag:05d}"
        folder.mkdir(exist_ok=True)
        cmd = [sys.executable, str(REPO / "scripts" / "label_puzzle_shard.py"),
               "--source", str(source), "--manifest", str(manifest),
               "--shard", str(tag), "--stockfish", str(binary),
               "--sl-repo", str(SL),
               "--out", str(folder / "train_set.npz"),
               "--partial", str(folder / "partial.npz"),
               "--stop-after-seconds", str(max(1, remaining - 600)),
               "--save-every-seconds", "600", "--hf-repo", HF_REPO,
               "--hf-prefix", prefix]
        if max_roots:
            cmd += ["--max-roots", str(max_roots)]
        run(cmd, cwd=REPO)
        if not (folder / "train_set.npz").exists():
            print(f"[tactical] shard {tag:05d} partial; next kernel resumes", flush=True)
            return
        files.add(f"{remote}/train_set.npz")
        files.add(f"{remote}/meta.json")
        shutil.rmtree(folder)


def teacher_phase(python: Path, checkpoint: Path, prefix: str,
                  tags=None) -> None:
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    for tag in (shard_tags() if tags is None else tags):
        remote = f"{prefix}/shard-{tag:05d}"
        if f"{remote}/teacher_logp.npy" in files:
            continue
        if f"{remote}/train_set.npz" not in files or f"{remote}/meta.json" not in files:
            raise ValueError(f"missing raw labels for {remote}")
        if time.time() - SESSION_STARTED > SESSION_BUDGET - 1800:
            print("[tactical] teacher session budget reached", flush=True)
            return
        folder = WORK / f"teacher-{tag:05d}"
        run([str(python), str(REPO / "scripts" / "label_puzzle_teacher.py"),
             "--shard", str(tag), "--hf-prefix", prefix, "--hf-repo", HF_REPO,
             "--teacher-checkpoint", str(checkpoint), "--sl-repo", str(SL),
             "--workdir", str(folder), "--batch", "1024"], cwd=REPO)
        files.add(f"{remote}/teacher_logp.npy")
        shutil.rmtree(folder)


def trainer_command(prefix: str, run_name: str, rows: dict[str, int],
                    session_steps: int = 0) -> list[str]:
    tags_file = WORK / "frozen-tags.json"
    tags_file.write_text(json.dumps({"tags": sorted(rows), "rows": rows,
                                     "source_sha256": CFG["source_sha256"]},
                                    sort_keys=True) + "\n")
    cmd = [sys.executable, str(REPO / "scripts" / "train_ccgavn.py"),
           "--hf-shards", prefix, "--shard-tags-file", str(tags_file),
           "--hf-repo", HF_REPO, "--hf-run", run_name,
           "--outdir", str(WORK / run_name), "--sl-repo", str(SL),
           "--searchless-commit", CFG["searchless_commit"],
           "--exclusion-puzzles", str(official_puzzles()),
           "--init-from-hf-run", CFG["functional_run"],
           "--init-checkpoint", f"checkpoint-{CFG['base_step']}",
           "--stage-start-step", str(CFG["base_step"]),
           "--steps", str(CFG["final_step"]),
           "--dim", "208", "--layers", "8", "--heads", "8",
           "--batch", "2048", "--lr", "0.0005", "--warmup", "2000",
           "--temperature", "1.0", "--w-dist", "1.0", "--w-ce", "0.25",
           "--reflect-prob", "0.5", "--dev-mod", "100", "--dev-fold", "0",
           "--dev-batch", "8192", "--seed", "0", "--ckpt-every", "50000",
           "--hf-upload-every", "1800", "--session-seconds", "39000",
           "--milestone-every", "50000", "--expect-shards", str(len(rows)),
           "--model-version", CFG["model_version"], "--resume-from-hf"]
    if session_steps:
        cmd += ["--session-steps", str(session_steps)]
    return cmd


def prepare_functional_checkpoint() -> None:
    from scripts.prepare_tactical_checkpoint import prepare
    report = prepare(CFG["base_run"], int(CFG["base_step"]),
                     CFG["functional_run"], HF_REPO, SL, WORK / "transplant")
    if (int(report.get("new_parameters", 0)) != int(CFG["parameter_count"])
            or float(report.get("max_logit_delta", 1.0)) > 1e-4):
        raise ValueError("3.9M functional checkpoint failed its parameter/parity gate")
    print("[tactical] exact 3.9M transplant verified with optimizer state", flush=True)


def demo_phase(source: Path, manifest: Path) -> None:
    prefix = CFG["demo_data_prefix"]
    run_name = CFG["demo_run"]
    prepare_functional_checkpoint()
    binary = stockfish16()
    label_phase(source, manifest, binary, prefix, max_roots=500)
    teacher_python = install_teacher()
    checkpoint = teacher_checkpoint()
    teacher_phase(teacher_python, checkpoint, prefix, tags=[0])
    remote = f"{prefix}/shard-00000/meta.json"
    meta_path = Path(hf_hub_download(HF_REPO, remote, repo_type="dataset", token=TOKEN))
    rows = {"00000": int(json.loads(meta_path.read_text())["rows"])}
    run([sys.executable, "-c", "import torch; assert torch.cuda.is_available()"])
    run(trainer_command(prefix, run_name, rows, session_steps=2), cwd=REPO)
    ckpt = WORK / run_name / f"checkpoint-{CFG['base_step'] + 2}"
    if not (ckpt / "state.pt").is_file():
        raise ValueError("demo learner did not write full checkpoint")
    mate = REPO / "data" / "positions" / "mate-selection-test.json"
    summary_path = WORK / "demo-eval-summary.json"
    run([sys.executable, str(REPO / "scripts" / "eval_gavn.py"),
         "--checkpoint", str(ckpt), "--sl-repo", str(SL),
         "--eval", str(mate), "--max-rows", "10",
         "--puzzles", str(official_puzzles()), "--num-puzzles", "20",
         "--score", "auto", "--summary-out", str(summary_path)], cwd=REPO)
    summary = json.loads(summary_path.read_text())
    if summary.get("mate", {}).get("total") != 10 or summary.get("puzzles", {}).get("total") != 20:
        raise ValueError("demo evaluator did not score its declared rows")
    api.upload_file(path_or_fileobj=str(summary_path),
                    path_in_repo=f"{CFG['demo_prefix']}/eval-summary.json",
                    repo_id=HF_REPO, repo_type="dataset",
                    commit_message="Complete tactical pipeline demo")
    print("[tactical] DEMO COMPLETE", flush=True)


def main() -> None:
    install_base()
    pinned_git("https://github.com/Vedang-P/chess-slm-benchmark.git", REPO,
               CFG["source_commit"])
    pinned_git("https://github.com/google-deepmind/searchless_chess.git", SL,
               CFG["searchless_commit"])
    prepare_searchless_tokenizer()
    sys.path.insert(0, str(REPO))
    if PHASE in ("labels", "demo"):
        source, manifest = source_files()
        if PHASE == "demo":
            demo_phase(source, manifest)
        else:
            label_phase(source, manifest, stockfish16(), CFG["data_prefix"])
    elif PHASE == "teacher":
        teacher_phase(install_teacher(), teacher_checkpoint(), CFG["data_prefix"])
    elif PHASE == "train":
        prepare_functional_checkpoint()
        run([sys.executable, "-c", "import torch; assert torch.cuda.is_available()"])
        run(trainer_command(CFG["data_prefix"], CFG["run"], CFG["rows"]), cwd=REPO)
    else:
        raise ValueError(f"unknown phase {PHASE}")


try:
    main()
    state = "DONE" if PHASE == "demo" else "IN PROGRESS"
    api.upload_file(path_or_fileobj=(
        f"{state} {PHASE} source_commit={CFG['source_commit']} "
        f"account_index={ACCOUNT_INDEX}").encode(),
        path_in_repo=f"{CFG['status_prefix']}/{PHASE}-acct-{ACCOUNT_INDEX}.txt",
        repo_id=HF_REPO, repo_type="dataset",
        commit_message=f"Tactical {PHASE} account {ACCOUNT_INDEX} session finished")
except Exception as exc:
    import traceback
    status = (f"FAILED {PHASE} source_commit={CFG['source_commit']} "
              f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-4000:]}")
    try:
        api.upload_file(path_or_fileobj=status.encode(),
                        path_in_repo=f"{CFG['status_prefix']}/{PHASE}-acct-{ACCOUNT_INDEX}.txt",
                        repo_id=HF_REPO, repo_type="dataset",
                        commit_message=f"Tactical {PHASE} account {ACCOUNT_INDEX} failed")
    except Exception as upload_exc:
        print(f"[tactical] could not persist failure: {upload_exc}", flush=True)
    raise
