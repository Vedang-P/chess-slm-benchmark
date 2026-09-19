"""2B-pair build labeler: SHARD_LIST_A / SHARD_LIST_B on the two T4s.

Runs two build_full_dataset.py processes (one per GPU) over the assigned slices;
state is on HF, so kernel restarts resume and skip finished shards. Re-pushed by
the watch-build-2b GitHub workflow until every assigned shard exists on HF.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SLICE_A = ["02084", "00036", "02085", "02097", "02022", "00182", "01959", "02104", "01896", "02122", "02021", "01890", "01757"]
SLICE_B = ["02011", "00209", "01963", "02086", "02102", "02013", "01731", "01886", "01370", "01785", "02125", "01698", "02138"]
WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
SL = WORK / "searchless_chess"
V = WORK / "slvenv"

if not (REPO / "scripts" / "build_full_dataset.py").exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Vedang-P/chess-slm-benchmark.git", str(REPO)], check=True)
if not SL.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/google-deepmind/searchless_chess.git", str(SL)], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "virtualenv"], check=True)
subprocess.run([sys.executable, "-m", "virtualenv", str(V)], check=True)
subprocess.run([f"{V}/bin/pip", "install", "--quiet", "-U", "pip"], check=True)
subprocess.run([f"{V}/bin/pip", "install", "--quiet",
                "-f", "https://storage.googleapis.com/jax-releases/jax_releases.html",
                "jax==0.4.35", "jaxlib==0.4.35", "jax_cuda12_pjrt==0.4.35", "jax_cuda12_plugin==0.4.35",
                "orbax-checkpoint==0.5.5", "dm-haiku==0.0.11", "numpy==1.26.4", "pandas==2.2.3",
                "jaxtyping", "typing-extensions", "python-chess", "zstandard",
                "apache-beam", "grain", "wrapt", "huggingface_hub"], check=True)

CK = WORK / "checkpoints"
CK.mkdir(exist_ok=True)
TEACHER = CK / "9M/6400000/params_ema"
if not TEACHER.exists():
    z = CK / "9M.zip"
    subprocess.run(["curl", "-sL", "--retry", "5", "-o", str(z),
                    "https://storage.googleapis.com/searchless_chess/checkpoints/9M.zip"], check=True)
    subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(CK)], check=True)

# HF token from the attached credentials dataset
import glob  # noqa: E402
hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
if hits:
    os.environ["HF_WRITE_TOKEN"] = Path(hits[0]).read_text().strip()
assert os.environ.get("HF_WRITE_TOKEN"), "no HF token"

def run(slice_ids, gpu, workdir):
    cmd = [f"{V}/bin/python", str(REPO / "scripts" / "build_full_dataset.py"),
           "--shard-list", ",".join(slice_ids), "--sl-repo", str(SL),
           "--workdir", workdir, "--teacher-checkpoint", str(TEACHER),
           "--teacher-dim", "256", "--teacher-layers", "8", "--teacher-heads", "8",
           "--teacher-batch", "1024",
           "--hf-repo", "vedangfake/chess-slm-benchmark", "--hf-run", "chessbench-full-build",
           "--resume-from-hf"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[2b] launching slice {slice_ids[:3]}... on GPU {gpu}", flush=True)
    return subprocess.Popen(cmd, env=env)

p1 = run(SLICE_A, 0, "/kaggle/working/build-a")
p2 = run(SLICE_B, 1, "/kaggle/working/build-b")
rc1 = p1.wait()
rc2 = p2.wait()
print(f"[2b] done rc={rc1},{rc2}", flush=True)
