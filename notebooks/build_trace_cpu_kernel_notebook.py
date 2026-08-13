"""Generate kaggle_traces_cpu.ipynb — verified lucid trace generation on a
Kaggle CPU kernel (overnight, resume-safe, sharded by --offset/--count).

    python notebooks/build_trace_cpu_kernel_notebook.py --offset 0 --count 1000
    kaggle kernels push -p notebooks/push_traces_0

Each shard owns 1000 positions with its own API key. 3 shards = 3k traces
in one overnight. Run merge locally afterwards.

Secrets injected at build time: GITHUB_TOKEN, HF_WRITE_TOKEN, OPENCODE_API_KEY,
WANDB_API_KEY.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
OWNER = os.environ.get("TRACE_OWNER", "vedanggggg")

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
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        return None

url = "https://github.com/Vedang-P/chess-slm-benchmark.git"
url = url.replace("https://", f"https://x-access-token:{find_token()}@")
res = subprocess.run(["git", "clone", "--quiet", "-b", "main", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed: " + res.stderr[-300:])
os.chdir(REPO)
print("cwd:", Path.cwd())
'''.strip()

DEPS_CELL = r'''
import subprocess, sys, torch
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "python-chess"], check=True)
print("cpu ready")
'''.strip()

SELECT_CELL = r'''
import os, subprocess, sys, time
# build the selected-position list (idempotent; overwrites local only)
cmd = [sys.executable, "scripts/build_lucid_traces.py",
       "--stage", "select", "--k", "3000",
       "--out", "data/positions/noexplain-slice/traces"]
print("running:", " ".join(cmd))
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
if res.returncode != 0:
    raise RuntimeError("select failed")
# push selected.jsonl to HF so all shards share one list
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
api.upload_file(
    path_or_fileobj="data/positions/noexplain-slice/traces/selected.jsonl",
    path_in_repo="noexplain-slice/traces/selected.jsonl",
    repo_id="vedangfake/chess-slm-benchmark", repo_type="dataset")
print("selected.jsonl uploaded", flush=True)
'''.strip()

FETCH_CELL = r'''
import os
from huggingface_hub import hf_hub_download
import subprocess, sys
from pathlib import Path
# fetch the shared selected list
Path("data/positions/noexplain-slice/traces").mkdir(parents=True, exist_ok=True)
hf_hub_download(
    repo_id="vedangfake/chess-slm-benchmark",
    filename="noexplain-slice/traces/selected.jsonl",
    repo_type="dataset",
    local_dir="/kaggle/working/chess-slm-benchmark",
    token=os.environ.get("HF_WRITE_TOKEN", ""))
print("selected.jsonl fetched", flush=True)
'''.strip()

TRACES_CELL = r'''
import os, subprocess, sys, time
from pathlib import Path

# gateway serializes per API key: the injected shard key drives this
# kernel's quota while the generic env slot stays untouched for other runs
os.environ["OPENCODE_API_KEY"] = os.environ.get("%(key_name)s", "")

cmd = [sys.executable, "scripts/build_lucid_traces.py",
       "--stage", "traces",
       "--offset", "%(offset)s",
       "--count", "%(count)s",
       "--out", "data/positions/noexplain-slice/traces"]
print("running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
print(f"traces exited rc={res.returncode} after {(time.time()-t0)/3600:.2f}h", flush=True)
if res.returncode != 0:
    raise RuntimeError("trace gen failed")

# push shard to HF so merge can run anywhere
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
shard = f"data/positions/noexplain-slice/traces/traces-%(offset)s.jsonl"
api.upload_file(path_or_fileobj=shard,
                path_in_repo=f"noexplain-slice/traces/traces-%(offset)s.jsonl",
                repo_id="vedangfake/chess-slm-benchmark", repo_type="dataset")
print("shard uploaded", flush=True)
'''.strip()


def load_env() -> dict:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def inject_secrets(nb: dict, env: dict, names: list[str]) -> None:
    lines = ["import os\n"]
    for name in names:
        if name not in env or not env[name]:
            raise RuntimeError(f"missing secret {name} in .env")
        lines.append(f'os.environ[{name!r}] = {env[name]!r}\n')
    lines.append("print('secrets set:', "
                 + ", ".join(f"bool(os.environ.get({n!r}))" for n in names)
                 + ")\n")
    cell = {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": lines,
    }
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source") or [])
        if "secrets are injected at build time" in src:
            nb["cells"][i] = cell
            return
    raise RuntimeError("placeholder secrets cell not found")


def main() -> None:
    offset = 0
    count = 1000
    key_name = "OPENCODE_API_KEY"
    for a in sys.argv[1:]:
        if a.startswith("--offset="):
            offset = int(a.split("=")[1])
        elif a.startswith("--count="):
            count = int(a.split("=")[1])
        elif a.startswith("--key-name="):
            key_name = a.split("=")[1]
    slug = f"traces-cpu-{offset}"
    cells = [
        _md("# Verified lucid trace generation (CPU, overnight)\n\n"
            f"Shard offset={offset} count={count}. Resume-safe: each "
            "position is appended + pushed to HF, so a session restart "
            "continues where it stopped."),
        _md("## 1. Secrets (injected at build time)"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies"),
        _code(DEPS_CELL),
        _md("## 4. Fetch the shared selected list"),
        _code(FETCH_CELL),
        _md("## 5. Generate + verify lucid traces (resume-safe)"),
        _code(TRACES_CELL % {"offset": offset, "count": count,
                             "key_name": key_name}),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env,
                   ["GITHUB_TOKEN", "HF_WRITE_TOKEN", key_name,
                    "WANDB_API_KEY"])

    push_dir = NB_DIR / f"push_traces_{offset}"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"kaggle_traces_{offset}.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{slug}",
        "title": f"Trace CPU shard {offset}",
        "code_file": code_file,
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
    }, indent=1))
    print(f"wrote {push_dir}/{code_file} (offset={offset} count={count})")
    print(f"push with: kaggle kernels push -p notebooks/push_traces_{offset}")


if __name__ == "__main__":
    main()
