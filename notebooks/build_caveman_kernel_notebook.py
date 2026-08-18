"""Generate kaggle_caveman_traces_demo.ipynb / _full.ipynb — deepseek
writes caveman explanations of Stockfish lines (the SFT training data).

Kaggle CPU kernel, internet enabled. Flow: clone repo -> light deps ->
download lines-2000.jsonl + the current trace shard from HF (resume) ->
run synthesize (gateway + HF uploads every 25 rows, all from secrets).

DEMO first (--count 10 --offset 16): verify the whole loop on Kaggle
before spending the full budget. FULL (--count 0 = all remaining):
re-run the same notebook as often as needed; it resumes from the HF
shard each time (Kaggle kernels die at ~12h; ~1 min/row => several runs).

    python3 notebooks/build_caveman_kernel_notebook.py --count 10 --offset 16
    kaggle kernels push -p notebooks/push_caveman_traces_demo
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
OWNER = os.environ.get("CAVEMAN_OWNER", "vedanggggg")
REPO_ID = "vedangfake/chess-slm-benchmark"

CLONE_CELL = r'''
import os, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
if REPO.exists():
    import shutil; shutil.rmtree(REPO)

# Kaggle's git demands credentials even for public repos (measured
# 2026-08-19: anonymous clone -> "could not read Username"). Use the
# GITHUB_TOKEN secret when present, plain clone otherwise.
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
tok = find_token()
if tok:
    url = url.replace("https://", f"https://x-access-token:{tok}@")
res = subprocess.run(["git", "clone", "--quiet", "-b", "main", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed: " + res.stderr[-300:])
os.chdir(REPO)
print("cwd:", Path.cwd())
'''.strip()

DEPS_CELL = r'''
import subprocess, sys
# Light install: only what synthesize needs. torch comes with Kaggle.
# (src.models imports NO torch at module level; the HF helper is
# src.hf_push, not the trl trainer.)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                "transformers", "huggingface_hub", "python-chess"], check=True)
print("deps installed")
'''.strip()

FETCH_CELL = r'''
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "vedangfake/chess-slm-benchmark"
WORK = Path("/kaggle/working")
os.chdir(WORK / "chess-slm-benchmark")

lines = hf_hub_download(repo_id=REPO_ID,
                        filename="caveman/lines-2000.jsonl",
                        repo_type="dataset")
import shutil
lines_local = WORK / "lines.jsonl"
shutil.copy(lines, lines_local)
print("lines:", lines_local)

out = WORK / "traces.jsonl"
if not out.exists():
    try:
        shard = hf_hub_download(repo_id=REPO_ID,
                                filename="%HF_PATH%",
                                repo_type="dataset")
        out.write_bytes(Path(shard).read_bytes())
        print(f"resume: {len(out.read_text().splitlines())} rows already done")
    except Exception as e:
        out.write_text("")
        print("no remote shard yet, starting empty:", type(e).__name__)
print("out:", out)
'''.strip()

RUN_CELL = r'''
import subprocess, sys
from pathlib import Path

cmd = [sys.executable, "scripts/synthesize_caveman_traces.py",
       "--lines", LINES, "--out", OUT,
       "--hf-path", "%HF_PATH%", %RUN_ARGS%]
print("running:", " ".join(cmd[:6]), "...")
r = subprocess.run(cmd)
if r.returncode != 0:
    raise SystemExit(f"synthesize failed: {r.returncode}")
done = len(Path(OUT).read_text().splitlines())
print(f"rows on HF shard now: {done}")
'''.strip()


def load_env() -> dict:
    env_path = NB_DIR.parent / ".env"
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def inject_secrets(nb: dict, env: dict, names: list[str],
                   remap: dict | None = None) -> None:
    remap = remap or {}
    lines = ["import os\n"]
    for name in names:
        src = remap.get(name, name)
        if src not in env or not env[src]:
            raise RuntimeError(f"missing secret {src} in .env")
        lines.append(f'os.environ[{name!r}] = {env[src]!r}\n')
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=0,
                    help="rows to synthesize (0 = all remaining)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--hf-path", default="caveman-traces/traces.jsonl",
                    help="HF path for this notebook's resume+upload shard "
                         "(sharded runs use caveman-traces/shards/shard-N.jsonl)")
    ap.add_argument("--slug", default="",
                    help="kernel slug override (default derived from count)")
    ap.add_argument("--key-from", default="OPENCODE_API_KEY",
                    help=".env key whose value is injected as OPENCODE_API_KEY")
    args = ap.parse_args()

    if args.slug:
        slug = args.slug
        title = slug.replace("-", " ")
    elif args.count > 0:
        slug = f"caveman-traces-demo-c{args.count}"
        title = f"caveman traces demo c{args.count}"
    else:
        slug = "caveman-traces-full"
        title = "caveman traces full"
    run_args = (f'"--offset", "{args.offset}", "--count", "{args.count}"'
                if args.count > 0 else '"--offset", "0", "--count", "0"')

    cells = [
        _md(f"# {title}\n\n"
            "Deepseek writes caveman explanations of Stockfish lines. "
            "Resumes from the HF shard (already-done fens are skipped). "
            "Uploads the shard to HF every 25 rows."),
        _md("## 1. Secrets"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies"),
        _code(DEPS_CELL),
        _md("## 4. Fetch lines + resume shard from HF"),
        _code(FETCH_CELL.replace("%HF_PATH%", args.hf_path)),
        _md("## 5. Synthesize"),
        _code(RUN_CELL.replace("LINES", '"/kaggle/working/lines.jsonl"')
                   .replace("OUT", '"/kaggle/working/traces.jsonl"')
                   .replace("%HF_PATH%", args.hf_path)
                   .replace("%RUN_ARGS%", run_args)),
        _md("## 6. Notes\n\n"
            "- Re-run this notebook to continue: it resumes from the HF "
            "shard and never re-generates a done fen.\n"
            "- ~1 min/row (deepseek streaming); a 12h kernel does "
            "~500-700 rows, so the full 2000 takes ~3-4 runs.\n"
            "- Budget: hard abort at 8,192 thinking tokens; every row "
            "stores raw thinking + checks + usage."),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env, ["OPENCODE_API_KEY", "HF_WRITE_TOKEN",
                             "GITHUB_TOKEN"],
                   remap={"OPENCODE_API_KEY": args.key_from})

    push_dir = NB_DIR / f"push_{slug}"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"kaggle_{slug}.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{slug}",
        "title": title,
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
    print(f"wrote {push_dir}/{code_file}")
    print(f"push with: kaggle kernels push -p notebooks/{push_dir.name}")


if __name__ == "__main__":
    main()
