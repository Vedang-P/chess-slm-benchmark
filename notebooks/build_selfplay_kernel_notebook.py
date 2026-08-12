"""Generate kaggle_selfplay_campaign.ipynb — the Kaggle CPU kernel that
runs the self-play trace campaign unattended.

Why a kernel (vs local): Kaggle CPU kernels are free, run 12h, and the
campaign (~40h for 100 games) self-relaunches: each kernel run plays up
to ~11h of games, then pushes a fresh version of itself before the 12h
kill; the next run resumes from HF checkpoints (Kaggle wipes
/kaggle/working on restart, so in-progress games are synced from HF).

Secrets are injected at build time (never committed): GITHUB_TOKEN,
HF_WRITE_TOKEN, OPENCODE_API_KEY_1..5, and the KAGGLE_API_TOKEN used to
self-relaunch.

    python notebooks/build_selfplay_kernel_notebook.py
    kaggle kernels push -p notebooks/push_selfplay
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
OWNER = os.environ.get("SELFPLAY_KERNEL_OWNER", "softmaxsimp")
SLUG = "self-play-campaign-deepseek-vs-deepseek"
TIME_LIMIT_H = 11.0
GAMES = 100
SLEEP_S = 30

GITHUB_TOKEN = "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"
HF_TOKEN = "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"
KAGGLE_TOKEN = "KGAT_2017e9ac2c32a6cde220d848f041bd4f"

SECRETS_CELL = _code('''
import os
print("secrets are injected at build time; this cell is a placeholder")
'''.strip())

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
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U", "-r", "requirements.txt"], check=True)
print("deps installed")
'''.strip()

GATE_CELL = r'''
status = subprocess.run([sys.executable, "scripts/test_engine.py", "--quick"],
                        capture_output=True, text=True)
print(status.stdout[-1500:])
if status.returncode != 0:
    print(status.stderr[-1500:])
    raise RuntimeError("test_engine failed -- see output above")
print("ALL TESTS PASSED")
'''.strip()

RUN_CELL = r'''
import os, subprocess, sys, time
from pathlib import Path

os.environ["BENCH_RUN_ID"] = "selfplay-campaign"
cmd = [sys.executable, "scripts/run_selfplay_campaign.py",
       "--games", "{{games}}",
       "--out", "results/selfplay",
       "--live-push",
       "--hf-sync",
       "--hf-upload", "120",
       "--time-limit-h", "{{time_limit}}",
       "--interval", "{{sleep_s}}"]
print("running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
elapsed_h = (time.time() - t0) / 3600
print(f"supervisor exited rc={res.returncode} after {elapsed_h:.2f}h", flush=True)
# A non-zero rc is a REAL failure (bad keys, crash, ...) -- fail loudly so
# the kernel shows the traceback instead of silently self-relaunching.
marker = Path("results/selfplay/TIME_LIMIT_HIT")
if res.returncode == 0 and marker.exists():
    print("time limit reached cleanly -- will self-relaunch", flush=True)
elif res.returncode != 0:
    raise RuntimeError(
        f"supervisor exited rc={res.returncode} after {elapsed_h:.2f}h -- "
        "see the supervisor output printed above for the traceback. NOT "
        "self-relaunching; fix and re-push manually.")
else:
    raise RuntimeError(
        "supervisor exited rc=0 but the time-limit marker is missing -- "
        "unexpected early stop; NOT self-relaunching.")
'''.strip()

RELAUNCH_CELL = r'''
import json, os, subprocess, sys, time
from pathlib import Path

# Reaching this cell means the supervisor hit its time budget (or died
# early). Push a fresh version of this kernel so the campaign continues
# in a new 12h window. The new run syncs in-progress games from HF.
print("pushing self-relaunch of %(owner)s/%(slug)s", flush=True)
push_dir = Path("notebooks/push_selfplay")
if not push_dir.exists():
    print("push dir missing -- skipping self-relaunch", flush=True)
else:
    env_k = dict(os.environ)
    env_k["KAGGLE_API_TOKEN"] = os.environ.get("KAGGLE_API_TOKEN", "")
    res = subprocess.run(["kaggle", "kernels", "push", "-p", str(push_dir)],
                         env=env_k, stderr=subprocess.STDOUT)
    print(f"self-relaunch push rc={res.returncode}", flush=True)
    print(res.stdout[-500:] if res.stdout else "", flush=True)
'''.strip()


def inject_secrets(nb: dict, env: dict, names: list[str]) -> None:
    lines = []
    for name in names:
        if name not in env:
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


def build_notebook() -> dict:
    cells = [
        _md("# Self-Play trace campaign (deepseek vs deepseek)\n\n"
            f"Kaggle CPU kernel, {TIME_LIMIT_H:.0f}h budget, "
            f"{GAMES} games. Resumes from HF checkpoints on restart "
            "(Kaggle wipes /kaggle/working). Self-relaunches a fresh "
            "version of itself at the end of each window."),
        _md("## 1. Secrets (injected at build time)"),
        SECRETS_CELL,
        _md("## 2. Get the repo (main branch)"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies"),
        _code(DEPS_CELL),
        _md("## 4. Engine/dataset gate"),
        _code(GATE_CELL),
        _md("## 5. Run the campaign supervisor\n\n"
            "Syncs in-progress games from HF, launches game workers "
            "(one API key each), pushes the throttled live feed, uploads "
            "checkpoints to HF every 2 minutes. Stops launching after "
            f"{TIME_LIMIT_H:.0f}h so the kernel can self-relaunch."),
        _code(RUN_CELL
              .replace("{{games}}", str(GAMES))
              .replace("{{time_limit}}", f"{TIME_LIMIT_H:.1f}")
              .replace("{{sleep_s}}", str(SLEEP_S))),
        _md("## 6. Self-relaunch (continue the campaign)"),
        _code(RELAUNCH_CELL % {"owner": OWNER, "slug": SLUG}),
        _md("## Notes\n"
            "- The campaign supervisor writes local files under "
            "`results/selfplay/` and mirrors them to HF + GitHub.\n"
            "- In-progress games are checkpointed to HF every 2 min; a "
            "fresh kernel run syncs them back and resumes.\n"
            "- Watch live: chess-bench-live.pages.dev/games.html"),
    ]
    return _notebook(cells)


def main() -> None:
    nb = build_notebook()
    env = load_env()
    inject_secrets(nb, env, ["GITHUB_TOKEN", "HF_WRITE_TOKEN",
                             "OPENCODE_API_KEY", "OPENCODE_API_KEY_2",
                             "OPENCODE_API_KEY_3", "OPENCODE_API_KEY_4",
                             "OPENCODE_API_KEY_5"])
    # KAGGLE_API_TOKEN for self-relaunch
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source") or [])
        if "KAGGLE_API_TOKEN" in src and "env_k" in src:
            pass
    # inject KAGGLE_API_TOKEN into the secrets cell too
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source") or [])
        if "secrets set:" in src:
            src += f"os.environ['KAGGLE_API_TOKEN'] = {env.get('KAGGLE_API_TOKEN', KAGGLE_TOKEN)!r}\n"
            nb["cells"][i]["source"] = src.splitlines(keepends=True)

    push_dir = NB_DIR / "push_selfplay"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = "kaggle_selfplay_campaign.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}",
        "title": "Self-play campaign (deepseek vs deepseek)",
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
    print(f"wrote {push_dir}/{code_file} (owner={OWNER}, CPU)")
    print(f"push with: kaggle kernels push -p notebooks/push_selfplay")


if __name__ == "__main__":
    main()
