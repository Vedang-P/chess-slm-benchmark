"""Generate the Kaggle campaign watcher notebook.

The watcher is a CPU kernel under the softmaxsimp account that drives the
3-wave MATE deepseek campaign after the laptop is closed:

  wave 1 (noexplain): wait for noexplain w2-w5 (running under vedangpandeyyy)
      to reach 200/200 by polling the LIVE GitHub repo's worker states.
  wave 2 (tactic):   regenerate + push 5 kernels AS softmaxsimp, wait for
      all 5 to finish.
  wave 3 (both):     same as wave 2.

Why softmaxsimp: vedangpandeyyy's 5 CPU slots are occupied by the deepseek
workers and its GPU quota is exhausted. softmaxsimp has 5 free CPU slots and
a fresh quota; the worker notebooks are account-agnostic (they clone the
repo and push results to the same GitHub/HF).

Why GitHub polling instead of `kaggle kernels status`: the watcher cannot
see vedangpandeyyy's private kernels, but every worker publishes
monitor/workers/<tag>.state.json to Vedang-P/chess-bench-live regardless of
which account runs it.

Self-resurrection: Kaggle kernels die at 12h. The watcher tracks elapsed
time and, before the limit, regenerates + pushes ITSELF (new kernel version
auto-starts), so a long tactic+both campaign survives a single session.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import reduce
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent


def _read_env() -> dict:
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
    return env


ENV = _read_env()
GITHUB_TOKEN = ENV.get("GITHUB_TOKEN", "")
KAGGLE_TOKEN_VEDANG = "KGAT_883bdea9f208985ab20e63a836a4f753"  # vedangpandeyyy
KAGGLE_TOKEN_SMAX = "KGAT_2017e9ac2c32a6cde220d848f041bd4f"  # softmaxsimp

WATCHER_CELLS = [
    _md("# MATE campaign watcher (softmaxsimp)\n\n"
        "Drives the full campaign after the laptop is closed:\n\n"
        "1. wait for noexplain w2-w5 to reach 200/200\n"
        "2. push + wait for tactic w1-5 (deepseek CPU)\n"
        "3. push + wait for both w1-5 (deepseek CPU)\n"
        "4. push 4 gemma GPU kernels (2 per account: w1 -> vedangpandeyyy, "
        "w2 -> softmaxsimp), wait, then the remaining 2\n\n"
        "Progress comes from the LIVE repo's `monitor/{workers,gemma/workers}/*.state.json` "
        "via the GitHub contents API. Self-relaunches before the 12h kernel "
        "limit by pushing a new version of itself."),
    _code('''
import base64, json, os, subprocess, sys, time
from pathlib import Path

GH = "@@GITHUB_TOKEN@@"
TOK_VEDANG = "@@KAGGLE_TOKEN_VEDANG@@"
TOK_SMAX = "@@KAGGLE_TOKEN_SMAX@@"
OWNER = "softmaxsimp"
REPO = "Vedang-P/chess-slm-benchmark"
LIVE = "Vedang-P/chess-bench-live"
TARGET = 200
GEM_TARGET = 500
STALL_MIN = 45  # minutes without progress before relaunching a worker
SLEEP_S = 90
RELAUNCH_H = 11.5  # push a new watcher version before this many hours

WORK = Path("/kaggle/working")
REPO_DIR = WORK / "chess-slm-benchmark"

def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, stderr=subprocess.STDOUT, **kw)

def api(url):
    # direct urllib call (no subprocess embedding)
    import urllib.request
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {GH}", "User-Agent": "watcher"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None

def worker_done(tag: str, ns: str = "") -> int:
    base = f"monitor/{ns}/workers" if ns else "monitor/workers"
    url = f"https://api.github.com/repos/{LIVE}/contents/{base}/{tag}.state.json"
    d = api(url)
    if not d or "content" not in d:
        return 0
    try:
        state = json.loads(base64.b64decode(d["content"]).decode())
    except Exception:
        return 0
    p = state.get("progress", {})
    return int(p.get("done", 0))
'''.strip()),
    _code('''
def push_wave(prefix: str, n_workers: int = 5, token: str | None = None) -> None:
    env_k = dict(os.environ)
    if token:
        env_k["KAGGLE_API_TOKEN"] = token
    for n in range(1, n_workers + 1):
        push_dir = REPO_DIR / f"notebooks/push_mate_{prefix}_w{n}"
        if not push_dir.exists():
            raise RuntimeError(f"missing push dir {push_dir}")
        res = sh(["kaggle", "kernels", "push", "-p", str(push_dir)], env=env_k)
        if res.returncode != 0:
            print(f"push of {prefix} w{n} failed rc={res.returncode}", flush=True)

def push_gemma_worker(subset: str, n: int, token: str | None = None) -> None:
    push_dir = REPO_DIR / f"notebooks/push_gemma_{subset}/w{n}"
    if not push_dir.exists():
        raise RuntimeError(f"missing push dir {push_dir}")
    env_k = dict(os.environ)
    if token:
        env_k["KAGGLE_API_TOKEN"] = token
    res = sh(["kaggle", "kernels", "push", "-p", str(push_dir)], env=env_k)
    if res.returncode != 0:
        print(f"push of gemma {subset} w{n} failed rc={res.returncode}", flush=True)

def relaunch_self() -> None:
    print("relaunching watcher (12h limit approaching)", flush=True)
    sh([sys.executable, "notebooks/build_campaign_watcher_notebook.py"])
    res = sh(["kaggle", "kernels", "push", "-p",
              str(REPO_DIR / "notebooks/push_watcher")])
    print(f"watcher self-push rc={res.returncode}", flush=True)
'''.strip()),
    _code('''
# --- clone the repo (has the builders) ---
if not REPO_DIR.exists():
    url = "https://github.com/Vedang-P/chess-slm-benchmark.git".replace(
        "https://", f"https://x-access-token:{GH}@")
    sh(["git", "clone", "--quiet", "-b", "main", url, str(REPO_DIR)])
os.chdir(REPO_DIR)

# regenerate all deepseek push dirs under softmaxsimp
env_ds = dict(os.environ, MATE_KERNEL_OWNER=OWNER)
res = sh([sys.executable, "notebooks/build_mate1000_variants_notebook.py"], env=env_ds)
if res.returncode != 0:
    raise SystemExit("deepseek builder failed -- see output above")

started = time.time()
last_seen = {}  # (ns, tag) -> (done, epoch)

def wait_wave(tags: list[str], prefix: str, ns: str = "",
              target: int | None = None) -> None:
    global last_seen, started
    target = target or TARGET
    while True:
        if time.time() - started > RELAUNCH_H * 3600:
            relaunch_self()
            return
        now = time.time()
        done_map = {}
        for tag in tags:
            d = worker_done(tag, ns)
            done_map[tag] = d
            key = (ns, tag)
            prev, pts = last_seen.get(key, (d, now))
            if d > prev:
                last_seen[key] = (d, now)
            elif now - pts > STALL_MIN * 60 and d < target:
                # stalled: push a fresh version of this worker (resumes from HF)
                n = int(tag.split("-w")[-1])
                if ns == "gemma":
                    subset = tag.rsplit("-", 1)[0]
                    token = TOK_VEDANG if n == 1 else TOK_SMAX
                    push_gemma_worker(subset, n, token)
                else:
                    push_dir = REPO_DIR / f"notebooks/push_mate_{prefix}_w{n}"
                    res = sh(["kaggle", "kernels", "push", "-p", str(push_dir)])
                    print(f"relaunched stalled {tag} (rc={res.returncode})", flush=True)
                last_seen[key] = (d, now)
        print(f"[{time.strftime('%H:%M:%S')}] {prefix}: " +
              " ".join(f"{t}={d}" for t, d in done_map.items()), flush=True)
        if all(d >= target for d in done_map.values()):
            return
        time.sleep(SLEEP_S)

# wave 1: noexplain w2-w5 (w1 already complete)
wait_wave([f"noexplain-w{n}" for n in (2, 3, 4, 5)], "noexplain")
print("NOEXPLAIN DONE -- pushing tactic", flush=True)
push_wave("tactic")
wait_wave([f"tactic-w{n}" for n in (1, 2, 3, 4, 5)], "tactic")
print("TACTIC DONE -- pushing both", flush=True)
push_wave("both")
wait_wave([f"both-w{n}" for n in (1, 2, 3, 4, 5)], "both")
print("BOTH DONE -- deepseek Table 1 complete", flush=True)
print("launching gemma on 4 GPUs (2 per account)", flush=True)

# regenerate gemma push dirs: w1 -> vedangpandeyyy, w2 -> softmaxsimp
env_gem = dict(os.environ, GEM_OWNER_W1="vedangpandeyyy", GEM_OWNER_W2="softmaxsimp")
res = sh([sys.executable, "notebooks/build_gemma1000_variants_notebook.py"], env=env_gem)
if res.returncode != 0:
    raise SystemExit("gemma builder failed -- see output above")

# wave 4: 4 gemma workers first (2 per account): noexplain-w1, noexplain-w2,
# tactic-w1, tactic-w2 (w1 of each subset -> vedangpandeyyy, w2 -> softmaxsimp)
for subset, n in (("noexplain", 1), ("noexplain", 2),
                  ("tactic", 1), ("tactic", 2)):
    push_gemma_worker(subset, n, TOK_VEDANG if n == 1 else TOK_SMAX)
wait_wave([f"noexplain-w{n}" for n in (1, 2)] + [f"tactic-w{n}" for n in (1, 2)],
          "gemma-nt", ns="gemma", target=GEM_TARGET)
print("GEMMA NOEXPLAIN+TACTIC DONE -- pushing both", flush=True)

# wave 5: remaining 2 gemma workers
for n in (1, 2):
    push_gemma_worker("both", n, TOK_VEDANG if n == 1 else TOK_SMAX)
wait_wave([f"both-w{n}" for n in (1, 2)], "gemma-both",
          ns="gemma", target=GEM_TARGET)
print("GEMMA BOTH DONE -- campaign complete", flush=True)
'''.strip()),
]


def main() -> None:
    nb = _notebook(WATCHER_CELLS)
    # inject the GitHub + Kaggle tokens into the placeholders (avoids
    # f-string brace collisions with the dict literals in generated code)
    repl = {
        "@@GITHUB_TOKEN@@": GITHUB_TOKEN,
        "@@KAGGLE_TOKEN_VEDANG@@": KAGGLE_TOKEN_VEDANG,
        "@@KAGGLE_TOKEN_SMAX@@": KAGGLE_TOKEN_SMAX,
    }
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = [
                reduce(lambda line, kv: line.replace(*kv), repl.items(), line)
                for line in cell["source"]
            ]
    push_dir = NB_DIR / "push_watcher"
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "kaggle_campaign_watcher.ipynb").write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": "softmaxsimp/mate-campaign-watcher",
        "title": "MATE campaign watcher",
        "code_file": "kaggle_campaign_watcher.ipynb",
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
    print(f"wrote {push_dir}/kernel-metadata.json (CPU, secrets injected)")
    print("push with: kaggle-softmaxsimp kernels push -p notebooks/push_watcher")


if __name__ == "__main__":
    main()
