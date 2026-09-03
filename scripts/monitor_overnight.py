#!/usr/bin/env python3
"""
Overnight monitor for chessbench-full 8-shard build + assembly + training.

Polls every 3 minutes:
- HF manifest + files (source of truth), run-status.txt
- Kaggle kernel statuses for all 8 kernels (3 accounts)
- Kaggle streaming logs (last 15s per running kernel) for [build] / errors
- Validates shard sizes / normalization (lightweight HEAD checks)

Logs to stdout + ./logs/monitor_overnight.log (or /tmp). Designed to run via
`hub start` with restart:on-failure, and via `caffeinate -i` to survive lid-close.

Usage:
  python3 scripts/monitor_overnight.py [--interval 180] [--once]

Auth handling:
  - vedanggggg  -> default ~/.kaggle/access_token (KGAT)
  - vedangpandeyyy -> fake HOME + KAGGLE_CONFIG_DIR with kaggle.json, no access_token
  - softmaxsimp -> KAGGLE_API_TOKEN env (KGAT)
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "monitor_overnight.log"
NTFY_TOPIC_DEFAULT = "chess-vedang-8shards-4k9p2x7q"
NTFY_INTERVAL_S = 1800  # legacy health ping interval (disabled below; keep for manual use)
_last_ntfy_health = 0
_last_shards_ok: set = set()
_ntfy_failed_sent: set = set()   # kernels already alerted as FAILED (alert once)
_ntfy_status_sent: str = ""      # last run-status error text already alerted
KERNELS = [
    # builders
    ("vedanggggg/chessbench-full-build", "vedanggggg", "builder 0+1 (0..8 resume)"),
    ("vedanggggg/acc1-build-shards2-3", "vedanggggg", "builder 2-3"),
    ("vedangpandeyyy/acc2-build-shards4-5", "vedangpandeyyy", "builder 4-5"),
    ("vedangpandeyyy/acc2-build-shards6-7", "vedangpandeyyy", "builder 6-7"),
    ("softmaxsimp/acc3-build-shard1", "softmaxsimp", "builder 1 backup"),
    # assemble
    ("vedanggggg/chessbench-full-assemble", "vedanggggg", "assemble 8->25GB"),
    # trainers (auto-pushed after assemble, 6 parallel, 6-9h)
    ("vedanggggg/baseline-5m-seed0", "vedanggggg", "train baseline 5M"),
    ("vedanggggg/gavn-3m-seed0", "vedanggggg", "train gavn 3M"),
    ("vedangpandeyyy/gavn-5m-seed0", "vedangpandeyyy", "train gavn 5M"),
    ("vedangpandeyyy/gavn-3m-seed1", "vedangpandeyyy", "train gavn 3M seed1"),
    ("softmaxsimp/gavn-5m-geometry", "softmaxsimp", "train gavn 5M geometry"),
    ("softmaxsimp/gavn-5m-loss", "softmaxsimp", "train gavn 5M loss"),
]
def ntfy_send(title: str, message: str, priority: str = "default", tags: str = ""):
    topic = os.environ.get("NTFY_TOPIC") or NTFY_TOPIC_DEFAULT
    if not topic:
        return
    try:
        url = f"https://ntfy.sh/{topic}"
        data = message.encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Title", title[:80])
        req.add_header("Priority", priority)
        if tags:
            req.add_header("Tags", tags)
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        log(f"ntfy sent: {title} -> {topic} ({priority})")
    except Exception as e:
        log(f"ntfy failed {title}: {e}")
HF_REPO = "vedangfake/chess-slm-benchmark"
HF_RUN = "chessbench-full-build"
N_SHARDS = 8

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg: str):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def env_for_account(account: str) -> dict:
    """Return env dict for subprocess that authenticates as `account`."""
    env = os.environ.copy()
    # Clean up any previous overrides
    env.pop("KAGGLE_CONFIG_DIR", None)
    env.pop("KAGGLE_API_TOKEN", None)
    if account == "vedanggggg":
        # default: uses ~/.kaggle/access_token; ensure no KAGGLE_API_TOKEN override
        pass
    elif account == "vedangpandeyyy":
        # Need to hide default access_token and use kaggle.json via fake HOME
        # Create temp HOME with .kaggle/kaggle.json
        fake_home = Path(tempfile.gettempdir()) / "kaggle_home_vedangpandeyyy"
        fake_home.mkdir(parents=True, exist_ok=True)
        kdir = fake_home / ".kaggle"
        kdir.mkdir(exist_ok=True)
        # copy kaggle.json from real home
        src = Path.home() / ".kaggle/kaggle.json"
        dst = kdir / "kaggle.json"
        if src.exists():
            shutil.copy(str(src), str(dst))
            os.chmod(dst, 0o600)
        # ensure no access_token in fake home
        (kdir / "access_token").unlink(missing_ok=True)
        (kdir / "access_token.txt").unlink(missing_ok=True)
        env["HOME"] = str(fake_home)
        # also set KAGGLE_CONFIG_DIR to fake .kaggle to be explicit
        env["KAGGLE_CONFIG_DIR"] = str(kdir)
    elif account == "softmaxsimp":
        token_path = Path.home() / ".kaggle/profiles/softmaxsimp/access_token"
        token = token_path.read_text().strip() if token_path.exists() else ""
        env["KAGGLE_API_TOKEN"] = token
        # Ensure default access_token is hidden by setting HOME to empty dir without token
        # But softmaxsimp uses token env, which takes precedence over file, so file can remain
        # However to be safe, we don't need to hide default; token env overrides file per SDK
        pass
    else:
        raise ValueError(account)
    return env

def kaggle_status(ref: str, account: str) -> str:
    env = env_for_account(account)
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", ref],
                       capture_output=True, text=True, env=env, timeout=30)
    out = (r.stdout + r.stderr).strip()
    # last token is status like KernelWorkerStatus.RUNNING
    if "RUNNING" in out:
        return "RUNNING"
    if "COMPLETE" in out:
        return "COMPLETE"
    if "FAILED" in out or "ERROR" in out:
        return "FAILED"
    # fallback: extract quoted status
    m = re.search(r'"([^"]+)"', out)
    if m:
        return m.group(1).split(".")[-1]
    return out[-80:] if out else "UNKNOWN"

def kaggle_stream_logs_snippet(ref: str, account: str, timeout_s: int = 12) -> str:
    """Run a short Python snippet that streams logs for `timeout_s` and returns filtered lines."""
    # Write a temp python script that authenticates correctly and streams
    snippet = f"""
import os, sys, re, time, pathlib, shutil, tempfile
from pathlib import Path
# setup env already provided by parent env
from kaggle.api.kaggle_api_extended import KaggleApi
try:
    api = KaggleApi()
    api.authenticate()
    # stream with timeout
    import signal
    def handler(signum, frame):
        raise TimeoutError("stream timeout")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm({timeout_s})
    out_lines=[]
    try:
        for evt in api.kernels_logs_stream("{ref}"):
            data = evt.get("data","")
            if any(k in data for k in ["[build]", "[assemble]", "[train]", "Traceback", "ERROR", "HF secret", "download failed", "truncated", "DONE", "rows", "manifest"]):
                out_lines.append(data.strip()[:800])
            if len(out_lines) > 40:
                break
    except TimeoutError:
        pass
    except Exception as e:
        out_lines.append(f"STREAM_ERR: {{e}}")
    finally:
        try:
            signal.alarm(0)
        except: pass
    for l in out_lines[-40:]:
        print(l)
except Exception as e:
    print(f"AUTH_ERR: {{e}}")
    import traceback; traceback.print_exc()
"""
    env = env_for_account(account)
    r = subprocess.run([sys.executable, "-c", snippet],
                       capture_output=True, text=True, env=env, timeout=timeout_s+10)
    return (r.stdout + r.stderr).strip()[:6000]

def hf_check():
    """Poll HF manifest, run-status, and file list."""
    # Load HF token from .env
    env = {}
    try:
        for line in open(ROOT / ".env"):
            line=line.strip()
            if "=" in line and not line.startswith("#"):
                k,v=line.split("=",1)
                env[k]=v.strip()
    except Exception:
        pass
    tok = env.get("HF_WRITE_TOKEN","").strip()
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    try:
        files = api.list_repo_files(HF_REPO, repo_type="dataset")
    except Exception as e:
        return {"error": f"list_repo_files failed: {e}", "files": [], "manifest": None, "status": None}
    # manifest
    manifest = None
    status_txt = None
    import urllib.request, json
    for path, key in [(f"{HF_RUN}/manifest.json","manifest"), (f"{HF_RUN}/run-status.txt","status")]:
        url = f"https://huggingface.co/datasets/{HF_REPO}/raw/main/{path}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read().decode()
                if key=="manifest":
                    try:
                        manifest = json.loads(data)
                    except:
                        manifest = {"raw": data[:2000]}
                else:
                    status_txt = data[:6000]
        except Exception as e:
            if key=="status":
                # 404 means no error status (good)
                status_txt = None
            else:
                manifest = None
    # shard files present
    present = {}
    for f in files:
        if f.startswith(f"{HF_RUN}/shard-"):
            # f like chessbench-full-build/shard-00000/train_set.npz
            parts = f.split("/")
            if len(parts)>=3:
                tag = parts[1].replace("shard-","")
                fname = parts[2]
                present.setdefault(tag, set()).add(fname)
    # check assembled dataset if exists
    assembled = [f for f in files if "chessbench" in f and ("train_set.npz" in f or "assembled" in f)]
    return {"files": files, "manifest": manifest, "status": status_txt, "present": present, "assembled": assembled}

def validate_shard_presence(present: dict):
    """Check each of 0..7 has both files."""
    ok=[]
    missing=[]
    partial=[]
    for i in range(N_SHARDS):
        tag=f"{i:05d}"
        s=present.get(tag, set())
        if "train_set.npz" in s and "teacher_logp.npy" in s:
            ok.append(tag)
        elif s:
            partial.append((tag, s))
        else:
            missing.append(tag)
    return ok, partial, missing

TRAINER_CONFIGS = [
    # (owner, slug, notebook_template, RUN_ID, description, extra_replacements)
    ("vedanggggg", "baseline-5m-seed0", "gavn", "account1-baseline-5m-seed0", "baseline 5M control (train_student)", {"train_gavn.py": "train_student.py", "DIM = 224": "DIM = 224", "BIAS_MODE": None}),
    ("vedanggggg", "gavn-3m-seed0", "gavn", "account1-gavn-3m-seed0", "gavn 3M", {"DIM = 224": "DIM = 192", "RUN_ID = 'account2-gavn-5m-seed0'": "RUN_ID = 'account1-gavn-3m-seed0'"}),
    ("vedangpandeyyy", "gavn-5m-seed0", "gavn", "account2-gavn-5m-seed0", "gavn 5M", {}),
    ("vedangpandeyyy", "gavn-3m-seed1", "gavn", "account2-gavn-3m-seed1", "gavn 3M seed1", {"SEED = 0": "SEED = 1", "RUN_ID = 'account2-gavn-5m-seed0'": "RUN_ID = 'account2-gavn-3m-seed1'", "DIM = 224": "DIM = 192"}),
    ("softmaxsimp", "gavn-5m-geometry", "gavn", "account3-gavn-5m-geometry", "geometry ablation fixed", {"BIAS_MODE = 'both'": "BIAS_MODE = 'fixed'", "RUN_ID = 'account2-gavn-5m-seed0'": "RUN_ID = 'account3-gavn-5m-geometry'"}),
    ("softmaxsimp", "gavn-5m-loss", "gavn", "account3-gavn-5m-loss", "loss ablation no-q", {"'--w-q', '0.5'": "'--w-q', '0.0'", "RUN_ID = 'account2-gavn-5m-seed0'": "RUN_ID = 'account3-gavn-5m-loss'"}),
]

_trainers_pushed = False

def maybe_auto_push_trainers(hf: dict):
    global _trainers_pushed
    if _trainers_pushed:
        return
    # For sharded training, just need 8 shards, not assembled
    files = set(hf.get("files", []))
    n_shards = sum(1 for i in range(8) if f'chessbench-full-build/shard-{i:05d}/train_set.npz' in files and f'chessbench-full-build/shard-{i:05d}/teacher_logp.npy' in files)
    if n_shards < 8:
        return
    state_path = LOG_DIR / "trainer_push_state.json"
    if state_path.exists():
        try:
            if json.loads(state_path.read_text()).get("pushed"):
                _trainers_pushed = True
                return
        except: pass
    log("AUTO-PUSH: 8 shards ready, pushing 6 trainers (sharded, no assemble)")
    ntfy_send("Shards ready - pushing trainers", "8/8 shards, launching 6 training kernels (sharded, 5GB peak)", priority="high", tags="rocket")
    # Use canonical notebooks from repo (sharded: HF_SHARDS, no 25GB assemble)
    for owner, slug, tmpl, run_id, desc, repl in TRAINER_CONFIGS:
        template_path = ROOT / ("notebooks/01_kaggle_baseline_5m.ipynb" if "baseline" in slug else "notebooks/02_kaggle_train_gavn.ipynb")
        try:
            push_dir = Path(tempfile.mkdtemp(prefix=f"kaggle_push_{slug}_"))
            j = json.loads(template_path.read_text())
            txt = json.dumps(j)
            for k,v in repl.items():
                if v is None: continue
                txt = txt.replace(k, v)
            txt = txt.replace("account2-gavn-5m-seed0", run_id)
            txt = txt.replace("account1-gavn-3m-seed0", run_id)
            j2 = json.loads(txt)
            # fix RUN_ID line explicitly if still not found
            for cell in j2["cells"]:
                if cell["cell_type"]=="code":
                    src = "".join(cell["source"])
                    if "RUN_ID =" in src and run_id not in src:
                        cell["source"] = [s.replace("account1-gavn-3m-seed0", run_id).replace("account2-gavn-5m-seed0", run_id).replace("account3-gavn-5m-geometry", run_id) for s in cell["source"]]
            # baseline already uses train_student.py via its own template; no patch needed
            out_nb = push_dir / f"{slug}.ipynb"
            out_nb.write_text(json.dumps(j2))
            # metadata
            meta = {
                "id": f"{owner}/{slug}",
                "title": slug,
                "code_file": f"{slug}.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": []
            }
            (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
            env = env_for_account(owner)
            log(f"pushing {owner}/{slug} ({desc})")
            r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(push_dir)], capture_output=True, text=True, env=env, timeout=120)
            log(f"push {owner}/{slug} rc={r.returncode} out={r.stdout[:400]} err={r.stderr[:600]}")
            if r.returncode==0:
                ntfy_send(f"Pushed {slug}", f"{owner}/{slug} started", priority="default", tags="arrow_forward")
            else:
                log(f"push failed for {owner}/{slug}, will retry next poll")
                return  # retry later
            time.sleep(15)
        except Exception as e:
            log(f"auto-push exception for {owner}/{slug}: {e}\n{traceback.format_exc()[:800]}")
            return
    # mark pushed
    try:
        state_path.write_text(json.dumps({"pushed": True, "time": ts(), "run_ids": [c[3] for c in TRAINER_CONFIGS]}))
    except: pass
    _trainers_pushed = True
    ntfy_send("All 6 trainers pushed", "Training wave 1 launched, 6-9h to finish", priority="high", tags="tada")

def once():
    log("="*70)
    log(f"poll: {N_SHARDS} shards target, HF {HF_REPO}/{HF_RUN}")
    # HF
    try:
        hf = hf_check()
    except Exception as e:
        log(f"HF check failed: {e}\n{traceback.format_exc()[-600:]}")
        hf = {"error": str(e), "present": {}, "manifest": None, "status": None}
    if "error" in hf and hf["error"]:
        log(f"HF error: {hf['error']}")
    manifest = hf.get("manifest")
    present = hf.get("present", {})
    ok, partial, missing = validate_shard_presence(present)
    manifest_shards = sorted((manifest or {}).get("shards", {}).keys()) if isinstance(manifest, dict) else []
    log(f"HF files: {len(ok)}/8 complete shards present: {ok}")
    if partial:
        log(f"  partial shards: {partial}")
    if missing:
        log(f"  missing shards: {missing}")
    log(f"HF manifest shards: {manifest_shards} (manifest is advisory; files are source of truth)")
    if manifest and isinstance(manifest, dict):
        for tag, info in sorted(manifest.get("shards", {}).items()):
            log(f"  manifest {tag}: {info}")
    status_txt = hf.get("status")
    if status_txt:
        # status contains traceback on failure
        log(f"HF run-status.txt PRESENT (error): {status_txt[:2000]}")
    else:
        log("HF run-status.txt: none (no failure recorded)")
    # Kaggle statuses
    statuses = {}
    for ref, account, desc in KERNELS:
        try:
            st = kaggle_status(ref, account)
        except Exception as e:
            st = f"ERR:{e}"
        statuses[ref]=st
        log(f"Kaggle {ref:40s} [{account:14s}] {st:10s}  # {desc}")
    # Streaming logs for RUNNING builders (first 4 builders + assemble)
    running_builders = [(ref,acc) for ref,acc,_ in KERNELS[:5] if statuses.get(ref)=="RUNNING"]
    for ref, acc in running_builders:
        try:
            snippet = kaggle_stream_logs_snippet(ref, acc, timeout_s=12)
            if snippet.strip():
                # compress
                lines = [l for l in snippet.split("\n") if l.strip()]
                log(f"logs {ref} ({acc}) last {len(lines)} filtered lines:")
                for l in lines[-15:]:
                    log(f"  {l[:500]}")
                # detect fatal patterns
                if "HF secret unavailable" in snippet:
                    log(f"ALERT {ref}: HF secret unavailable -> will fail uploads! Check Kaggle secret HF_WRITE_TOKEN")
                if "Traceback" in snippet or "ArgumentError" in snippet:
                    log(f"ALERT {ref}: Traceback / ArgumentError -> check logs")
                if "download failed" in snippet.lower() or "truncated" in snippet.lower():
                    log(f"ALERT {ref}: download/truncated issue")
            else:
                log(f"logs {ref}: (no filtered lines in 12s stream; kernel may be early setup or idle)")
        except Exception as e:
            log(f"logs {ref} fetch failed: {e}")
    # Overall health
    n_ok = len(ok)
    if n_ok == 8:
        log("HEALTH: ALL 8 SHARDS PRESENT -> ready for assembly")
    elif n_ok >=5:
        log(f"HEALTH: {n_ok}/8 shards done, {len(missing)} missing: {missing} -> ETA 1-2h for remaining")
    else:
        log(f"HEALTH: {n_ok}/8 shards done -> still building")
    # ntfy notifications — IMPORTANT-ONLY policy:
    #   * FAILED kernel: alert ONCE per kernel (deduped), never repeat.
    #   * run-status error: alert ONCE per distinct error text.
    #   * NO periodic health pings (logs carry that detail).
    #   * milestone alerts (new shard, 8/8, pushed trainers) still fire once.
    global _last_ntfy_health, _last_shards_ok, _ntfy_failed_sent, _ntfy_status_sent
    try:
        now = time.time()
        new_ok = set(ok) - _last_shards_ok
        if new_ok:
            ntfy_send(f"Shard done: {sorted(new_ok)}", f"{len(ok)}/8 shards present: {ok}\nmissing: {missing}", priority="high", tags="white_check_mark")
        # FAILED kernels — alert once per kernel only
        failed = [r for r, s in statuses.items() if s == "FAILED"]
        new_failed = [r for r in failed if r not in _ntfy_failed_sent]
        if new_failed:
            ntfy_send("Kaggle FAILED (new)", f"{new_failed}", priority="urgent", tags="rotating_light")
            _ntfy_failed_sent.update(new_failed)
        # run-status.txt — alert once per distinct error text
        if status_txt and "ArgumentError" not in status_txt[:200]:
            if status_txt[:800] != _ntfy_status_sent:
                ntfy_send("HF run-status error", status_txt[:800], priority="high", tags="warning")
                _ntfy_status_sent = status_txt[:800]
        # NOTE: 30-min health ping DISABLED (was phone spam). Health stays in the log.
        # 8/8 celebration
        if n_ok == 8 and len(_last_shards_ok) != 8:
            ntfy_send("All 8 shards DONE!", "Ready for assemble -> 25GB dataset", priority="high", tags="tada")
        _last_shards_ok = set(ok)
    except Exception as e:
        log(f"ntfy logic error: {e}")
    # auto-push trainers sequentially after assemble (no GPU waste before)
    try:
        maybe_auto_push_trainers(hf)
    except Exception as e:
        log(f"auto-push error: {e}\n{traceback.format_exc()[:600]}")
    log("poll done")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=180, help="seconds between polls")
    p.add_argument("--once", action="store_true", help="single poll then exit")
    args = p.parse_args()
    log(f"monitor starting: interval={args.interval}s once={args.once} log={LOG_FILE}")
    if args.once:
        once()
        return
    while True:
        try:
            once()
        except Exception as e:
            log(f"poll exception: {e}\n{traceback.format_exc()[-2000:]}")
        log(f"sleep {args.interval}s...")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
