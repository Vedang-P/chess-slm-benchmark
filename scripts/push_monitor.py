"""Push the full dashboard snapshot to the Cloudflare monitor.

The Worker cannot fetch hundreds of HF files per invocation (subrequest
limits), so this local/CI helper assembles the heavy sections - training
curve, eval summaries, corpus progress, PGN games, kernel statuses, quotas -
and POSTs them to /api/ingest. Intended to run from the scheduled pipeline
every 15 minutes; harmless to run by hand.

Requires in .env: MONITOR_URL, MONITOR_INGEST_KEY, HF_WRITE_TOKEN.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "vedangfake/chess-slm-benchmark"
# Historical prefix first, corrected-v2 continuation second; steps continue
# across both, so the curve merges by step and never rewrites legacy points.
RUNS = ["ccgavn-5m-seed0", "ccgavn-5m-seed0-v2"]
LIVE_RUNS = ["ccgavn-5m-seed0", "ccgavn-5m-seed0-v2", "chessbench-full-build",
             "ccgavn-1b", "puzzle-curriculum", "refutation-r0",
             "refutation-data-r0", "ccgavn-refutation-r0-seed0",
             "ccgavn-refutation-r0-seed0-v2"]
REF_KERNEL = "ccgavn-refutation-r0"
REF_DATA = "refutation-data-r0"
REF_TRAIN = "ccgavn-refutation-r0-seed0-v2"
REF_EVAL = "eval-results/ccgavn-refutation-r0"
REF_STEPS = 12500
ACCOUNTS = ["vedanggggg", "vedangpandeyyy", "softmaxsimp", "samaltmannnn", "shoumikmitra"]
KERNELS = [
    ("vedanggggg", "build-2b-slice"), ("softmaxsimp", "build-2b-slice"),
    ("samaltmannnn", "build-2b-slice"), ("shoumikmitra", "build-2b-slice"),
] + [(acct, "ccgavn-1b") for acct in ACCOUNTS] + [(acct, "ccgavn-2b") for acct in ACCOUNTS] + [(acct, REF_KERNEL) for acct in ACCOUNTS]


def load_env() -> dict:
    """Real environment first (CI), then .env as a local convenience."""
    import os
    env = dict(os.environ)
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def hf_files(api) -> list[str]:
    return api.list_repo_files(HF_REPO, repo_type="dataset")


def stage1_schedule():
    """Reproduce the deterministic step->shard schedule of the 0..320k phase
    (seed 0, the 10-shard mix) so the chart can shade puzzle-shard blocks for
    checkpoints produced before the trainer logged its shard tag."""
    import numpy as np
    rows_cfg = json.loads((ROOT / "configs" / "chessbench_rows.json").read_text())
    tags = list(rows_cfg)
    rows = np.array([rows_cfg[t] for t in tags], dtype=np.float64)
    steps = 320000
    rng = np.random.default_rng(0)
    frac = rows / rows.sum()
    counts = np.maximum(1, np.round(frac * steps).astype(int))
    counts[int(np.argmax(counts))] += steps - counts.sum()
    order = rng.permutation(len(tags))
    return np.concatenate([np.repeat(tags[i], counts[i]) for i in order])


def checkpoint_times(token: str, min_step: int) -> dict:
    """(run, step) -> ISO upload time for metrics.json files, from commit
    history. Gives throughput charts a real time axis even before the trainer
    logs it."""
    import requests as rq
    out = {}
    for page in range(20):
        r = rq.get(f"https://huggingface.co/api/datasets/{HF_REPO}/commits/main",
                   params={"p": page}, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        if r.status_code != 200:
            break
        commits = r.json()
        if not commits:
            break
        for x in commits:
            t = x.get("title", "")
            if not t.endswith("/metrics.json with huggingface_hub"):
                continue
            for run in RUNS:
                prefix = f"Upload {run}/checkpoint-"
                if t.startswith(prefix):
                    step = t.split("checkpoint-")[1].split("/")[0]
                    d = x["date"]
                    key = (run, step)
                    if key not in out or d < out[key]:
                        out[key] = d
    return out


def fetch_previous(url: str) -> dict:
    if not url:
        return {}
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/api/snapshot",
                                     headers={"user-agent": "chess-slm-monitor/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except Exception as exc:
        print(f"[monitor] previous snapshot unavailable: {exc}", flush=True)
        return {}


def build_curve(api, token: str, files: list[str], prior: list[dict] | None = None) -> list[dict]:
    from huggingface_hub import snapshot_download
    entries = sorted({(run, int(m.group(1))) for f in files for run in RUNS
                      if (m := re.search(rf"{re.escape(run)}/checkpoint-(\d+)/metrics\.json$", f))},
                     key=lambda x: x[1])
    sched = stage1_schedule()
    times = checkpoint_times(token, entries[0][1] if entries else 0)
    # Reuse the curve already in KV and fetch only newly uploaded metrics.
    # HEADing all 1,559 historical files every ten minutes trips HF's rate
    # limit (429) and stalls the whole monitor tick.
    prior = [dict(p) for p in (prior or [])]
    known = {(str(p.get("run")), int(p.get("step"))) for p in prior
             if p.get("run") is not None and p.get("step") is not None}
    new = [(run, s) for run, s in entries if (run, s) not in known]
    cache = None
    if new:
        cache = Path(snapshot_download(
            HF_REPO, repo_type="dataset", token=token,
            allow_patterns=[f"{run}/checkpoint-{s}/metrics.json" for run, s in new]))
    curve = prior
    for run, s in new:
        try:
            p = cache / f"{run}/checkpoint-{s}/metrics.json"
            m = json.loads(p.read_text())
            point = {"step": m.get("step", s), "run": run,
                     "train": m.get("train_loss"), "dev": m.get("dev_loss")}
            for k in ("dev_dist", "dev_ce", "lr", "grad_norm", "samples_per_s", "shard"):
                if m.get(k) is not None:
                    point[k] = m[k]
            if "shard" not in point and run != "ccgavn-5m-seed0-v2":
                idx = int(point["step"]) - 1
                if 0 <= idx < len(sched):
                    point["shard"] = str(sched[idx])
            t = times.get((run, str(point["step"])))
            if t:
                point["t"] = t
            curve.append(point)
        except Exception:
            continue
    curve.sort(key=lambda p: p.get("step", 0))
    return curve


def build_evals(api, token: str, files: list[str], prior: dict | None = None) -> dict:
    from huggingface_hub import snapshot_download
    prior = dict(prior or {})
    paths = [f for f in files if re.match(r"^eval-results/.*/eval-summary\.json$", f) or (
        f.startswith(f"{REF_EVAL}/") and f.endswith("/summary.json"))]
    new = [f for f in paths if f not in prior]
    if not new:
        return prior
    cache = Path(snapshot_download(
        HF_REPO, repo_type="dataset", token=token, allow_patterns=new))
    out = prior
    for f in new:
        try:
            out[f] = json.loads((cache / f).read_text())
        except Exception:
            continue
    return out


def build_corpus(api, files: list[str]) -> dict:
    rows_map = json.loads((ROOT / "kernels" / "build-2b" / "shard_rows.json").read_text())
    done = [s for s in rows_map if f"chessbench-full-build/shard-{s}/teacher_logp.npy" in files]
    labeled = sum(int(rows_map[s]) for s in done)
    return {"target_rows": sum(int(v) for v in rows_map.values()), "labeled_rows": labeled,
            "shards_planned": len(rows_map), "shards_done": len(done),
            "original_rows": 94_277_038,
            "done_tags": sorted(done),
            "all_tags": sorted(rows_map),
            "rows_by_tag": {t: int(rows_map[t]) for t in sorted(rows_map)}}


def build_games(token: str, limit: int = 6) -> list[dict]:
    import chess.pgn
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(HF_REPO, "elo-results/ccgavn-160k-stockfish-ladder/games-1400.pgn",
                            repo_type="dataset", token=token)
    except Exception:
        return []
    games = []
    with open(p) as fh:
        while len(games) < limit:
            g = chess.pgn.read_game(fh)
            if g is None:
                break
            board = g.board()
            moves = []
            for mv in g.mainline_moves():
                san = board.san(mv)
                board.push(mv)
                moves.append({"san": san, "fen": board.fen(),
                              "from": chess.square_name(mv.from_square),
                              "to": chess.square_name(mv.to_square)})
            games.append({"event": g.headers.get("Event", ""),
                          "white": g.headers.get("White", "?"),
                          "black": g.headers.get("Black", "?"),
                          "result": g.headers.get("Result", "*"), "moves": moves})
    return games


def sanitize_status(txt: str) -> str:
    t = " ".join(txt.split())
    if any(w in t.lower() for w in ("traceback", "error", "exception", "killed")):
        return "FAILED — log on HF"
    return t[:120]


ALIASES = {a: f"acct-{i+1}" for i, a in enumerate(ACCOUNTS)}


def mask_accounts(payload: dict) -> dict:
    s = json.dumps(payload)
    for real, alias in ALIASES.items():
        s = s.replace(real, alias)
    return json.loads(s)


def build_runs(api, token: str) -> dict:
    from huggingface_hub import hf_hub_download
    out = {}
    for f in hf_files(api):
        if re.match(r"^[^/]+/run-status\.txt$", f) and f.split("/")[0] in LIVE_RUNS:
            try:
                p = hf_hub_download(HF_REPO, f, repo_type="dataset", token=token)
                out[f] = sanitize_status(Path(p).read_text()[:400])
            except Exception:
                continue
    return out


def _ref_json(token: str, path: str):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(HF_REPO, path, repo_type="dataset", token=token)
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _ref_text(token: str, path: str) -> str:
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(HF_REPO, path, repo_type="dataset", token=token)
        return Path(p).read_text()
    except Exception:
        return ""


def _eval_counts(summary):
    if not summary:
        return None
    def one(m, key):
        if not m:
            return None
        if isinstance(m, list):
            m = m[0] if m else None
        if not m:
            return None
        n = m.get(key, m.get("correct"))
        total = m.get("total", m.get("expected"))
        if n is None or not total:
            return None
        return {"n": n, "total": total, "pct": round(100*n/total, 2)}
    return {"mate": one(summary.get("mate"), "correct"),
            "puzzles": one(summary.get("puzzles"), "solved"),
            "complete": bool(summary.get("complete"))}


def build_refutation(token: str, files: list[str], kernels: list) -> dict:
    source_ready = f"{REF_DATA}/input/parents-r0.jsonl" in files
    data_progress = _ref_json(token, f"{REF_DATA}/data/progress.json") or {}
    source_progress = _ref_json(token, f"{REF_DATA}/source-progress.json") or {}
    train_progress = _ref_json(token, f"{REF_TRAIN}/progress.json") or {}
    top_status = _ref_text(token, "refutation-r0/run-status.txt")
    data_status = _ref_text(token, f"{REF_DATA}/run-status.txt")
    train_status = _ref_text(token, f"{REF_TRAIN}/run-status.txt")
    accepted = int(data_progress.get("accepted") or 0)
    rejected = int(data_progress.get("rejected") or 0)
    total = int(data_progress.get("total_rows") or (1944 if source_ready else 0))
    done = int(data_progress.get("input_rows") or accepted+rejected)
    label_rate = float(data_progress.get("rows_per_second") or 0)
    train_step = int(train_progress.get("step") or 0)
    if not train_step:
        steps = [int(m.group(1)) for f in files
                 if (m := re.search(rf"{re.escape(REF_TRAIN)}/checkpoint-(\d+)/config\.json$", f))]
        train_step = max(steps, default=0)
    eval_summary = None
    for path in files:
        if path.startswith(f"{REF_EVAL}/") and path.endswith("/summary.json"):
            eval_summary = _ref_json(token, path) or eval_summary
    eval_counts = _eval_counts(eval_summary)
    if eval_counts and eval_counts.get("complete"):
        phase = "complete"
    elif train_step >= REF_STEPS or train_status.upper().startswith("DONE"):
        phase = "evaluation"
    elif done and total and done >= total:
        phase = "training"
    elif done:
        phase = "labeling"
    elif source_ready:
        phase = "labeling"
    else:
        phase = "source"
    if "FAILED" in top_status.upper() or "FAILED" in train_status.upper():
        phase = "failed"
    live = next((k for k in kernels if k.get("kernel") == REF_KERNEL
                 and any(x in str(k.get("status", "")) for x in ("RUNNING", "QUEUED", "PENDING"))), None)
    if live and phase == "source" and "QUEUED" in live.get("status", ""):
        phase = "queued"
    eta = None
    if phase == "labeling" and label_rate > 0 and total > done:
        eta = (total-done)/label_rate
    session_start = train_progress.get("session_start_step")
    session_s = float(train_progress.get("session_seconds") or 0)
    if phase == "training" and session_start is not None and session_s > 0:
        rate = (train_step-int(session_start))/session_s
        if rate > 0:
            eta = (REF_STEPS-train_step)/rate
    source = {k: source_progress[k] for k in
              ("stage", "pct", "bytes_done", "bytes_total", "games", "rows",
               "rows_scanned", "rows_total", "found", "updated_at")
              if k in source_progress}
    return {"phase": phase, "source_ready": source_ready, "source": source,
            "kernel": live, "kernel_status": (live or {}).get("status", ""),
            "top_status": top_status[:240], "data_status": data_status[:240],
            "train_status": train_status[:240],
            "accepted": accepted, "rejected": rejected,
            "label_done": done, "label_total": total,
            "label_pct": round(100*done/total, 2) if total else 0,
            "label_rate": round(label_rate, 3), "eta_seconds": round(eta) if eta else None,
            "train_step": train_step, "train_total": REF_STEPS,
            "train_pct": round(100*train_step/REF_STEPS, 2),
            "train_loss": train_progress.get("loss"),
            "train_dev": (train_progress.get("dev") or {}).get("oracle_group_top1") if isinstance(train_progress.get("dev"), dict) else None,
            "eval": eval_counts, "updated_at": data_progress.get("updated_at") or train_progress.get("updated_at")}


def build_kernels_quota() -> tuple[list, dict, str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_accounts import env_for_account
    kernels, quota = [], {}
    reset = ""
    for acct, slug in KERNELS:
        try:
            r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status",
                                f"{acct}/{slug}"], env=env_for_account(acct),
                               capture_output=True, text=True, timeout=60)
            out = (r.stdout or r.stderr).strip()
            if "Cannot access kernel" in out:
                st = "not created"
            elif '"' in out:
                st = out.split('"')[-2].split(".")[-1]
            else:
                st = out[:40]
        except Exception as exc:
            st = f"err {str(exc)[:30]}"
        # The labeling fleet is finished; keep its rows only while actually
        # running (the corpus card reports the labeled-shard state instead).
        if slug == "build-2b-slice" and st not in ("RUNNING", "QUEUED", "PENDING"):
            continue
        if slug == REF_KERNEL and st == "not created":
            continue
        kernels.append({"account": acct, "kernel": slug, "status": st})
    for acct in ACCOUNTS:
        try:
            r = subprocess.run([sys.executable, "-m", "kaggle", "quota"],
                               env=env_for_account(acct), capture_output=True,
                               text=True, timeout=90)
            for line in (r.stdout or "").splitlines():
                parts = line.split()
                if len(parts) >= 4 and parts[0] == "GPU":
                    quota[acct] = parts[2].rstrip("h")
                    if len(parts) >= 5 and not reset:
                        # kaggle prints e.g. 2026-09-26T00:00:00 (UTC)
                        stamp = parts[4]
                        if not stamp.endswith(("Z", "z")) and "+" not in stamp:
                            stamp += "Z"
                        reset = stamp
        except Exception:
            continue
    return kernels, quota, reset


def main() -> None:
    from huggingface_hub import HfApi
    env = load_env()
    url = env.get("MONITOR_URL", "").rstrip("/")
    key = env.get("MONITOR_INGEST_KEY", "")
    token = env.get("HF_WRITE_TOKEN", "")
    if not (url and key and token):
        print("[monitor] MONITOR_URL / MONITOR_INGEST_KEY / HF_WRITE_TOKEN missing; skip")
        return
    api = HfApi(token=token)
    files = hf_files(api)
    print(f"[monitor] {len(files)} files", flush=True)
    prior = fetch_previous(url)
    kernels, quota, quota_reset = build_kernels_quota()
    print(f"[monitor] kernels={len(kernels)} accounts={len(quota)}", flush=True)
    curve = build_curve(api, token, files, prior.get("curve"))
    print(f"[monitor] curve={len(curve)}", flush=True)
    evals = build_evals(api, token, files, prior.get("evals"))
    print(f"[monitor] evals={len(evals)}", flush=True)
    payload = {
        "curve": curve,
        "evals": evals,
        "corpus": build_corpus(api, files),
        "games": build_games(token),
        "runs": build_runs(api, token),
        "kernels": kernels,
        "refutation": build_refutation(token, files, kernels),
        "quota": quota,
        "quota_reset": quota_reset,
    }
    payload = mask_accounts(payload)
    req = urllib.request.Request(f"{url}/api/ingest", method="POST",
                                 data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json",
                                          "x-ingest-key": key,
                                          "user-agent": "chess-slm-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        print(f"[monitor] ingested: {r.status} curve={len(payload['curve'])} "
              f"evals={len(payload['evals'])} games={len(payload['games'])} "
              f"corpus={payload['corpus']['labeled_rows']/1e6:.0f}M rows")


if __name__ == "__main__":
    main()
