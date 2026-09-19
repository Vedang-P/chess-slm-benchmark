"""Push the full dashboard snapshot to the Cloudflare monitor.

The Worker cannot fetch hundreds of HF files per invocation (subrequest
limits), so this local/CI helper assembles the heavy sections - training
curve, eval summaries, corpus progress, PGN games, kernel statuses, quotas -
and POSTs them to /api/ingest. Intended to run from scripts/supervise_all.py
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
RUN = "ccgavn-5m-seed0"
ACCOUNTS = ["vedanggggg", "vedangpandeyyy", "softmaxsimp", "samaltmannnn", "shoumikmitra"]
KERNELS = [
    ("vedanggggg", "build-2b-slice"), ("softmaxsimp", "build-2b-slice"),
    ("samaltmannnn", "build-2b-slice"), ("shoumikmitra", "build-2b-slice"),
    ("vedangpandeyyy", "ccgavn-5m-seed0"), ("vedangpandeyyy", "eval-ccgavn-320k-auto"),
]


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


def build_curve(api, token: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    files = hf_files(api)
    steps = sorted({int(m.group(1)) for f in files
                    if (m := re.search(rf"{RUN}/checkpoint-(\d+)/metrics\.json$", f))})
    curve = []
    for s in steps:
        try:
            p = hf_hub_download(HF_REPO, f"{RUN}/checkpoint-{s}/metrics.json",
                                repo_type="dataset", token=token)
            m = json.loads(Path(p).read_text())
            curve.append({"step": m.get("step", s), "train": m.get("train_loss"),
                          "dev": m.get("dev_loss")})
        except Exception:
            continue
    return curve


def build_evals(api, token: str) -> dict:
    from huggingface_hub import hf_hub_download
    out = {}
    for f in hf_files(api):
        if re.match(r"^eval-results/.*/eval-summary\.json$", f):
            try:
                p = hf_hub_download(HF_REPO, f, repo_type="dataset", token=token)
                out[f] = json.loads(Path(p).read_text())
            except Exception:
                continue
    return out


def build_corpus(api, files: list[str]) -> dict:
    rows_map = json.loads((ROOT / "kernels" / "build-2b" / "shard_rows.json").read_text())
    done = [s for s in rows_map if f"chessbench-full-build/shard-{s}/teacher_logp.npy" in files]
    labeled = sum(int(rows_map[s]) for s in done)
    return {"target_rows": 920_000_000, "labeled_rows": labeled,
            "shards_planned": len(rows_map), "shards_done": len(done),
            "original_rows": 94_277_038}


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
                moves.append({"san": san, "fen": board.fen()})
            games.append({"event": g.headers.get("Event", ""),
                          "white": g.headers.get("White", "?"),
                          "black": g.headers.get("Black", "?"),
                          "result": g.headers.get("Result", "*"), "moves": moves})
    return games


def build_runs(api, token: str) -> dict:
    from huggingface_hub import hf_hub_download
    out = {}
    for f in hf_files(api):
        if re.match(r"^[^/]+/run-status\.txt$", f) or f.startswith(f"{RUN}/run-status"):
            try:
                p = hf_hub_download(HF_REPO, f, repo_type="dataset", token=token)
                out[f] = Path(p).read_text()[:400]
            except Exception:
                continue
    return out


def build_kernels_quota() -> tuple[list, dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from launch_trainers import env_for_account
    kernels, quota = [], {}
    for acct, slug in KERNELS:
        try:
            r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status",
                                f"{acct}/{slug}"], env=env_for_account(acct),
                               capture_output=True, text=True, timeout=60)
            out = (r.stdout or r.stderr).strip()
            st = out.split('"')[-2].split(".")[-1] if '"' in out else out[:40]
        except Exception as exc:
            st = f"err {str(exc)[:30]}"
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
        except Exception:
            continue
    return kernels, quota


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
    payload = {
        "curve": build_curve(api, token),
        "evals": build_evals(api, token),
        "corpus": build_corpus(api, files),
        "games": build_games(token),
        "runs": build_runs(api, token),
        "kernels": build_kernels_quota()[0],
        "quota": build_kernels_quota()[1],
    }
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
