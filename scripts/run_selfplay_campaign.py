"""Self-play campaign supervisor — the ONLY process that touches the
network for telemetry. Each game worker writes local files; this process:

  - launches/relaunches/resumes game workers (one API key each)
  - aggregates their live-*.json snapshots into ONE GitHub file
    (monitor/games/live.json), throttled, so the website has a live feed
    without spamming the repo per move
  - uploads completed games + traces to HF, and (with --hf-upload
    interval) in-progress game checkpoints so a killed/relaunched kernel
    can resume from HF (Kaggle wipes /kaggle/working on restart)

    python3 scripts/run_selfplay_campaign.py --games 100 --out results/selfplay
    python3 scripts/run_selfplay_campaign.py --games 100 --out results/selfplay \
        --hf-sync --hf-upload 120 --time-limit-h 11

Local files (source of truth): {out}/*.json, {out}/live-*.json,
{out}/*.traces.jsonl. GitHub/HF are derived mirrors only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.hf_push import HF_REPO, _api  # noqa: E402
from src.live_push import resolve_token, upload_file  # noqa: E402

GH_PUSH_INTERVAL_S = 10.0
GH_LIVE_PATH = "monitor/games/live.json"
HF_RUN_DIR = "runs/selfplay"


def _env_keys() -> list[str]:
    """API keys: first from os.environ (Kaggle secrets inject env vars),
    then from the gitignored .env file (OPENCODE_API_KEY, _2.._6)."""
    keys = []
    for k in sorted(os.environ):
        if k.startswith("OPENCODE_API_KEY"):
            v = os.environ[k]
            if v and v not in keys:
                keys.append(v)
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENCODE_API_KEY"):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v and v not in keys:
                    keys.append(v)
    return keys


def _game_status(out_dir: Path, gid: str) -> str:
    cp = out_dir / f"{gid}.json"
    if not cp.exists():
        return "not_started"
    try:
        return json.loads(cp.read_text()).get("status", "running")
    except Exception:
        return "not_started"


def _live_snapshot(out_dir: Path, gid: str) -> dict | None:
    p = out_dir / f"live-{gid}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _push_live(out_dir: Path, games: list[str], live_token: str | None) -> None:
    rows = []
    for gid in games:
        snap = _live_snapshot(out_dir, gid)
        cp = out_dir / f"{gid}.json"
        status, result = "not_started", "*"
        if cp.exists():
            try:
                g = json.loads(cp.read_text())
                status, result = g.get("status", "running"), g.get("result", "*")
            except Exception:
                pass
        if snap:
            status = snap.get("status", status)
            rows.append({
                "id": gid, "status": status, "result": result,
                "plies": snap.get("plies", 0), "by": snap.get("by"),
                "fen": snap.get("fen"), "turn": snap.get("turn"),
                "last_san": snap.get("last_san"),
                "history": snap.get("history", []),
                "thinking": snap.get("thinking", {}),
                "updated_at": snap.get("updated_at"),
            })
        else:
            rows.append({"id": gid, "status": status, "result": result,
                         "plies": 0})
    live = {
        "campaign": "selfplay-pilot",
        "total": len(games),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "games": rows,
    }
    if live_token:
        upload_file(live_token, GH_LIVE_PATH,
                    json.dumps(live).encode())


def _upload_game_to_hf(out_dir: Path, gid: str) -> None:
    cp = out_dir / f"{gid}.json"
    traces = out_dir / f"{gid}.traces.jsonl"
    if not cp.exists():
        return
    api = _api()
    try:
        api.upload_file(
            path_or_fileobj=cp.read_bytes(),
            path_in_repo=f"{HF_RUN_DIR}/{gid}.json",
            repo_id=HF_REPO, repo_type="dataset",
            commit_message=f"selfplay {gid} complete")
        if traces.exists():
            api.upload_file(
                path_or_fileobj=traces.read_bytes(),
                path_in_repo=f"{HF_RUN_DIR}/{gid}.traces.jsonl",
                repo_id=HF_REPO, repo_type="dataset",
                commit_message=f"selfplay {gid} traces")
        print(f"hf: uploaded {gid}", flush=True)
    except Exception as e:
        print(f"hf: upload {gid} failed: {type(e).__name__}: {e}", flush=True)


def _sync_from_hf(out_dir: Path) -> None:
    """Pull every selfplay game + trace file from HF into the local out
    dir (Kaggle wipes /kaggle/working on restart, so a fresh kernel
    resumes from here). Idempotent: local files are never overwritten."""
    from huggingface_hub import hf_hub_download

    from src.hf_push import resolve_hf_token

    out_dir.mkdir(parents=True, exist_ok=True)
    token = resolve_hf_token()
    try:
        api = _api()
        files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    except Exception as e:
        print(f"hf: sync failed: {type(e).__name__}: {e}", flush=True)
        return
    got = 0
    for f in files:
        if not f.startswith(HF_RUN_DIR + "/"):
            continue
        if not (f.endswith(".json") or f.endswith(".traces.jsonl")):
            continue
        name = f.rsplit("/", 1)[-1]
        dest = out_dir / name
        if dest.exists():
            continue
        try:
            hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset",
                filename=f, local_dir=str(out_dir), token=token)
            got += 1
        except Exception as e:
            print(f"hf: sync {f} failed: {type(e).__name__}: {e}", flush=True)
    if got:
        print(f"hf: synced {got} game/trace files from HF", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--out", default="results/selfplay")
    ap.add_argument("--live-push", action="store_true")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--hf-sync", action="store_true",
                    help="pull existing games/traces from HF before launching")
    ap.add_argument("--hf-upload", type=float, default=0.0,
                    help="upload in-progress game checkpoints to HF every N "
                         "seconds (0 disables; required for Kaggle resume)")
    ap.add_argument("--time-limit-h", type=float, default=0.0,
                    help="stop launching new games after N hours (0 = no "
                         "limit; ~11 for a 12h Kaggle kernel so it can "
                         "self-relaunch before being killed)")
    args = ap.parse_args()

    keys = _env_keys()
    if not keys:
        print("no OPENCODE_API_KEY* found in .env", file=sys.stderr)
        sys.exit(1)
    print(f"{len(keys)} API keys -> up to {len(keys)} concurrent games",
          flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    live_token = resolve_token() if args.live_push else None

    if args.hf_sync:
        _sync_from_hf(out_dir)

    game_ids = [f"g{i:04d}" for i in range(1, args.games + 1)]
    procs: dict[str, subprocess.Popen] = {}
    hf_uploaded = set()
    last_gh_push = 0.0
    last_hf_upload = 0.0
    t0 = time.time()

    try:
        while True:
            if args.time_limit_h and \
                    time.time() - t0 > args.time_limit_h * 3600:
                print(f"time limit ({args.time_limit_h}h) reached — "
                      f"stopping. {len([p for p in procs.values() if p.poll() is None])} "
                      "games still running will resume next run.", flush=True)
                (out_dir / "TIME_LIMIT_HIT").write_text(
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                break

            for gid in list(procs):
                if procs[gid].poll() is not None:
                    procs.pop(gid)
                    # upload freshly finished games to HF (once each)
                    if gid not in hf_uploaded and \
                            _game_status(out_dir, gid) not in ("not_started", "running"):
                        _upload_game_to_hf(out_dir, gid)
                        hf_uploaded.add(gid)

            for gid in game_ids:
                if len(procs) >= len(keys):
                    break
                status = _game_status(out_dir, gid)
                if status in ("not_started", "running"):
                    opening = (int(gid[1:]) - 1) % 20
                    from_move_one = (int(gid[1:]) - 1) % 2 == 0
                    env = dict(os.environ)
                    env["OPENCODE_API_KEY"] = keys[len(procs)]
                    cmd = [
                        sys.executable, str(ROOT / "scripts" / "play_selfplay.py"),
                        "--game-id", gid, "--out", str(out_dir),
                        "--opening", str(opening),
                    ]
                    if from_move_one:
                        cmd.append("--from-move-one")
                    log = open(out_dir / f"{gid}.log", "a")
                    procs[gid] = subprocess.Popen(
                        cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                    print(f"launched {gid} (key {len(procs)})", flush=True)

            now = time.time()
            if args.live_push and now - last_gh_push >= GH_PUSH_INTERVAL_S:
                last_gh_push = now
                _push_live(out_dir, game_ids, live_token)

            if args.hf_upload and now - last_hf_upload >= args.hf_upload:
                last_hf_upload = now
                for gid in game_ids:
                    if gid in hf_uploaded:
                        continue
                    status = _game_status(out_dir, gid)
                    if status == "running":
                        _upload_game_to_hf(out_dir, gid)

            if not procs:
                done = all(
                    _game_status(out_dir, gid) not in ("not_started", "running")
                    for gid in game_ids)
                if done:
                    print("campaign complete", flush=True)
                    break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        for p in procs.values():
            p.terminate()
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
