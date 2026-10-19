"""Build versioned counterfactual labels; Stockfish is used ONLY offline.

Input JSONL: {"fen": ..., "game_id": ..., "source": ...}. Optional student
checkpoint selects hard candidate moves and supplies on-policy continuations.
Outputs are resumable JSONL journals with full label provenance and exclusions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chess
import chess.engine
import numpy as np
import torch

from scripts.data_hygiene import load_exclusions, position_hashes
from scripts.eval_gavn import tokenize_fen, _sha256_file
from scripts.kaggle_checkpoint import api, write_status
from scripts.refutation import ARCHITECTURE, load_network, score_moves


def hashes(fens):
    return [str(int(x)) for x in position_hashes(np.stack([tokenize_fen(f) for f in fens]))]


def mate_category(score):
    m = score.mate()
    return 0 if m is None else (0 if m > 0 else 16) + min(16, max(1, abs(m)))


def label(engine, board, nodes, multipv=1, root_moves=None):
    end = board.outcome(claim_draw=False)
    if end is not None:
        q = .5 if end.winner is None else float(end.winner == board.turn)
        return [{"q": q, "utility": 2*q-1, "mate": 17 if board.is_checkmate() else 0,
                 "move": None, "pv": [], "nodes": 0}]
    # Clear hash so labels do not depend on interrupted worker history.
    engine.configure({"Clear Hash": None})
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes),
                          multipv=min(multipv, board.legal_moves.count()), root_moves=root_moves)
    result = []
    for item in info:
        if item.get("lowerbound") or item.get("upperbound"):
            raise ValueError("oracle returned a bound, not an exact score")
        pv = item.get("pv", [])
        if not pv or (root_moves is not None and pv[0] not in root_moves):
            raise ValueError("oracle PV missing or violates forced-root move")
        copy = board.copy(stack=True)
        for move in pv:
            if move not in copy.legal_moves:
                raise ValueError("oracle returned illegal PV")
            copy.push(move)
        score = item["score"].pov(board.turn)
        cp = score.score(mate_score=100000)
        # Label conversion is explicitly versioned; not silently mixed with
        # ChessBench's historical 50ms label transformation.
        q = score.wdl(model="sf16", ply=board.ply()).expectation()
        m = score.mate()
        utility = float(np.tanh(cp/600)) if m is None else float(np.sign(cp)*(1.1-.002*min(abs(m), 49)))
        result.append({"q": q, "utility": utility, "mate": mate_category(score),
                       "move": pv[0].uci(), "pv": [m.uci() for m in pv],
                       "nodes": item.get("nodes"), "depth": item.get("depth")})
    return result


def build_group(record, engine, args, student, utils):
    board = chess.Board(record["fen"])
    if not board.is_valid() or board.is_game_over():
        raise ValueError("source must be a valid, nonterminal standard chess position")
    oracle = label(engine, board, args.nodes, multipv=2)
    candidates = [x["move"] for x in oracle]
    legal = sorted(board.legal_moves, key=lambda m: m.uci())
    if student is not None:
        scores = score_moves(student, board, legal, utils, mode=args.student_mode)
        candidates += [legal[i].uci() for i in np.argsort(-scores, kind="stable")[:2]]
    # Deterministic diversity independent of process RNG and resume boundaries.
    candidates += sorted((m.uci() for m in legal), key=lambda m: hashlib.sha256(
        (record["fen"]+m).encode()).digest())[:2]
    branches = []
    for uci in dict.fromkeys(candidates):
        move = chess.Move.from_uci(uci)
        target = label(engine, board, args.nodes, root_moves=[move])[0]
        child = board.copy(stack=True); child.push(move)
        reply_labels = label(engine, child, args.nodes, multipv=args.width)
        if args.on_policy and not child.is_game_over():
            device = next(student.parameters()).device
            with torch.inference_mode():
                _, policy, _, _ = student.state(torch.tensor(tokenize_fen(child.fen())[None], device=device))
            replies = sorted(child.legal_moves, key=lambda m: m.uci())
            order = torch.argsort(policy[0, [utils.MOVE_TO_ACTION[m.uci()] for m in replies]],
                                  descending=True, stable=True).tolist()
            first_replies = [replies[i].uci() for i in order[:args.width]]
        else:
            first_replies = [r["move"] for r in reply_labels]
        while len(first_replies) < args.width:
            first_replies.append(first_replies[-1])
        paths = []
        for first in first_replies:
            future = child.copy(stack=True)
            path = []
            for depth in range(args.horizon):
                truth = reply_labels[0] if depth == 0 else label(engine, future, args.nodes)[0]
                path.append({"fen": future.fen(), **truth})
                if future.is_game_over() or depth == args.horizon-1:
                    break
                if depth == 0:
                    chosen = first
                elif args.on_policy:
                    device = next(student.parameters()).device
                    with torch.inference_mode():
                        _, pi, _, _ = student.state(torch.tensor(tokenize_fen(future.fen())[None], device=device))
                    lm = sorted(future.legal_moves, key=lambda m: m.uci())
                    idx = int(pi[0, [utils.MOVE_TO_ACTION[m.uci()] for m in lm]].argmax())
                    chosen = lm[idx].uci()
                else:
                    chosen = truth["move"]
                path[-1]["played"] = chosen
                future.push_uci(chosen)
            paths.append(path)
        branches.append({"action": uci, "target": target, "paths": paths})
    return {"fen": board.fen(), "game_id": record.get("game_id"),
            "source": record["source"], "branches": branches}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--stockfish", required=True)
    p.add_argument("--engine-sha256", required=True)
    p.add_argument("--nodes", type=int, default=50000)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--width", type=int, choices=[1, 2], default=2)
    p.add_argument("--student-checkpoint")
    p.add_argument("--student-mode", choices=["base", "planner"], default="base")
    p.add_argument("--on-policy", action="store_true")
    p.add_argument("--prior-data", nargs="*", default=[], help="prior round journals to enforce cross-round split exclusion")
    p.add_argument("--sl-repo", default="/kaggle/working/searchless_chess")
    p.add_argument("--split-unit", choices=["game", "position"], default="game")
    p.add_argument("--exclusion-puzzles", required=True)
    p.add_argument("--mate-dir", default=str(ROOT / "data/positions"))
    p.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    p.add_argument("--hf-run", required=True)
    p.add_argument("--hf-upload-every", type=float, default=900)
    p.add_argument("--resume-from-hf", action="store_true")
    p.add_argument("--session-hours", type=float, default=10.5)
    args = p.parse_args()
    if args.horizon < 1 or args.nodes < 1 or args.hf_upload_every <= 0 or args.session_hours <= 0:
        raise ValueError("budgets must be positive")
    if not args.hf_run.startswith("refutation-data-"):
        raise ValueError("use a fresh refutation-data-* prefix")
    if _sha256_file(Path(args.stockfish)) != args.engine_sha256:
        raise ValueError("Stockfish binary hash mismatch")
    if args.on_policy and not args.student_checkpoint:
        raise ValueError("on-policy labeling requires a planner checkpoint")
    client = api(ROOT)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    if args.resume_from_hf:
        from huggingface_hub import snapshot_download
        import shutil
        snap = snapshot_download(args.hf_repo, repo_type="dataset", token=client.token,
                                 allow_patterns=[f"{args.hf_run}/data/*"])
        remote = Path(snap) / args.hf_run / "data"
        if not remote.exists():
            raise ValueError("requested data resume has no remote journal")
        for path in remote.iterdir():
            if (out/path.name).exists() and (out/path.name).read_bytes() != path.read_bytes():
                raise ValueError("local/remote dataset conflict; choose a fresh outdir")
            shutil.copy2(path, out/path.name)
    excluded, digest = load_exclusions(Path(args.mate_dir), Path(args.exclusion_puzzles))
    excluded = set(str(int(x)) for x in excluded)
    config = {k: getattr(args, k) for k in ("nodes", "horizon", "width", "split_unit", "on_policy", "student_mode")}
    config.update(schema=1, input_sha256=_sha256_file(Path(args.input)), exclusion_sha256=digest,
                  engine_sha256=args.engine_sha256, value_transform="python-chess-sf16-wdl",
                  python_chess=chess.__version__,
                  student_sha256=_sha256_file(Path(args.student_checkpoint)/"state.pt") if args.student_checkpoint else None,
                  student_config_sha256=_sha256_file(Path(args.student_checkpoint)/"config.json") if args.student_checkpoint else None,
                  builder_sha256=_sha256_file(Path(__file__)),
                  dependencies_sha256={name: _sha256_file(ROOT/"scripts"/name) for name in (
                      "refutation.py", "data_hygiene.py", "eval_gavn.py", "train_ccgavn.py", "train_gavn.py")},
                  prior_data_sha256=[_sha256_file(Path(p)) for p in args.prior_data])
    manifest = out/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("dataset resume identity mismatch")
    if not manifest.exists() and list(out.glob("*.jsonl")):
        raise ValueError("unidentified existing dataset")
    manifest.write_text(json.dumps(config, indent=2))
    db = sqlite3.connect(out/"positions.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS positions (hash TEXT PRIMARY KEY, split TEXT)")
    # Rebuild ownership from the authoritative journal after interrupted writes.
    db.execute("DELETE FROM positions")
    for prior in args.prior_data:
        with Path(prior).open() as stream:
            for line in stream:
                record = json.loads(line)
                if "branches" not in record:
                    continue
                for h in record["hashes"]:
                    previous = db.execute("SELECT split FROM positions WHERE hash=?", (h,)).fetchone()
                    if previous not in (None, (record["split"],)):
                        raise ValueError("prior datasets have conflicting train/dev ownership")
                    db.execute("INSERT OR IGNORE INTO positions VALUES (?, ?)", (h, record["split"]))
    journal = out/"groups.jsonl"
    consumed = 0
    if journal.exists():
        with journal.open() as f:
            for line in f:
                row = json.loads(line)  # partial/corrupt rows fail closed
                consumed = row["input_row"]+1
                if "branches" in row:
                    db.executemany("INSERT OR IGNORE INTO positions VALUES (?, ?)",
                                   ((h, row["split"]) for h in row["hashes"]))
    db.commit()
    student = utils = None
    if args.student_checkpoint:
        student, student_cfg, utils = load_network(args.student_checkpoint, args.sl_repo,
                                          "cuda" if torch.cuda.is_available() else "cpu")
        if args.on_policy and student_cfg.get("architecture") != ARCHITECTURE:
            raise ValueError("on-policy continuation requires trained planner heads")
        student.eval()
    started, last = time.monotonic(), 0.
    engine = None
    def upload():
        client.upload_folder(repo_id=args.hf_repo, repo_type="dataset", folder_path=str(out),
                             path_in_repo=f"{args.hf_run}/data", allow_patterns=["*.json", "*.jsonl"])
    try:
        write_status(client, args.hf_repo, args.hf_run, "IN PROGRESS: oracle labeling")
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        if not engine.id.get("name", "").startswith("Stockfish 16"):
            raise ValueError(f"expected pinned Stockfish 16, got {engine.id}")
        engine.configure({"Threads": 1, "Hash": 128})
        paused = False
        with Path(args.input).open() as source, journal.open("a") as sink:
            for i, line in enumerate(source):
                if i < consumed:
                    continue
                record = json.loads(line)
                key = record["game_id"] if args.split_unit == "game" else hashes([record["fen"]])[0]
                split = "dev" if int(hashlib.sha256(str(key).encode()).hexdigest()[:16], 16) % 20 == 0 else "train"
                root_excluded = hashes([record["fen"]])[0] in excluded
                group = {"rejected": "benchmark root"} if root_excluded else build_group(record, engine, args, student, utils)
                if not root_excluded:
                    fens = [group["fen"]] + [s["fen"] for b in group["branches"] for path in b["paths"] for s in path]
                    keys = set(hashes(fens))
                    conflict = any(db.execute("SELECT split FROM positions WHERE hash=?", (h,)).fetchone()
                                   not in (None, (split,)) for h in keys)
                    if keys & excluded or conflict:
                        group = {"rejected": "benchmark future or cross-split transposition"}
                    else:
                        group.update(hashes=sorted(keys), split=split)
                        db.executemany("INSERT OR IGNORE INTO positions VALUES (?, ?)", ((h, split) for h in keys))
                        db.commit()
                sink.write(json.dumps({"input_row": i, **group})+"\n")
                sink.flush(); os.fsync(sink.fileno())
                if time.monotonic()-last >= args.hf_upload_every:
                    upload(); last = time.monotonic()
                    print(json.dumps({"input_rows": i+1, "elapsed_s": last-started}), flush=True)
                if time.monotonic()-started >= args.session_hours*3600:
                    paused = True
                    break
        upload()
        write_status(client, args.hf_repo, args.hf_run, "PAUSED: resume required" if paused else "DONE: labels complete")
    except BaseException as exc:
        write_status(client, args.hf_repo, args.hf_run, f"FAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        db.close()
        if engine is not None:
            engine.quit()


if __name__ == "__main__":
    main()
