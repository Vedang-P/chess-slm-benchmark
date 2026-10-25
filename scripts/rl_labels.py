"""Create exact-depth, all-legal-move verifier rewards for root-policy RL.

Each of five CPU Kaggle jobs owns a disjoint position-hash partition.  Labels
and a commit-marker state are uploaded to HF every 15 minutes; restart reads
the marker and truncates any labels uploaded after an interrupted commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import chess
import chess.engine
import numpy as np
from huggingface_hub import hf_hub_download

from scripts.data_hygiene import load_exclusions, position_hashes
from scripts.eval_gavn import _CHARS, tokenize_fen
from scripts.kaggle_checkpoint import UploadTimer, api as make_api, write_status

ROOT = Path(__file__).resolve().parent.parent
ACCOUNTS = ("vedanggggg", "vedangpandeyyy", "softmaxsimp", "samaltmannnn", "shoumikmitra")
CHARS = {value: key for key, value in _CHARS.items()}


def fen_from_tokens(tokens: np.ndarray) -> str:
    chars = [CHARS[int(x)] for x in tokens]
    if len(chars) != 77 or chars[0] not in ("w", "b"):
        raise ValueError("invalid source tokens")
    ranks = []
    for start in range(1, 65, 8):
        rank, empty = "", 0
        for char in chars[start:start + 8]:
            if char == ".":
                empty += 1
            else:
                if empty:
                    rank += str(empty); empty = 0
                rank += char
        if empty:
            rank += str(empty)
        ranks.append(rank)
    castling = "".join(c for c in chars[65:69] if c != ".") or "-"
    ep = "".join(chars[69:71]) if chars[69] != "." else "-"
    halfmoves = "".join(chars[71:74]).rstrip(".") or "0"
    fullmoves = "".join(chars[74:77]).rstrip(".") or "1"
    if not halfmoves.isdigit() or not fullmoves.isdigit() or int(fullmoves) < 1:
        raise ValueError("source tokens contain invalid FEN move counters")
    fen = (f"{'/'.join(ranks)} {chars[0]} {castling} {ep} "
           f"{int(halfmoves)} {int(fullmoves)}")
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"source tokens decoded to invalid chess board: {fen}")
    if not np.array_equal(tokenize_fen(fen), np.asarray(tokens)):
        raise ValueError("source token round trip failed")
    return fen


def reward_for_cp(cp: int, utils) -> float:
    return float(utils.centipawns_to_win_probability(cp))


def label_root(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int,
               move_to_action: dict[str, int], utils) -> dict:
    """Rescore each legal root move at the same exact depth with no bounds."""
    actions, uci_moves, cp_scores, rewards = [], [], [], []
    for move in board.legal_moves:
        info = engine.analyse(board, chess.engine.Limit(depth=depth),
                              root_moves=[move])
        if int(info.get("depth", -1)) != depth or info.get("lowerbound") or info.get("upperbound"):
            raise ValueError(f"inexact Stockfish root score at depth {depth}: {info}")
        score = info["score"].pov(board.turn).score(mate_score=100_000)
        if score is None:
            raise ValueError("Stockfish returned a missing root score")
        uci = move.uci()
        actions.append(int(move_to_action[uci]))
        uci_moves.append(uci)
        cp_scores.append(int(score))
        rewards.append(reward_for_cp(int(score), utils))
    if len(actions) < 2:
        raise ValueError("root has fewer than two legal moves")
    return {"actions": actions, "moves": uci_moves, "scores_cp": cp_scores,
            "rewards": rewards}


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def restore(api, repo: str, prefix: str, outdir: Path, identity: dict) -> tuple[int, int]:
    files = set(api.list_repo_files(repo, repo_type="dataset"))
    marker = f"{prefix}/state.json"
    if marker not in files:
        if f"{prefix}/labels.jsonl" in files:
            raise ValueError("orphan labels without a resume marker")
        return 0, 0
    source = hf_hub_download(repo, marker, repo_type="dataset", token=api.token)
    state = json.loads(Path(source).read_text(encoding="utf-8"))
    if state["identity"] != identity:
        raise ValueError("label-job identity changed on resume")
    labels_source = hf_hub_download(repo, f"{prefix}/labels.jsonl", repo_type="dataset",
                                    token=api.token)
    lines = Path(labels_source).read_bytes().splitlines(keepends=True)
    count = int(state["count"])
    if len(lines) < count:
        raise ValueError("remote label file is shorter than its commit marker")
    labels = outdir / "labels.jsonl"
    labels.write_bytes(b"".join(lines[:count]))
    if file_digest(labels) != state["sha256"]:
        raise ValueError("remote label digest disagrees with its commit marker")
    return int(state["cursor"]), count


def persist(api, repo: str, prefix: str, outdir: Path, identity: dict,
            cursor: int, count: int) -> None:
    labels = outdir / "labels.jsonl"
    marker = {"identity": identity, "cursor": cursor, "count": count,
              "sha256": file_digest(labels)}
    api.upload_file(path_or_fileobj=str(labels), path_in_repo=f"{prefix}/labels.jsonl",
                    repo_id=repo, repo_type="dataset")
    # The state is the commit marker and must be uploaded last.
    api.upload_file(path_or_fileobj=json.dumps(marker, sort_keys=True).encode(),
                    path_in_repo=f"{prefix}/state.json", repo_id=repo,
                    repo_type="dataset")
    write_status(api, repo, prefix, f"IN PROGRESS {count}/{identity['target']} labels")
    print(f"[labels] uploaded {count}/{identity['target']} roots at cursor {cursor}", flush=True)


def run(args) -> None:
    if args.account not in ACCOUNTS:
        raise ValueError(f"unknown account {args.account}")
    if args.depth < 1 or args.target < 1:
        raise ValueError("depth and target must be positive")
    from searchless_chess.src import utils

    api = make_api(ROOT)
    prefix = f"{args.prefix}/{args.account}"
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    excluded, exclusion_digest = load_exclusions(args.mate_dir, args.puzzles)
    identity = {"account": args.account, "shard_tag": args.shard_tag,
                "depth": args.depth, "target": args.target, "seed": args.seed,
                "source_commit": args.source_commit,
                "exclusion_digest": exclusion_digest,
                "source": "chessbench-full-build/train_set.npz",
                "fen_counters": "preserved-v1",
                "reward": "searchless-centipawns-to-win-probability-v1",
                "searchless_commit": args.searchless_commit, "all_legal": True}
    cursor, count = restore(api, args.hf_repo, prefix, outdir, identity)
    labels = outdir / "labels.jsonl"
    if not labels.exists():
        labels.write_bytes(b"")
    existing = {int(json.loads(line)["position_hash"]) for line in labels.read_text().splitlines()}
    if len(existing) != count:
        raise ValueError("duplicate or missing committed position hashes")
    if count >= args.target:
        write_status(api, args.hf_repo, prefix, f"DONE {count} roots")
        return
    remote = f"chessbench-full-build/shard-{args.shard_tag}/train_set.npz"
    source = hf_hub_download(args.hf_repo, remote, repo_type="dataset",
                             token=api.token, local_dir=str(outdir / "source"))
    with np.load(source) as bundle:
        tokens = bundle["tokens"]
    rng = np.random.default_rng(args.seed)
    # Fixed random stream and cursor make account restarts deterministic.
    row_indices = rng.integers(0, len(tokens), size=args.target * 100,
                               dtype=np.int64)
    account_index = ACCOUNTS.index(args.account)
    timer = UploadTimer(args.upload_every)
    timer.mark()
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
        engine.configure({"Threads": 2, "Hash": 128})
        engine_id = dict(engine.id)
        with labels.open("a", encoding="utf-8") as out:
            while cursor < len(row_indices) and count < args.target:
                row = int(row_indices[cursor]); cursor += 1
                token_row = np.asarray(tokens[row])
                position_hash = int(position_hashes(token_row[None])[0])
                if position_hash % len(ACCOUNTS) != account_index or \
                        position_hash in existing or position_hash in excluded:
                    continue
                fen = fen_from_tokens(token_row)
                board = chess.Board(fen)
                if len(list(board.legal_moves)) < 2:
                    continue
                result = label_root(board, engine, args.depth, utils.MOVE_TO_ACTION, utils)
                record = {"fen": fen, "position_hash": position_hash,
                          "source_tag": args.shard_tag, "source_row": row,
                          "engine": engine_id, "depth": args.depth, **result}
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
                out.flush()
                existing.add(position_hash); count += 1
                if count % 10 == 0:
                    print(f"[labels] {args.account} {count}/{args.target}", flush=True)
                if timer.due():
                    persist(api, args.hf_repo, prefix, outdir, identity, cursor, count)
                    timer.mark()
    persist(api, args.hf_repo, prefix, outdir, identity, cursor, count)
    if count != args.target:
        raise RuntimeError(f"candidate row stream exhausted at {count}/{args.target}")
    write_status(api, args.hf_repo, prefix, f"DONE {count} roots")
    print(f"[labels] DONE {args.account}: {count} roots", flush=True)
    shutil.rmtree(outdir / "source", ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--account", required=True)
    p.add_argument("--shard-tag", required=True)
    p.add_argument("--stockfish", default="/usr/games/stockfish")
    p.add_argument("--depth", type=int, default=14)
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--searchless-commit", required=True)
    p.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    p.add_argument("--prefix", default="rl-root-v2-labels")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--mate-dir", type=Path, default=ROOT / "data/positions")
    p.add_argument("--puzzles", type=Path, required=True)
    p.add_argument("--upload-every", type=float, default=900)
    args = p.parse_args()
    api = make_api(ROOT)
    try:
        run(args)
    except BaseException as exc:
        write_status(api, args.hf_repo, f"{args.prefix}/{args.account}",
                     f"FAILED {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
