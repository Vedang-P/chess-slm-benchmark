"""Build ChessBench-format action-value rows from frozen Lichess puzzle roots.

One row is produced for every legal action at the puzzle-player root. Each
action gets the original Stockfish 16, 50 ms root-move evaluation. The result
can be passed unchanged to teacher_label.py and train_ccgavn.py.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import signal
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import zstandard as zstd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.freeze_puzzle_corpus import file_sha256

ROOTS_PER_SHARD = 5_000
STOCKFISH_SECONDS = 0.05
STOP = False


def _signal_stop(_number, _frame) -> None:
    global STOP
    STOP = True


def roots_in_shard(source: Path, excluded: set[str], shard: int,
                   roots_per_shard: int = ROOTS_PER_SHARD):
    """Yield rows in the exact included-source order, starting at offset zero."""
    if shard < 0 or roots_per_shard <= 0:
        raise ValueError("invalid shard index or size")
    start = shard * roots_per_shard
    end = start + roots_per_shard
    included = 0
    with source.open("rb") as raw:
        stream = zstd.ZstdDecompressor().stream_reader(raw)
        with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
            for row in csv.DictReader(text):
                if row["PuzzleId"] in excluded:
                    continue
                if included >= end:
                    break
                if included >= start:
                    yield row
                included += 1


def label_root(row: dict, engine: chess.engine.SimpleEngine, tokenizer, utils,
               seconds: float = STOCKFISH_SECONDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use one exact 50 ms Stockfish search per legal root action."""
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    if len(moves) < 2:
        raise ValueError(f"{row['PuzzleId']}: puzzle has no player move")
    setup = chess.Move.from_uci(moves[0])
    if setup not in board.legal_moves:
        raise ValueError(f"{row['PuzzleId']}: illegal setup move {moves[0]}")
    board.push(setup)
    solution = chess.Move.from_uci(moves[1])
    if solution not in board.legal_moves:
        raise ValueError(f"{row['PuzzleId']}: illegal solution move {moves[1]}")
    legal = list(board.legal_moves)
    token = np.asarray(tokenizer.tokenize(board.fen()), dtype=np.uint8)
    if token.shape != (77,):
        raise ValueError(f"{row['PuzzleId']}: bad token shape {token.shape}")
    tokens = np.repeat(token[None, :], len(legal), axis=0)
    actions = np.empty(len(legal), dtype=np.uint16)
    winprob = np.empty(len(legal), dtype=np.float32)
    for idx, move in enumerate(legal):
        analysis = engine.analyse(board, chess.engine.Limit(time=seconds),
                                  root_moves=[move], info=chess.engine.INFO_SCORE)
        score = analysis["score"].pov(board.turn).score(mate_score=10_000)
        if score is None:
            raise ValueError(f"{row['PuzzleId']}: Stockfish returned no score for {move}")
        actions[idx] = utils.MOVE_TO_ACTION[move.uci()]
        winprob[idx] = utils.centipawns_to_win_probability(score)
    if not np.isfinite(winprob).all() or np.any((winprob < 0) | (winprob > 1)):
        raise ValueError(f"{row['PuzzleId']}: invalid win probabilities")
    return tokens, actions, winprob


def _valid_engine(engine: chess.engine.SimpleEngine) -> None:
    name = str(engine.id.get("name", ""))
    if not re.search(r"\bStockfish\s+16(?:\b|\.)", name):
        raise ValueError(f"Stockfish 16 required; found {name!r}")


def _load_partial(path: Path, identity: dict) -> tuple[int, list, list, list]:
    if not path.is_file():
        return 0, [], [], []
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["identity_json"].item()))
        if meta != identity:
            raise ValueError("partial shard identity mismatch")
        roots_done = int(data["roots_done"].item())
        tokens = [data["tokens"].copy()]
        actions = [data["actions"].copy()]
        winprob = [data["winprob"].copy()]
    return roots_done, tokens, actions, winprob


def _save_partial(path: Path, identity: dict, roots_done: int,
                  tokens: list, actions: list, winprob: list) -> None:
    if roots_done == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, identity_json=json.dumps(identity, sort_keys=True),
                        roots_done=roots_done, tokens=np.concatenate(tokens),
                        actions=np.concatenate(actions), winprob=np.concatenate(winprob))
    temporary.replace(path)


def build(source: Path, manifest_path: Path, shard: int, stockfish: Path,
          output: Path, partial: Path, sl_repo: Path, max_roots: int = 0,
          save_every_seconds: float = 900.0,
          stop_after_seconds: float = 39_000.0,
          checkpoint_callback=None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "lichess-puzzle-corpus-v1":
        raise ValueError("unexpected puzzle corpus manifest format")
    source_digest = file_sha256(source)
    if source_digest != manifest.get("source_sha256"):
        raise ValueError(f"source digest mismatch: {source_digest}")
    excluded = set(manifest["excluded_ids"])
    if len(excluded) != 9_995 or manifest.get("included_roots") != 6_147_346:
        raise ValueError("frozen corpus identity/count mismatch")
    identity = {"source_sha256": source_digest, "shard": shard,
                "roots_per_shard": ROOTS_PER_SHARD, "max_roots": max_roots,
                "engine": "Stockfish 16", "seconds_per_action": STOCKFISH_SECONDS}
    roots_done, token_parts, action_parts, wp_parts = _load_partial(partial, identity)
    sys.path.insert(0, str(sl_repo.resolve().parent))
    from searchless_chess.src import tokenizer, utils
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    started = time.monotonic()
    last_save = started
    target = min(ROOTS_PER_SHARD, manifest["included_roots"] - shard * ROOTS_PER_SHARD)
    if target <= 0:
        raise ValueError("shard index exceeds the frozen corpus")
    if max_roots:
        target = min(target, max_roots)
    with chess.engine.SimpleEngine.popen_uci(str(stockfish)) as engine:
        _valid_engine(engine)
        for offset, row in enumerate(roots_in_shard(source, excluded, shard)):
            if offset >= target or STOP or time.monotonic() - started >= stop_after_seconds:
                break
            if offset < roots_done:
                continue
            tokens, actions, winprob = label_root(row, engine, tokenizer, utils)
            token_parts.append(tokens)
            action_parts.append(actions)
            wp_parts.append(winprob)
            roots_done += 1
            if time.monotonic() - last_save >= save_every_seconds:
                _save_partial(partial, identity, roots_done,
                              token_parts, action_parts, wp_parts)
                if checkpoint_callback is not None:
                    checkpoint_callback(partial, roots_done,
                                        sum(len(x) for x in action_parts))
                print(f"[labels] shard={shard:05d} roots={roots_done}/{target} "
                      f"rows={sum(len(x) for x in action_parts):,}", flush=True)
                last_save = time.monotonic()
    if roots_done != target:
        _save_partial(partial, identity, roots_done,
                      token_parts, action_parts, wp_parts)
        if roots_done and checkpoint_callback is not None:
            checkpoint_callback(partial, roots_done,
                                sum(len(x) for x in action_parts))
        return {"complete": False, "roots": roots_done, "target": target}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, tokens=np.concatenate(token_parts),
                        actions=np.concatenate(action_parts),
                        winprob=np.concatenate(wp_parts))
    temporary.replace(output)
    return {"complete": True, "roots": roots_done, "target": target,
            "rows": sum(len(x) for x in action_parts),
            "seconds": time.monotonic() - started, "identity": identity,
            "train_set_sha256": file_sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--sl-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--save-every-seconds", type=float, default=900.0)
    parser.add_argument("--stop-after-seconds", type=float, default=39_000.0)
    parser.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    parser.add_argument("--hf-prefix", default="",
                        help="remote corpus prefix; enables periodic HF persistence")
    args = parser.parse_args()
    api = None
    remote = f"{args.hf_prefix.strip('/')}/shard-{args.shard:05d}"
    if args.hf_prefix:
        from huggingface_hub import HfApi, hf_hub_download
        from scripts.kaggle_checkpoint import hf_token
        api = HfApi(token=hf_token(ROOT))
        files = set(api.list_repo_files(args.hf_repo, repo_type="dataset"))
        if f"{remote}/train_set.npz" in files and f"{remote}/meta.json" in files:
            print(f"[labels] {remote} already complete on HF", flush=True)
            return
        if f"{remote}/partial.npz" in files:
            hf_hub_download(args.hf_repo, f"{remote}/partial.npz",
                            repo_type="dataset", local_dir=str(args.partial.parent),
                            token=api.token)
            downloaded = args.partial.parent / remote / "partial.npz"
            downloaded.replace(args.partial)

    def checkpoint(path: Path, roots_done: int, rows_done: int) -> None:
        assert api is not None
        progress = path.parent / "progress.json"
        progress.write_text(json.dumps({"roots_done": roots_done,
                                        "rows_done": rows_done,
                                        "updated_at": time.time(),
                                        "identity": {"shard": args.shard,
                                                     "source_sha256": file_sha256(args.source)}},
                                       sort_keys=True) + "\n")
        api.upload_folder(folder_path=str(path.parent), path_in_repo=remote,
                          repo_id=args.hf_repo, repo_type="dataset",
                          allow_patterns=[path.name, progress.name],
                          commit_message=f"Save {remote} partial labels")

    try:
        result = build(args.source, args.manifest, args.shard, args.stockfish,
                       args.out, args.partial, args.sl_repo, args.max_roots,
                       args.save_every_seconds, args.stop_after_seconds,
                       checkpoint if api else None)
        if result["complete"] and api:
            meta = args.out.parent / "meta.json"
            meta.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            api.upload_folder(folder_path=str(args.out.parent), path_in_repo=remote,
                              repo_id=args.hf_repo, repo_type="dataset",
                              allow_patterns=[args.out.name, meta.name],
                              commit_message=f"Complete {remote} Stockfish labels")
        print(json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        if api:
            from scripts.kaggle_checkpoint import write_status
            write_status(api, args.hf_repo, remote, f"FAILED {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
