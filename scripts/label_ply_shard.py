"""Build ChessBench-format action-value rows for later-ply Lichess puzzle roots.

Adapted from scripts/label_puzzle_shard.py, which stays frozen. The root is the
board after pushing ``Moves[0..ply-1]`` (ply 3 and 5: the solver's second and third
decision) for puzzles with ``len(Moves) > ply``. One row is produced for every legal
action, each with the original Stockfish 16, 50 ms root-move evaluation and the same
win-probability transform. Several Stockfish processes label different roots of the
same shard at once; roots are consumed in order, so a partial shard is always a
prefix and resumes exactly.

Official puzzles (by id), the validation/test puzzles (by id) and every position in
the benchmark exclusion set (by reflection-orbit hash) never reach a shard.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import queue
import re
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cont_data import ROOTS_PER_SHARD  # noqa: E402
from scripts.data_hygiene import position_hashes  # noqa: E402
from scripts.freeze_puzzle_corpus import file_sha256  # noqa: E402
from scripts.ply_data import (PREFIX, SECONDS_PER_ACTION, check_ply, read_ids,  # noqa: E402
                              shard_candidates, shard_dir, shard_identity)

STOCKFISH_SECONDS = SECONDS_PER_ACTION
OFFICIAL_COUNT, INCLUDED_ROOTS = 9_995, 6_147_346       # the frozen corpus identity
STOP = False
PARTIAL_ARRAYS = ("tokens", "actions", "winprob", "lens", "index", "rating", "puzzle_id")


def _signal_stop(_number, _frame) -> None:
    global STOP
    STOP = True


def root_board(row: dict, ply: int) -> chess.Board:
    """Board after ``Moves[0..ply-1]``; the stored solution move ``Moves[ply]`` must be legal."""
    check_ply(ply)
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    if len(moves) <= ply:
        raise ValueError(f"{row['PuzzleId']}: puzzle has no move at ply {ply}")
    for index, uci in enumerate(moves[:ply]):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"{row['PuzzleId']}: illegal move {uci} at index {index}")
        board.push(move)
    solution = chess.Move.from_uci(moves[ply])
    if solution not in board.legal_moves:
        raise ValueError(f"{row['PuzzleId']}: illegal solution move {moves[ply]} at ply {ply}")
    return board


def label_board(board: chess.Board, puzzle_id: str, engine: chess.engine.SimpleEngine,
                tokenizer, utils, seconds: float = STOCKFISH_SECONDS
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use one exact 50 ms Stockfish search per legal action of `board`."""
    legal = list(board.legal_moves)
    token = np.asarray(tokenizer.tokenize(board.fen()), dtype=np.uint8)
    if token.shape != (77,):
        raise ValueError(f"{puzzle_id}: bad token shape {token.shape}")
    tokens = np.repeat(token[None, :], len(legal), axis=0)
    actions = np.empty(len(legal), dtype=np.uint16)
    winprob = np.empty(len(legal), dtype=np.float32)
    for idx, move in enumerate(legal):
        analysis = engine.analyse(board, chess.engine.Limit(time=seconds),
                                  root_moves=[move], info=chess.engine.INFO_SCORE)
        score = analysis["score"].pov(board.turn).score(mate_score=10_000)
        if score is None:
            raise ValueError(f"{puzzle_id}: Stockfish returned no score for {move}")
        actions[idx] = utils.MOVE_TO_ACTION[move.uci()]
        winprob[idx] = utils.centipawns_to_win_probability(score)
    if not np.isfinite(winprob).all() or np.any((winprob < 0) | (winprob > 1)):
        raise ValueError(f"{puzzle_id}: invalid win probabilities")
    return tokens, actions, winprob


def label_root(row: dict, ply: int, engine: chess.engine.SimpleEngine, tokenizer, utils,
               seconds: float = STOCKFISH_SECONDS):
    return label_board(root_board(row, ply), row["PuzzleId"], engine, tokenizer, utils, seconds)


def _valid_engine(engine: chess.engine.SimpleEngine) -> None:
    name = str(engine.id.get("name", ""))
    if not re.search(r"\bStockfish\s+16(?:\b|\.)", name):
        raise ValueError(f"Stockfish 16 required; found {name!r}")


def _concat(parts: list, dtype, empty_shape: tuple = (0,)) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(empty_shape, dtype=dtype)


def _load_partial(path: Path, identity: dict) -> dict | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["identity_json"].item()))
        if meta != identity:
            raise ValueError("partial shard identity mismatch")
        state = {k: data[k].copy() for k in PARTIAL_ARRAYS}
        state["offset_done"] = int(data["offset_done"].item())
        state["dropped"] = int(data["dropped"].item())
    return state


def _save_partial(path: Path, identity: dict, state: dict) -> None:
    if state["offset_done"] == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, identity_json=json.dumps(identity, sort_keys=True),
                        offset_done=state["offset_done"], dropped=state["dropped"],
                        **{k: state[k] for k in PARTIAL_ARRAYS})
    temporary.replace(path)


class Accumulator:
    """Labeled roots in candidate order, plus how many candidates have been consumed."""

    def __init__(self, restored: dict | None = None):
        self.tokens: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.winprob: list[np.ndarray] = []
        self.lens: list[int] = []
        self.index: list[int] = []
        self.rating: list[int] = []
        self.puzzle_id: list[str] = []
        self.offset_done = 0
        self.dropped = 0
        if restored is not None:
            self.offset_done, self.dropped = restored["offset_done"], restored["dropped"]
            self.tokens, self.actions = [restored["tokens"]], [restored["actions"]]
            self.winprob = [restored["winprob"]]
            self.lens = restored["lens"].tolist()
            self.index = restored["index"].tolist()
            self.rating = restored["rating"].tolist()
            self.puzzle_id = restored["puzzle_id"].tolist()

    @property
    def roots(self) -> int:
        return len(self.lens)

    @property
    def rows(self) -> int:
        return int(sum(self.lens))

    def add(self, labeled, included: int, rating: int, puzzle_id: str) -> None:
        if len(puzzle_id) > 8:
            raise ValueError(f"puzzle id {puzzle_id!r} does not fit the shard id column")
        tokens, actions, winprob = labeled
        self.tokens.append(tokens)
        self.actions.append(actions)
        self.winprob.append(winprob)
        self.lens.append(len(actions))
        self.index.append(included)
        self.rating.append(rating)
        self.puzzle_id.append(puzzle_id)

    def arrays(self) -> dict:
        return {"tokens": _concat(self.tokens, np.uint8, (0, 77)),
                "actions": _concat(self.actions, np.uint16),
                "winprob": _concat(self.winprob, np.float32),
                "lens": np.asarray(self.lens, dtype=np.int32),
                "index": np.asarray(self.index, dtype=np.uint32),
                "rating": np.asarray(self.rating, dtype=np.uint16),
                "puzzle_id": np.asarray(self.puzzle_id, dtype="U8")}

    def state(self) -> dict:
        return {**self.arrays(), "offset_done": self.offset_done, "dropped": self.dropped}


def build(source: Path, manifest_path: Path, ply: int, shard: int, stockfish: Path,
          output: Path, partial: Path, sl_repo: Path, *, val_csv: Path, test_csv: Path,
          exclusion_path: Path, max_roots: int = 0, engines: int = 1,
          save_every_seconds: float = 900.0, stop_after_seconds: float = 39_000.0,
          checkpoint_callback=None, engine_factory=None, modules=None,
          roots_per_shard: int = ROOTS_PER_SHARD) -> dict:
    check_ply(ply)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "lichess-puzzle-corpus-v1":
        raise ValueError("unexpected puzzle corpus manifest format")
    source_digest = file_sha256(source)
    if source_digest != manifest.get("source_sha256"):
        raise ValueError(f"source digest mismatch: {source_digest}")
    official_ids = set(manifest["excluded_ids"])
    if len(official_ids) != OFFICIAL_COUNT or manifest.get("included_roots") != INCLUDED_ROOTS:
        raise ValueError("frozen corpus identity/count mismatch")
    skip_ids = read_ids(val_csv, test_csv)
    if skip_ids & official_ids:
        raise ValueError("validation/test puzzles overlap the official set")
    hashes = np.load(exclusion_path, allow_pickle=False)
    if hashes.dtype != np.uint64 or hashes.ndim != 1:
        raise ValueError("exclusion hashes must be a 1-D uint64 array")
    excluded_hashes = {int(h) for h in hashes}
    identity = shard_identity(
        source_sha256=source_digest, ply=ply, shard=shard, max_roots=max_roots,
        val_csv_sha256=file_sha256(val_csv), test_csv_sha256=file_sha256(test_csv),
        exclusion_sha256=file_sha256(exclusion_path), roots_per_shard=roots_per_shard)

    candidates = shard_candidates(source, official_ids, skip_ids, ply, shard, roots_per_shard)
    if not candidates:
        raise ValueError(f"shard {shard} exceeds the ply-{ply} corpus")
    if max_roots:
        candidates = candidates[:max_roots]
    target = len(candidates)

    acc = Accumulator(_load_partial(partial, identity))
    if acc.offset_done > target:
        raise ValueError("partial shard is longer than the shard")
    if modules is None:
        sys.path.insert(0, str(sl_repo.resolve().parent))
        from searchless_chess.src import tokenizer, utils
    else:
        tokenizer, utils = modules
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    started = time.monotonic()
    last_save = last_log = started
    factory = engine_factory or (lambda path: chess.engine.SimpleEngine.popen_uci(str(path)))
    stopping = False
    engine_name = ""

    def report(tag: str) -> None:
        print(f"[ply-labels] {tag} ply={ply} shard={shard:05d} candidates={acc.offset_done}/{target} "
              f"roots={acc.roots} rows={acc.rows:,}", flush=True)

    with contextlib.ExitStack() as stack:
        free: queue.Queue = queue.Queue()
        for _ in range(max(1, engines)):
            engine = stack.enter_context(contextlib.closing(factory(stockfish)))
            _valid_engine(engine)
            engine_name = str(engine.id.get("name", ""))
            free.put(engine)
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=max(1, engines)))

        def label(board: chess.Board, puzzle_id: str):
            engine = free.get()
            try:
                return label_board(board, puzzle_id, engine, tokenizer, utils)
            finally:
                free.put(engine)

        window: deque = deque()          # (included, rating, puzzle_id, future | None), in order
        position = acc.offset_done       # next candidate to put in flight
        try:
            while True:
                stopping = stopping or STOP or time.monotonic() - started >= stop_after_seconds
                while (not stopping and position < target and len(window) < 2 * max(1, engines)):
                    _, included, row = candidates[position]
                    position += 1
                    board = root_board(row, ply)
                    token = np.asarray(tokenizer.tokenize(board.fen()), dtype=np.uint8)
                    if int(position_hashes(token[None, :])[0]) in excluded_hashes:
                        window.append((included, 0, row["PuzzleId"], None))
                        continue
                    window.append((included, int(row["Rating"]), row["PuzzleId"],
                                   pool.submit(label, board, row["PuzzleId"])))
                if not window:
                    break
                included, rating, puzzle_id, future = window.popleft()
                if future is None:
                    acc.dropped += 1
                else:
                    acc.add(future.result(), included, rating, puzzle_id)
                acc.offset_done += 1
                now = time.monotonic()
                if now - last_save >= save_every_seconds:
                    _save_partial(partial, identity, acc.state())
                    if checkpoint_callback is not None and acc.roots:
                        checkpoint_callback(partial, acc.roots, acc.rows)
                    last_save = now
                if now - last_log >= 300.0:
                    report("progress")
                    last_log = now
        except BaseException:
            # In-flight labels are discarded; the consumed prefix is still exact.
            for _, _, _, future in window:
                if future is not None:
                    future.cancel()
            _save_partial(partial, identity, acc.state())
            raise
    if acc.offset_done != target:
        _save_partial(partial, identity, acc.state())
        if acc.roots and checkpoint_callback is not None:
            checkpoint_callback(partial, acc.roots, acc.rows)
        report("stopped")
        return {"complete": False, "ply": ply, "shard": shard, "candidates": acc.offset_done,
                "roots": acc.roots, "target": target}
    arrays = acc.arrays()
    arrays["ply"] = np.full(acc.roots, ply, dtype=np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    return {"complete": True, "format": identity["format"], "ply": ply, "shard": shard,
            "candidates": target, "roots": acc.roots, "target": target,
            "dropped_benchmark": acc.dropped, "rows": acc.rows,
            "seconds": time.monotonic() - started, "identity": identity, "engines": engines,
            "engine_name": engine_name, "train_set_sha256": file_sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--exclusion-hashes", type=Path, required=True,
                        help="sorted unique uint64 reflection-orbit hashes (.npy)")
    parser.add_argument("--ply", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--sl-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--engines", type=int, default=1, help="Stockfish processes in parallel")
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--save-every-seconds", type=float, default=900.0)
    parser.add_argument("--stop-after-seconds", type=float, default=39_000.0)
    parser.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    parser.add_argument("--hf-prefix", default="",
                        help=f"remote corpus prefix ({PREFIX}); enables periodic HF persistence")
    args = parser.parse_args()
    api = None
    retry_hf_call = None
    remote = shard_dir(args.hf_prefix, args.ply, args.shard) if args.hf_prefix else ""
    if args.hf_prefix:
        from huggingface_hub import HfApi, hf_hub_download
        from scripts.kaggle_checkpoint import hf_token
        from scripts.hf_upload import retry_hf_call
        api = HfApi(token=hf_token(ROOT))
        files = set(api.list_repo_files(args.hf_repo, repo_type="dataset"))
        if f"{remote}/train_set.npz" in files and f"{remote}/meta.json" in files:
            print(f"[ply-labels] {remote} already complete on HF", flush=True)
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
        progress.write_text(json.dumps({"roots_done": roots_done, "rows_done": rows_done,
                                        "updated_at": time.time(),
                                        "identity": {"ply": args.ply, "shard": args.shard,
                                                     "source_sha256": file_sha256(args.source)}},
                                       sort_keys=True) + "\n")
        retry_hf_call(
            lambda: api.upload_folder(folder_path=str(path.parent), path_in_repo=remote,
                                      repo_id=args.hf_repo, repo_type="dataset",
                                      allow_patterns=[path.name, progress.name],
                                      commit_message=f"Save {remote} partial labels"),
            f"partial labels for {remote}")

    try:
        result = build(args.source, args.manifest, args.ply, args.shard, args.stockfish,
                       args.out, args.partial, args.sl_repo, val_csv=args.val_csv,
                       test_csv=args.test_csv, exclusion_path=args.exclusion_hashes,
                       max_roots=args.max_roots, engines=args.engines,
                       save_every_seconds=args.save_every_seconds,
                       stop_after_seconds=args.stop_after_seconds,
                       checkpoint_callback=checkpoint if api else None)
        if result["complete"] and api:
            meta = args.out.parent / "meta.json"
            meta.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            assert retry_hf_call is not None
            # One commit publishes the labels and retires the shard's scratch files.
            retry_hf_call(
                lambda: api.upload_folder(folder_path=str(args.out.parent),
                                          path_in_repo=remote, repo_id=args.hf_repo,
                                          repo_type="dataset",
                                          allow_patterns=[args.out.name, meta.name],
                                          delete_patterns=["partial.npz", "progress.json",
                                                           "run-status.txt"],
                                          commit_message=f"Complete {remote} Stockfish labels"),
                f"completed labels for {remote}")
        print(json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        if api:
            from scripts.kaggle_checkpoint import write_status
            write_status(api, args.hf_repo, remote, f"FAILED {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
