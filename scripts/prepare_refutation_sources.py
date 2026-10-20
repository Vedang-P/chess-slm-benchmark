"""Create the fixed 50/25/25 CC-GAVN-R pilot input from public sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import multiprocessing
import os
import random
import re
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.pgn
import zstandard

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import SourceProgress


_HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$')
_SOURCE_BLOCK = 128


def _header_map(header_text: str) -> dict[str, str]:
    headers = {}
    for line in header_text.splitlines():
        if not line.startswith("["):
            continue
        match = _HEADER_RE.match(line)
        if match:
            headers[match.group(1)] = match.group(2)
    return headers


def _iter_game_blocks(stream):
    """Yield (headers, movetext) without asking python-chess to parse a game."""
    headers = []
    moves = []
    state = "headers"
    for line in stream:
        if state == "headers":
            if line.startswith("["):
                headers.append(line)
            elif not line.strip() and headers:
                state = "moves"
        else:
            if line.strip():
                moves.append(line)
            elif headers:
                yield "".join(headers), "".join(moves)
                headers = []
                moves = []
                state = "headers"
    if headers:
        yield "".join(headers), "".join(moves)


def _parse_blocks(blocks):
    """Parse eligible PGN blocks in a worker; return None for filtered games."""
    parsed = []
    for header_text, move_text in blocks:
        game = chess.pgn.read_game(io.StringIO(header_text + "\n" + move_text))
        if game is None or game.errors or len(game.variations) != 1:
            parsed.append(None)
            continue
        if game.board().chess960:
            parsed.append(None)
            continue
        moves = [move.uci() for move in game.mainline_moves()]
        if len(moves) < 24:
            parsed.append(None)
            continue
        headers = game.headers
        parsed.append((headers.get("Site") or "", headers.get("Date", ""),
                       headers.get("White", ""), headers.get("Black", ""), moves))
    return parsed


def _eligible_blocks(stream, raw, total_bytes, progress, state):
    """Header-only prefilter; only ELO/variant-eligible games reach a parser."""
    blocks = []
    header_eligible = 0
    last_progress = time.monotonic()
    for header_text, move_text in _iter_game_blocks(stream):
        headers = _header_map(header_text)
        if headers.get("Variant", "Standard") != "Standard":
            continue
        try:
            elo = min(int(headers.get("WhiteElo", 0)), int(headers.get("BlackElo", 0)))
        except ValueError:
            elo = 0
        if elo < 1800:
            continue
        header_eligible += 1
        blocks.append((header_text, move_text))
        if len(blocks) >= _SOURCE_BLOCK:
            state["bytes"], state["header_eligible"] = raw.tell(), header_eligible
            yield blocks
            blocks = []
        if progress and time.monotonic() - last_progress >= 5:
            state["bytes"], state["header_eligible"] = raw.tell(), header_eligible
            progress.update("scanning_games", bytes_done=state["bytes"],
                            bytes_total=total_bytes,
                            pct=100 * state["bytes"] / total_bytes,
                            games=header_eligible)
            last_progress = time.monotonic()
    state["bytes"], state["header_eligible"] = total_bytes, header_eligible
    if blocks:
        yield blocks


def game_positions(path: Path, count: int, seed: int, progress: SourceProgress | None = None,
                   workers: int | None = None):
    rng = random.Random(seed)
    chosen: list[dict] = []
    total_bytes = path.stat().st_size
    if workers is None:
        workers = int(os.environ.get("SOURCE_SCAN_WORKERS", "0"))
    if not workers:
        try:
            workers = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            workers = os.cpu_count() or 1
    workers = max(1, min(4, workers))
    print(f"[source] scanning games with {workers} workers", flush=True)
    state = {"bytes": 0, "header_eligible": 0}
    eligible = 0
    last_progress = time.monotonic()
    executor = None
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        blocks = _eligible_blocks(text, raw, total_bytes, progress, state)
        try:
            if workers == 1:
                results = itertools.chain.from_iterable(_parse_blocks(block) for block in blocks)
            else:
                try:
                    mp_context = multiprocessing.get_context("fork")
                except ValueError:
                    mp_context = None
                executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp_context)
                pending = deque()
                for _ in range(workers * 2):
                    try:
                        pending.append(executor.submit(_parse_blocks, next(blocks)))
                    except StopIteration:
                        break
                def completed():
                    while pending:
                        future = pending.popleft()
                        yield from future.result()
                        try:
                            pending.append(executor.submit(_parse_blocks, next(blocks)))
                        except StopIteration:
                            pass
                results = completed()
            for record in results:
                if record is None:
                    continue
                site, date, white, black, moves = record
                ply = rng.randrange(18, min(len(moves) - 4, 54))
                board = chess.Board()
                for uci in moves[:ply]:
                    board.push(chess.Move.from_uci(uci))
                if board.is_game_over() or not board.is_valid():
                    continue
                game_id = site or hashlib.sha256(
                    (date + white + black + " ".join(moves)).encode()).hexdigest()
                row = {"fen": board.fen(), "game_id": f"lichess:{game_id}",
                       "source": "lichess-rated-game"}
                eligible += 1
                if len(chosen) < count:
                    chosen.append(row)
                else:
                    j = rng.randrange(eligible)
                    if j < count:
                        chosen[j] = row
                if progress and time.monotonic() - last_progress >= 5:
                    progress.update("scanning_games", bytes_done=state["bytes"],
                                    bytes_total=total_bytes,
                                    pct=100 * state["bytes"] / total_bytes, games=eligible)
                    last_progress = time.monotonic()
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
    if progress:
        progress.update("scanning_games", force=True, bytes_done=total_bytes,
                        bytes_total=total_bytes, games=eligible, pct=100)
    if len(chosen) != count:
        raise RuntimeError(f"only found {len(chosen)}/{count} eligible game positions")
    return chosen


def tactical_positions(path: Path, count: int, seed: int, progress: SourceProgress | None = None):
    rng = random.Random(seed + 1)
    chosen: list[dict] = []
    eligible = 0
    total_bytes = path.stat().st_size
    last_progress = time.monotonic()
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        rows = csv.DictReader(text)
        for row in rows:
            if row.get("OpeningTags"):
                continue
            moves = row["Moves"].split()
            board = chess.Board(row["FEN"])
            if not moves:
                continue
            first = chess.Move.from_uci(moves[0])
            if first not in board.legal_moves:
                continue
            board.push(first)
            if board.is_game_over() or not board.is_valid():
                continue
            item = {"fen": board.fen(), "game_id": f"lichess-puzzle:{row['PuzzleId']}",
                    "source": "lichess-tactical-puzzle"}
            eligible += 1
            if len(chosen) < count:
                chosen.append(item)
            else:
                j = rng.randrange(eligible)
                if j < count:
                    chosen[j] = item
            if progress and time.monotonic() - last_progress >= 5:
                progress.update("scanning_tactics", bytes_done=raw.tell(), bytes_total=total_bytes,
                                rows=eligible)
                last_progress = time.monotonic()
    if progress:
        progress.update("scanning_tactics", force=True, bytes_done=total_bytes,
                        bytes_total=total_bytes, rows=eligible, pct=100)
    if len(chosen) != count:
        raise RuntimeError(f"only found {len(chosen)}/{count} tactical positions")
    return chosen


def disagreement_positions(path: Path, count: int, seed: int, checkpoint: Path, sl_repo: Path,
                           progress: SourceProgress | None = None):
    import numpy as np
    import torch
    from scripts.eval_gavn import tokenize_fen
    from scripts.refutation import load_network, score_moves

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network, _, utils = load_network(checkpoint, str(sl_repo), device)
    network.eval()
    rng = random.Random(seed + 2)
    chosen: list[dict] = []
    seen: set[str] = set()
    total_rows = sum(1 for _ in path.open())
    with path.open(encoding="utf-8") as stream, torch.inference_mode():
        for row_number, line in enumerate(stream, 1):
            if progress and row_number % 25 == 0:
                progress.update("scoring_disagreements", rows_scanned=row_number,
                                rows_total=total_rows,
                                pct=round(100*row_number/max(1,total_rows), 2))
            row = json.loads(line)
            a, b = row.get("candidate_a"), row.get("candidate_b")
            if not a or not b or row.get("truth_label") not in ("A", "B"):
                continue
            board = chess.Board(row["fen"])
            moves = [chess.Move.from_uci(a), chess.Move.from_uci(b)]
            if any(move not in board.legal_moves for move in moves):
                continue
            scores = score_moves(network, board, moves, utils, mode="base")
            picked = int(np.argmax(scores))
            truth = 0 if row["truth_label"] == "A" else 1
            if picked == truth:
                continue
            key = board.fen()
            if key in seen:
                continue
            seen.add(key)
            item = {"fen": key, "game_id": None, "source": "student-oracle-disagreement"}
            if len(chosen) < count:
                chosen.append(item)
            else:
                j = rng.randrange(len(seen))
                if j < count:
                    chosen[j] = item
    if progress:
        progress.update("scoring_disagreements", force=True, rows_scanned=total_rows,
                        rows_total=total_rows, pct=100, found=len(chosen))
    if len(chosen) < count:
        print(json.dumps({"warning": "fewer genuine base/oracle disagreements than requested",
                          "requested": count, "found": len(chosen)}), flush=True)
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=Path, required=True)
    p.add_argument("--tactics", type=Path, required=True)
    p.add_argument("--disagreements", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sl-repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--progress-file", type=Path)
    p.add_argument("--hf-repo")
    p.add_argument("--hf-run")
    args = p.parse_args()
    if args.count % 4:
        raise ValueError("pilot size must be divisible by four")
    broad, tactical, disagreements = args.count // 2, args.count // 4, args.count // 4
    progress = SourceProgress(args.progress_file, args.hf_repo, args.hf_run) if args.progress_file else None
    rows = game_positions(args.games, broad, args.seed, progress)
    rows += tactical_positions(args.tactics, tactical, args.seed, progress)
    rows += disagreement_positions(args.disagreements, disagreements, args.seed,
                                   args.checkpoint, args.sl_repo, progress)
    if progress:
        progress.update("writing_source", force=True, rows=len(rows), pct=100)
    random.Random(args.seed).shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, separators=(",", ":")) + "\n")
    counts = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    print(json.dumps({"rows": len(rows), "source_counts": counts, "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
