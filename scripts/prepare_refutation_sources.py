"""Create the fixed 50/25/25 CC-GAVN-R pilot input from public sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import zstandard

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import SourceProgress


def game_positions(path: Path, count: int, seed: int, progress: SourceProgress | None = None):
    rng = random.Random(seed)
    chosen: list[dict] = []
    total_bytes = path.stat().st_size
    last_progress = time.monotonic()
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        eligible = 0
        while game := chess.pgn.read_game(text):
            if game.errors or game.end().board().chess960 or len(game.variations) != 1:
                continue
            headers = game.headers
            if headers.get("Variant", "Standard") != "Standard":
                continue
            try:
                elo = min(int(headers.get("WhiteElo", 0)), int(headers.get("BlackElo", 0)))
            except ValueError:
                elo = 0
            if elo < 1800:
                continue
            moves = list(game.mainline_moves())
            if len(moves) < 24:
                continue
            ply = rng.randrange(18, min(len(moves) - 4, 54))
            board = game.board()
            for move in moves[:ply]:
                board.push(move)
            if board.is_game_over() or not board.is_valid():
                continue
            game_id = headers.get("Site") or hashlib.sha256(
                (headers.get("Date", "") + headers.get("White", "") + headers.get("Black", "") +
                 " ".join(m.uci() for m in moves)).encode()).hexdigest()
            row = {"fen": board.fen(), "game_id": f"lichess:{game_id}", "source": "lichess-rated-game"}
            eligible += 1
            if len(chosen) < count:
                chosen.append(row)
            else:
                j = rng.randrange(eligible)
                if j < count:
                    chosen[j] = row
            if progress and time.monotonic() - last_progress >= 5:
                progress.update("scanning_games", bytes_done=raw.tell(), bytes_total=total_bytes,
                                games=eligible)
                last_progress = time.monotonic()
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
