"""Build puzzle-curriculum training rows from the Lichess puzzle DB (CPU).

Streams lichess_db_puzzle.csv.zst, applies the frozen-protocol leakage guards
(official 10K PuzzleIds, official pre- and query-positions, MATE positions at
every ply), and emits (tokens, action) rows for the solver's moves along each
solution. Teacher labels are added later on GPU (teacher_label.py).

Outputs (npz; tokens uint8 [N,77], actions uint16 [N]):
  <out>-0.npz, <out>-1.npz   training rows, split for parallel GPU labeling
  <out>-dev.npz              held-out puzzle-dev rows (never trained on)
  <out>-audit.json           exclusion counts and verification result
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def fen_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def build_exclusions(test_csv: Path, mate_dir: Path):
    import chess

    test_ids, test_pre, test_positions = set(), set(), set()
    with open(test_csv) as fh:
        for row in csv.DictReader(fh):
            test_ids.add(row["PuzzleId"])
            board = chess.Board(row["FEN"])
            test_pre.add(fen_key(board.fen()))
            for i, uci in enumerate(row["Moves"].split(" ")):
                if i % 2 == 1:
                    test_positions.add(fen_key(board.fen()))
                board.push(chess.Move.from_uci(uci))
    mate_positions = set()
    for jf in Path(mate_dir).glob("*.json"):
        for row in json.loads(jf.read_text()):
            fen = row.get("fen") or row.get("position")
            if fen:
                mate_positions.add(fen_key(fen))
    return test_ids, test_pre, test_positions, mate_positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="lichess_db_puzzle.csv.zst")
    ap.add_argument("--test", required=True, help="official puzzles.csv (excluded)")
    ap.add_argument("--mate-dir", default=str(ROOT / "data" / "positions"))
    ap.add_argument("--out", required=True, help="output prefix, e.g. /tmp/puzzle_rows")
    ap.add_argument("--dev-every", type=int, default=200, help="hold out every Nth puzzle for puzzle-dev")
    ap.add_argument("--sl-repo", required=True, help="official searchless_chess clone")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    import chess
    import zstandard
    sys.path.insert(0, str(Path(args.sl_repo).parent))
    from searchless_chess.src import utils  # type: ignore
    from scripts.eval_gavn import tokenize_fen  # noqa: E402  (verified bit-exact)

    test_ids, test_pre, test_positions, mate_positions = build_exclusions(
        Path(args.test), Path(args.mate_dir))
    print(f"[build] exclusions: {len(test_ids)} ids, {len(test_pre)} pre, "
          f"{len(test_positions)} test positions, {len(mate_positions)} mate positions", flush=True)

    train_tokens, train_actions = [], []
    dev_tokens, dev_actions = [], []
    stats = dict(total=0, skipped_excluded=0, skipped_unsafe=0, skipped_bad=0,
                 puzzles_used=0, train_rows=0, dev_rows=0)
    t0 = time.time()
    dctx = zstandard.ZstdDecompressor()
    with open(args.db, "rb") as fh, dctx.stream_reader(fh) as rd:
        r = csv.reader(io.TextIOWrapper(rd, encoding="utf-8"))
        header = next(r)
        i_id, i_fen, i_moves = (header.index(k) for k in ("PuzzleId", "FEN", "Moves"))
        for row in r:
            stats["total"] += 1
            pid, fen, moves = row[i_id], row[i_fen], row[i_moves]
            try:
                if pid in test_ids or fen_key(fen) in test_pre or fen_key(fen) in mate_positions:
                    stats["skipped_excluded"] += 1
                    continue
                board = chess.Board(fen)
                solution = moves.split(" ")
                rows = []
                safe = True
                for i, uci in enumerate(solution):
                    if i % 2 == 1:
                        key = fen_key(board.fen())
                        if key in test_positions or key in mate_positions:
                            safe = False
                            break
                        rows.append((int(utils.MOVE_TO_ACTION[uci]),
                                     tokenize_fen(board.fen())))
                    board.push(chess.Move.from_uci(uci))
                if not safe:
                    stats["skipped_unsafe"] += 1
                    continue
            except Exception:
                stats["skipped_bad"] += 1
                continue
            stats["puzzles_used"] += 1
            dev = (stats["puzzles_used"] % args.dev_every == 0)
            for action, tokens in rows:
                if dev:
                    dev_tokens.append(tokens)
                    dev_actions.append(action)
                else:
                    train_tokens.append(tokens)
                    train_actions.append(action)
            if args.max_rows and len(train_tokens) >= args.max_rows:
                break
            if stats["total"] % 500000 == 0:
                print(f"[build] {stats['total']/1e6:.1f}M scanned, "
                      f"{len(train_tokens)/1e6:.1f}M train rows, {time.time()-t0:.0f}s", flush=True)

    def save(prefix: str, toks, acts):
        if not toks:
            return
        np.savez_compressed(prefix + ".npz",
                            tokens=np.asarray(toks, dtype=np.uint8),
                            actions=np.asarray(acts, dtype=np.uint16))

    n = len(train_tokens)
    half = n // 2
    save(f"{args.out}-0", train_tokens[:half], train_actions[:half])
    save(f"{args.out}-1", train_tokens[half:], train_actions[half:])
    save(f"{args.out}-dev", dev_tokens, dev_actions)
    stats["train_rows"] = n
    stats["dev_rows"] = len(dev_tokens)
    stats["seconds"] = round(time.time() - t0, 1)
    Path(f"{args.out}-audit.json").write_text(json.dumps(stats, indent=1))
    print(f"[build] done: {stats}", flush=True)


if __name__ == "__main__":
    main()
