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

    train_tok_path = Path(f"{args.out}-train-tokens.bin")
    train_act_path = Path(f"{args.out}-train-actions.bin")
    dev_tok_path = Path(f"{args.out}-dev-tokens.bin")
    dev_act_path = Path(f"{args.out}-dev-actions.bin")
    train_tok_fh = open(train_tok_path, "wb")
    train_act_fh = open(train_act_path, "wb")
    dev_tok_fh = open(dev_tok_path, "wb")
    dev_act_fh = open(dev_act_path, "wb")
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
                    dev_tok_fh.write(tokens.astype(np.uint8).tobytes())
                    dev_act_fh.write(np.uint16(action).tobytes())
                    stats["dev_rows"] += 1
                else:
                    train_tok_fh.write(tokens.astype(np.uint8).tobytes())
                    train_act_fh.write(np.uint16(action).tobytes())
                    stats["train_rows"] += 1
            if args.max_rows and stats["train_rows"] >= args.max_rows:
                break
            if stats["total"] % 500000 == 0:
                print(f"[build] {stats['total']/1e6:.1f}M scanned, "
                      f"{stats['train_rows']/1e6:.1f}M train rows, {time.time()-t0:.0f}s", flush=True)

    for fh in (train_tok_fh, train_act_fh, dev_tok_fh, dev_act_fh):
        fh.close()

    def load_bin(tok_path, act_path, n_rows):
        if n_rows == 0:
            return None, None
        toks = np.fromfile(tok_path, dtype=np.uint8).reshape(n_rows, 77)
        acts = np.fromfile(act_path, dtype=np.uint16)
        assert len(acts) == n_rows, (len(acts), n_rows)
        return toks, acts

    n = stats["train_rows"]
    toks, acts = load_bin(train_tok_path, train_act_path, n)
    half = n // 2
    np.savez_compressed(f"{args.out}-0.npz", tokens=toks[:half], actions=acts[:half])
    np.savez_compressed(f"{args.out}-1.npz", tokens=toks[half:], actions=acts[half:])
    dev_toks, dev_acts = load_bin(dev_tok_path, dev_act_path, stats["dev_rows"])
    if dev_toks is not None:
        np.savez_compressed(f"{args.out}-dev.npz", tokens=dev_toks, actions=dev_acts)
    for pth in (train_tok_path, train_act_path, dev_tok_path, dev_act_path):
        pth.unlink(missing_ok=True)
    stats["seconds"] = round(time.time() - t0, 1)
    Path(f"{args.out}-audit.json").write_text(json.dumps(stats, indent=1))
    print(f"[build] done: {stats}", flush=True)


if __name__ == "__main__":
    main()
