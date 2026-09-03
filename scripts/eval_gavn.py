"""Evaluate a GAVN checkpoint on MATE and the official puzzle protocol."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_gavn import GAVN, action_tables, relation_types  # noqa: E402


_CHARS = {c: i for i, c in enumerate([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'p', 'n', 'r', 'k', 'q', 'P', 'B', 'N', 'R', 'Q',
    'K', 'w', '.'])}


def tokenize_fen(fen: str) -> np.ndarray:
    """Dependency-free equivalent of the official 77-token tokenizer."""
    board, side, castling, en_passant, halfmoves, fullmoves = fen.split(' ')
    chars = [side] + list(board.replace('/', ''))
    expanded = []
    for char in chars:
        if char in '12345678':
            expanded.extend(['.'] * int(char))
        else:
            expanded.append(char)
    expanded += ['.'] * 4 if castling == '-' else list(castling) + ['.'] * (4 - len(castling))
    expanded += ['.', '.'] if en_passant == '-' else list(en_passant)
    expanded += list((halfmoves + '...')[:3])
    expanded += list((fullmoves + '...')[:3])
    if len(expanded) != 77:
        raise ValueError(f"tokenizer produced {len(expanded)} tokens for {fen}")
    return np.asarray([_CHARS[x] for x in expanded], dtype=np.int64)


def main():
    import chess
    import chess.pgn
    import pandas as pd
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="checkpoint-N directory")
    p.add_argument("--sl-repo", default=os.environ.get("SL_REPO", "/kaggle/working/searchless_chess"))
    p.add_argument("--eval", default="", help="comma-separated MATE JSON files")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--puzzles", default="", help="official puzzles.csv path")
    p.add_argument("--num-puzzles", type=int, default=10000)
    p.add_argument("--score", choices=["q", "dist"], default="q")
    args = p.parse_args()

    cp = Path(args.checkpoint)
    cfg = json.loads((cp / "config.json").read_text(encoding="utf-8"))
    src, dst, promo, _ = action_tables(Path(args.sl_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAVN(torch, int(cfg["dim"]), int(cfg["layers"]), int(cfg["heads"]),
                 src, dst, promo, relation_types(),
                 bias_mode=cfg.get("bias_mode", "both")).to(device)
    state = torch.load(cp / "state.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    sys.path.insert(0, str(Path(args.sl_repo).parent))
    from searchless_chess.src import utils  # type: ignore
    bucket_values = torch.as_tensor(
        np.asarray(utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32),
        device=device)

    def scores(board):
        moves = list(board.legal_moves)
        action_ids = [utils.MOVE_TO_ACTION[m.uci()] for m in moves]
        tokens = np.repeat(tokenize_fen(board.fen())[None, :], len(moves), axis=0)
        with torch.inference_mode():
            logits, q = model(torch.as_tensor(tokens, dtype=torch.long, device=device),
                              torch.as_tensor(action_ids, dtype=torch.long, device=device))
            if args.score == "q":
                values = q
            else:
                values = torch.softmax(logits, -1) @ bucket_values
        return moves, values.detach().cpu().numpy()

    if args.eval:
        rows = []
        for filename in args.eval.split(","):
            rows.extend(json.loads(Path(filename.strip()).read_text(encoding="utf-8")))
        if args.max_rows:
            rows = rows[:args.max_rows]
        correct = total = 0
        for row in rows:
            fen = row.get("fen") or row.get("position")
            ca = row.get("candidate_a") or row.get("move_a")
            cb = row.get("candidate_b") or row.get("move_b")
            truth = row.get("truth_label") or row.get("label")
            extra = row.get("task_extra") or {}
            ca, cb, truth = ca or extra.get("candidate_a"), cb or extra.get("candidate_b"), truth or extra.get("truth_label")
            if not (fen and ca and cb and truth):
                continue
            board = chess.Board(fen)
            moves, values = scores(board)
            by_uci = {m.uci(): float(v) for m, v in zip(moves, values)}
            pred = "A" if by_uci[ca] > by_uci[cb] else "B"
            total += 1
            correct += pred == truth
        if total == 0:
            print("[gavn] MATE: no parseable rows (check eval file schema)")
        else:
            print(f"[gavn] MATE: {correct}/{total} = {100*correct/total:.2f}%")

    if args.puzzles:
        puzzles = pd.read_csv(args.puzzles, nrows=args.num_puzzles)
        solved = 0
        for _, puzzle in puzzles.iterrows():
            game = chess.pgn.read_game(io.StringIO(puzzle["PGN"]))
            board = game.end().board()
            moves = puzzle["Moves"].split(" ")
            ok = True
            for i, uci in enumerate(moves):
                if i % 2 == 1:
                    legal, values = scores(board)
                    predicted = legal[int(np.argmax(values))].uci()
                    if predicted != uci:
                        board.push(chess.Move.from_uci(predicted))
                        ok = board.is_checkmate()
                        break
                board.push(chess.Move.from_uci(uci))
            solved += ok
            if (int(_) + 1) % 100 == 0:
                print(f"[gavn] puzzle progress row={_+1} solved={solved}", flush=True)
        print(f"[gavn] puzzles: {solved}/{len(puzzles)} = {100*solved/len(puzzles):.2f}%")


if __name__ == "__main__":
    main()
