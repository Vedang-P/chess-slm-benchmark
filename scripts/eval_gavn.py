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
    p.add_argument("--score", choices=["auto", "q", "dist"], default="auto",
                   help="Decision score. auto/dist uses the trained return\n"
                        "distribution expectation; q is only a diagnostic\n"
                        "for checkpoints trained with a nonzero --w-q.")
    args = p.parse_args()

    cp = Path(args.checkpoint)
    cfg = json.loads((cp / "config.json").read_text(encoding="utf-8"))
    src, dst, promo, _ = action_tables(Path(args.sl_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("architecture") == "cc-gavn-v1":
        from scripts.train_ccgavn import CCGAVN, candidate_relation_types
        model = CCGAVN(torch, int(cfg["dim"]), int(cfg["layers"]), int(cfg["heads"]),
                      src, dst, promo, candidate_relation_types()).to(device)
        architecture = "cc-gavn-v1"
    else:
        legacy_relations = cfg.get("relation_schema") is None
        model = GAVN(torch, int(cfg["dim"]), int(cfg["layers"]), int(cfg["heads"]),
                     src, dst, promo, relation_types(legacy=legacy_relations),
                     bias_mode=cfg.get("bias_mode", "both")).to(device)
        architecture = "gavn"
    state = torch.load(cp / "state.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    sys.path.insert(0, str(Path(args.sl_repo).parent))
    from searchless_chess.src import utils  # type: ignore
    from searchless_chess.src.engines import engine as engine_lib  # type: ignore
    bucket_values = torch.as_tensor(
        np.asarray(utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32),
        device=device)

    score_mode = "dist" if args.score == "auto" else args.score
    if score_mode == "q" and not cfg.get("q_head_trained", cfg.get("w_q", 0.0)):
        raise ValueError(
            "This checkpoint's q_head was not trained (w_q=0). "
            "Use --score dist, which is the canonical action-value output.")
    if architecture == "cc-gavn-v1" and score_mode == "q":
        raise ValueError("CC-GAVN has no scalar q_head; use --score dist")
    print(f"[gavn] architecture={architecture} decision score={score_mode}", flush=True)

    def scores(board):
        # The official engine specifies its own stable ordering.  Ordering only
        # affects ties, but exact puzzle sequences make those ties observable.
        moves = engine_lib.get_ordered_legal_moves(board)
        action_ids = [utils.MOVE_TO_ACTION[m.uci()] for m in moves]
        tokens = np.repeat(tokenize_fen(board.fen())[None, :], len(moves), axis=0)
        with torch.inference_mode():
            outputs = model(torch.as_tensor(tokens, dtype=torch.long, device=device),
                            torch.as_tensor(action_ids, dtype=torch.long, device=device))
            if architecture == "cc-gavn-v1":
                logits, q = outputs, None
            else:
                logits, q = outputs
            if score_mode == "q":
                values = q
            else:
                values = torch.softmax(logits, -1) @ bucket_values
        values = values.detach().cpu().numpy()
        # Match ActionValueEngine's rule-based score for claimable repetitions.
        for i, move in enumerate(moves):
            board.push(move)
            if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                values[i] = 0.5
            board.pop()
        return moves, values

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
