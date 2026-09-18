"""Play with a trained CC-GAVN checkpoint: rank legal moves or play a game.

Searchless: every legal move is scored in one batched forward pass, the
highest-scoring move is played. Scores are win-probability expectations over
the 128 return bins, with the official repetition override (0.5).

Examples:
  python3 scripts/play_ccgavn.py --checkpoint ckpt/ccgavn-5m-seed0/checkpoint-160000 \
      --sl-repo /path/to/searchless_chess --fen startpos --top-k 6
  python3 scripts/play_ccgavn.py --checkpoint ... --sl-repo ... --play 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_gavn import tokenize_fen  # noqa: E402
from scripts.train_gavn import action_tables  # noqa: E402


def main() -> None:
    import chess
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="checkpoint-N directory with state.pt + config.json")
    p.add_argument("--sl-repo", required=True, help="path to the official searchless_chess clone")
    p.add_argument("--fen", default="", help="position FEN, or 'startpos' / empty for the initial position")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--play", type=int, default=0, help="play this many plies (model vs model)")
    args = p.parse_args()

    cp = Path(args.checkpoint)
    cfg = __import__("json").loads((cp / "config.json").read_text())
    if cfg.get("architecture") != "cc-gavn-v1":
        raise SystemExit("this demo expects a cc-gavn-v1 checkpoint")
    src, dst, promo, _ = action_tables(Path(args.sl_repo))
    from scripts.train_ccgavn import CCGAVN, candidate_relation_types
    model = CCGAVN(torch, int(cfg["dim"]), int(cfg["layers"]), int(cfg["heads"]),
                   src, dst, promo, candidate_relation_types())
    state = torch.load(cp / "state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    sys.path.insert(0, str(Path(args.sl_repo).parent))
    from searchless_chess.src import utils  # type: ignore
    from searchless_chess.src.engines import engine as engine_lib  # type: ignore
    bucket_values = torch.as_tensor(
        np.asarray(utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32))

    def score_moves(board):
        moves = engine_lib.get_ordered_legal_moves(board)
        action_ids = [utils.MOVE_TO_ACTION[m.uci()] for m in moves]
        tokens = np.repeat(tokenize_fen(board.fen())[None, :], len(moves), axis=0)
        with torch.inference_mode():
            logits = model(torch.as_tensor(tokens, dtype=torch.long),
                           torch.as_tensor(action_ids, dtype=torch.long))
            values = (torch.softmax(logits, -1) @ bucket_values).numpy()
        for i, move in enumerate(moves):
            board.push(move)
            if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                values[i] = 0.5
            board.pop()
        order = np.argsort(-values)
        return [(moves[i], float(values[i])) for i in order]

    if args.play:
        board = chess.Board()
        print(f"initial position, {args.play} plies, model vs model")
        for ply in range(args.play):
            ranked = score_moves(board)
            move, value = ranked[0]
            print(f"{ply+1:2d}. {board.san(move):8s} score={value:.3f}")
            board.push(move)
        print(board.fen())
        return

    fen = args.fen
    board = chess.Board() if fen in ("", "startpos") else chess.Board(fen)
    print(f"position: {board.fen()}")
    print(f"side to move: {'white' if board.turn else 'black'}")
    for rank, (move, value) in enumerate(score_moves(board)[:args.top_k], 1):
        san = board.san(move)
        print(f"{rank:2d}. {san:8s} ({move.uci()}) score={value:.4f}")


if __name__ == "__main__":
    main()
