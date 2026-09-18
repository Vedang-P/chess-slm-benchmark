"""Frozen-protocol evaluation with search on top of a CC-GAVN checkpoint.

Modes:
  raw    - argmax over model move scores (searchless; identical decision rule
           to scripts/eval_gavn.py, including the repetition override)
  kbest  - negamax: every ply keeps the top-K moves by the side-to-move's own
           model value (which is the opponent's worst-K after negation), model
           value at the horizon, bounded check/capture extensions
  mcts   - PUCT with the model as prior (softmax of move scores) and value
           (expected win probability), one search per root position

Terminal handling: mate (1.0/0.0), stalemate, insufficient material, fifty-move
and claimable/fivefold repetition (0.5). "With search" numbers are a separate
track from the searchless protocol.

Usage:
  python3 scripts/eval_gavn_search.py --checkpoint CKPT --sl-repo SL \
      --mate data/positions/mate-selection-test-noexplain.json --max-rows 200 \
      --mode kbest --depth 4 --width 4
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_gavn import tokenize_fen  # noqa: E402
from scripts.train_gavn import action_tables  # noqa: E402


def main() -> None:
    import chess
    import chess.pgn
    import pandas as pd
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sl-repo", required=True)
    ap.add_argument("--mate", default="")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--puzzles", default="")
    ap.add_argument("--num-puzzles", type=int, default=0)
    ap.add_argument("--mode", choices=["raw", "kbest", "mcts"], default="kbest")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--root-top", type=int, default=6, help="kbest: only search the top-N root moves; others keep raw scores")
    ap.add_argument("--extensions", type=int, default=2)
    args = ap.parse_args()

    cp = Path(args.checkpoint)
    cfg = json.loads((cp / "config.json").read_text())
    src, dst, promo, _ = action_tables(Path(args.sl_repo))
    from scripts.train_ccgavn import CCGAVN, candidate_relation_types
    model = CCGAVN(torch, int(cfg["dim"]), int(cfg["layers"]), int(cfg["heads"]),
                   src, dst, promo, candidate_relation_types())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(cp / "state.pt", map_location=device,
                                     weights_only=False)["model"])
    model = model.to(device)
    model.eval()
    print(f"[search] device={device}", flush=True)
    sys.path.insert(0, str(Path(args.sl_repo).parent))
    from searchless_chess.src import utils  # type: ignore
    from searchless_chess.src.engines import engine as engine_lib  # type: ignore
    bucket_values = torch.as_tensor(
        np.asarray(utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32), device=device)

    counters = {"evals": 0}

    def score_moves(board, repetition_rule=False):
        moves = engine_lib.get_ordered_legal_moves(board)
        if not moves:
            return []
        action_ids = [utils.MOVE_TO_ACTION[m.uci()] for m in moves]
        tokens = np.repeat(tokenize_fen(board.fen())[None, :], len(moves), axis=0)
        with torch.inference_mode():
            logits = model(torch.as_tensor(tokens, dtype=torch.long, device=device),
                           torch.as_tensor(action_ids, dtype=torch.long, device=device))
            values = (torch.softmax(logits, -1) @ bucket_values).numpy()
        counters["evals"] += len(moves)
        out = []
        for move, value in zip(moves, values):
            v = float(value)
            if repetition_rule:
                board.push(move)
                if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                    v = 0.5
                board.pop()
            out.append((move, v))
        return out

    def terminal(board):
        if board.is_checkmate():
            return 0.0  # side to move is mated
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_fivefold_repetition()
                or board.can_claim_threefold_repetition()
                or board.can_claim_fifty_moves()):
            return 0.5
        return None

    def negamax(board, depth, ext):
        t = terminal(board)
        if t is not None:
            return t
        scored = score_moves(board)
        if not scored:
            return 0.5
        if depth <= 0:
            return max(v for _m, v in scored)
        scored.sort(key=lambda x: x[1], reverse=True)  # top-K for side to move
        best = -1.0
        for move, _value in scored[:args.width]:
            extend = ext > 0 and (board.is_check() or board.is_capture(move))
            board.push(move)
            child = negamax(board, depth if extend else depth - 1, ext - 1 if extend else ext)
            board.pop()
            best = max(best, 1.0 - child)
        return best

    def kbest_roots(board, restrict=None):
        raw_scores = score_moves(board)
        if restrict is not None:
            raw_scores = [(m, v) for m, v in raw_scores if m.uci() in restrict]
        found = {}
        for move, raw in raw_scores:
            found[move.uci()] = raw
        ranked = sorted(raw_scores, key=lambda x: x[1], reverse=True)
        searched = ranked if restrict is not None else ranked[:args.root_top]
        out = []
        for move, raw in searched:
            board.push(move)
            value = 1.0 - negamax(board, args.depth - 1, args.extensions)
            board.pop()
            out.append((move, value))
        for move, raw in ranked[len(searched):]:
            out.append((move, raw - 0.02))
        return out

    def mcts_roots(board):
        class Node:
            __slots__ = ("board", "children", "visits", "wins", "priors", "expanded")

            def __init__(self, b):
                self.board = b
                self.children = {}
                self.visits = 0
                self.wins = 0.0
                self.priors = {}
                self.expanded = False

        root = Node(board.copy(stack=False))
        for _ in range(args.nodes):
            node = root
            path = [node]
            # selection
            while node.expanded and node.children:
                total = max(1, node.visits)
                best_move, best_score = None, -1e9
                for move, child in node.children.items():
                    q = 1.0 - child.wins / max(1, child.visits)  # parent's perspective
                    u = 1.5 * node.priors[move] * math.sqrt(total) / (1 + child.visits)
                    if q + u > best_score:
                        best_move, best_score = move, q + u
                node = node.children[best_move]
                path.append(node)
            # expansion / evaluation
            t = terminal(node.board)
            if t is not None:
                value = t
            else:
                scored = score_moves(node.board)
                if not scored:
                    value = 0.5
                else:
                    vals = np.array([v for _m, v in scored], dtype=np.float64)
                    pri = np.exp((vals - vals.max()) * 8.0)
                    pri /= pri.sum()
                    for (move, _v), pr in zip(scored, pri):
                        child = Node(node.board.copy(stack=False))
                        child.board.push(move)
                        node.children[move] = child
                        node.priors[move] = float(pr)
                    node.expanded = True
                    value = float(vals.max())
            # backup (flip perspective at every level)
            for i, nd in enumerate(reversed(path)):
                nd.visits += 1
                nd.wins += value if i % 2 == 0 else (1.0 - value)
        out = []
        for move, child in root.children.items():
            out.append((move, 1.0 - child.wins / max(1, child.visits)))
        for move, raw in score_moves(board):
            if move not in root.children:
                out.append((move, raw - 0.05))
        return out

    def root_values(board, restrict=None):
        if args.mode == "raw":
            return score_moves(board, repetition_rule=True)
        if args.mode == "kbest":
            return kbest_roots(board, restrict=restrict)
        return mcts_roots(board)

    t0 = time.time()
    if args.mate:
        rows = []
        for name in args.mate.split(","):
            rows.extend(json.loads(Path(name.strip()).read_text()))
        if args.max_rows:
            rows = rows[:args.max_rows]
        correct = total = 0
        for row in rows:
            fen = row.get("fen") or row.get("position")
            ca = row.get("candidate_a") or row.get("move_a")
            cb = row.get("candidate_b") or row.get("move_b")
            truth = row.get("truth_label") or row.get("label")
            extra = row.get("task_extra") or {}
            ca = ca or extra.get("candidate_a")
            cb = cb or extra.get("candidate_b")
            truth = truth or extra.get("truth_label")
            if not (fen and ca and cb and truth):
                continue
            board = chess.Board(fen)
            legal = {m.uci() for m in engine_lib.get_ordered_legal_moves(board)}
            if ca not in legal or cb not in legal:
                continue
            restrict = {ca, cb}
            values = {m.uci(): v for m, v in root_values(board, restrict=restrict)}
            pred = "A" if values[ca] > values[cb] else "B"
            correct += pred == truth
            total += 1
            if total % 25 == 0:
                print(f"[search] MATE {total} acc={correct/total:.3f} evals={counters['evals']} "
                      f"t={time.time()-t0:.0f}s", flush=True)
        print(f"[search] mode={args.mode} MATE: {correct}/{total} = {100*correct/max(total,1):.2f}%")

    if args.puzzles and args.num_puzzles:
        puzzles = pd.read_csv(args.puzzles, nrows=args.num_puzzles)
        solved = 0
        for idx, puzzle in puzzles.iterrows():
            game = chess.pgn.read_game(io.StringIO(puzzle["PGN"]))
            board = game.end().board()
            moves = puzzle["Moves"].split(" ")
            ok = True
            for i, uci in enumerate(moves):
                if i % 2 == 1:
                    ranked = root_values(board)
                    predicted = max(ranked, key=lambda x: x[1])[0].uci()
                    if predicted != uci:
                        board.push(chess.Move.from_uci(predicted))
                        ok = board.is_checkmate()
                        break
                board.push(chess.Move.from_uci(uci))
            solved += ok
            if (idx + 1) % 50 == 0:
                print(f"[search] puzzles {idx+1}/{len(puzzles)} solved={solved} "
                      f"evals={counters['evals']} t={time.time()-t0:.0f}s", flush=True)
        print(f"[search] mode={args.mode} puzzles: {solved}/{len(puzzles)} = {100*solved/len(puzzles):.2f}%")
    print(f"[search] total evals={counters['evals']} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
