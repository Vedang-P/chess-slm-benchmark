"""Search-diagnostic: same rows, all modes, plus known-answer mate-in-1 tests.

The 20-40 row smoke slice was biased (first rows of the file). This script
compares raw / kbest / mcts on the SAME rows (random sample, fixed seed) and
runs a known-answer set of mate-in-1 puzzles to catch search bugs.
"""
from __future__ import annotations

import argparse
import io
import json
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
    ap.add_argument("--mate", required=True)
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--n-rows", type=int, default=60)
    ap.add_argument("--n-mates", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="raw,kbest,mcts")
    args = ap.parse_args()

    cp = Path(args.checkpoint)
    cfg = json.loads((cp / "config.json").read_text())
    src, dst, promo, _ = action_tables(Path(args.sl_repo))
    from scripts.train_ccgavn import CCGAVN, candidate_relation_types
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCGAVN(torch, int(cfg["dim"]), int(cfg["layers"]), int(cfg["heads"]),
                   src, dst, promo, candidate_relation_types()).to(device)
    model.load_state_dict(torch.load(cp / "state.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    sys.path.insert(0, str(Path(args.sl_repo).parent))
    from searchless_chess.src import utils  # type: ignore
    from searchless_chess.src.engines import engine as engine_lib  # type: ignore
    bucket_values = torch.as_tensor(
        np.asarray(utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32), device=device)

    evals = {"n": 0}

    def score_moves(board, repetition_rule=False):
        moves = engine_lib.get_ordered_legal_moves(board)
        if not moves:
            return []
        ids = [utils.MOVE_TO_ACTION[m.uci()] for m in moves]
        tokens = np.repeat(tokenize_fen(board.fen())[None, :], len(moves), axis=0)
        with torch.inference_mode():
            logits = model(torch.as_tensor(tokens, dtype=torch.long, device=device),
                           torch.as_tensor(ids, dtype=torch.long, device=device))
            values = (torch.softmax(logits, -1) @ bucket_values).detach().cpu().numpy()
        evals["n"] += len(moves)
        out = []
        for mv, v in zip(moves, values):
            val = float(v)
            if repetition_rule:
                board.push(mv)
                if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                    val = 0.5
                board.pop()
            out.append((mv, val))
        return out

    def terminal(board):
        if board.is_checkmate():
            return 0.0
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_fivefold_repetition() or board.can_claim_threefold_repetition()
                or board.can_claim_fifty_moves()):
            return 0.5
        return None

    def negamax(board, depth, ext, width):
        t = terminal(board)
        if t is not None:
            return t
        scored = score_moves(board)
        if not scored:
            return 0.5
        if depth <= 0:
            return max(v for _m, v in scored)
        scored.sort(key=lambda x: x[1], reverse=True)
        best = -1.0
        for mv, _v in scored[:width]:
            extend = ext > 0 and (board.is_check() or board.is_capture(mv))
            board.push(mv)
            child = negamax(board, depth if extend else depth - 1, ext - 1 if extend else ext, width)
            board.pop()
            best = max(best, 1.0 - child)
        return best

    def kbest(board, depth, width, restrict=None):
        scored = score_moves(board)
        if restrict is not None:
            scored = [(m, v) for m, v in scored if m.uci() in restrict]
        out = []
        for mv, _v in scored:
            board.push(mv)
            out.append((mv, 1.0 - negamax(board, depth - 1, 2, width)))
            board.pop()
        return out

    def mcts(board, nodes):
        class N:
            __slots__ = ("board", "children", "visits", "wins", "priors", "expanded", "player")

            def __init__(self, b, player):
                self.board, self.children, self.visits, self.wins = b, {}, 0, 0.0
                self.priors, self.expanded, self.player = {}, False, player

        root = N(board.copy(stack=False), board.turn)
        for _ in range(nodes):
            node, path = root, [root]
            while node.expanded and node.children:
                total = max(1, node.visits)
                best_m, best_s = None, -1e9
                for mv, ch in node.children.items():
                    q = 1.0 - ch.wins / max(1, ch.visits)
                    u = 1.5 * node.priors[mv] * (total ** 0.5) / (1 + ch.visits)
                    if q + u > best_s:
                        best_m, best_s = mv, q + u
                node = node.children[best_m]
                path.append(node)
            t = terminal(node.board)
            if t is not None:
                value = t
            else:
                scored = score_moves(node.board)
                if not scored:
                    value = 0.5
                else:
                    vals = np.array([v for _m, v in scored])
                    pri = np.exp((vals - vals.max()) * 8.0)
                    pri /= pri.sum()
                    for (mv, _v), pr in zip(scored, pri):
                        ch = N(node.board.copy(stack=False), not node.board.turn)
                        ch.board.push(mv)
                        node.children[mv] = ch
                        node.priors[mv] = float(pr)
                    node.expanded = True
                    value = float(vals.max())
            for i, nd in enumerate(reversed(path)):
                nd.visits += 1
                nd.wins += value if i % 2 == 0 else 1.0 - value
        return [(mv, 1.0 - ch.wins / max(1, ch.visits)) for mv, ch in root.children.items()]

    def choose(mode, board, restrict=None):
        if mode == "raw":
            scored = score_moves(board, repetition_rule=True)
        elif mode == "kbest":
            scored = kbest(board, 3, 3, restrict=restrict)
        else:
            scored = mcts(board, 100)
        if not scored:
            return None, {}
        if restrict is not None:
            scored = [(m, v) for m, v in scored if m.uci() in restrict]
        vals = {m.uci(): v for m, v in scored}
        best = max(scored, key=lambda x: x[1])[0]
        return best, vals

    # ---------- same-rows MATE comparison ----------
    rows = json.loads(Path(args.mate).read_text())
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(rows), size=min(args.n_rows, len(rows)), replace=False)
    idx = sorted(int(i) for i in idx)
    print(f"[diag] MATE rows: {len(idx)} random rows (seed {args.seed})", flush=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {m: {"correct": 0, "total": 0, "flips_vs_raw": 0} for m in modes}
    raw_pred = {}
    t0 = time.time()
    for n, i in enumerate(idx, 1):
        row = rows[i]
        fen = row.get("fen") or row.get("position")
        extra = row.get("task_extra") or {}
        ca = (row.get("candidate_a") or row.get("move_a") or extra.get("candidate_a"))
        cb = (row.get("candidate_b") or row.get("move_b") or extra.get("candidate_b"))
        truth = (row.get("truth_label") or row.get("label") or extra.get("truth_label"))
        if not (fen and ca and cb and truth):
            continue
        board = chess.Board(fen)
        raw_vals = {m.uci(): v for m, v in score_moves(board, repetition_rule=True)}
        # MATE protocol compares the two candidates only (not a global argmax)
        raw_pred[i] = "A" if raw_vals.get(ca, -1) > raw_vals.get(cb, -1) else "B"
        for mode in [m for m in modes if m != "raw"]:
            best, vals = choose(mode, board, restrict={ca, cb})
            pred = "A" if best and best.uci() == ca else "B"
            if pred != raw_pred[i]:
                results[mode]["flips_vs_raw"] += 1
            results[mode]["correct"] += pred == truth
            results[mode]["total"] += 1
        results["raw"]["correct"] += raw_pred[i] == truth
        results["raw"]["total"] += 1
        if n % 10 == 0:
            print(f"[diag] row {n}/{len(idx)} t={time.time()-t0:.0f}s evals={evals['n']}", flush=True)
    print(f"[diag] same-rows MATE (n={results['raw']['total']}):")
    for mode in modes:
        r = results[mode]
        print(f"   {mode:6s} acc={100*r['correct']/max(1,r['total']):.1f}% "
              f"({r['correct']}/{r['total']}) flips_vs_raw={r['flips_vs_raw']}")

    # ---------- known-answer mate-in-1 set ----------
    df = pd.read_csv(args.puzzles, nrows=4000)
    mates = []
    for _, p in df.iterrows():
        game = chess.pgn.read_game(io.StringIO(p["PGN"]))
        b = game.end().board()
        mv = p["Moves"].split(" ")
        if len(mv) != 2:
            continue
        b.push(chess.Move.from_uci(mv[0]))
        solution = chess.Move.from_uci(mv[1])
        if b.is_legal(solution):
            solver_fen = b.fen()
            b.push(solution)
            if b.is_checkmate():
                mates.append((solver_fen, solution.uci()))
        if len(mates) >= args.n_mates:
            break
    print(f"[diag] mate-in-1 sanity set: {len(mates)} positions", flush=True)
    for mode in modes:
        ok = 0
        for fen, sol in mates:
            b = chess.Board(fen)
            best, _ = choose(mode, b)
            if best is None:
                continue
            if best.uci() == sol:
                ok += 1
            else:
                b.push(best)
                if b.is_checkmate():
                    ok += 1
        print(f"   {mode:6s} mate-in-1 solves: {ok}/{len(mates)}")
    print(f"[diag] total evals={evals['n']} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
