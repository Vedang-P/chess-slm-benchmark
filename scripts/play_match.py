"""Stockfish-anchored Elo ladder for CC-GAVN (searchless play).

Plays the model against Stockfish with UCI_LimitStrength at a list of Elo
anchors, ECO openings, colors swapped, draw adjudication for very long games.
Stops the ladder early when the score at an anchor is too low.

Usage:
  python3 scripts/play_match.py --checkpoint CKPT --sl-repo SL \
      --stockfish /usr/games/stockfish --anchors 1400,1600,1800 \
      --games 40 --out-dir matches --demo
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_gavn import tokenize_fen  # noqa: E402
from scripts.train_gavn import action_tables  # noqa: E402

MAX_PLIES = 220


def main() -> None:
    import chess
    import chess.engine
    import chess.pgn
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sl-repo", required=True)
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--anchors", default="1400,1600,1800,2000,2200,2500")
    ap.add_argument("--games", type=int, default=40, help="games per anchor (half each color)")
    ap.add_argument("--move-time", type=float, default=0.1, help="Stockfish seconds per move")
    ap.add_argument("--openings", default="", help="eco_openings.pgn path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-score", type=float, default=0.30,
                    help="stop the ladder when score at an anchor is below this")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--demo", action="store_true", help="2 games per anchor, first anchor only")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    anchors = [int(x) for x in args.anchors.split(",") if x.strip()]
    games_per_anchor = 4 if args.demo else args.games
    if args.demo:
        anchors = anchors[:1]

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

    def choose(board):
        moves = engine_lib.get_ordered_legal_moves(board)
        ids = [utils.MOVE_TO_ACTION[m.uci()] for m in moves]
        tokens = np.repeat(tokenize_fen(board.fen())[None, :], len(moves), axis=0)
        with torch.inference_mode():
            logits = model(torch.as_tensor(tokens, dtype=torch.long, device=device),
                           torch.as_tensor(ids, dtype=torch.long, device=device))
            values = (torch.softmax(logits, -1) @ bucket_values).detach().cpu().numpy()
        vals = list(zip(moves, (float(v) for v in values)))
        for i, (mv, _v) in enumerate(vals):
            board.push(mv)
            rep = board.is_fivefold_repetition() or board.can_claim_threefold_repetition()
            board.pop()
            if rep:
                vals[i] = (mv, 0.5)
        return max(vals, key=lambda x: x[1])[0]

    openings = []
    if args.openings and Path(args.openings).exists():
        with open(args.openings) as fh:
            while True:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                moves = [m for m in g.mainline_moves()]
                if moves:
                    openings.append(moves[:8])
                if len(openings) >= 400:
                    break
    if not openings:
        openings = [[]]
    rng = random.Random(args.seed)
    results = {"anchors": {}, "checkpoint": str(cp), "demo": args.demo,
               "settings": {"move_time": args.move_time, "games_per_anchor": games_per_anchor}}
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    t0 = time.time()
    try:
        for elo in anchors:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1, "Hash": 16})
            wins = losses = draws = 0
            pgn_path = out / f"games-{elo}.pgn"
            with open(pgn_path, "w") as pgn_out:
                for gi in range(games_per_anchor):
                    model_white = (gi % 2 == 0)
                    board = chess.Board()
                    game = chess.pgn.Game()
                    game.headers.update({
                        "Event": f"CC-GAVN vs Stockfish UCI_Elo={elo}",
                        "White": "CC-GAVN" if model_white else f"Stockfish-{elo}",
                        "Black": f"Stockfish-{elo}" if model_white else "CC-GAVN",
                        "Result": "*",
                    })
                    node = game
                    prefix = rng.choice(openings)
                    for mv in prefix:
                        if board.is_legal(mv):
                            node = node.add_variation(mv)
                            board.push(mv)
                    game.headers["PlyCount"] = str(board.ply())
                    outcome = None
                    while not board.is_game_over(claim_draw=True):
                        if board.ply() >= MAX_PLIES:
                            outcome = "1/2-1/2"
                            break
                        if (board.turn == chess.WHITE) == model_white:
                            mv = choose(board)
                        else:
                            mv = engine.play(board, chess.engine.Limit(time=args.move_time)).move
                        node = node.add_variation(mv)
                        board.push(mv)
                    if outcome is None:
                        res = board.result(claim_draw=True)
                    else:
                        res = outcome
                    game.headers["Result"] = res
                    pgn_out.write(str(game) + "\n\n")
                    if res == "1/2-1/2":
                        draws += 1
                        sym = "D"
                    elif (res == "1-0" and model_white) or (res == "0-1" and not model_white):
                        wins += 1
                        sym = "W"
                    else:
                        losses += 1
                        sym = "L"
                    print(f"[match] elo={elo} game {gi+1}/{games_per_anchor} {sym} "
                          f"({wins}W {draws}D {losses}L) t={time.time()-t0:.0f}s", flush=True)
            played = wins + draws + losses
            score = (wins + 0.5 * draws) / max(1, played)
            results["anchors"][str(elo)] = {"wins": wins, "draws": draws, "losses": losses,
                                            "score": score, "games": played}
            print(f"[match] anchor {elo}: score {score:.3f} ({wins}W {draws}D {losses}L)", flush=True)
            if score < args.stop_score:
                print(f"[match] score {score:.3f} < {args.stop_score}: stopping ladder", flush=True)
                break
    finally:
        engine.quit()
        (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[match] done in {time.time()-t0:.0f}s -> {out/'results.json'}")


if __name__ == "__main__":
    main()
