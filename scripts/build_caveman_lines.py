"""Stage A of the caveman pipeline: engine lines for both candidates.

For every pool position, Stockfish analyzes the position after EACH
candidate move at --depth and records the best continuation line + eval
(from the mover's perspective). This is the oracle brief that deepseek
explains in caveman style (Stage B) — the chess content is engine-given,
so deepseek can never invent a line.

    python3 scripts/build_caveman_lines.py \
        --pool results/rlvr-pool/smoke.jsonl \
        --out results/caveman-pilot/lines.jsonl

Output rows:
  {fen, candidate_a, candidate_b, truth_label,
   eval_a, line_a, eval_b, line_b, engine_preferred}

engine_preferred = the candidate with the better eval after it is played
(argmax of the two) — the ground truth the trace must end with.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _analyze(fen: str, move: str, engine, depth: int, max_plies: int):
    """Play `move` from `fen`, return (eval_cp_from_mover_pov, main_line)
    or (None, []) on any engine failure."""
    import chess
    board = chess.Board(fen)
    try:
        mv = chess.Move.from_uci(move)
    except ValueError:
        return None, []
    if mv not in board.legal_moves:
        return None, []
    board.push(mv)
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
    except Exception:
        return None, []
    score = info.get("score")
    if score is None:
        return None, []
    cp = score.white().score(mate_score=100000)
    eval_cp = -cp if board.turn else cp  # mover's perspective
    pv = info.get("pv") or []
    line = [m.uci() for m in pv[:max_plies]]
    return eval_cp, line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--max-plies", type=int, default=10,
                    help="plies of engine continuation per candidate line. "
                         "Longer lines give deepseek more verified material "
                         "to explain, so it is less likely to invent "
                         "continuation moves (measured: grounding violations "
                         "dropped once the supplied line covered the moves "
                         "the model wanted to mention)")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip fens already in --out")
    args = ap.parse_args()

    import chess
    import chess.engine

    rows = [json.loads(l) for l in Path(args.pool).read_text().splitlines()
            if l.strip()]
    if args.max_rows > 0:
        rows = rows[:args.max_rows]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["fen"])
    todo = [r for r in rows if r["fen"] not in done]
    print(f"positions: {len(rows)} (resume: {len(done)} done, "
          f"{len(todo)} to analyze)", flush=True)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    n_ok = n_skip = 0
    with out.open("a") as fout:
        for i, row in enumerate(todo):
            fen = row["fen"]
            eval_a, line_a = _analyze(fen, row["candidate_a"], engine,
                                      args.depth, args.max_plies)
            eval_b, line_b = _analyze(fen, row["candidate_b"], engine,
                                      args.depth, args.max_plies)
            if eval_a is None or eval_b is None:
                n_skip += 1
                print(f"  [{i}] skip {fen} (analysis failed)", flush=True)
                continue
            rec = {**row, "eval_a": round(eval_a), "line_a": line_a,
                   "eval_b": round(eval_b), "line_b": line_b,
                   "engine_preferred": "A" if eval_a >= eval_b else "B"}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_ok += 1
            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(todo)}] ok={n_ok} skip={n_skip}",
                      flush=True)
    engine.quit()
    print(f"lines done: {n_ok} analyzed, {n_skip} skipped -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
