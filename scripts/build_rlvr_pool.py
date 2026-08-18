"""Build the RLVR position pool from the noexplain SFT slice (rlvr-plan.md).

Rows: {fen, candidate_a, candidate_b, truth_label} — the MATE selection
tuple the GRPO trainer samples from. Optional difficulty gating on the
Stockfish eval gap (|evalA - evalB| <= --max-gap-cp) per the plan: the
deciding cases are near-equal evals, where reasoning pays off.

    python3 scripts/build_rlvr_pool.py \
        --train data/positions/noexplain-slice/train.jsonl \
        --out results/rlvr-pool/train.jsonl \
        --max-rows 20000

    # difficulty-gated subset (Stockfish d12 on the candidates):
    python3 scripts/build_rlvr_pool.py --train ... --out ... \
        --difficulty-gate --max-gap-cp 60 --max-rows 20000

Input format is the noexplain-slice rows ({fen, messages}), where the
user message carries "MoveA:<uci> MoveB:<uci>" and the assistant message
carries the expert choice "MoveX:<uci>". Rows whose truth can't be parsed
are dropped (never guessed).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

USER_RE = re.compile(r"MoveA:\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)"
                     r"\s+MoveB:\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)")
ASSIST_RE = re.compile(r"\bMove\s*([AB])\b\s*[:.\-]?\s*"
                       r"([a-h][1-8][a-h][1-8][qrbnQRBN]?)?", re.I)


def parse_row(row: dict) -> dict | None:
    """Extract (fen, candidate_a, candidate_b, truth_label) from one
    noexplain-slice row. Returns None when the truth is unparseable —
    a pool row must never carry a guessed label."""
    msgs = row.get("messages") or []
    if len(msgs) < 2:
        return None
    user_text = msgs[0].get("content", "")
    m = USER_RE.search(user_text)
    if not m:
        return None
    candidate_a, candidate_b = m.group(1), m.group(2)
    am = list(ASSIST_RE.finditer(msgs[1].get("content", "")))
    if not am:
        return None
    match = am[-1]  # last mention wins, same rule as run_mate_eval.parse_choice
    label = match.group(1).upper()
    move = match.group(2)
    if move and move not in (candidate_a, candidate_b):
        return None  # assistant said a move that is not a candidate — bad row
    if not move:
        move = candidate_a if label == "A" else candidate_b
    truth_label = label
    return {"fen": row.get("fen"), "candidate_a": candidate_a,
            "candidate_b": candidate_b, "truth_label": truth_label}


def _score_candidate(fen: str, move: str, engine) -> float | None:
    """cp eval of the position AFTER playing `move` (from the side-to-move
    pov), or None on error."""
    import chess
    board = chess.Board(fen)
    if not chess.Move.from_uci(move) in board.legal_moves:
        return None
    board.push_uci(move)
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=12))
    except Exception:
        return None
    score = info.get("score")
    if score is None:
        return None
    # board.turn is now the opponent; cp from the mover's pov = -white pov
    cp = score.white().score(mate_score=100000)
    return -cp if board.turn else cp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="noexplain-slice "
                    "train.jsonl (or the small smoke slice)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap pool size (0 = all parsed rows)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--difficulty-gate", action="store_true",
                    help="keep only rows where |evalA - evalB| <= --max-gap-cp "
                         "at Stockfish d12 (near-equal deciding cases)")
    ap.add_argument("--max-gap-cp", type=int, default=60)
    ap.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    args = ap.parse_args()

    rows = []
    dropped = 0
    for line in Path(args.train).read_text().splitlines():
        if not line.strip():
            continue
        row = parse_row(json.loads(line))
        if row is None:
            dropped += 1
        else:
            rows.append(row)
    print(f"parsed {len(rows)} pool rows, dropped {dropped}", flush=True)

    if args.difficulty_gate:
        import chess
        import chess.engine

        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        kept = []
        skipped = 0
        for i, row in enumerate(rows):
            ea = _score_candidate(row["fen"], row["candidate_a"], engine)
            eb = _score_candidate(row["fen"], row["candidate_b"], engine)
            if ea is None or eb is None:
                skipped += 1
                continue
            if abs(ea - eb) <= args.max_gap_cp:
                row["eval_a"] = round(ea)
                row["eval_b"] = round(eb)
                kept.append(row)
            if (i + 1) % 5000 == 0:
                print(f"  gated {i + 1}/{len(rows)} rows, kept {len(kept)}",
                      flush=True)
        engine.quit()
        rows = kept
        print(f"difficulty gate: {len(rows)} kept (|gap| <= {args.max_gap_cp}cp), "
              f"{skipped} unscoreable dropped", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_rows > 0:
        rows = rows[:args.max_rows]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"pool written: {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
