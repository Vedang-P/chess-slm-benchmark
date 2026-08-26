"""Build verbalized-search traces for SEARCH-IN-LANGUAGE (egsd C pivot,
searchlang-plan.md).

For each pool position, run Stockfish MultiPV (--multipv) at --depth and
verbalize the top lines into a natural-language search trace:

    Position: <FEN> (Black to move). Candidates: MoveA:e4e1 MoveB:g3h3.
    Line MoveA:e4e1: e4e1 (mate) ...
    Line MoveB:g3h3: g3h3 d7d1 h3g3 d1g1 ... (eval -436)
    Verdict: MoveA:e4e1 wins because ...
    MoveA:e4e1

Every number in the trace comes from the engine (no self-judgment). The
final answer line is parseable by the standard MoveA/MoveB regex. Output
both the raw trace rows and the SFT-ready messages form.

Usage:
    python3 scripts/build_search_traces.py \
        --pool results/rlvr-pool/train-5k.jsonl \
        --out results/searchlang-traces.jsonl \
        --n-positions 5000 --depth 14 --multipv 4 \
        --stockfish /opt/homebrew/bin/stockfish
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import chess.engine

MATE = 100000


def pov_cp(score, side: chess.Color) -> int:
    """Centipawns from the side-to-move's perspective; mate = +/-100000.
    analyse() returns PovScore; unwrap with .pov(side)."""
    try:
        s = score.pov(side)
    except Exception:
        s = score
    if s.is_mate():
        return MATE if s.mate() > 0 else -MATE
    return s.score(mate_score=MATE)


def uci_verbose(u: str) -> str:
    """e4e1 -> 'e4-e1'; e7e8q -> 'e7-e8=Q' (friendlier NL)."""
    promo = u[4] if len(u) == 5 else ""
    return f"{u[:2]}-{u[2:4]}" + (f"={promo.upper()}" if promo else "")


def verbalize(pv, evals, side) -> str:
    """One line: first move + engine eval after it, then 2-6 plies of
    lookahead with the terminal eval. evals are side-to-move POV."""
    side_name = "White" if side == chess.WHITE else "Black"
    parts = [f"Line {uci_verbose(pv[0])}:"]
    parts.append(f"{uci_verbose(pv[0])}")
    for m in pv[1:6]:
        parts.append(uci_verbose(m))
    ev = evals[-1] if evals else None
    if ev is not None:
        if abs(ev) == MATE:
            parts.append(f"(mate for {side_name})")
        else:
            parts.append(f"(eval {ev:+d})")
    return " ".join(parts)


def build_trace(row: dict, lines: list) -> str:
    """Compose the full NL trace for one position."""
    fen = row["fen"]
    ca, cb = row["candidate_a"], row["candidate_b"]
    truth = row["truth_label"]
    board = chess.Board(fen)
    side = board.turn  # True=White
    side_name = "White" if side == chess.WHITE else "Black"

    parts = [f"Position: {fen} ({side_name} to move).",
             f"Candidates: MoveA:{ca} MoveB:{cb}.",
             "I search each candidate with engine evaluations."]
    for pv, evals in lines:
        parts.append(verbalize(pv, evals, side))

    # verdict from the engine's best line
    best_pv, best_evals = lines[0]
    best_move = best_pv[0]
    best_label = "A" if best_move == ca else ("B" if best_move == cb else None)
    best_ev = best_evals[-1] if best_evals else 0
    if best_label is not None:
        if abs(best_ev) == MATE:
            why = f"Move{best_label}:{best_move} wins by mate"
        else:
            why = f"Move{best_label}:{best_move} is best (eval {best_ev:+d})"
        parts.append(f"Verdict: {why}.")
        parts.append(f"Move{best_label}:{best_move}")
        answer = f"Move{best_label}:{best_move}"
    else:
        # engine best is not one of the candidates; still answer truth-adjacent
        parts.append(f"Verdict: neither candidate is the engine's best "
                     f"({best_move}); choose the better of the two given.")
        parts.append(f"Move{truth}:{'ca' if truth == 'A' else 'cb'}")
        answer = f"Move{truth}:{'ca' if truth == 'A' else 'cb'}"

    return "\n".join(parts), answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-positions", type=int, default=0)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--multipv", type=int, default=4)
    ap.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip first N pool rows (resume)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pool)]
    if args.skip:
        rows = rows[args.skip:]
    if args.n_positions > 0:
        rows = rows[:args.n_positions]

    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    eng.configure({"Threads": 4, "Hash": 512})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sft_out = out.with_name(out.stem + "_sft.jsonl")
    t0 = time.time()
    n_trace, n_skip = 0, 0
    with out.open("w") as f, sft_out.open("w") as sf:
        for i, row in enumerate(rows):
            fen = row["fen"]
            board = chess.Board(fen)
            try:
                infos = eng.analyse(board, chess.engine.Limit(
                    depth=args.depth), multipv=args.multipv)
                seen = {}  # first move -> (pv, [evals])
                for info in infos:
                    pv = [m.uci() for m in info.get("pv", [])]
                    score = info.get("score")
                    if not pv or score is None:
                        continue
                    ev = pov_cp(score, board.turn)
                    first = pv[0]
                    # one entry per first move, deepest pv + eval
                    seen[first] = (pv, [ev])
            except Exception as e:
                n_skip += 1
                continue
            if not seen:
                n_skip += 1
                continue

            # order: engine-best first (highest eval from side to move)
            lines = sorted(seen.values(),
                           key=lambda kv: kv[1][-1] if kv[1] else -MATE,
                           reverse=True)[:args.multipv]
            trace, answer = build_trace(row, lines)
            rec = {"fen": fen, "candidate_a": row["candidate_a"],
                   "candidate_b": row["candidate_b"],
                   "truth": row["truth_label"],
                   "trace": trace, "answer": answer,
                   "lines": [[pv, evs] for pv, evs in lines]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sft_row = {
                "fen": fen,
                "messages": [
                    {"role": "user", "content":
                     f"You are an expert chess player. {fen} is the position "
                     f"({ 'White' if board.turn else 'Black'} to move). "
                     f"Two candidate moves: MoveA:{row['candidate_a']} "
                     f"MoveB:{row['candidate_b']}. Search each candidate by "
                     f"looking ahead, then answer with exactly one of "
                     f"MoveA:<move> or MoveB:<move>."},
                    {"role": "assistant", "content": trace},
                ],
            }
            sf.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            n_trace += 1
            if (i + 1) % 200 == 0:
                print(f"[build] {i+1}/{len(rows)} | traces={n_trace} "
                      f"skip={n_skip} | {time.time()-t0:.0f}s", flush=True)

    eng.quit()
    print(f"[build] DONE: {n_trace} traces, {n_skip} skipped -> {out}",
          flush=True)
    print(f"[build] SFT-ready -> {sft_out}", flush=True)


if __name__ == "__main__":
    main()
