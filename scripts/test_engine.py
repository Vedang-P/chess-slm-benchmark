"""Engine/dataset gate tests for the chess study.

All 8x8 scoring runs on python-chess (standard chess). This suite verifies:
  1. The committed task sets are self-consistent under python-chess
     (FEN parses, pieces match, oracles legal, mate moves actually mate).
  2. Standard-rules regression: castling, en passant, double-step legal.
  3. Answer extraction: from-to + SAN parsing never invents moves.

Usage:
    python scripts/test_engine.py [--quick]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"ok   {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


def test_dataset_standard_chess() -> None:
    """Committed 8x8 datasets must be fully consistent under standard chess."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "positions"
    for name in ("mate1-lichess", "mate2-lichess", "bestmove-8x8"):
        recs = json.loads((data_dir / f"{name}.json").read_text())
        check(f"dataset {name} non-empty", len(recs) >= 40)
        for rec in recs:
            try:
                board = chess.Board(rec["presented_fen"])
            except ValueError:
                check(f"{rec['id']} fen parses", False, rec["presented_fen"])
                continue
            fen_map = {chess.square_name(sq): p.symbol()
                       for sq in chess.SQUARES if (p := board.piece_at(sq)) is not None}
            rec_map = {p["sq"]: (p["kind"] if p["color"] == "w" else p["kind"].lower())
                       for p in rec["pieces"]}
            check(f"{rec['id']} pieces match fen", fen_map == rec_map)
            check(f"{rec['id']} turn matches",
                  ("w" if board.turn == chess.WHITE else "b") == rec["turn"])
            extra = rec["task_extra"]
            if name == "mate1-lichess":
                for uci in extra.get("mate_moves", []):
                    b = chess.Board(rec["presented_fen"])
                    b.push_uci(uci)
                    check(f"{rec['id']} {uci} mates", b.is_checkmate())
            elif name == "mate2-lichess":
                mv = chess.Move.from_uci(extra["first_move"])
                check(f"{rec['id']} first_move legal", mv in board.legal_moves)
            else:  # bestmove
                mv = chess.Move.from_uci(extra["best_move"])
                check(f"{rec['id']} best_move legal", mv in board.legal_moves)

    # MATE move-selection set: FEN parses, candidates are legal, truth label sane
    mate = json.loads((data_dir / "mate-selection-test.json").read_text())
    check("dataset mate-selection-test non-empty", len(mate) >= 100)
    for rec in mate:
        extra = rec["task_extra"]
        board = chess.Board(rec["fen"])
        ca = chess.Move.from_uci(extra["candidate_a"])
        cb = chess.Move.from_uci(extra["candidate_b"])
        check(f"{rec['id']} candidate_a legal", ca in board.legal_moves)
        check(f"{rec['id']} candidate_b legal", cb in board.legal_moves)
        check(f"{rec['id']} truth label sane",
              extra["truth_label"] in ("A", "B"))
        check(f"{rec['id']} candidates differ", extra["candidate_a"] != extra["candidate_b"])

    # standard-rules regression: these moves MUST be legal
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    check("standard: castling e1g1 legal", chess.Move.from_uci("e1g1") in b.legal_moves)
    b = chess.Board("rnbqkbnr/pppppppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    check("standard: en passant d4e3 legal", chess.Move.from_uci("d4e3") in b.legal_moves)
    b = chess.Board()
    check("standard: double-step e2e4 legal", chess.Move.from_uci("e2e4") in b.legal_moves)


def test_answer_extraction() -> None:
    """Extraction must never invent a move. from-to first (last MOVE: line),
    then line-anchored, then SAN via python-chess (legal-only)."""
    from src.benchmarks.games.tasks import parse_move_output

    def board_from_fen(fen):
        return chess.Board(fen)

    cases = [
        # (fen, output, expected_uci, expected_fmt)
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: d2f3", "d2f3", "fromto"),
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: Nf3", "d2f3", "san"),
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "I think Rc8 is best.", None, None),  # prose without an answer
        ("2kr1b1r/p1p2pp1/2pqN3/7p/6n1/2NPB3/PPP2PPP/R2Q1RK1 b - - 0 1",
         "MOVE: bKb8#", "c8b8", "san"),      # color-prefixed king move
        ("4r3/1k6/pp3P2/1b5p/3R1p2/P1R2P2/1P4PP/6K1 b - - 0 1",
         "MOVE: Rc8", "e8c8", "san"),
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "candidate MOVE: a2a3 then MOVE: e1g1", "e1g1", "fromto"),
        # last MOVE: line wins over earlier drafts
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: h2h4 MOVE: d2f3", "d2f3", "fromto"),
        # line-anchored bare from-to
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "the answer is\nd2f3", "d2f3", "fromto"),
        # prose containing a move string must NOT parse (no MOVE:, not a line)
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "e2e4 would be bad here.", None, None),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
         "MOVE: O-O", "e1g1", "san"),
        ("8/1P6/8/8/8/8/8/k1K5 w - - 0 1",
         "MOVE: b8=Q", "b7b8q", "san"),
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "", None, None),
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: z9z9", None, None),   # malformed squares never parse
    ]
    for fen, out, exp_uci, exp_fmt in cases:
        board = board_from_fen(fen) if fen else None
        uci, fmt = parse_move_output(out, board)
        check(f"extract {out!r}",
              uci == exp_uci and fmt == exp_fmt,
              f"got {(uci, fmt)} want {(exp_uci, exp_fmt)}")


def main() -> None:
    test_dataset_standard_chess()
    test_answer_extraction()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES", flush=True)
        sys.exit(1)
    print("\nALL TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
