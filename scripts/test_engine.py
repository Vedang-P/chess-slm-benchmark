"""Engine + dataset invariant tests. Runs on CPU anywhere (no torch).

Usage:  python scripts/test_engine.py            # full
        python scripts/test_engine.py --quick    # fast subset (check notebook)
Exit code 0 = all pass; any failure raises with a message.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.games.oracles import (
    checkmate_moves,
    clear_cache,
    mobility_stats,
    solve,
)
from src.benchmarks.games.rules import Board, algebraic_to_sq, sq_to_algebraic

FAILURES: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILURES.append(f"{name}: {detail}")
        print(f"FAIL {name} {detail}", flush=True)
    else:
        print(f"ok   {name}", flush=True)


def test_movegen_known() -> None:
    b = Board(8, {(7, 7): ("b", "K"), (5, 6): ("w", "Q"), (6, 5): ("w", "K")}, "w")
    mates = [m.uci for m in checkmate_moves(b)]
    check("8x8 mate-in-1 Qg7-style", "g6g7" in mates, f"mates={mates}")

    b = Board(3, {(0, 0): ("w", "K"), (2, 2): ("b", "K")}, "w")
    check("3x3 KvK 2 legal moves", len(b.legal_moves()) == 2, f"n={len(b.legal_moves())}")


def test_square_helpers() -> None:
    check("sq_to_algebraic a1", sq_to_algebraic((0, 0)) == "a1")
    check("algebraic_to_sq h8", algebraic_to_sq("h8") == (7, 7))
    check("algebraic_to_sq junk", algebraic_to_sq("zz") is None)
    check("algebraic_to_sq short", algebraic_to_sq("a") is None)


def test_known_values() -> None:
    cases = [
        ("KvK draw", Board(3, {(0, 0): ("w", "K"), (2, 2): ("b", "K")}, "w"), 0),
        ("3x3 KQvK win", Board(3, {(0, 0): ("w", "K"), (2, 2): ("b", "K"), (0, 1): ("w", "Q")}, "w"), 1),
        ("5x5 KQvK win", Board(5, {(0, 0): ("w", "K"), (4, 4): ("b", "K"), (0, 1): ("w", "Q")}, "w"), 1),
        ("blocked pawn draw", Board(5, {(0, 0): ("w", "K"), (1, 1): ("w", "P"), (2, 1): ("b", "K")}, "w"), 0),
        ("bK can capture, draw", Board(5, {(1, 0): ("w", "K"), (3, 1): ("w", "P"), (4, 1): ("b", "K")}, "b"), 0),
        ("far pawn promotes, win", Board(5, {(0, 0): ("w", "K"), (1, 1): ("w", "P"), (4, 4): ("b", "K")}, "w"), 1),
        ("KQvKQ 3x3 draw", Board(3, {(0, 0): ("w", "K"), (2, 2): ("b", "K"), (0, 1): ("w", "Q"), (2, 1): ("b", "Q")}, "w"), 0),
    ]
    for name, b, expect in cases:
        clear_cache()
        r = solve(b)
        check(f"value {name}", r.value == expect, f"got {r.value} want {expect}")


def test_win_lose_move_semantics() -> None:
    b = Board(5, {(0, 0): ("w", "K"), (4, 4): ("b", "K"), (0, 1): ("w", "Q")}, "w")
    clear_cache()
    r = solve(b)
    legal = {m.uci for m in b.legal_moves()}
    check("win moves subset of legal", set(r.win_moves) <= legal)
    check("lose moves subset of legal", set(r.lose_moves) <= legal)
    check("win and lose disjoint", not (set(r.win_moves) & set(r.lose_moves)))
    check("win position has win moves", len(r.win_moves) >= 1)
    check("win position has non-win moves", len(r.win_moves) < len(legal))


def test_dataset_invariants() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data" / "positions"
    for path in sorted(data_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        recs = json.loads(path.read_text())
        check(f"dataset {path.stem} non-empty", len(recs) >= 3)
        for rec in recs:
            pid = rec["id"]
            check(f"{pid} has oracle data", "win_moves" in rec and "lose_moves" in rec)
            if path.stem.startswith("sm"):
                check(f"{pid} value matches oracles",
                      (len(rec["win_moves"]) > 0) == (rec["value"] == "win"),
                      f"value={rec['value']} win_moves={len(rec['win_moves'])}")
                check(f"{pid} lose moves non-trivial", len(rec["lose_moves"]) >= 1)
                if rec["value"] == "win":
                    check(f"{pid} win non-vacuous",
                          len(rec["win_moves"]) < _n_legal_moves(rec),
                          "all moves win")
            if path.stem.startswith("mate1"):
                extra = rec["task_extra"]
                check(f"{pid} has mate moves", len(extra["mate_moves"]) >= 1)
                check(f"{pid} mate non-vacuous",
                      len(extra["mate_moves"]) < _n_legal_moves(rec))
            if path.stem.startswith("mate2"):
                extra = rec["task_extra"]
                check(f"{pid} has first_move", "first_move" in extra)
                from src.benchmarks.games.rules import Board

                pieces = {(int(p["sq"][1:]) - 1, ord(p["sq"][0]) - ord("a")): (p["color"], p["kind"])
                          for p in rec["pieces"]}
                b = Board(rec["n"], pieces, rec["turn"])
                check(f"{pid} first_move legal", extra["first_move"] in {m.uci for m in b.legal_moves()})
                check(f"{pid} mate2 non-vacuous", len(b.legal_moves()) >= 2)
            if path.stem.startswith("bestmove"):
                extra = rec["task_extra"]
                check(f"{pid} has best_move", "best_move" in extra)
                from src.benchmarks.games.rules import Board

                pieces = {(int(p["sq"][1:]) - 1, ord(p["sq"][0]) - ord("a")): (p["color"], p["kind"])
                          for p in rec["pieces"]}
                b = Board(rec["n"], pieces, rec["turn"])
                legal = {m.uci for m in b.legal_moves()}
                check(f"{pid} best_move legal in our variant",
                      extra["best_move"] in legal, f"best={extra['best_move']}")
                check(f"{pid} bestmove non-vacuous", len(legal) >= 2)
            if path.stem.startswith("mob"):
                s = rec["task_extra"]["mobility"]
                check(f"{pid} mobility varies", s["min"] < s["max"])
    clear_cache()


def _n_legal_moves(rec) -> int:
    from src.benchmarks.games.rules import Board

    pieces = {(int(p["sq"][1:]) - 1, ord(p["sq"][0]) - ord("a")): (p["color"], p["kind"])
              for p in rec["pieces"]}
    b = Board(rec["n"], pieces, rec["turn"])
    return len(b.legal_moves())


def test_fuzz_legality(rounds: int = 200) -> None:
    import random

    rng = random.Random(42)
    for _ in range(rounds):
        n = rng.choice([3, 5])
        pieces = {}
        sqs = [(r, c) for r in range(n) for c in range(n)]
        rng.shuffle(sqs)
        pieces[sqs[0]] = ("w", "K")
        pieces[sqs[1]] = ("b", "K")
        for sq in sqs[2:]:
            if len(pieces) >= 5 or rng.random() < 0.6:
                break
            pieces[sq] = (rng.choice("wb"), rng.choice("QRBNP"))
        b = Board(n, pieces, rng.choice("wb"))
        for m in b.legal_moves()[:40]:
            after = b.apply(m)
            if after.in_check(b.turn):  # the MOVER's king must not be in check
                FAILURES.append(f"fuzz: move {m.uci} leaves own king in check")
                return
            wk, bk = after.king_square("w"), after.king_square("b")
            if wk and bk and max(abs(wk[0] - bk[0]), abs(wk[1] - bk[1])) <= 1:
                FAILURES.append(f"fuzz: adjacent kings after {m.uci}")
                return
    check("fuzz legality 200 rounds", not FAILURES or all("fuzz" not in f for f in FAILURES))


def test_python_chess_parity() -> None:
    """Cross-validate 8x8 legality + mate detection against python-chess.
    Skipped (silently passes) if python-chess is not installed."""
    try:
        import chess  # noqa
    except ImportError:
        print("ok   python-chess parity (skipped: not installed)")
        return
    from src.benchmarks.games.fen import parse_fen, fen_of_board

    data_dir = Path(__file__).resolve().parent.parent / "data" / "positions"
    recs = json.loads((data_dir / "mate1-lichess.json").read_text())[:40]

    # 1) legal-move parity on the lichess positions. Our documented variant
    #    excludes double-step pawn pushes and en-passant; everything else
    #    must agree with python-chess exactly.
    mismatches = 0
    double_steps = 0
    for rec in recs:
        board = parse_fen(rec["fen"])
        ours = {m.uci for m in board.legal_moves()}
        theirs = {m.uci() for m in chess.Board(rec["fen"]).legal_moves}
        for uci in theirs - ours:
            fr_r, fr_c = int(uci[1]) - 1, ord(uci[0]) - ord("a")
            to_r, to_c = int(uci[3]) - 1, ord(uci[2]) - ord("a")
            piece = board.at((fr_r, fr_c))
            # documented variant differences: no double-step pushes, no
            # castling (puzzle FENs carry castling rights python-chess honors)
            if piece and piece[1] == "P" and abs(to_r - fr_r) == 2:
                double_steps += 1
                continue
            if piece and piece[1] == "K" and abs(to_c - fr_c) == 2:
                double_steps += 1
                continue
            mismatches += 1
            if mismatches <= 2:
                print(f"  parity diff at {rec['id']}: ours-only={ours - theirs} "
                      f"theirs-only={uci}")
        if ours - theirs:
            mismatches += 1
            print(f"  ours-extra at {rec['id']}: {ours - theirs}")
    check("python-chess legal-move parity (8x8)", mismatches == 0,
          f"{mismatches} mismatches ({double_steps} double-step pushes, documented variant)")

    # 2) mate-move parity on the presented positions
    mate_mismatch = 0
    for rec in recs:
        board = parse_fen(rec["fen"])
        first = rec["task_extra"]["presented_after"]
        m0 = next(m for m in board.legal_moves() if m.uci == first)
        presented = board.apply(m0)
        ours_mates = {m.uci for m in checkmate_moves(presented)}
        cb = chess.Board(rec["fen"])
        cb.push(chess.Move.from_uci(first))
        theirs_mates = {m.uci() for m in cb.legal_moves
                        if cb.copy().push(m) or (cb.is_checkmate() and True)}
        # python-chess: is_checkmate() is valid only after the move; recompute:
        theirs_mates = set()
        for m in cb.legal_moves:
            nb = cb.copy()
            nb.push(m)
            if nb.is_checkmate():
                theirs_mates.add(m.uci())
        if ours_mates != theirs_mates:
            mate_mismatch += 1
            if mate_mismatch <= 2:
                print(f"  mate diff at {rec['id']}: ours={ours_mates} theirs={theirs_mates}")
    check("python-chess mate-move parity (8x8)", mate_mismatch == 0, f"{mate_mismatch} mismatches")


def test_san_parsing() -> None:
    """SAN/lenient move parsing — the format models like Gemma actually
    produce ('Nf3', 'Rc8', 'bKb8#', 'b7b8Q'). Regression cases from the
    check run."""
    from src.benchmarks.games.tasks import _board, parse_move_output

    def board_from_fen(fen):
        from src.benchmarks.games.fen import parse_fen

        return parse_fen(fen)

    cases = [
        # (fen, output, expected_uci, expected_fmt)
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: d2f3", "d2f3", "fromto"),   # knight f3, from-to form
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: Nf3", "d2f3", "san"),       # gemma-style SAN
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "I think Rc8 is best.", None, None),  # black's move: SAN fails (white to move)
        ("2kr1b1r/p1p2pp1/2pqN3/7p/6n1/2NPB3/PPP2PPP/R2Q1RK1 b - - 0 1",
         "MOVE: bKb8#", "c8b8", "san"),     # color-prefixed king move
        ("4r3/1k6/pp3P2/1b5p/3R1p2/P1R2P2/1P4PP/6K1 b - - 0 1",
         "MOVE: Rc8", "e8c8", "san"),       # rook e8-c8 along rank 8 is legal
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: e7-e5", "e7e5", "fromto"),  # parses as from-to (illegal later)
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: b7b8Q", "b7b8", "fromto"),  # promotion suffix stripped
        ("2kr1b1r/p1p2pp1/2pqb3/7p/3N2n1/2NPB3/PPP2PPP/R2Q1RK1 w - - 2 13",
         "MOVE: d6h2", "d6h2", "fromto"),   # bishop mate
    ]
    for fen, out, exp_uci, exp_fmt in cases:
        b = board_from_fen(fen)
        uci, fmt = parse_move_output(out, b)
        check(f"san-parse {out!r}",
              uci == exp_uci and fmt == exp_fmt,
              f"got {(uci, fmt)} want {(exp_uci, exp_fmt)}")

    # 3x3 small boards: from-to regex catches 'Ka1-b1' inside the text — fine
    from src.benchmarks.games.rules import Board

    b3 = Board(3, {(0, 0): ("w", "K"), (2, 2): ("b", "K")}, "w")
    uci, fmt = parse_move_output("MOVE: Ka1-b1", b3)
    check("san 3x3 king move", uci == "a1b1", f"got {(uci, fmt)}")


def main() -> None:
    quick = "--quick" in sys.argv
    test_square_helpers()
    test_movegen_known()
    test_known_values()
    test_win_lose_move_semantics()
    test_dataset_invariants()
    test_python_chess_parity()
    test_san_parsing()
    if not quick:
        t0 = time.time()
        test_fuzz_legality()
        print(f"fuzz took {time.time() - t0:.1f}s", flush=True)
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES", flush=True)
        sys.exit(1)
    print("\nALL TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
