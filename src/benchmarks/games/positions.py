"""Position generation for the anti-goal benchmark.

Rejection sampling over random legal material/placement, with explicit
legality filters and non-vacuity conditions checked against the oracles.
The ORACLE DATA IS EMBEDDED in each position record (computed once at
generation time, by solve()): runtime compliance checks are O(1) lookups.

Position record (JSON-safe):
{"id", "n", "turn", "value", "pieces": [{"sq","color","kind"}, ...],
 "win_moves": [uci], "lose_moves": [uci], "over_budget": bool,
 "task_extra": {...}}   # task-specific (mate moves / mobility stats)
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from src.benchmarks.games.oracles import (
    checkmate_moves,
    mobility_stats,
    solve,
)
from src.benchmarks.games.rules import Board, COLORS, PIECES, Square

PIECE_WEIGHTS = {"Q": 1, "R": 2, "B": 2, "N": 2, "P": 3}
MAX_ATTEMPTS = 40000


def _place_kings(n: int, rng: random.Random) -> Tuple[Square, Square]:
    for _ in range(1000):
        wk = (rng.randrange(n), rng.randrange(n))
        bk = (rng.randrange(n), rng.randrange(n))
        if wk == bk:
            continue
        if max(abs(wk[0] - bk[0]), abs(wk[1] - bk[1])) <= 1:
            continue
        return wk, bk
    raise RuntimeError("king placement failed")


def _random_material(rng: random.Random, max_pieces: int) -> List[Tuple[str, str]]:
    total = rng.randint(1, max_pieces)
    return [
        (rng.choice(COLORS), rng.choices(list(PIECE_WEIGHTS), weights=list(PIECE_WEIGHTS.values()))[0])
        for _ in range(total)
    ]


def _scatter(
    n: int, rng: random.Random, pieces: List[Tuple[str, str]],
    blocked: List[Square],
) -> Optional[Dict[Square, Tuple[str, str]]]:
    squares = [(r, c) for r in range(n) for c in range(n) if (r, c) not in blocked]
    rng.shuffle(squares)
    if len(squares) < len(pieces):
        return None
    return {sq: piece for sq, piece in zip(squares, pieces)}


def _base_position(
    n: int, rng: random.Random, extra_pieces: List[Tuple[str, str]],
    turn: Optional[str] = None,
) -> Optional[Board]:
    wk, bk = _place_kings(n, rng)
    scattered = _scatter(n, rng, extra_pieces, [wk, bk])
    if scattered is None:
        return None
    pieces = dict(scattered)
    pieces[wk] = ("w", "K")
    pieces[bk] = ("b", "K")
    board = Board(n, pieces, turn if turn is not None else rng.choice(COLORS))
    if board.in_check(board.turn):
        return None
    if board.in_check("b" if board.turn == "w" else "w"):
        return None  # side not to move was left in check
    terminal, _ = board.outcome()
    if terminal:
        return None
    return board


def _record(board: Board, pid: str, res, task_extra: Optional[dict] = None) -> Dict[str, object]:
    rec = {
        "id": pid,
        "n": board.n,
        "turn": board.turn,
        "value": {1: "win", 0: "draw", -1: "loss"}.get(res.value, "unsolved"),
        "pieces": [
            {"sq": f"{chr(ord('a') + c)}{r + 1}", "color": color, "kind": kind}
            for (r, c), (color, kind) in sorted(board.pieces.items())
        ],
        "win_moves": res.win_moves,
        "lose_moves": res.lose_moves,
        "over_budget": res.over_budget,
    }
    if task_extra:
        rec["task_extra"] = task_extra
    return rec


def single_move_positions(
    n: int, seed: int, n_positions: int,
    want_value: str = "win",
) -> List[Dict[str, object]]:
    """Paired single-move task positions (3x3/5x5 only -- exact oracles).

    Material is pool-restricted per target value so generation is fast and
    every attempt has a high hit rate:
      - 'win':  side to move owns Q or R vs bare king (KQvK / KRvK are
                always exact wins on these boards).
      - 'draw': pawn(s) -- KPvK draws arise when the pawn is blockaded;
                on 3x3 the pool is widened to any piece since solves there
                are instant.

    Non-vacuity: at least one WIN move AND at least one non-WIN move must
    exist (so the LOSE condition is meaningful); for 'draw', at least one
    LOSE move must exist.
    """
    rng = random.Random(seed)
    pool = ("Q", "R") if want_value == "win" else ("P",)
    out: List[Dict[str, object]] = []
    attempts = 0
    while len(out) < n_positions and attempts < MAX_ATTEMPTS:
        attempts += 1
        if want_value == "win":
            turn = rng.choice(COLORS)
            extra = [(turn, rng.choice(pool))]
        else:
            turn = None
            extra = [(rng.choice(COLORS), rng.choice(pool))]
        board = _base_position(n, rng, extra, turn=turn)
        if board is None or not board.legal_moves():
            continue
        res = solve(board)
        if res is None or res.over_budget:
            continue
        value = {1: "win", 0: "draw", -1: "loss"}[res.value]
        if value != want_value:
            continue
        if value == "win":
            if not res.win_moves or len(res.win_moves) == len(board.legal_moves()):
                continue
        if value == "draw" and not res.lose_moves:
            continue
        out.append(_record(board, f"sm-{n}x{n}-{len(out):04d}", res))
    if len(out) < n_positions:
        raise RuntimeError(
            f"single_move_positions({n}, want={want_value}): only {len(out)}/{n_positions} "
            f"after {attempts} attempts"
        )
    return out


def mate1_positions(
    n: int, seed: int, n_positions: int, max_pieces: int,
) -> List[Dict[str, object]]:
    """8x8 puzzle task: side to move has >=1 checkmating move AND >=1 non-mate
    legal move. Material biased toward the side to move (else generation
    starves)."""
    rng = random.Random(seed)
    out: List[Dict[str, object]] = []
    attempts = 0
    while len(out) < n_positions and attempts < MAX_ATTEMPTS * 5:
        attempts += 1
        board = _mate1_candidate(n, rng, max_pieces)
        if board is None:
            continue
        mates = checkmate_moves(board)
        if not mates:
            continue
        if len(mates) == len(board.legal_moves()):
            continue
        rec = _record(board, f"mate1-{n}x{n}-{len(out):04d}", _dummy_solve())
        rec["task_extra"] = {"mate_moves": [m.uci for m in mates]}
        out.append(rec)
    if len(out) < n_positions:
        raise RuntimeError(f"mate1_positions: only {len(out)}/{n_positions} after attempts")
    return out


def _std_record(board, pid: str, task_extra: Optional[dict] = None) -> Dict[str, object]:
    """Record for an 8x8 position under STANDARD chess (python-chess)."""
    import chess

    rec = {
        "id": pid,
        "n": 8,
        "turn": "w" if board.turn == chess.WHITE else "b",
        "value": "cap",
        "fen": board.fen(),
        "presented_fen": board.fen(),
        "pieces": [
            {"sq": chess.square_name(sq),
             "color": "w" if board.piece_at(sq).color == chess.WHITE else "b",
             "kind": board.piece_at(sq).symbol().upper()}
            for sq in chess.SQUARES if board.piece_at(sq) is not None
        ],
        "win_moves": [],
        "lose_moves": [],
        "over_budget": False,
    }
    if task_extra:
        rec["task_extra"] = task_extra
    return rec


def _std_mate1_candidate(rng, max_pieces: int):
    """Random legal 8x8 position with a mate-in-1 (python-chess). Returns a
    chess.Board or None. Material biased toward the side to move."""
    import chess

    turn = rng.choice(COLORS)
    other = "b" if turn == "w" else "w"
    for _ in range(200):
        b = chess.Board(None)
        wk = chess.square(rng.randint(0, 7), rng.randint(0, 7))
        bk = chess.square(rng.randint(0, 7), rng.randint(0, 7))
        if wk == bk:
            continue
        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        b.turn = chess.WHITE if turn == "w" else chess.BLACK
        n_attack = rng.randint(2, min(max_pieces, 4))
        material = [("Q" if turn == "w" else "q")] + [
            (rng.choice(["R", "R", "B", "N"]) if turn == "w"
             else rng.choice(["r", "r", "b", "n"])) for _ in range(n_attack - 1)
        ]
        if rng.random() < 0.4:
            material.append(rng.choice(["B", "b", "N", "n", "P", "p"]))
        occupied = {wk, bk}
        ok = True
        for sym in material:
            for _ in range(50):
                sq = chess.square(rng.randint(0, 7), rng.randint(0, 7))
                if sq not in occupied:
                    b.set_piece_at(sq, chess.Piece.from_symbol(sym))
                    occupied.add(sq)
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        if not (b.king(chess.WHITE) and b.king(chess.BLACK)):
            continue
        if b.is_check() or b.is_game_over():
            continue
        # require at least one mating move and non-vacuity
        mates = []
        for mv in b.legal_moves:
            b.push(mv)
            if b.is_checkmate():
                mates.append(mv.uci())
            b.pop()
        if not mates or len(mates) == len(list(b.legal_moves)):
            continue
        return b, mates
    return None


def std_mate1_positions(
    n: int, seed: int, n_positions: int, max_pieces: int,
) -> List[Dict[str, object]]:
    """8x8 mate-in-1 task under STANDARD chess (python-chess). The custom
    engine's mate1_positions is only valid for NxN small boards."""
    rng = random.Random(seed)
    out: List[Dict[str, object]] = []
    attempts = 0
    while len(out) < n_positions and attempts < MAX_ATTEMPTS * 5:
        attempts += 1
        cand = _std_mate1_candidate(rng, max_pieces)
        if cand is None:
            continue
        board, mates = cand
        rec = _std_record(board, f"mate1-{n}x{n}-{len(out):04d}")
        rec["task_extra"] = {"mate_moves": mates}
        out.append(rec)
    if len(out) < n_positions:
        raise RuntimeError(f"std_mate1_positions: only {len(out)}/{n_positions} after attempts")
    return out


def std_mobility_positions(
    n: int, seed: int, n_positions: int, max_pieces: int,
) -> List[Dict[str, object]]:
    """8x8 opponent-mobility task under STANDARD chess (python-chess)."""
    import chess

    rng = random.Random(seed)
    out: List[Dict[str, object]] = []
    attempts = 0
    while len(out) < n_positions and attempts < MAX_ATTEMPTS * 3:
        attempts += 1
        b = chess.Board(None)
        wk = chess.square(rng.randint(0, 7), rng.randint(0, 7))
        bk = chess.square(rng.randint(0, 7), rng.randint(0, 7))
        if wk == bk:
            continue
        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        b.turn = rng.choice((chess.WHITE, chess.BLACK))
        occupied = {wk, bk}
        ok = True
        for _ in range(rng.randint(2, max_pieces - 2)):
            sym = rng.choice(["Q", "q", "R", "r", "B", "b", "N", "n", "P", "p"])
            for _ in range(50):
                sq = chess.square(rng.randint(0, 7), rng.randint(0, 7))
                if sq not in occupied:
                    b.set_piece_at(sq, chess.Piece.from_symbol(sym))
                    occupied.add(sq)
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        if b.is_check() or b.is_game_over():
            continue
        if len(list(b.legal_moves)) < 2:
            continue
        moves = []
        for mv in b.legal_moves:
            b.push(mv)
            replies = len(list(b.legal_moves))
            b.pop()
            moves.append({"move": mv.uci(), "opp_replies": replies})
        replies = [m["opp_replies"] for m in moves]
        if min(replies) == max(replies):
            continue
        rec = _std_record(b, f"mob-{n}x{n}-{len(out):04d}",
                          {"mobility": {"min": min(replies), "max": max(replies), "moves": moves}})
        out.append(rec)
    if len(out) < n_positions:
        raise RuntimeError(f"std_mobility_positions: only {len(out)}/{n_positions} after attempts")
    return out


class _Dummy:
    value = 0
    win_moves = []
    lose_moves = []
    over_budget = False


def _dummy_solve():
    return _Dummy()


def _mate1_candidate(n: int, rng: random.Random, max_pieces: int) -> Optional[Board]:
    turn = rng.choice(COLORS)
    other = "b" if turn == "w" else "w"
    wk, bk = _place_kings(n, rng)
    defender_extra: List[Tuple[str, str]] = []
    if rng.random() < 0.4:
        defender_extra.append((other, rng.choice(["B", "N", "P"])))
    n_attack = rng.randint(2, min(max_pieces, 4))
    attacker_extra = [(turn, "Q")] + [
        (turn, rng.choice(["R", "R", "B", "N"])) for _ in range(n_attack - 1)
    ]
    scattered = _scatter(n, rng, attacker_extra + defender_extra, [wk, bk])
    if scattered is None:
        return None
    pieces = dict(scattered)
    pieces[wk] = ("w", "K")
    pieces[bk] = ("b", "K")
    board = Board(n, pieces, turn)
    if board.in_check(turn):
        return None
    if board.in_check(other):
        return None
    terminal, _ = board.outcome()
    if terminal:
        return None
    return board


def mobility_positions(
    n: int, seed: int, n_positions: int, max_pieces: int,
) -> List[Dict[str, object]]:
    """8x8 puzzle task: opponent-mobility varies across legal moves."""
    rng = random.Random(seed)
    out: List[Dict[str, object]] = []
    attempts = 0
    while len(out) < n_positions and attempts < MAX_ATTEMPTS * 3:
        attempts += 1
        board = _base_position(n, rng, _random_material(rng, max_pieces))
        if board is None or not board.legal_moves():
            continue
        stats = mobility_stats(board)
        if len(stats["moves"]) < 2 or stats["min"] == stats["max"]:
            continue
        rec = _record(board, f"mob-{n}x{n}-{len(out):04d}", _dummy_solve())
        rec["task_extra"] = {"mobility": stats}
        out.append(rec)
    if len(out) < n_positions:
        raise RuntimeError(f"mobility_positions: only {len(out)}/{n_positions} after attempts")
    return out
