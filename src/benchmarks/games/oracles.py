"""Exact game-theoretic oracles for the small-board tasks.

Method: RETROGRADE ANALYSIS over the reachable state space of the start
position (BFS + backward value propagation). This is the standard exact
technique for tiny endgames: shuffling lines are naturally draws, no horizon
heuristics, no alpha-beta pathologies. Bounded by state budgets so heavy
material (e.g. KQ vs KQ on 5x5, ~600k states) is reported as "over budget"
and callers skip such positions.

For every solved position we store, per legal move, whether it is a WIN move
(child value == -1, opponent loses), a LOSE move (strictly worsens the
position), or neither -- this is the anti-goal ground truth:

  game_value / classify     exact W/D/L for the side to move
  win_moves(board)          moves to a position where the opponent is losing
  lose_moves(board)         moves that strictly worsen the position
                            (empty when already lost)
  checkmate_moves           exact local-task oracle (8x8 puzzles)
  mobility_stats            exact local-task oracle (8x8 puzzles)
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from src.benchmarks.games.rules import Board, Move

STATE_BUDGET = 100_000
TIME_BUDGET_S = 30.0


class SolveResult:
    __slots__ = ("value", "win_moves", "lose_moves", "states", "over_budget")

    def __init__(self, value, win_moves, lose_moves, states, over_budget):
        self.value = value  # 1 win / 0 draw / -1 loss for side to move
        self.win_moves = win_moves  # List[str] uci
        self.lose_moves = lose_moves  # List[str] uci
        self.states = states
        self.over_budget = over_budget


_SOLVE_CACHE: Dict[str, SolveResult] = {}


def clear_cache() -> None:
    _SOLVE_CACHE.clear()


def solve(board: Board) -> Optional[SolveResult]:
    """Exact solve of the reachable state space. None if over budget."""
    start_key = board.key()
    cached = _SOLVE_CACHE.get(start_key)
    if cached is not None:
        return cached

    import time

    t0 = time.time()
    key_to_idx: Dict[str, int] = {start_key: 0}
    boards: List[Board] = [board]
    # edges[idx] = list of (child_idx, uci)
    edges: List[List] = [[]]
    rev: List[List] = [[]]
    queue = deque([0])
    while queue:
        if len(boards) >= STATE_BUDGET:
            _SOLVE_CACHE[start_key] = SolveResult(0, [], [], len(boards), True)
            return _SOLVE_CACHE[start_key]
        if time.time() - t0 > TIME_BUDGET_S:
            _SOLVE_CACHE[start_key] = SolveResult(0, [], [], len(boards), True)
            return _SOLVE_CACHE[start_key]
        idx = queue.popleft()
        b = boards[idx]
        for m in b.legal_moves():
            after = b.apply(m)
            ck = after.key()
            ci = key_to_idx.get(ck)
            if ci is None:
                ci = len(boards)
                key_to_idx[ck] = ci
                boards.append(after)
                edges.append([])
                rev.append([])
                queue.append(ci)
            edges[idx].append((ci, m.uci))
            rev[ci].append(idx)

    n = len(boards)
    # values: 1 win / -1 loss / 0 draw (side to move), -9 unresolved
    val = [-9] * n
    unknown = [0] * n
    saw_draw_child = [False] * n
    work: deque = deque()

    for i, b in enumerate(boards):
        terminal, winner = b.outcome()
        unknown[i] = len(edges[i])
        if terminal:
            if winner is None:
                val[i] = 0
            else:
                val[i] = 1 if winner == b.turn else -1
            work.append(i)
        elif unknown[i] == 0:
            val[i] = 0  # no legal moves, not terminal (defensive)
            work.append(i)

    while work:
        c = work.popleft()
        vc = val[c]
        for p in rev[c]:
            if val[p] != -9:
                continue
            if vc == -1:
                # child's side (the opponent) loses -> parent has a winning move
                val[p] = 1
                work.append(p)
                continue
            if vc == 0:
                # child is drawn -> parent has a drawing move
                saw_draw_child[p] = True
            unknown[p] -= 1
            if unknown[p] == 0:
                # no winning move (else resolved above); draw if any drawing
                # move exists, otherwise every move loses
                val[p] = -1 if not saw_draw_child[p] else 0
                work.append(p)

    # fill remaining (cycle) states as draws
    for i in range(n):
        if val[i] == -9:
            val[i] = 0

    start_edges = edges[0]
    win_moves = [uci for (ci, uci) in start_edges if val[ci] == -1]
    lose_moves = []
    for (ci, uci) in start_edges:
        # strictly worse for the side to move: -val_child < val_parent
        if -val[ci] < val[0]:
            lose_moves.append(uci)
    res = SolveResult(val[0], win_moves, lose_moves, n, False)
    _SOLVE_CACHE[start_key] = res
    return res


def game_value(board: Board) -> int:
    res = solve(board)
    if res is None or res.over_budget:
        return 0  # caller should have filtered over-budget positions
    return res.value


def win_moves(board: Board) -> List[str]:
    res = solve(board)
    return res.win_moves if res else []


def lose_moves(board: Board) -> List[str]:
    res = solve(board)
    return res.lose_moves if res else []


def classify(board: Board) -> str:
    res = solve(board)
    if res is None or res.over_budget:
        return "unsolved"
    return {1: "win", 0: "draw", -1: "loss"}[res.value]


def checkmate_moves(board: Board) -> List[Move]:
    out = []
    for m in board.legal_moves():
        after = board.apply(m)
        terminal, winner = after.outcome()
        if terminal and winner == board.turn:
            out.append(m)
    return out


def opponent_mobility(board: Board, m: Move) -> int:
    return len(board.apply(m).legal_moves())


def mobility_stats(board: Board) -> Dict[str, object]:
    stats = [{"move": m.uci, "opp_replies": opponent_mobility(board, m)}
             for m in board.legal_moves()]
    return {
        "moves": stats,
        "max": max(s["opp_replies"] for s in stats),
        "min": min(s["opp_replies"] for s in stats),
    }
