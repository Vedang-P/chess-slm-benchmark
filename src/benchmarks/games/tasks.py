"""Task definitions: prompt construction, output parsing, oracle-based
compliance scoring for the anti-goal benchmark.

Every metric is computed against EXTERNAL oracles embedded in the position
records (see positions.py): no model self-judgment anywhere.

Taxonomy per (position, condition) sample:
  status: no_answer | parse_error | illegal | legal
  compliance (only for legal): True/False/None (None = condition undefined,
  e.g. LOSE on an already-lost position)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from src.benchmarks.games.prompts import (
    build_cap_prompt,
    build_mate1_prompt,
    build_mobility_prompt,
    build_single_move_prompt,
)
from src.benchmarks.games.rules import algebraic_to_sq

_MOVE_RE = re.compile(r"([a-h][1-8])\s*[- ]?\s*([a-h][1-8])")
_SAN_RE = re.compile(r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?(?:[+#])?")
_COLOR_PREFIX_RE = re.compile(r"\b[wb](?=[KQRBN])")


def _uci(fr: tuple, to: tuple) -> str:
    return (f"{chr(ord('a') + fr[1])}{fr[0] + 1}"
            f"{chr(ord('a') + to[1])}{to[0] + 1}")


def _from_to(text: str) -> Optional[str]:
    m = _MOVE_RE.search(text)
    if not m:
        return None
    fr, to = algebraic_to_sq(m.group(1)), algebraic_to_sq(m.group(2))
    if fr is None or to is None:
        return None
    return _uci(fr, to)


def _san_to_uci(text: str, board) -> Optional[str]:
    """Resolve SAN-style moves (Nf3, Rc8, Kb8, b7b8Q, e2e4) against OUR
    engine's legal moves. Returns the uci if exactly one legal move matches.
    Handles piece-letter moves, from-to with promotion suffix, and a leading
    color prefix ('bKb8' -> Kb8) that the model may add."""
    for token in _SAN_RE.findall(text):
        t = token.rstrip("#+")
        m = re.match(r"^(?:(K|Q|R|B|N|P)?)([a-h]?)([1-8]?)?x?([a-h])([1-8])(?:=([QRBN]))?$", t)
        if not m:
            # color-prefixed king move: 'bKb8'
            stripped = _COLOR_PREFIX_RE.sub("", text)
            if stripped != text:
                return _san_to_uci(stripped, board)
            continue
        kind, df, dr, tf, tr = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        target = (int(tr) - 1, ord(tf) - ord("a"))
        cands = []
        for mv in board.legal_moves():
            if mv.to != target:
                continue
            if kind and mv.piece != kind:
                continue
            if df and mv.fr[1] != ord(df) - ord("a"):
                continue
            if dr and mv.fr[0] != int(dr) - 1:
                continue
            cands.append(mv.uci)
        if len(cands) == 1:
            return cands[0]
    return None


def parse_move_output(text: str, board=None) -> tuple:
    """Parse a model move. Returns (uci, fmt) where fmt is 'fromto', 'san',
    or None. From-to is tried first (the format the prompt demands); SAN
    resolution is the lenient fallback that needs the board (models like
    Gemma 4 answer 'Nf3' / 'Rc8' / 'bKb8#')."""
    if not text:
        return (None, None)
    uci = _from_to(text)
    if uci:
        return (uci, "fromto")
    if board is not None:
        uci = _san_to_uci(text, board)
        if uci:
            return (uci, "san")
    return (None, None)


def _board(rec: Dict[str, object]):
    from src.benchmarks.games.rules import Board

    pieces = {(int(p["sq"][1:]) - 1, ord(p["sq"][0]) - ord("a")): (p["color"], p["kind"])
              for p in rec["pieces"]}
    return Board(rec["n"], pieces, rec["turn"])



# ---------------------------------------------------------------------- #
# single-move task (3x3/5x5, exact game values)
# ---------------------------------------------------------------------- #
def score_single_move(rec: Dict[str, object], condition: str,
                      model_text: str) -> Dict[str, object]:
    board = _board(rec)
    uci, fmt = parse_move_output(model_text, board)
    if uci is None:
        return {"status": "no_answer" if not model_text.strip() else "parse_error", "format": fmt}
    legal = {m.uci for m in board.legal_moves()}
    if uci not in legal:
        return {"status": "illegal", "move": uci, "format": fmt}
    if condition == "win":
        compliant = uci in set(rec["win_moves"])
    else:  # lose
        if rec["value"] == "loss":
            compliant = None  # already lost: nothing to worsen
        else:
            compliant = uci in set(rec["lose_moves"])
    return {"status": "legal", "move": uci, "compliance": compliant, "format": fmt}


# ---------------------------------------------------------------------- #
# mate-in-1 task (8x8)
# ---------------------------------------------------------------------- #
def score_mate1(rec: Dict[str, object], condition: str,
                model_text: str) -> Dict[str, object]:
    board = _board(rec)
    uci, fmt = parse_move_output(model_text, board)
    if uci is None:
        return {"status": "no_answer" if not model_text.strip() else "parse_error", "format": fmt}
    legal = {m.uci for m in board.legal_moves()}
    if uci not in legal:
        return {"status": "illegal", "move": uci, "format": fmt}
    mates = set(rec["task_extra"]["mate_moves"])
    if condition == "win":
        compliant = uci in mates
    else:  # lose: avoid the mate
        compliant = uci not in mates
    return {"status": "legal", "move": uci, "compliance": compliant, "format": fmt}


# ---------------------------------------------------------------------- #
# max/min opponent mobility task (8x8)
# ---------------------------------------------------------------------- #
def score_mobility(rec: Dict[str, object], condition: str,
                   model_text: str) -> Dict[str, object]:
    board = _board(rec)
    uci, fmt = parse_move_output(model_text, board)
    if uci is None:
        return {"status": "no_answer" if not model_text.strip() else "parse_error", "format": fmt}
    legal = {m.uci for m in board.legal_moves()}
    if uci not in legal:
        return {"status": "illegal", "move": uci, "format": fmt}
    stats = {s["move"]: s["opp_replies"] for s in rec["task_extra"]["mobility"]["moves"]}
    replies = stats[uci]
    target = min(stats.values()) if condition == "win" else max(stats.values())
    return {"status": "legal", "move": uci, "compliance": replies == target,
            "format": fmt, "opp_replies": replies}


# ---------------------------------------------------------------------- #
# cap task (pure legality probe, no objective)
# ---------------------------------------------------------------------- #
def score_cap(rec: Dict[str, object], condition: str,
              model_text: str) -> Dict[str, object]:
    board = _board(rec)
    uci, fmt = parse_move_output(model_text, board)
    if uci is None:
        return {"status": "no_answer" if not model_text.strip() else "parse_error", "format": fmt}
    legal = {m.uci for m in board.legal_moves()}
    if uci not in legal:
        return {"status": "illegal", "move": uci, "format": fmt}
    return {"status": "legal", "move": uci, "compliance": None, "format": fmt}


# ---------------------------------------------------------------------- #
# best-move task (8x8, Stockfish ground truth from lichess eval DB / cloud)
# ---------------------------------------------------------------------- #
def score_bestmove(rec: Dict[str, object], condition: str,
                   model_text: str) -> Dict[str, object]:
    board = _board(rec)
    uci, fmt = parse_move_output(model_text, board)
    if uci is None:
        return {"status": "no_answer" if not model_text.strip() else "parse_error", "format": fmt}
    legal = {m.uci for m in board.legal_moves()}
    if uci not in legal:
        return {"status": "illegal", "move": uci, "format": fmt}
    best = rec["task_extra"]["best_move"]
    return {"status": "legal", "move": uci, "compliance": uci == best, "format": fmt}


# ---------------------------------------------------------------------- #
# mate-in-2 task (8x8, lichess puzzles, mateIn2 theme)
# ---------------------------------------------------------------------- #
def score_mate2(rec: Dict[str, object], condition: str,
                model_text: str) -> Dict[str, object]:
    board = _board(rec)
    uci, fmt = parse_move_output(model_text, board)
    if uci is None:
        return {"status": "no_answer" if not model_text.strip() else "parse_error", "format": fmt}
    legal = {m.uci for m in board.legal_moves()}
    if uci not in legal:
        return {"status": "illegal", "move": uci, "format": fmt}
    first = rec["task_extra"]["first_move"]  # the lichess 'only move' of the mate line
    return {"status": "legal", "move": uci, "compliance": uci == first, "format": fmt}


SCORERS = {
    "sm": score_single_move,
    "mate1": score_mate1,
    "mate2": score_mate2,
    "mob": score_mobility,
    "cap": score_cap,
    "bestmove": score_bestmove,
}

PROMPT_BUILDERS = {
    "sm": build_single_move_prompt,
    "mate1": build_mate1_prompt,
    "mate2": build_mate1_prompt,  # same strong-move framing; solution checked in scorer
    "mob": build_mobility_prompt,
    "cap": build_cap_prompt,
    "bestmove": build_single_move_prompt,
}

CONDITIONS = ("win", "lose")
CAP_CONDITIONS = ("win",)  # cap tasks have no objective; single condition


def task_kind(record_id: str) -> str:
    return record_id.split("-")[0]


def score_record(rec: Dict[str, object], condition: str, model_text: str,
                 kind: Optional[str] = None) -> Dict[str, object]:
    """kind defaults to the record-id prefix (works for generated records);
    pass it explicitly when record ids don't encode the task (e.g. lichess)."""
    return SCORERS[kind or task_kind(rec["id"])](rec, condition, model_text)


def get_correct(rec: Dict[str, object], kind: str) -> Optional[dict]:
    """The oracle answer for a record, for the live viewer. Returns
    {"move": uci-or-None, "note": human explanation}."""
    extra = rec.get("task_extra") or {}
    if kind == "bestmove":
        return {"move": extra.get("best_move"), "note": "stockfish best move"}
    if kind == "mate1":
        mates = extra.get("mate_moves") or []
        return {"move": mates[0] if mates else None,
                "note": "any move delivering checkmate"}
    if kind == "mate2":
        return {"move": extra.get("first_move"), "note": "first move of the mate line"}
    if kind == "sm":
        wins = rec.get("win_moves") or []
        if wins:
            return {"move": wins[0], "note": "a game-theoretically winning move"}
        return {"move": None, "note": "no winning move in this position"}
    if kind == "mob":
        stats = (extra.get("mobility") or {}).get("moves") or []
        if stats:
            target = min(s["opp_replies"] for s in stats)
            move = next(s["move"] for s in stats if s["opp_replies"] == target)
            return {"move": move, "note": "move minimizing opponent replies"}
        return {"move": None, "note": ""}
    if kind == "cap":
        return {"move": None, "note": "any legal move"}
    return {"move": None, "note": ""}