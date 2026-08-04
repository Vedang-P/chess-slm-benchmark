"""Task definitions: parsing and oracle-based scoring for the chess study.

Rules engine: STANDARD chess via python-chess — the same library the
lichess ecosystem uses. No custom engine, no simplified variant.

Answer extraction is deliberately STRICT. For every sample we record the
exact model text; parsing happens in this priority order:

1. `MOVE:`-prefixed from-to moves — the format the prompt demands. We take
   the LAST such token (models sometimes draft candidates before the final
   answer). Example: "...MOVE: e2e4" -> e2e4.
2. A line whose entire content is a from-to move (no prose around it).
3. SAN via python-chess `parse_san`, which by construction only resolves
   to a LEGAL move in the position (disambiguation, promotion, castling,
   check/mate suffixes included).

Nothing is ever invented: if no move parses, the sample is a parse_error
or no_answer. There is NO fallback that fabricates an answer — an answer
recorded on a sample is always the model's own text.

The only residual risk is intent (a model writing "not MOVE: e2e4"), which
the prompt's output spec prevents; we document this rather than hide it.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import chess

from src.benchmarks.games.prompts import (
    build_mate1_prompt,
    build_mate2_prompt,
    build_single_move_prompt,
)

# a from-to move with optional promotion piece, e.g. e2e4, b7b8q
_MOVE_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?")
_MOVE_LINE_RE = re.compile(r"^\s*[a-h][1-8][a-h][1-8][qrbnQRBN]?\s*$")
_SAN_RE = re.compile(r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?(?:[+#])?|[O0]-[O0](?:-[O0])?")
_COLOR_PREFIX_RE = re.compile(r"\b[wb](?=[KQRBN])")

TASK_CATEGORIES = {
    "mate1-lichess": "Short Tactics",
    "mate2-lichess": "Short Tactics",
    "bestmove-8x8": "Position Judgment",
    "mate-selection-test": "Short Tactics",
}


def task_category(task_name: str) -> str:
    return TASK_CATEGORIES.get(task_name, "Uncategorized")


def _board(rec: Dict[str, object]) -> chess.Board:
    return chess.Board(rec.get("presented_fen") or rec.get("fen"))


def _legal_ucis(board: chess.Board) -> set:
    return {m.uci() for m in board.legal_moves}


def _moves_with_prefix(text: str) -> List[str]:
    """All from-to moves preceded by MOVE: (case-insensitive), in order.
    \\b anchors the keyword so 'remove: e2e4' is not read as an answer."""
    return re.findall(r"\bMOVE\s*[:.\-]?\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)",
                      text, re.I)


def _san_to_uci(text: str, board: chess.Board) -> Optional[str]:
    """Resolve SAN tokens (Nf3, Rc8, Kb8, b7b8Q, O-O, e2e4) with
    python-chess's own parser. Only legal moves can come out of it. A
    leading color prefix ('bKb8' -> Kb8) is stripped first.

    The LAST legal token wins, for the same reason the MOVE: rule takes the
    last match: a model that weighs candidates before committing writes its
    answer last. Taking the first legal token scored the first move a model
    *considered*, not the one it chose."""
    stripped = _COLOR_PREFIX_RE.sub("", text)
    if stripped != text:
        return _san_to_uci(stripped, board)
    resolved = None
    for token in _SAN_RE.findall(stripped):
        t = token.rstrip("#+")
        if not t:
            continue
        try:
            resolved = board.parse_san(t).uci()
        except ValueError:
            continue
    return resolved


def _no_move_status(model_text: Optional[str], api_error: Optional[str]) -> Dict[str, object]:
    """Classify a sample with no parseable move. An api_error is a TRANSPORT
    failure (gateway 5xx, timeout, rate limit) — it is not the model failing
    to answer, and counting it as parse_error understates parse/legal rates
    with infrastructure noise. It gets its own status and is excluded from
    every denominator in src/report.aggregate_samples."""
    if api_error:
        return {"status": "api_error", "format": None, "error": api_error}
    return {"status": "no_answer" if not (model_text or "").strip() else "parse_error",
            "format": None}


def parse_move_output(text: str, board: chess.Board = None) -> tuple:
    """Extract the model's move. Returns (uci, fmt) with fmt in
    fromto / san / None. Strict priority; never invents a move."""
    if not text:
        return (None, None)

    # 1) last MOVE:-prefixed from-to token (the demanded output format)
    prefixed = _moves_with_prefix(text)
    if prefixed:
        return (prefixed[-1].lower(), "fromto")

    # 2) a whole line that is exactly a from-to move
    for line in text.splitlines():
        if _MOVE_LINE_RE.match(line):
            return (line.strip().lower(), "fromto")

    # 3) SAN resolved against the actual position (always a legal move)
    if board is not None:
        uci = _san_to_uci(text, board)
        if uci:
            return (uci, "san")
    return (None, None)


# ---------------------------------------------------------------------- #
# mate-in-1 task (8x8, python-chess)
# ---------------------------------------------------------------------- #
def score_mate1(rec: Dict[str, object], condition: str,
                model_text: str, api_error: Optional[str] = None) -> Dict[str, object]:
    board = _board(rec)
    uci, fmt = parse_move_output(model_text, board)
    if uci is None:
        return _no_move_status(model_text, api_error)
    legal = _legal_ucis(board)
    if uci not in legal:
        return {"status": "illegal", "move": uci, "format": fmt}
    mates = set(rec["task_extra"]["mate_moves"])
    compliant = uci in mates
    return {"status": "legal", "move": uci, "compliance": compliant, "format": fmt}


# ---------------------------------------------------------------------- #
# best-move task (8x8, Stockfish ground truth from lichess eval DB)
# ---------------------------------------------------------------------- #
def score_bestmove(rec: Dict[str, object], condition: str,
                   model_text: str, api_error: Optional[str] = None) -> Dict[str, object]:
    board = _board(rec)
    uci, fmt = parse_move_output(model_text, board)
    if uci is None:
        return _no_move_status(model_text, api_error)
    legal = _legal_ucis(board)
    if uci not in legal:
        return {"status": "illegal", "move": uci, "format": fmt}
    best = rec["task_extra"]["best_move"]
    return {"status": "legal", "move": uci, "compliance": uci == best, "format": fmt}


# ---------------------------------------------------------------------- #
# mate-in-2 task (8x8, lichess puzzles, mateIn2 theme; python-chess)
# ---------------------------------------------------------------------- #
def score_mate2(rec: Dict[str, object], condition: str,
                model_text: str, api_error: Optional[str] = None) -> Dict[str, object]:
    board = _board(rec)
    uci, fmt = parse_move_output(model_text, board)
    if uci is None:
        return _no_move_status(model_text, api_error)
    legal = _legal_ucis(board)
    if uci not in legal:
        return {"status": "illegal", "move": uci, "format": fmt}
    first = rec["task_extra"]["first_move"]  # the lichess 'only move' of the mate line
    return {"status": "legal", "move": uci, "compliance": uci == first, "format": fmt}


SCORERS = {
    "mate1": score_mate1,
    "mate2": score_mate2,
    "bestmove": score_bestmove,
}

PROMPT_BUILDERS = {
    "mate1": build_mate1_prompt,
    "mate2": build_mate2_prompt,  # asks for the first move of the forced mate
    "bestmove": build_single_move_prompt,
}

CONDITIONS = ("win",)


def task_kind(record_id: str) -> str:
    return record_id.split("-")[0]


def score_record(rec: Dict[str, object], condition: str, model_text: str,
                 kind: Optional[str] = None,
                 api_error: Optional[str] = None) -> Dict[str, object]:
    return SCORERS[kind or task_kind(rec["id"])](rec, condition, model_text,
                                                 api_error)


def get_correct(rec: Dict[str, object], kind: str) -> Optional[dict]:
    """The oracle answer for a record, for the live viewer."""
    extra = rec.get("task_extra") or {}
    if kind == "bestmove":
        return {"move": extra.get("best_move"), "note": "stockfish best move"}
    if kind == "mate1":
        mates = extra.get("mate_moves") or []
        return {"move": mates[0] if mates else None,
                "note": "any move delivering checkmate"}
    if kind == "mate2":
        return {"move": extra.get("first_move"), "note": "first move of the mate line"}
    return {"move": None, "note": ""}
