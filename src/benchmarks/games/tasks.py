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


def parse_move_output(text: str) -> Optional[str]:
    """Extract a from-to move in algebraic notation ("e2e4" / "e2 e4" /
    "e2-e4") from a model response. Returns canonical uci or None."""
    if not text:
        return None
    m = _MOVE_RE.search(text)
    if not m:
        return None
    fr, to = algebraic_to_sq(m.group(1)), algebraic_to_sq(m.group(2))
    if fr is None or to is None:
        return None
    return f"{chr(ord('a') + fr[1])}{fr[0] + 1}{chr(ord('a') + to[1])}{to[0] + 1}"


def _legal_uci(rec: Dict[str, object]) -> set:
    from src.benchmarks.games.rules import Board

    pieces = {(ord(p["sq"][0]) - ord("a"), int(p["sq"][1:]) - 1): (p["color"], p["kind"])
              for p in rec["pieces"]}
    b = Board(rec["n"], pieces, rec["turn"])
    return {m.uci for m in b.legal_moves()}


# ---------------------------------------------------------------------- #
# single-move task (3x3/5x5, exact game values)
# ---------------------------------------------------------------------- #
def score_single_move(rec: Dict[str, object], condition: str,
                      model_text: str) -> Dict[str, object]:
    uci = parse_move_output(model_text)
    if uci is None:
        return {"status": "no_answer" if not model_text.strip() else "parse_error"}
    legal = _legal_uci(rec)
    if uci not in legal:
        return {"status": "illegal", "move": uci}
    if condition == "win":
        compliant = uci in set(rec["win_moves"])
    else:  # lose
        if rec["value"] == "loss":
            compliant = None  # already lost: nothing to worsen
        else:
            compliant = uci in set(rec["lose_moves"])
    return {"status": "legal", "move": uci, "compliance": compliant}


# ---------------------------------------------------------------------- #
# mate-in-1 task (8x8)
# ---------------------------------------------------------------------- #
def score_mate1(rec: Dict[str, object], condition: str,
                model_text: str) -> Dict[str, object]:
    uci = parse_move_output(model_text)
    if uci is None:
        return {"status": "no_answer" if not model_text.strip() else "parse_error"}
    legal = _legal_uci(rec)
    if uci not in legal:
        return {"status": "illegal", "move": uci}
    mates = set(rec["task_extra"]["mate_moves"])
    if condition == "win":
        compliant = uci in mates
    else:  # lose: avoid the mate
        compliant = uci not in mates
    return {"status": "legal", "move": uci, "compliance": compliant}


# ---------------------------------------------------------------------- #
# max/min opponent mobility task (8x8)
# ---------------------------------------------------------------------- #
def score_mobility(rec: Dict[str, object], condition: str,
                   model_text: str) -> Dict[str, object]:
    uci = parse_move_output(model_text)
    if uci is None:
        return {"status": "no_answer" if not model_text.strip() else "parse_error"}
    legal = _legal_uci(rec)
    if uci not in legal:
        return {"status": "illegal", "move": uci}
    stats = {s["move"]: s["opp_replies"] for s in rec["task_extra"]["mobility"]["moves"]}
    replies = stats[uci]
    target = min(stats.values()) if condition == "win" else max(stats.values())
    return {"status": "legal", "move": uci, "compliance": replies == target,
            "opp_replies": replies}


# ---------------------------------------------------------------------- #
# cap task (pure legality probe, no objective)
# ---------------------------------------------------------------------- #
def score_cap(rec: Dict[str, object], condition: str,
              model_text: str) -> Dict[str, object]:
    uci = parse_move_output(model_text)
    if uci is None:
        return {"status": "no_answer" if not model_text.strip() else "parse_error"}
    legal = _legal_uci(rec)
    if uci not in legal:
        return {"status": "illegal", "move": uci}
    return {"status": "legal", "move": uci, "compliance": None}


SCORERS = {
    "sm": score_single_move,
    "mate1": score_mate1,
    "mob": score_mobility,
    "cap": score_cap,
}

PROMPT_BUILDERS = {
    "sm": build_single_move_prompt,
    "mate1": build_mate1_prompt,
    "mob": build_mobility_prompt,
    "cap": build_cap_prompt,
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