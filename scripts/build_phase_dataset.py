"""Phase classifier + position sampler for the MATE train corpus.

Phase classification for chess ML is under-specified in the literature; we
publish this deterministic classifier with the dataset. Boundaries are
documented, monotone, and cheap to compute from a FEN.

Phase rules (in order):
  1. ENDGAME   — non-king material <= 13 points AND <= 6 non-king pieces.
                 (Standard "reduced material" endgame heuristic; K+P vs K+P
                 etc. are obviously endgames regardless of moves.)
  2. OPENING   — fullmove_number <= 12 AND non-king pieces >= 16.
                 (Book/mobilization phase: both sides still near full force
                 and the game is young. The piece guard prevents a quick
                 tactical annihilation at move 8 from being called an
                 opening.)
  3. MIDDLEGAME — everything else (pieces >= 7 or moves > 12).
  4. SPARSE (flag, not a phase): non-king pieces <= 3 — ultra-thin
                 (e.g., K+P vs K) — flagged so the sampler can optionally
                 oversample or exclude them; many engines/tablebases treat
                 these specially.

Material weights (standard): P=1 N=3 B=3 R=5 Q=9.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import chess

MATERIAL = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}
ENDGAME_MATERIAL_MAX = 13
ENDGAME_PIECES_MAX = 6
OPENING_MOVES_MAX = 12
OPENING_PIECES_MIN = 16
SPARSE_PIECES_MAX = 3


def fen_parts(fen: str) -> list[str]:
    return fen.strip().split()


def classify_fen(fen: str) -> str:
    """Return 'opening' | 'middlegame' | 'endgame' (or 'sparse' for the
    ultra-thin flag). Callers that want only the 3-way phase should map
    'sparse' -> 'endgame' (they are endgames by any definition)."""
    board = chess.Board(fen)
    pts = 0
    n_pieces = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.piece_type == chess.KING:
            continue
        n_pieces += 1
        pts += MATERIAL.get(p.symbol().upper(), 0)
    if n_pieces <= SPARSE_PIECES_MAX:
        return "sparse"
    if pts <= ENDGAME_MATERIAL_MAX and n_pieces <= ENDGAME_PIECES_MAX:
        return "endgame"
    if board.fullmove_number <= OPENING_MOVES_MAX and n_pieces >= OPENING_PIECES_MIN:
        return "opening"
    return "middlegame"


def _fen_from_mate_input(input_text: str) -> str | None:
    m = re.search(r'"([^"]+)"', input_text)
    return m.group(1) if m else None


def _move_from_mate_output(output: str) -> str | None:
    m = re.search(r"Move([AB]):\s*([a-hNBRQK][a-h1-8xO-]{1,7})", output)
    return f"{m.group(1)}:{m.group(2)}" if m else None


def scan_mate_file(path: Path, limit: int | None = None,
                   max_records: int | None = None) -> list[dict]:
    """Scan one MATE subset jsonl into deduped phase-tagged records."""
    seen = set()
    rows = []
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            fen = _fen_from_mate_input(d.get("input", ""))
            if not fen:
                continue
            if fen in seen:
                continue
            seen.add(fen)
            phase = classify_fen(fen)
            rows.append({
                "fen": fen,
                "phase": phase,
                "move": _move_from_mate_output(d.get("output", "")),
                "input": d.get("input", ""),
                "subset": path.parent.name,
            })
            n += 1
            if max_records and n >= max_records:
                break
    return rows


def load_test_fens() -> set[str]:
    """FENs of the committed MATE test sets — must never appear in a
    training/benchmark split (contamination hygiene)."""
    test = set()
    for name in ("mate-selection-test.json", "mate-selection-test-noexplain.json",
                 "mate-selection-test-tactic.json", "mate-selection-test-both.json"):
        p = Path("data/positions") / name
        if p.exists():
            for r in json.loads(p.read_text()):
                test.add(r["fen"])
    return test


def build_sample(subset: str, out: Path, n_per_phase: int,
                 include_sparse: bool = False, seed: int = 42,
                 exclude_fens: set[str] | None = None) -> dict:
    """Sample n_per_phase positions per phase from one MATE subset,
    deduped by FEN (globally, across phases) and excluded from the
    provided test FEN set, and write the phase-tagged manifest.

    Returns the per-phase counts actually achieved (fewer if the subset
    does not contain enough of a phase)."""
    rng = random.Random(seed)
    subset_dir = Path("data/raw/mate-train") / subset
    jsonl_paths = sorted(subset_dir.glob("*/*.jsonl"))
    if not jsonl_paths:
        raise RuntimeError(f"no jsonl under {subset_dir}")
    buckets: dict[str, list[dict]] = {p: [] for p in
                                      ("opening", "middlegame", "endgame", "sparse")}
    seen_fens = set(exclude_fens) if exclude_fens else set()
    for p in jsonl_paths:
        for row in scan_mate_file(p):
            if row["fen"] in seen_fens:
                continue
            seen_fens.add(row["fen"])
            buckets[row["phase"]].append(row)

    out_manifest = {
        "classifier_version": "v1",
        "rules": {
            "endgame": (f"material<={ENDGAME_MATERIAL_MAX} and "
                        f"nonking_pieces<={ENDGAME_PIECES_MAX}"),
            "opening": (f"fullmove<={OPENING_MOVES_MAX} and "
                        f"nonking_pieces>={OPENING_PIECES_MIN}"),
            "middlegame": "else",
            "sparse": f"nonking_pieces<={SPARSE_PIECES_MAX} (flag)",
        },
        "subset": subset,
        "seed": seed,
        "n_per_phase": n_per_phase,
        "phases": {},
    }
    kept = {}
    for phase in ("opening", "middlegame", "endgame"):
        pool = buckets[phase]
        if phase == "endgame" and include_sparse:
            pool = pool + buckets["sparse"]
        rng.shuffle(pool)
        chosen = pool[:n_per_phase]
        kept[phase] = chosen
        out_manifest["phases"][phase] = len(chosen)

    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"{subset}.phase-manifest.json"
    manifest_path.write_text(json.dumps(out_manifest, indent=1))

    rows = []
    for phase, recs in kept.items():
        for r in recs:
            rows.append({"phase": phase, "fen": r["fen"],
                         "move": r["move"], "subset": r["subset"]})
    rng.shuffle(rows)
    (out / f"{subset}.positions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} positions -> {out / (subset + '.positions.jsonl')}")
    for k, v in out_manifest["phases"].items():
        print(f"  {k}: {v}")
    return out_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="strategy",
                    choices=["strategy", "noexplain", "tactic", "both"])
    ap.add_argument("--out", default="data/positions/phase-train")
    ap.add_argument("--n-per-phase", type=int, default=2000)
    ap.add_argument("--include-sparse", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_sample(args.subset, Path(args.out), args.n_per_phase,
                 include_sparse=args.include_sparse, seed=args.seed,
                 exclude_fens=load_test_fens())


if __name__ == "__main__":
    main()
