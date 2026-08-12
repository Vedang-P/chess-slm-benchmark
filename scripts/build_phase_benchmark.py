"""Build the phase-stratified benchmark holdout from MATE train positions.

The benchmark is the PRIMARY artifact of the phase-segregation project: a
phase-stratified instrument (opening/middlegame/endgame) for rating any
chess model, with Stockfish ground truth and per-position metadata.

Design:
  - 300 positions per phase per source (strategy + tactic) = 1,800 total.
  - FEN-deduped against the phase-train set AND the MATE test sets.
  - Ground truth: the MATE expert move (kept) + Stockfish best move + eval
    (computed at build time, cached in the manifest).
  - Metadata per position: phase, source subset, fullmove, material,
    non-king piece count, halfmove clock.

    python3 scripts/build_phase_benchmark.py --out data/positions/phase-benchmark
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_phase_dataset import (  # noqa: E402
    classify_fen,
    load_test_fens,
    scan_mate_file,
)

N_PER_PHASE_PER_SOURCE = 300
SOURCES = ["strategy", "tactic"]
PHASES = ["opening", "middlegame", "endgame"]


def stockfish_best(board: chess.Board, depth: int = 10,
                   timeout_s: float = 10.0):
    """Best move + eval from a local stockfish binary (engines/). Returns
    None if no binary is available."""
    import chess.engine

    for cand in (Path("engines/stockfish"), Path("engines/stockfish-mac"),
                 Path("/usr/local/bin/stockfish"), Path("/opt/homebrew/bin/stockfish")):
        if cand.exists():
            try:
                eng = chess.engine.SimpleEngine.popen_uci(str(cand))
                try:
                    info = eng.analyse(board, chess.engine.Limit(
                        depth=depth, time=timeout_s))
                    score = info.get("score")
                    pv = info.get("pv") or []
                    cp = None
                    if score is not None:
                        # score is PovScore: .relative for the side to move
                        try:
                            rel = score.relative
                            if rel.is_mate():
                                cp = 10000 if rel.mate() > 0 else -10000
                            else:
                                cp = rel.score()
                        except Exception:
                            cp = None
                    return {
                        "best_move": pv[0].uci() if pv else None,
                        "cp": cp,
                        "depth": info.get("depth"),
                    }
                finally:
                    eng.quit()
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}
    return None


def material_stats(board: chess.Board) -> dict:
    pts = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}
    total, n = 0, 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type != chess.KING:
            total += pts.get(p.symbol().upper(), 0)
            n += 1
    return {"material": total, "nonking_pieces": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/positions/phase-benchmark")
    ap.add_argument("--n", type=int, default=N_PER_PHASE_PER_SOURCE)
    ap.add_argument("--sources", default=",".join(SOURCES),
                    help="comma-separated MATE subsets (positions overlap "
                         "between subsets; tactic ⊂ strategy ~72%, so "
                         "strategy-only avoids near-duplicate benchmark rows)")
    ap.add_argument("--exclude-train", default="data/positions/phase-train",
                    help="dir of training positions.jsonl to exclude")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sources = [s for s in args.sources.split(",") if s]
    excluded = load_test_fens()
    if args.exclude_train:
        for p in Path(args.exclude_train).glob("*.positions.jsonl"):
            for line in p.read_text().splitlines():
                if line.strip():
                    excluded.add(json.loads(line)["fen"])

    # bucket all sources by phase
    buckets: dict[str, dict[str, list[dict]]] = {
        s: {p: [] for p in PHASES} for s in sources}
    seen_global = set(excluded)
    for source in sources:
        for p in sorted((Path("data/raw/mate-train") / source).glob("*/*.jsonl")):
            for row in scan_mate_file(p):
                if row["fen"] in seen_global:
                    continue
                seen_global.add(row["fen"])
                ph = classify_fen(row["fen"])
                if ph == "sparse":
                    ph = "endgame"
                if ph in buckets[source]:
                    buckets[source][ph].append(row)
        print(f"  scanned {source}: " + ", ".join(
            f"{ph}={len(buckets[source][ph])}" for ph in PHASES), flush=True)

    records = []
    for source in sources:
        for ph in PHASES:
            pool = buckets[source][ph]
            rng.shuffle(pool)
            for row in pool[: args.n]:
                board = chess.Board(row["fen"])
                stats = material_stats(board)
                sf = stockfish_best(board)
                records.append({
                    "id": f"{source}-{ph}-{len(records):04d}",
                    "fen": row["fen"],
                    "phase": ph,
                    "source": source,
                    "mate_move": row["move"],
                    "fullmove": board.fullmove_number,
                    "halfmove_clock": board.halfmove_clock,
                    **stats,
                    "stockfish": sf,
                })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    manifest = {
        "classifier_version": "v1",
        "n_per_phase_per_source": args.n,
        "sources": sources,
        "phases": PHASES,
        "seed": args.seed,
        "excluded": {
            "mate_test_sets": True,
            "phase_train_dir": str(args.exclude_train),
        },
        "counts": {},
    }
    from collections import Counter

    c = Counter((r["source"], r["phase"]) for r in records)
    for (s, p), n in sorted(c.items()):
        manifest["counts"][f"{s}/{p}"] = n
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(records)} benchmark positions -> {out}")
    for k, v in manifest["counts"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
