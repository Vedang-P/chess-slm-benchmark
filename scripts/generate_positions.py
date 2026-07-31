"""Generate the anti-goal benchmark position datasets.

Deterministic (seeded) and committed to the repo so Kaggle runs never
regenerate oracles at runtime. Output: data/positions/*.json.

Usage:
    python scripts/generate_positions.py [--out data/positions] [--n 40]
    python scripts/generate_positions.py --check   # tiny n for the check notebook
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.games import positions as P  # noqa: E402

TASKS = [
    # (name, generator callable kwargs)
    ("sm-3x3-win", lambda n, seed: P.single_move_positions(3, seed=seed, n_positions=n, want_value="win")),
    ("sm-3x3-draw", lambda n, seed: P.single_move_positions(3, seed=seed, n_positions=n, want_value="draw")),
    ("sm-5x5-win", lambda n, seed: P.single_move_positions(5, seed=seed, n_positions=n, want_value="win")),
    ("sm-5x5-draw", lambda n, seed: P.single_move_positions(5, seed=seed, n_positions=n, want_value="draw")),
    ("mate1-8x8", lambda n, seed: P.mate1_positions(8, seed=seed, n_positions=n, max_pieces=6)),
    ("mob-8x8", lambda n, seed: P.mobility_positions(8, seed=seed, n_positions=n, max_pieces=6)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/positions")
    ap.add_argument("--n", type=int, default=40, help="positions per task (default 40)")
    ap.add_argument("--check", action="store_true", help="tiny dataset for notebook check mode")
    args = ap.parse_args()

    n = 3 if args.check else args.n
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    total = time.time()
    for name, gen in TASKS:
        t = time.time()
        positions = gen(n, seed=hash(name) % (2**31))
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(positions, indent=1))
        values = [p["value"] for p in positions]
        summary[name] = {
            "n": len(positions),
            "values": {v: values.count(v) for v in set(values)},
            "seconds": round(time.time() - t, 1),
        }
        print(f"{name}: {len(positions)} positions {summary[name]['seconds']}s", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"total {time.time() - total:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
