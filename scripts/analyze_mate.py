"""Position-clustered MATE analysis from per-example JSONL records.

MATE rows are not independent: 4,000 rows cover 2,952 unique positions, and
1,030 positions appear in more than one subset (often with different candidate
pairs). Reporting a plain row-level binomial interval overstates precision.
This tool reports:

  - row-level accuracy with a position-clustered bootstrap CI (resample unique
    positions, keep all rows of each sampled position),
  - per-subset accuracy and the position-multiplicity distribution,
  - paired comparisons between runs on the intersection of (file, row) keys,
    with a clustered bootstrap CI on the difference and an exact McNemar test.

Usage:
  python3 scripts/analyze_mate.py --run a@320k=m320k.jsonl --run b@1.6m=m1620k.jsonl \
      --out analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_run(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path}: no per-example records")
    for r in records:
        for key in ("position", "correct", "file", "row"):
            if key not in r:
                raise ValueError(f"{path}: record missing {key!r}: {r}")
    return records


def cluster_bootstrap(positions: np.ndarray, correct: np.ndarray,
                      rng: np.random.Generator, replicates: int) -> np.ndarray:
    """Row-level accuracy replicates, resampling unique positions with replacement."""
    unique, inverse = np.unique(positions, return_inverse=True)
    n_clusters = len(unique)
    out = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        weights = np.bincount(rng.integers(0, n_clusters, n_clusters),
                              minlength=n_clusters).astype(np.float64)
        row_w = weights[inverse]
        denom = row_w.sum()
        out[b] = (row_w * correct).sum() / denom if denom else np.nan
    return out


def ci(values: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, [100 * (1 - alpha) / 2, 100 * (1 + alpha) / 2])
    return float(lo), float(hi)


def summarize(records: list[dict], rng: np.random.Generator,
              replicates: int) -> dict:
    positions = np.asarray([r["position"] for r in records])
    correct = np.asarray([bool(r["correct"]) for r in records], dtype=np.float64)
    unique, counts = np.unique(positions, return_counts=True)
    per_file = {}
    for record in records:
        bucket = per_file.setdefault(record["file"], {"rows": 0, "correct": 0})
        bucket["rows"] += 1
        bucket["correct"] += int(bool(record["correct"]))
    for bucket in per_file.values():
        bucket["accuracy"] = bucket["correct"] / bucket["rows"]
    boot = cluster_bootstrap(positions, correct, rng, replicates)
    lo, hi = ci(boot)
    return {
        "rows": int(len(records)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "ci95_clustered": [lo, hi],
        "unique_positions": int(len(unique)),
        "multiplicity": {str(k): int(v) for k, v in
                         zip(*np.unique(counts, return_counts=True))},
        "per_file": per_file,
    }


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(a_records: list[dict], b_records: list[dict],
            rng: np.random.Generator, replicates: int) -> dict:
    a = {(r["file"], r["row"]): r for r in a_records}
    b = {(r["file"], r["row"]): r for r in b_records}
    common = sorted(set(a) & set(b))
    if not common:
        return {"common_rows": 0, "error": "no common (file, row) keys"}
    a_correct = np.asarray([bool(a[k]["correct"]) for k in common], dtype=np.float64)
    b_correct = np.asarray([bool(b[k]["correct"]) for k in common], dtype=np.float64)
    positions = np.asarray([a[k]["position"] for k in common])
    delta = b_correct - a_correct
    boot = cluster_bootstrap(positions, delta, rng, replicates)
    lo, hi = ci(boot)
    b_wins = int(((a_correct == 0) & (b_correct == 1)).sum())
    a_wins = int(((a_correct == 1) & (b_correct == 0)).sum())
    return {
        "common_rows": len(common),
        "a_accuracy": float(a_correct.mean()),
        "b_accuracy": float(b_correct.mean()),
        "delta_b_minus_a": float(delta.mean()),
        "ci95_clustered": [lo, hi],
        "mcnemar": {"a_correct_b_wrong": a_wins, "b_correct_a_wrong": b_wins,
                    "exact_p": mcnemar_exact(a_wins, b_wins)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="NAME=PATH.jsonl (repeatable)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--run must be NAME=PATH.jsonl, got {spec!r}")
        runs[name] = load_run(Path(path))

    rng = np.random.default_rng(args.seed)
    analysis = {"runs": {}, "pairs": []}
    for name, records in runs.items():
        analysis["runs"][name] = summarize(records, rng, args.bootstrap)
    names = list(runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            analysis["pairs"].append({
                "a": names[i], "b": names[j],
                **compare(runs[names[i]], runs[names[j]], rng, args.bootstrap)})

    for name, result in analysis["runs"].items():
        lo, hi = result["ci95_clustered"]
        print(f"{name}: {result['correct']}/{result['rows']} = "
              f"{100*result['accuracy']:.2f}%  95% clustered CI "
              f"[{100*lo:.2f}, {100*hi:.2f}]  "
              f"unique positions {result['unique_positions']}")
    for pair in analysis["pairs"]:
        if "error" in pair:
            print(f"{pair['a']} vs {pair['b']}: {pair['error']}")
            continue
        lo, hi = pair["ci95_clustered"]
        print(f"{pair['a']} vs {pair['b']}: delta={100*pair['delta_b_minus_a']:+.2f}pp "
              f"95% CI [{100*lo:+.2f}, {100*hi:+.2f}], common={pair['common_rows']}, "
              f"McNemar p={pair['mcnemar']['exact_p']:.4g}")
    if args.out:
        Path(args.out).write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
