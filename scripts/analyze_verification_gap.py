"""Verification-gap measurement (rlvr-plan.md §5, the paper claim).

Per eval sample: does the model's OWN process-verified trace correlate
with correctness? Reuses the training-time process reward exactly
(train_mate_grpo._verify_trace + the stockfish oracle), so the analysis
scores the same "verify your reasoning" signal RL optimizes.

    python3 scripts/analyze_verification_gap.py \
        --samples results/rlvr-probe/<model>_<task>.samples.jsonl \
        [--include-reasoning] [--out results/verification-gap.json]

Input: run_mate_eval samples.jsonl rows {position_metadata:{fen,
truth_label,candidate_a,candidate_b}, output, reasoning, correct}.
Verified fraction is computed over output only (the completion RL
sees) unless --include-reasoning adds the thinking channel — for
thinking-ON gemma runs, reasoning+output is the model's full trace.

The claim to test: verified_fraction should predict correctness (high
verified -> correct at a higher rate than base accuracy).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_mate_grpo import Oracle, _verify_trace  # noqa: E402


def _row_text(row: dict, include_reasoning: bool) -> str:
    out = row.get("output") or ""
    if include_reasoning:
        reason = row.get("reasoning") or ""
        return (reason + "\n" + out).strip()
    return out


def _bucket(v: float) -> str:
    if v <= 0.0:
        return "0 (nothing verified)"
    if v < 0.5:
        return "(0, 0.5)"
    if v < 1.0:
        return "[0.5, 1)"
    return "1 (fully verified)"


def _corr(xs: list[float], ys: list[float]) -> float:
    """Pearson r (point-biserial for the 0/1 correctness vector)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True,
                    help="run_mate_eval samples.jsonl")
    ap.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--depth", type=int, default=12,
                    help="engine depth, same as training (d12)")
    ap.add_argument("--include-reasoning", action="store_true",
                    help="verify over reasoning+output (thinking-ON full "
                         "trace) instead of output only")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out", default="",
                    help="optional json summary path")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.samples).read_text().splitlines()
            if l.strip()]
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    print(f"samples: {len(rows)} (reasoning included: "
          f"{args.include_reasoning})", flush=True)

    oracle = Oracle("stockfish", depth=args.depth, stockfish=args.stockfish)
    memo: dict[str, float] = {}

    def eval_cp_memo(fen: str) -> float | None:
        if fen not in memo:
            memo[fen] = oracle.eval_cp(fen)
        return memo[fen]

    orig_eval = oracle.eval_cp
    oracle.eval_cp = eval_cp_memo  # type: ignore[method-assign]

    verified: list[float] = []
    correct: list[float] = []
    parsed = 0
    skipped = 0
    for i, row in enumerate(rows):
        meta = row.get("position_metadata") or {}
        fen = meta.get("fen")
        text = _row_text(row, args.include_reasoning)
        if not fen or not text:
            skipped += 1
            continue
        # process reward on the model's own trace; empty -> 0.0 by the
        # same rule the trainer applies
        v = _verify_trace(text, fen, oracle)
        if text.strip():
            parsed += 1
        verified.append(v)
        correct.append(1.0 if row.get("correct") else 0.0)
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(rows)}", flush=True)

    oracle.eval_cp = orig_eval  # type: ignore[method-assign]
    oracle.close()

    if not verified:
        print("no scorable rows (missing fen/output) — nothing to report")
        return

    n = len(verified)
    base_acc = sum(correct) / n
    r = _corr(verified, correct)
    print(f"\nscored {n} samples ({skipped} skipped, {parsed} with text)")
    print(f"base accuracy: {base_acc:.3f}")
    print(f"mean verified fraction: {sum(verified) / n:.3f}")
    print(f"correlation(verified_fraction, correct): {r:+.3f} "
          f"(point-biserial)")

    buckets: dict[str, list[int]] = {}
    for v, c in zip(verified, correct):
        b = buckets.setdefault(_bucket(v), [0, 0])
        b[0] += 1
        b[1] += int(c)
    print("\nbucket                  n    accuracy")
    for b in sorted(buckets):
        n_b, ok = buckets[b]
        print(f"{b:<22} {n_b:>4}   {ok / n_b:.3f}")

    if args.out:
        out = {
            "n": n, "base_accuracy": base_acc,
            "mean_verified_fraction": sum(verified) / n,
            "correlation": r, "buckets": {
                b: {"n": buckets[b][0], "correct": buckets[b][1]}
                for b in buckets},
        }
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nsummary written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
