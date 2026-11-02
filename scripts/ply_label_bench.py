"""Does running several Stockfish processes at once change the 50 ms labels?

The first-decision labelers ran one engine on a four-core kernel. The later-ply
fleet runs four, so each 50 ms search may see less CPU. This measures the engines'
nodes per second alone and under load, then labels the same roots three ways (one
engine; four at once; four at once again) and compares them. Two identical
configurations still differ, because a time-limited search is not deterministic, so
the run-to-run gap between the two four-engine passes is the noise floor that the
one-versus-four gap has to be read against.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.label_ply_shard import _valid_engine, label_board, root_board  # noqa: E402
from scripts.ply_data import read_ids, shard_candidates  # noqa: E402


def mean_nps(stockfish: Path, boards: list[chess.Board], parallel: int, repeats: int = 20) -> float:
    """Mean nodes per second of 50 ms searches, `parallel` engines searching at once."""
    engines = [chess.engine.SimpleEngine.popen_uci(str(stockfish)) for _ in range(parallel)]
    try:
        for engine in engines:
            _valid_engine(engine)

        def run(index: int) -> float:
            engine, speeds = engines[index], []
            for k in range(repeats):
                info = engine.analyse(boards[(index + k) % len(boards)],
                                      chess.engine.Limit(time=0.05), info=chess.engine.INFO_ALL)
                speeds.append(float(info.get("nps", 0)))
            return float(np.mean(speeds))

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            return float(np.mean(list(pool.map(run, range(parallel)))))
    finally:
        for engine in engines:
            engine.close()


def label_all(stockfish: Path, boards: list[chess.Board], ids: list[str], parallel: int,
              tokenizer, utils) -> tuple[list[tuple[np.ndarray, np.ndarray]], float]:
    engines = [chess.engine.SimpleEngine.popen_uci(str(stockfish)) for _ in range(parallel)]
    started = time.monotonic()
    try:
        def run(index: int):
            return [label_board(boards[i], ids[i], engines[index], tokenizer, utils)[1:]
                    for i in range(index, len(boards), parallel)]

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            chunks = list(pool.map(run, range(parallel)))
    finally:
        for engine in engines:
            engine.close()
    out: list = [None] * len(boards)
    for index, chunk in enumerate(chunks):
        for k, labeled in enumerate(chunk):
            out[index + k * parallel] = labeled
    return out, time.monotonic() - started


def compare(a: list, b: list) -> dict:
    """Win-probability gaps between two labelings (b - a) and how often the best move agrees.

    A tie for the best value counts as agreement. The signed mean is the bias of `b` against
    `a`; the absolute mean and 95th percentile are the size of the disagreement.
    """
    abs_gaps, signed, per_root, agree = [], [], [], 0
    for (actions_a, wp_a), (actions_b, wp_b) in zip(a, b):
        if not np.array_equal(actions_a, actions_b):
            raise ValueError("labeled actions differ between passes")
        abs_gaps.append(np.abs(wp_b - wp_a))
        signed.append(wp_b - wp_a)
        per_root.append(float(np.abs(wp_b - wp_a).mean()))
        agree += int(wp_b[int(np.argmax(wp_a))] >= wp_b.max() - 1e-6)
    flat_abs, flat_signed = np.concatenate(abs_gaps), np.concatenate(signed)
    return {"mean_abs_winprob_gap": float(np.mean(per_root)),
            "mean_signed_winprob_gap": float(flat_signed.mean()),
            "p95_abs_winprob_gap": float(np.percentile(flat_abs, 95)),
            "best_move_agreement": agree / len(a), "moves": int(len(flat_abs))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stockfish", type=Path, required=True)
    p.add_argument("--sl-repo", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--val-csv", type=Path, required=True)
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--ply", type=int, default=3)
    p.add_argument("--roots", type=int, default=60)
    p.add_argument("--engines", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.sl_repo.resolve().parent))
    from searchless_chess.src import tokenizer, utils

    official = set(json.loads(args.manifest.read_text(encoding="utf-8"))["excluded_ids"])
    rows = [row for _, _, row in shard_candidates(
        args.source, official, read_ids(args.val_csv, args.test_csv), args.ply, 0)][:args.roots]
    boards = [root_board(row, args.ply) for row in rows]
    ids = [row["PuzzleId"] for row in rows]
    report = {"roots": len(boards), "engines": args.engines, "ply": args.ply,
              "nps_single": mean_nps(args.stockfish, boards, 1),
              "nps_parallel": mean_nps(args.stockfish, boards, args.engines)}
    report["nps_ratio"] = report["nps_parallel"] / max(1.0, report["nps_single"])
    one, t_one = label_all(args.stockfish, boards, ids, 1, tokenizer, utils)
    four, t_four = label_all(args.stockfish, boards, ids, args.engines, tokenizer, utils)
    again, t_again = label_all(args.stockfish, boards, ids, args.engines, tokenizer, utils)
    report.update(seconds={"one_engine": t_one, "parallel": t_four, "parallel_again": t_again},
                  one_vs_parallel=compare(one, four), parallel_vs_parallel=compare(four, again),
                  one_vs_parallel_again=compare(one, again))
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
