"""Run the MATE benchmark across N parallel workers, then merge.

NOTE (measured 2026-08-03): the opencode-go gateway SERIALIZES heavy
generations per API key. Three concurrent 2048-budget requests: one served
in 22s, the other two queued to 82-89s and returned EMPTY content (budget
enforcement broke under queueing). Parallel workers do not increase
throughput and degrade quality. Keep --workers 1 unless the gateway's
concurrency behavior changes.
"""

Each worker scores a disjoint shard of the record set (via --offset/--n)
into its own output dir. When all workers finish, the shards are merged
into a single canonical samples.jsonl + summary.json, which is then
uploaded to the Hugging Face results repo.

Usage:
    python scripts/run_mate_parallel.py --workers 16 --n 1000
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
from src.report import aggregate_token_usage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--output_dir", default="results/mate-selection")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--thinking-budget", type=int, default=2048)
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    shards_dir = out_root / "shards"
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True)

    chunk = (args.n + args.workers - 1) // args.workers
    procs = []
    for w in range(args.workers):
        offset = w * chunk
        if offset >= args.n:
            break
        n = min(chunk, args.n - offset)
        shard_dir = shards_dir / f"shard-{w}"
        cmd = [
            sys.executable, str(ROOT / "scripts" / "run_mate_eval.py"),
            "--model", args.model, "--n", str(n), "--offset", str(offset),
            "--thinking-budget", str(args.thinking_budget),
            "--output_dir", str(shard_dir),
            "--live-push",
        ]
        log = open(shards_dir / f"shard-{w}.log", "w")
        procs.append((w, offset, n, subprocess.Popen(cmd, stdout=log, stderr=log), log))

    t0 = time.time()
    done_counts = {}
    while True:
        for w, offset, n, proc, log in procs:
            if proc.poll() is not None and w not in done_counts:
                done_counts[w] = True
                print(f"shard {w} done ({time.time()-t0:.0f}s)", flush=True)
        done = sum(1 for w, _, _, proc, _ in procs if proc.poll() is not None)
        if done == len(procs):
            break
        time.sleep(10)
    for _, _, _, proc, log in procs:
        proc.wait()
        log.close()

    # merge shards
    samples = {}
    for w, offset, n, _, _ in procs:
        shard_samples = shards_dir / f"shard-{w}" / f"{args.model}_mate-selection-test_strategy.samples.jsonl"
        if not shard_samples.exists():
            continue
        for line in shard_samples.read_text().splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            samples[s["position_id"]] = s
    ordered = [samples[k] for k in sorted(samples)]
    run_name = f"{args.model}_mate-selection-test_strategy"
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / f"{run_name}.samples.jsonl").open("w") as f:
        for s in ordered:
            f.write(json.dumps(s) + "\n")

    parsed = [s for s in ordered if s["status"] != "parse_error"]
    accuracy = {
        "n": len(ordered),
        "parse_rate": round(len(parsed) / len(ordered), 4),
        "accuracy_strict": round(sum(s["compliance"] for s in ordered) / len(ordered), 4),
        "accuracy_of_parsed": round(sum(s["compliance"] for s in parsed) / len(parsed), 4) if parsed else None,
        "correct": sum(s["compliance"] for s in ordered),
        "wrong": sum(not s["compliance"] for s in ordered),
        "parse_error": sum(s["status"] == "parse_error" for s in ordered),
    }
    metrics = {"accuracy": accuracy, "token_usage": aggregate_token_usage(ordered)}
    from src.report import ResultWriter

    writer = ResultWriter(out_root, run_name, {
        "model": args.model, "task": "mate-selection-test",
        "prompt_variant": "strategy", "task_category": "Short Tactics",
        "run_id": ordered[0].get("run_id") if ordered else None,
        "smoke": False, "thinking_enabled": True,
        "thinking_budget": args.thinking_budget,
        "parallel_workers": args.workers,
    })
    for s in ordered:
        writer.add(s)
    writer.finish(metrics)
    print(f"merged {len(ordered)} samples -> {out_root / run_name}.summary.json", flush=True)
    print(json.dumps(metrics, indent=1), flush=True)
    try:
        from src.hf_push import upload_cell

        upload_cell(out_root, ordered[0].get("run_id") or "parallel", writer.summary_path)
    except Exception as e:
        print(f"hf upload skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
