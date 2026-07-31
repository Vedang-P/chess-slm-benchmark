"""Run the model x task x prompt-variant matrix and produce the comparison table.

Usage:
    python scripts/run_suite.py                # full sweep (paper data)
    python scripts/run_suite.py --check        # tiny sanity sweep
    python scripts/run_suite.py --smoke        # stub models, no GPU
    python scripts/run_suite.py --models deepseek-r1-distill-qwen-1.5b smollm2-1.7b
    python scripts/run_suite.py --tasks mate1-lichess cap-legal-8x8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.report import write_comparison_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--output_dir", default="results/chess")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "suite.yaml").read_text())
    mode = cfg["check"] if args.check else cfg["full"]
    models = args.models or ([mode["models"][0]] if args.check else cfg["models"])
    tasks = args.tasks or list(cfg["tasks"])
    max_tokens = mode.get("max_new_tokens", 512)

    cells = []
    for task in tasks:
        t = cfg["tasks"][task]
        for variant in t.get("variants", ["grid"]):
            cells.append((task, variant))

    print(f"suite: {len(models)} models x {len(cells)} task-variant cells "
          f"({'CHECK' if args.check else 'FULL'} mode)", flush=True)
    rows = []
    t0 = time.time()
    for model in models:
        for task, variant in cells:
            n = cfg["tasks"][task]["check_n"] if args.check else cfg["tasks"][task]["full_n"]
            cmd = [
                sys.executable, str(ROOT / "scripts" / "run_chess.py"),
                "--model", model, "--task", task, "--prompt-variant", variant,
                "--n", str(n), "--max_new_tokens", str(max_tokens),
                "--output_dir", str(ROOT / args.output_dir),
            ]
            if args.smoke:
                cmd.append("--smoke")
            print(f"\n>>> {model} x {task}:{variant} (n={n})", flush=True)
            t = time.time()
            res = subprocess.run(cmd, cwd=ROOT)
            if res.returncode != 0:
                print(f"!!! {model} x {task}:{variant} FAILED rc={res.returncode}", flush=True)
                continue
            summary = json.loads(
                (ROOT / args.output_dir / f"{model}_{task}_{variant}.summary.json").read_text()
            )
            for cond, m in summary["metrics"]["conditions"].items():
                rows.append({
                    "model": model, "task": task, "variant": variant, "condition": cond,
                    "n": m["n"], "parse_rate": m["parse_rate"],
                    "legal_rate": m["legal_rate"],
                    "compliance_of_legal": m["compliance_of_legal"],
                    "compliance_strict": m["compliance_strict"],
                    "undefined": m["undefined"],
                })
            div = summary["metrics"].get("divergence_rate")
            if div is not None:
                rows.append({
                    "model": model, "task": task, "variant": variant,
                    "condition": "divergence", "n": "", "parse_rate": "",
                    "legal_rate": "", "compliance_of_legal": div,
                    "compliance_strict": "", "undefined": "",
                })
            print(f"    {time.time() - t:.0f}s", flush=True)
    csv_path = write_comparison_csv(ROOT / args.output_dir, rows)
    print(f"\ncomparison table: {csv_path} ({len(rows)} rows) "
          f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
