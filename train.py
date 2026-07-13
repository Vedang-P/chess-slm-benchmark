#!/usr/bin/env python3
"""Baseline evaluation harness for Gemma 4 E2B spatial reasoning.

Runs a model (base or a fine-tuned checkpoint) across GridRoute and Lost in
Aggregation, English-only, to establish/compare cross-benchmark performance.
This is the evaluation side of the GRPO fine-tuning project -- the actual
SFT/GRPO training follows Unsloth's documented Gemma 4 recipe directly
(https://unsloth.ai/docs/models/gemma-4/train) rather than a from-scratch
implementation here.

MazeEval integration is not yet built (no loader exists yet) -- GridRoute
and Lost in Aggregation are the two benchmarks currently wired up.

Entry point for both local smoke-testing (--smoke_test, CPU, stub model) and
real runs (local GPU via --backend ollama/hf, or Modal cloud).
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

if MODAL_AVAILABLE:
    app = modal.App("gemma4-spatial-reasoning")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "transformers>=5.13.0",
            "accelerate",
            "numpy",
            "sentencepiece",
            "protobuf",
            "huggingface_hub",
        )
        .add_local_dir("src", remote_path="/root/src")
        .add_local_dir("data", remote_path="/root/data")
        .add_local_python_source("hf_models")
    )

SEED = 42
MODEL_KEYS = ["deepseek-r1-distill-qwen-1.5b", "gemma4-e2b", "gemma4-e4b", "qwen2.5-1.5b", "qwen2.5-3b"]
BENCHMARKS = ["gridroute", "lost_in_aggregation"]
# deepseek-r1-distill-qwen-1.5b is AlphaMaze's own base model -- included as a
# replication sanity check (do we reproduce their reported numbers) before
# trusting results on models nobody's tested this recipe on yet.


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_gridroute_instances(n_tasks: int, seed: int = SEED):
    from src.grid_generator import generate_gridroute_maps
    tasks = generate_gridroute_maps(
        size=10, obstacle_size=3, num_obstacles=2,
        num_maps=100, pairs_per_map=5, seed=seed,
    )
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(tasks), size=min(n_tasks, len(tasks)), replace=False))
    instances = []
    for i in idx:
        t = tasks[i]
        instances.append({
            "task_id": t.task_id, "instruction": t.nl_variants["direct"],
            "start": t.start, "goal": t.goal, "grid": t.grid,
            "optimal_path": t.optimal_path, "optimal_length": t.optimal_length,
        })
    return instances


def load_lost_in_agg_instances(n_tasks: int, size: int = 7, seed: int = SEED):
    """Load Lost in Aggregation mazes. Instruction is templated from
    start/goal since the maze JSON doesn't include NL text directly."""
    path = Path("data/lost_in_aggregation") / f"size_{size}_150.json"
    with open(path) as f:
        mazes = json.load(f)
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(mazes), size=min(n_tasks, len(mazes)), replace=False))
    instances = []
    for i in idx:
        m = mazes[i]
        grid = np.array(m["grid"])
        h, w = grid.shape
        start = tuple(m.get("start", (1, 1)))
        goal = tuple(m.get("goal", (w - 2, h - 2)))
        instruction = (
            f"Navigate from ({start[0]},{start[1]}) to ({goal[0]},{goal[1]}) "
            f"on a {w}x{h} maze grid (1=wall, 0=path). Move up/down/left/right."
        )
        solution = m.get("topology", {}).get("shortest_path") or [list(start), list(goal)]
        optimal_path = [tuple(p) for p in solution]
        instances.append({
            "task_id": m["id"], "instruction": instruction,
            "start": start, "goal": goal, "grid": grid,
            "optimal_path": optimal_path, "optimal_length": len(optimal_path) - 1,
        })
    return instances


def run(args):
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    from hf_models import HFModel, OllamaModel, UnslothModel, parse_path_response
    from src.evaluation import PathResult, compute_metrics, print_report

    n_tasks = 3 if args.smoke_test else args.n_tasks
    model_keys = MODEL_KEYS[:1] if args.smoke_test else MODEL_KEYS
    benchmarks = BENCHMARKS[:1] if args.smoke_test else BENCHMARKS

    bench_instances = {}
    if "gridroute" in benchmarks:
        bench_instances["gridroute"] = load_gridroute_instances(n_tasks)
    if "lost_in_aggregation" in benchmarks:
        bench_instances["lost_in_aggregation"] = load_lost_in_agg_instances(n_tasks)

    all_results = {}
    for model_key in model_keys:
        print(f"\n{'='*60}\nModel: {model_key} (backend={args.backend})\n{'='*60}")
        if args.backend == "ollama":
            model = OllamaModel(model_key, smoke_test=args.smoke_test).load()
        elif args.backend == "unsloth":
            model = UnslothModel(model_key, smoke_test=args.smoke_test).load()
        else:
            model = HFModel(model_key, smoke_test=args.smoke_test).load()

        bench_reports = {}
        for bench_name, instances in bench_instances.items():
            results, grids = [], []
            for i, inst in enumerate(instances):
                gen = model.generate(inst["instruction"], max_new_tokens=1536, temperature=0.0)
                path = parse_path_response(gen["content"], inst["start"], inst["goal"])
                results.append(PathResult(
                    path=path, optimal_path=inst["optimal_path"],
                    optimal_length=inst["optimal_length"],
                    tokens_generated=gen["input_tokens"] + gen["output_tokens"],
                    latency_ms=gen["latency_ms"], raw_output=gen["content"][:500],
                    task_id=inst["task_id"],
                ))
                grids.append(inst["grid"])
                if (i + 1) % 25 == 0 or i == len(instances) - 1:
                    print(f"  [{bench_name}] {i+1}/{len(instances)}")

            report = compute_metrics(results, grids=grids)
            bench_reports[bench_name] = report.__dict__
            print_report(report, f"{model_key} / {bench_name}")

        all_results[model_key] = bench_reports
        with open(os.path.join(args.output_dir, f"{model_key}_results.json"), "w") as f:
            json.dump(bench_reports, f, indent=2, default=str)

    metrics = {"n_tasks": n_tasks, "model_keys": model_keys, "benchmarks": benchmarks,
               "results": all_results, "smoke_test": args.smoke_test}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nDone. Metrics written to {args.output_dir}/metrics.json")
    return metrics


if MODAL_AVAILABLE:
    @app.function(image=image, gpu="A100", timeout=3600,
                   secrets=[modal.Secret.from_name("huggingface-secret")])
    def train_remote(n_tasks: int = 100, smoke_test: bool = False, models: str = ""):
        global MODEL_KEYS
        if models:
            MODEL_KEYS = [m.strip() for m in models.split(",")]
        args = argparse.Namespace(
            data_dir="/root/data", output_dir="/output",
            n_tasks=n_tasks, smoke_test=smoke_test, backend="hf",
        )
        return run(args)

    @app.local_entrypoint()
    def main(n_tasks: int = 100, smoke_test: bool = False, models: str = ""):
        result = train_remote.remote(n_tasks=n_tasks, smoke_test=smoke_test, models=models)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./results/local_run")
    parser.add_argument("--n_tasks", type=int, default=100)
    parser.add_argument("--models", type=str, default="",
                         help="Comma-separated subset of gemma4-e2b,gemma4-e4b,qwen2.5-1.5b,"
                              "qwen2.5-3b,deepseek-r1-distill-qwen-1.5b (default: all)")
    parser.add_argument("--backend", type=str, default="unsloth",
                         choices=["unsloth", "hf", "ollama"],
                         help="'unsloth' (default, recommended) handles architecture quirks "
                              "raw transformers can't. 'hf' is plain transformers, "
                              "inference-only. 'ollama' talks to a local Ollama server.")
    cli_args = parser.parse_args()
    if cli_args.models:
        MODEL_KEYS = [m.strip() for m in cli_args.models.split(",")]
    run(cli_args)
