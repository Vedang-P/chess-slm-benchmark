#!/usr/bin/env python3
"""Cross-lingual spatial navigation gap measurement.

Runs Gemma 4 E2B + Qwen2.5-1.5B/3B on the same GridRoute navigation tasks
across English + 9 translated languages, measuring valid-path and optimality
rate per language, normalized against each model's own English baseline.

Entry point for both local smoke-testing (--smoke_test, CPU, stub model) and
Modal cloud execution (real models, A100 GPU).
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    # Modal is optional: this script also runs standalone on any machine with
    # a GPU (e.g. `python3 train.py --n_tasks 20`), no cloud dependency needed.
    MODAL_AVAILABLE = False

if MODAL_AVAILABLE:
    app = modal.App("rstack-crosslingual-nav")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "transformers>=4.46",
            "accelerate",
            "numpy",
            "sentencepiece",
            "protobuf",
            "huggingface_hub",
        )
        .add_local_dir("src", remote_path="/root/src")
        .add_local_dir("data", remote_path="/root/data")
        .add_local_python_source("hf_models", "multilingual_data")
    )

SEED = 42
MODEL_KEYS = ["gemma4-e2b", "qwen2.5-1.5b", "qwen2.5-3b"]


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


def run(args):
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data directory: {args.data_dir}")
    print(f"Contents: {os.listdir(args.data_dir) if os.path.isdir(args.data_dir) else 'MISSING'}")
    assert os.path.isdir(args.data_dir), f"Data dir not found: {args.data_dir}"

    from multilingual_data import build_instances
    from hf_models import HFModel, OllamaModel, parse_path_response
    from src.evaluation import PathResult, compute_metrics, print_report

    n_tasks = 3 if args.smoke_test else args.n_tasks
    model_keys = MODEL_KEYS[:1] if args.smoke_test else MODEL_KEYS

    instances = build_instances(n_tasks=n_tasks, seed=SEED)
    print(f"Built {len(instances)} (task, language) instances "
          f"from {n_tasks} tasks x {len(set(i['lang'] for i in instances))} languages")

    all_results = {}  # model_key -> lang -> list[PathResult]
    all_grids = {}     # model_key -> lang -> list[np.ndarray]

    for model_key in model_keys:
        print(f"\n{'='*60}\nModel: {model_key} (backend={args.backend})\n{'='*60}")
        if args.backend == "ollama":
            model = OllamaModel(model_key, smoke_test=args.smoke_test).load()
        else:
            model = HFModel(model_key, smoke_test=args.smoke_test).load()

        results_by_lang, grids_by_lang = {}, {}
        for i, inst in enumerate(instances):
            task = inst["task"]
            lang = inst["lang"]
            gen = model.generate(inst["instruction"], max_new_tokens=1536, temperature=0.0)
            path = parse_path_response(gen["content"], task.start, task.goal)

            r = PathResult(
                path=path,
                optimal_path=task.optimal_path,
                optimal_length=task.optimal_length,
                tokens_generated=gen["input_tokens"] + gen["output_tokens"],
                latency_ms=gen["latency_ms"],
                raw_output=gen["content"][:500],
                task_id=task.task_id,
            )
            results_by_lang.setdefault(lang, []).append(r)
            grids_by_lang.setdefault(lang, []).append(task.grid)

            if (i + 1) % 25 == 0 or i == len(instances) - 1:
                print(f"  Progress: {i+1}/{len(instances)}")

        lang_reports = {}
        for lang, results in results_by_lang.items():
            report = compute_metrics(results, grids=grids_by_lang[lang])
            lang_reports[lang] = report.__dict__
            print_report(report, f"{model_key} / {lang}")

        # Normalize each language's rates against this model's own English baseline.
        en = lang_reports.get("en", {})
        en_valid = en.get("feasibility_ratio", 0) or 1e-9
        en_optimal = en.get("optimal_ratio", 0) or 1e-9
        for lang, rep in lang_reports.items():
            rep["normalized_valid_gap"] = (en_valid - rep["feasibility_ratio"]) / en_valid
            rep["normalized_optimal_gap"] = (en_optimal - rep["optimal_ratio"]) / en_optimal

        all_results[model_key] = lang_reports

        with open(os.path.join(args.output_dir, f"{model_key}_results.json"), "w") as f:
            json.dump(lang_reports, f, indent=2, default=str)

    metrics = {
        "n_tasks": n_tasks,
        "model_keys": model_keys,
        "results": all_results,
        "smoke_test": args.smoke_test,
    }
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
                         help="Comma-separated subset of gemma4-e2b,qwen2.5-1.5b,qwen2.5-3b (default: all)")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "ollama"],
                         help="'hf' loads full weights via transformers (needs real VRAM, "
                              "used on Modal/A100). 'ollama' talks to a local Ollama server "
                              "with already-quantized models (fits 6GB laptop GPUs).")
    cli_args = parser.parse_args()
    if cli_args.models:
        MODEL_KEYS = [m.strip() for m in cli_args.models.split(",")]
    run(cli_args)
