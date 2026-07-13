"""GRPO training script -- Gemma 4 E2B only, for now.

This is deliberately a SHORT run (small step count) to get a real wall-clock/
GPU-hour timing number on the actual hardware before committing to the full
4-model x 3-training-condition plan. Not yet verified end-to-end -- run this
and report back exactly what happens, including any error, before trusting
the timing number or scaling up.

Single training condition: GridRoute only (the "single-format" condition
from experiment-plan.md). Mixed-format and consistency-reward conditions
come after this establishes a real timing baseline.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def build_dataset(n_tasks: int, seed: int = 42):
    """GridRoute tasks -> a datasets.Dataset with prompt + reward metadata."""
    from datasets import Dataset
    from src.grid_generator import generate_gridroute_maps

    tasks = generate_gridroute_maps(
        size=10, obstacle_size=3, num_obstacles=2,
        num_maps=100, pairs_per_map=5, seed=seed,
    )
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(tasks), size=min(n_tasks, len(tasks)), replace=False))

    rows = {"prompt": [], "start": [], "goal": [], "grid_json": [], "optimal_length": []}
    for i in idx:
        t = tasks[i]
        rows["prompt"].append(t.nl_variants["direct"])
        rows["start"].append(list(t.start))
        rows["goal"].append(list(t.goal))
        rows["grid_json"].append(json.dumps(t.grid.tolist()))
        rows["optimal_length"].append(t.optimal_length)
    return Dataset.from_dict(rows)


def make_reward_fn():
    from hf_models import parse_path_response
    from src.evaluation import _is_collision_free, _is_in_bounds, _is_valid_steps

    def reward_gridroute(completions, start, goal, grid_json, optimal_length, **kwargs):
        rewards = []
        for completion, s, g, gj, opt_len in zip(completions, start, goal, grid_json, optimal_length):
            text = completion if isinstance(completion, str) else str(completion)
            grid = np.array(json.loads(gj))
            path = parse_path_response(text, tuple(s), tuple(g))

            if path is None:
                rewards.append(-0.5)  # no parseable path at all
                continue
            if not (_is_collision_free(path, grid) and _is_in_bounds(path, grid.shape)
                     and _is_valid_steps(path)):
                rewards.append(0.0)  # parsed but invalid (collision/OOB/bad step)
                continue
            length = len(path) - 1
            rewards.append(1.0 if length == opt_len else 0.5)  # valid: optimal vs suboptimal
        return rewards

    return reward_gridroute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tasks", type=int, default=50,
                         help="Number of distinct GridRoute problems in the training set")
    parser.add_argument("--max_steps", type=int, default=50,
                         help="GRPO training steps -- kept small for a timing test")
    parser.add_argument("--num_generations", type=int, default=8,
                         help="GRPO group size: completions sampled per prompt")
    parser.add_argument("--output_dir", type=str, default="./results/grpo_timing_test")
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    print("Loading Gemma 4 E2B via Unsloth...")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-4-E2B-it", max_seq_length=2048,
        dtype=None, load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    print(f"Model + LoRA ready in {time.time()-t0:.1f}s")

    print(f"Building dataset ({args.n_tasks} GridRoute tasks)...")
    dataset = build_dataset(args.n_tasks)

    reward_fn = make_reward_fn()

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_completion_length=512,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.max_steps,  # only save at the end for this timing test
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting GRPO: {args.max_steps} steps, group size {args.num_generations}, "
          f"{args.n_tasks} training problems.")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"TIMING RESULT")
    print(f"{'='*60}")
    print(f"{args.max_steps} steps took {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Per-step: {elapsed/args.max_steps:.2f}s")
    print(f"Extrapolated to 1000 steps: {elapsed/args.max_steps*1000/3600:.2f} GPU-hours")

    trainer.save_model(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
