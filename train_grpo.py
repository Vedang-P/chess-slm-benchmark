"""GRPO training script — multi-model, on-device spatial reasoning.

Single training condition: GridRoute only (the "single-format" condition
from experiment-plan.md). Mixed-format and consistency-reward conditions
come after this establishes a real timing baseline.

Target: consumer-laptop GPUs (6 GB VRAM).  Defaults to 4-bit LoRA GRPO
on DeepSeek-R1-Distill-Qwen-1.5B (AlphaMaze's proven base model).  Use
--model <shorthand> to switch (see --help for presets).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch


# ── model presets ──────────────────────────────────────────────────
# All measured on RTX 4050 6GB with 4-bit LoRA. "marginal" = may need
# reduced num_generations or max_seq_length.
MODEL_PRESETS: dict[str, dict] = {
    # ── ✅ confirmed FEASIBLE on 6 GB (bitsandbytes 4-bit) ──
    "deepseek-r1-distill-qwen-1.5b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "fits_6gb": True,
        "measured_grpo_gb": 4.2,   # check_finetune_feasibility.py
    },
    "smollm2-1.7b": {
        "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "fits_6gb": True,
        "measured_grpo_gb": 3.0,   # check_finetune_feasibility.py
    },
    # ── ⚠️ marginal (tight on 6 GB, reduce generations) ──
    "qwen2.5-3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "fits_6gb": "marginal",
    },
    # ── inference-only (GRPO needs >6 GB) ──
    "gemma4-e2b": {
        "model_id": "google/gemma-4-E2B-it",
        "fits_6gb": False,
    },
    # ── gated models (needs HF access approval) ──
    # "gemma-3-1b": {"model_id": "google/gemma-3-1b-it", "fits_6gb": True},
}


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
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning for on-device spatial reasoning (4-bit LoRA).")
    parser.add_argument("--model", type=str, default="deepseek-r1-distill-qwen-1.5b",
                         help="Model shorthand (%s) or full HuggingFace model ID."
                              % ", ".join(MODEL_PRESETS))
    parser.add_argument("--n_tasks", type=int, default=50,
                         help="Number of distinct GridRoute problems in the training set")
    parser.add_argument("--max_steps", type=int, default=50,
                         help="GRPO training steps -- kept small for a timing test")
    parser.add_argument("--num_generations", type=int, default=8,
                         help="GRPO group size: completions sampled per prompt")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                         help="Load model in 4-bit quantization (default: True for 6 GB GPUs)")
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false",
                         help="Disable 4-bit loading (use full precision)")
    parser.add_argument("--lora_r", type=int, default=16,
                         help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                         help="LoRA alpha scaling factor (default: 16)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                         help="Maximum sequence length (default: 2048)")
    parser.add_argument("--max_completion_length", type=int, default=512,
                         help="Maximum completion length for GRPO generations (default: 512)")
    parser.add_argument("--output_dir", type=str, default="./results/grpo_timing_test")
    args = parser.parse_args()

    # Resolve model: shorthand → HF ID, or use raw ID if unrecognized.
    preset = MODEL_PRESETS.get(args.model)
    if preset:
        model_id = preset["model_id"]
        fits = preset["fits_6gb"]
    else:
        model_id = args.model
        fits = None  # unknown — user's responsibility

    # Warn if model is known-infeasible on 6 GB.
    if fits is False:
        print(f"⚠️  WARNING: '{args.model}' needs ~9+ GB for GRPO.")
        print("   This will likely OOM on a 6 GB GPU.  Use --model smollm2-1.7b or deepseek-r1-distill-qwen-1.5b.")
        if not args.load_in_4bit:
            print("   Also: you have --no_4bit set — this guarantees OOM on 6 GB.")
    elif fits == "marginal":
        print(f"⚠️  NOTE: '{args.model}' is tight on 6 GB.  Consider reducing --num_generations or --max_seq_length.")
    if fits is not False and not args.load_in_4bit:
        print("⚠️  NOTE: --no_4bit is set.  Full-precision weights may OOM on 6 GB GPUs.")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import GRPOConfig, GRPOTrainer

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {model_id} (4bit={args.load_in_4bit}, seq_len={args.max_seq_length})...")
    t0 = time.time()

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # needed for gradient checkpointing with PEFT
    print(f"Model + LoRA ready in {time.time()-t0:.1f}s")

    print(f"Building dataset ({args.n_tasks} GridRoute tasks)...")
    dataset = build_dataset(args.n_tasks)

    reward_fn = make_reward_fn()

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.max_steps,
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
