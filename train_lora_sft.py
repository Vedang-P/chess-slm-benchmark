#!/usr/bin/env python3
"""LoRA SFT fine-tune Gemma 4 E2B on MazeBench (token mazes) + GridRoute 5x5 (NL routing).

Phase 1 training pipeline. Uses Unsloth for Gemma 4 compatibility.
Trains on both benchmarks jointly, evaluates on both post-training.
Test AlphaMaze locally for comparison.

Usage:
  # Local (6GB GPU): need 4-bit
  python train_lora_sft.py --n_maze 100 --n_grid 100 --load_in_4bit --eval

  # Kaggle T4 (16GB): full bf16, no quantization
  python train_lora_sft.py --n_maze 500 --n_grid 500 --epochs 2 --eval
"""

import argparse, json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

MODEL_ID = "google/gemma-4-E2B-it"
ALPHAMAZE_PATH = "./data/models/alphamaze-v0.2-1.5b"
OUTPUT_DIR = "./results/lora_sft_gemma4"

# ══════════════════════════════════════════════════════════════════════
# Data builders
# ══════════════════════════════════════════════════════════════════════

def build_mazebench_sft_data(n: int, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Menlo/Maze-Bench-v0.2", split="test")
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    SYS = (
        "You are a helpful assistant that solves mazes. You will be given a maze represented by "
        "a series of tokens. The tokens represent: "
        "- Coordinates: <|row-col|> (e.g., <|0-0|>, <|2-4|>) "
        "- Walls: <|no_wall|>, <|up_wall|>, <|down_wall|>, <|left_wall|>, <|right_wall|>, etc. "
        "- Origin: <|origin|> - Target: <|target|> "
        "- Movement: <|up|>, <|down|>, <|left|>, <|right|>, <|blank|> "
        "Your task is to output the sequence of movement tokens "
        "(<|up|>, <|down|>, <|left|>, <|right|>) required to navigate "
        "from the origin to the target. Output ONLY the movement tokens, "
        "separated by spaces, nothing else."
    )
    rows = []
    for i in idx:
        row = ds[int(i)]
        prompt = f"{SYS}\n\nMAZE:\n{row['Prompt']}"
        completion = row["Response"]
        rows.append({"prompt": prompt, "completion": completion})
    return rows

def build_gridroute_sft_data(n: int, size: int = 5, seed: int = 42) -> list[dict]:
    from src.grid_generator import generate_gridroute_maps
    from src.astar_solver import astar
    tasks = generate_gridroute_maps(size=size, obstacle_size=1, num_obstacles=1,
                                     num_maps=100, pairs_per_map=5, seed=seed)
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(tasks), size=min(n, len(tasks)), replace=False))
    rows = []
    for i in idx:
        t = tasks[i]
        prompt = (t.nl_variants["direct"] +
                  "\n\nOutput the shortest path as a Python list of (row,col) tuples. "
                  "Output ONLY the list, nothing else.")
        path = astar(np.array(t.grid), t.start, t.goal)
        completion = str([(int(x), int(y)) for x, y in path])
        rows.append({"prompt": prompt, "completion": completion})
    return rows

def format_sft_chat(example):
    return {"text": [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]}

# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def eval_mazebench(model, tokenizer, n: int = 30, seed: int = 99) -> dict:
    from datasets import load_dataset
    ds = load_dataset("Menlo/Maze-Bench-v0.2", split="test")
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    SYS = (
        "You are a helpful assistant that solves mazes ... "
        "Output ONLY the movement tokens, separated by spaces, nothing else."
    )
    correct = 0
    for i in idx:
        row = ds[int(i)]
        prompt = f"{SYS}\n\nMAZE:\n{row['Prompt']}"
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                tokenize=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device) if isinstance(inputs, dict) else inputs.to(model.device)
        input_len = input_ids.shape[-1]
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=256, temperature=0.0,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        gt = re.findall(r'<\|(up|down|left|right)\|>', row["Response"])
        pred = re.findall(r'<\|(up|down|left|right)\|>', text)
        if pred == gt: correct += 1
    return {"correct": correct, "total": len(idx), "accuracy": 100 * correct / len(idx)}

def eval_gridroute(model, tokenizer, n: int = 30, seed: int = 99) -> dict:
    from src.grid_generator import generate_gridroute_maps
    from src.evaluation import _is_collision_free, _is_in_bounds, _is_valid_steps
    tasks = generate_gridroute_maps(size=5, obstacle_size=1, num_obstacles=1,
                                     num_maps=100, pairs_per_map=5, seed=seed)
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(tasks), size=min(n, len(tasks)), replace=False))
    valid, optimal = 0, 0
    for i in idx:
        t = tasks[i]
        prompt = (t.nl_variants["direct"] +
                  "\n\nOutput the shortest path as a Python list of (row,col) tuples. "
                  "Output ONLY the list, nothing else.")
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                tokenize=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device) if isinstance(inputs, dict) else inputs.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=256, temperature=0.0,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
        coords = re.findall(r'\((\d+),\s*(\d+)\)', text)
        path = [(int(x), int(y)) for x, y in coords]
        grid = np.array(t.grid)
        ib = _is_in_bounds(path, grid.shape) if path else False
        cf = _is_collision_free(path, grid) if path and ib else False
        vs = _is_valid_steps(path) if path and cf else False
        if ib and cf and vs:
            valid += 1
            if len(path) - 1 == t.optimal_length: optimal += 1
    return {"valid": valid, "optimal": optimal, "total": len(idx),
            "valid_rate": 100 * valid / len(idx), "optimal_rate": 100 * optimal / len(idx)}

def eval_alphamaze_mazebench(n: int = 30) -> dict:
    """Evaluate local AlphaMaze checkpoint on MazeBench."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n  Loading AlphaMaze from {ALPHAMAZE_PATH} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(ALPHAMAZE_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ALPHAMAZE_PATH, device_map="auto", torch_dtype=torch.bfloat16,
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    result = eval_mazebench(model, tokenizer, n=n, seed=99)
    del model; torch.cuda.empty_cache()
    return result

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="LoRA SFT Gemma 4 E2B on MazeBench + GridRoute")
    p.add_argument("--n_maze", type=int, default=200)
    p.add_argument("--n_grid", type=int, default=200)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--load_in_4bit", action="store_true", default=False,
                   help="Use 4-bit quantization (for GPUs < 10GB VRAM)")
    p.add_argument("--eval", action="store_true", help="Run eval after training")
    p.add_argument("--eval_only", action="store_true", help="Skip training, eval only")
    p.add_argument("--eval_alphamaze", action="store_true", help="Also test AlphaMaze locally")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("=" * 60)
    print("  LoRA SFT: Gemma 4 E2B on MazeBench + GridRoute 5x5")
    print("=" * 60)
    print(f"  Precision: {'4-bit' if args.load_in_4bit else 'bf16 (full)'}")
    print(f"  MazeBench: {args.n_maze} | GridRoute: {args.n_grid}")
    print(f"  Epochs: {args.epochs} | LR: {args.lr} | LoRA r={args.lora_r}")

    from unsloth import FastLanguageModel

    print(f"\n  Loading {MODEL_ID} via Unsloth"
          f"{' (4-bit)' if args.load_in_4bit else ' (bf16)'}...")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if not args.load_in_4bit else None,
        load_in_4bit=args.load_in_4bit,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if not args.eval_only:
        print(f"\n  Building training data...")
        maze_data = build_mazebench_sft_data(args.n_maze, args.seed)
        grid_data = build_gridroute_sft_data(args.n_grid, size=5, seed=args.seed)
        all_data = maze_data + grid_data
        rng = np.random.RandomState(args.seed); rng.shuffle(all_data)
        print(f"  Total: {len(all_data)} ({len(maze_data)} MazeBench + {len(grid_data)} GridRoute)")
        train_data = [format_sft_chat(x) for x in all_data]

        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import is_bfloat16_supported

        training_args = TrainingArguments(
            output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
            learning_rate=args.lr, logging_steps=10, save_strategy="epoch",
            bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
            report_to=[], remove_unused_columns=False)
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_data,
                              processing_class=tokenizer, max_seq_length=args.max_seq_length)

        print(f"\n  Training: {args.epochs} epochs x {len(train_data)} examples, "
              f"batch={args.batch_size}, accum={args.grad_accum}")
        t0 = time.time()
        trainer.train()
        print(f"  Done in {time.time()-t0:.1f}s")

        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"  Saved to {args.output_dir}")

    if args.eval or args.eval_only:
        FastLanguageModel.for_inference(model)
        model.eval()
        print(f"\n{'='*60}\n  EVALUATION\n{'='*60}")

        print(f"\n  MazeBench (30 mazes)...")
        mb = eval_mazebench(model, tokenizer, n=30)
        print(f"  MazeBench: {mb['correct']}/{mb['total']} = {mb['accuracy']:.1f}%")

        print(f"\n  GridRoute 5x5 (30 tasks)...")
        gr = eval_gridroute(model, tokenizer, n=30)
        print(f"  GridRoute: Valid={gr['valid']}/{gr['total']} ({gr['valid_rate']:.1f}%)  "
              f"Optimal={gr['optimal']}/{gr['total']} ({gr['optimal_rate']:.1f}%)")

        eval_results = {"mazebench": mb, "gridroute": gr}

        # Compare AlphaMaze
        if args.eval_alphamaze:
            print(f"\n  AlphaMaze (local) on MazeBench (30 mazes)...")
            am = eval_alphamaze_mazebench(n=30)
            print(f"  AlphaMaze MazeBench: {am['correct']}/{am['total']} = {am['accuracy']:.1f}%")
            eval_results["alphamaze_mazebench"] = am

        out = os.path.join(args.output_dir, "eval_results.json")
        with open(out, "w") as f: json.dump(eval_results, f, indent=2)
        print(f"\n  Saved to {out}")

    print(f"\n{'='*60}\n  DONE\n{'='*60}")

if __name__ == "__main__":
    main()
