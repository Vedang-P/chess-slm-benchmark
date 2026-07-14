"""SFT fine-tuning on GridRoute — natural language → coordinate path.

Phase 1 of the training pipeline. Runs before GRPO.
Uses TRL's SFTTrainer with 4-bit QLoRA.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent))

MODEL_PRESETS = {
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

def build_sft_dataset(n_tasks: int, grid_size: int = 10, seed: int = 42) -> Dataset:
    """GridRoute NL prompts → coordinate path pairs for SFT."""
    from src.grid_generator import generate_gridroute_maps
    from src.astar_solver import astar

    # Auto-adjust obstacle config for grid size
    obs_size = 1 if grid_size == 5 else 3
    n_obs = 1 if grid_size == 5 else 2

    tasks = generate_gridroute_maps(
        size=grid_size, obstacle_size=obs_size, num_obstacles=n_obs,
        num_maps=100, pairs_per_map=5, seed=seed,
    )
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(tasks), size=min(n_tasks, len(tasks)), replace=False))

    rows = {"prompt": [], "completion": []}
    for i in idx:
        t = tasks[i]
        instruction = t.nl_variants["direct"]
        path = astar(t.grid, t.start, t.goal)
        path_str = str([(int(x), int(y)) for x, y in path])
        rows["prompt"].append(instruction + "\n\nOutput the path as a Python list of (row,col) tuples.")
        rows["completion"].append(path_str)

    return Dataset.from_dict(rows)


def format_sft(example):
    """Format prompt+completion for SFT training."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return {"text": messages}  # SFTTrainer handles chat template


def main():
    parser = argparse.ArgumentParser(description="SFT fine-tuning for GridRoute spatial reasoning")
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--model_path", default="", help="Local path (overrides --model)")
    parser.add_argument("--n_tasks", type=int, default=400)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--output_dir", default="./results/sft_deepseek")
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    model_id = args.model_path if args.model_path else MODEL_PRESETS.get(args.model, args.model)
    print(f"Model: {model_id}")
    print(f"Tasks: {args.n_tasks}, Grid: {args.grid_size}x{args.grid_size}, Epochs: {args.epochs}, LR: {args.lr}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                              bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)

    # LoRA
    lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0, bias="none",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"],
                       task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()

    # Dataset
    print(f"Building SFT dataset ({args.n_tasks} tasks, {args.grid_size}x{args.grid_size})...")
    dataset = build_sft_dataset(args.n_tasks, grid_size=args.grid_size)
    dataset = dataset.map(format_sft, remove_columns=dataset.column_names)

    # Train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=dataset,
        tokenizer=tokenizer, max_seq_length=args.max_seq_length,
    )

    print(f"Starting SFT: {args.epochs} epoch(s), {len(dataset)} examples")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"SFT complete in {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
