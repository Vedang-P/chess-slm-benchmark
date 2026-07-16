"""SFT warm-start, before GRPO. Phase 1 of the training pipeline.

--format nl:     GridRoute NL coordinates only.
--format token:  our own token-maze format only (src/token_maze.py).
--format mixed:  NL + token interleaved over the same underlying grids --
                 matches train_grpo.py's mixed/consistency conditions' data
                 recipe, so SFT and GRPO warm-start from the same kind of data.

Uses hf_models.load_trainable_model() -- bitsandbytes+peft for every model,
Gemma 4 (E2B, E4B) included. NL prompts use the same GRIDROUTE_NL_ANSWER_SUFFIX
("FINAL ANSWER:" convention) that eval.py and train_grpo.py expect -- SFT
needs to actually teach that exact reporting format, not a different one, or
the model was never trained on the convention it gets scored against later.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent))

from src.grid_generator import GRIDROUTE_NL_ANSWER_SUFFIX

ALPHAMAZE_LOCAL_PATH = "./data/models/alphamaze-v0.2-1.5b"
MODEL_PRESETS = {
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "alphamaze": ALPHAMAZE_LOCAL_PATH,
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
}


def build_sft_dataset(fmt: str, n_tasks: int, grid_size: int = 5, seed: int = 42) -> Dataset:
    """GridRoute tasks -> (prompt, completion) pairs, A* ground-truth paths."""
    from src.astar_solver import astar
    from src.grid_generator import generate_gridroute_maps, gridroute_defaults
    from src import token_maze

    obs_size, n_obs = gridroute_defaults(grid_size)
    tasks = generate_gridroute_maps(
        size=grid_size, obstacle_size=obs_size, num_obstacles=n_obs,
        num_maps=100, pairs_per_map=5, seed=seed,
    )
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(tasks), size=min(n_tasks, len(tasks)), replace=False))

    rows = {"prompt": [], "completion": []}
    for i in idx:
        t = tasks[i]
        grid = np.array(t.grid)
        path = astar(grid, t.start, t.goal)

        formats = [fmt] if fmt != "mixed" else ["nl", "token"]
        for f in formats:
            if f == "token":
                prompt = token_maze.TASK_INSTRUCTION + "\n\n" + token_maze.grid_to_token_maze(grid, t.start, t.goal)
                completion = token_maze.path_to_moves(path)
            else:
                prompt = t.nl_variants["direct"] + GRIDROUTE_NL_ANSWER_SUFFIX
                completion = "FINAL ANSWER: " + str([(int(x), int(y)) for x, y in path])
            rows["prompt"].append(prompt)
            rows["completion"].append(completion)

    return Dataset.from_dict(rows)


def format_sft(example):
    """Format prompt+completion for SFT training."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return {"text": messages}  # SFTTrainer handles chat template


def main():
    from hf_models import configure_quiet_logging
    configure_quiet_logging()

    parser = argparse.ArgumentParser(description="SFT warm-start for GridRoute + MazeBench spatial reasoning")
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b",
                         help="Model shorthand (%s) or full HuggingFace model ID." % ", ".join(MODEL_PRESETS))
    parser.add_argument("--model_path", default="", help="Local path (overrides --model)")
    parser.add_argument("--format", choices=["nl", "token", "mixed"], default="nl")
    parser.add_argument("--n_tasks", type=int, default=400)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048,
                         help="Default raised from 1024: GridRoute completions now include a "
                              "'FINAL ANSWER:' preamble and token-format completions can be long.")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    args = parser.parse_args()
    model_id = args.model_path if args.model_path else MODEL_PRESETS.get(args.model, args.model)
    if not args.output_dir:
        args.output_dir = f"./results/sft_{args.model}_{args.format}"

    print(f"Model: {model_id}")
    print(f"Format: {args.format}  Tasks: {args.n_tasks}  Grid: {args.grid_size}x{args.grid_size}  "
          f"Epochs: {args.epochs}  LR: {args.lr}")

    from hf_models import is_gemma4, load_trainable_model
    model, tokenizer, backend = load_trainable_model(
        model_id, load_in_4bit=args.load_in_4bit, max_seq_length=args.max_seq_length,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        # --model_path replaces model_id above with a local path (e.g. a
        # full merged checkpoint like AlphaMaze's), which won't necessarily
        # contain "gemma4" in its name -- pass the real hint explicitly
        # rather than relying on is_gemma4() to guess from that path.
        force_gemma4=is_gemma4(args.model),
    )
    print(f"Model + LoRA ready (backend={backend})")

    print(f"Building SFT dataset ({args.format}, {args.n_tasks} tasks, {args.grid_size}x{args.grid_size})...")
    dataset = build_sft_dataset(args.format, args.n_tasks, grid_size=args.grid_size)
    print(f"  {len(dataset)} training rows.")
    dataset = dataset.map(format_sft, remove_columns=dataset.column_names)

    # bf16 checked against real hardware support, not hardcoded: a T4 (this
    # project's actual free-tier Kaggle GPU) has no bf16 tensor-core support,
    # and some transformers versions raise outright if bf16=True is requested
    # without it rather than silently falling back -- fp16 is what T4s
    # actually accelerate. This is untested territory as of this commit:
    # every prior Kaggle run went through check_finetune_feasibility.py's own
    # manual forward+backward, which never touches TrainingArguments/
    # SFTConfig's precision flags at all.
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # SFTConfig, not plain TrainingArguments -- newer TRL moved SFT-specific
    # settings out of being accepted as bare SFTTrainer() kwargs and into
    # this dedicated config class. Its length-limit kwarg is max_length, not
    # max_seq_length -- confirmed via a real Kaggle traceback (TRL renamed
    # it in 0.16.0; requirements.txt only pins a lower bound, so whatever
    # TRL Kaggle installs picks up the new name). --max_seq_length is kept
    # as this project's own CLI flag name (used for more than just this one
    # kwarg, see load_trainable_model()'s tokenizer.model_max_length) --
    # only the kwarg passed to SFTConfig itself needed to change.
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=bf16_ok,
        fp16=not bf16_ok,
        report_to=[],
        remove_unused_columns=False,
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=dataset,
        processing_class=tokenizer,
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
