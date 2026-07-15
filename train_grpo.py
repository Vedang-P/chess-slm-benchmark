"""GRPO training script -- building the best on-device SLM for GridRoute (NL)
and MazeBench (token) jointly.

Three training-recipe options to try on the SAME underlying GridRoute
problems, in two surface formats (NL coordinates vs. our own token-maze
encoding, see src/token_maze.py) -- these are engineering options in pursuit
of the best model, not a hypothesis-testing pipeline (see idea.md's
changelog: this project isn't claiming the consistency condition as a novel
proposed technique, just trying it as one candidate recipe):

  --condition single       GridRoute NL only. Does further single-format
                            training hurt the starting checkpoint's
                            other-format (token maze) skill?
  --condition mixed        NL + token examples interleaved, naive per-format
                            reward, no cross-format signal.
  --condition consistency  Same mixed data, but each reward call also scores
                            the model's OWN current completion on the paired
                            prompt in the other format, and adds a bonus when
                            both formats agree on validity/optimality for the
                            same underlying problem (adapts Elhady et al.
                            2606.01464's cross-lingual consistency reward
                            mechanism to cross-format spatial reasoning, as
                            one recipe among several -- not presented as this
                            project's headline contribution).

Target: Kaggle's free T4 (16 GB) for Gemma 4 E2B (primary) and E4B (if it
fits -- E4B LoRA needs ~17GB, right at/over the T4's ceiling, unconfirmed
until you actually run check_finetune_feasibility.py with it). Also runs
DeepSeek-R1-Distill-Qwen-1.5B / SmolLM2-1.7B / AlphaMaze's checkpoint on a
6 GB consumer GPU. Gemma 4 loads via Unsloth (hf_models.load_trainable_model);
everything else via bitsandbytes+peft -- see that function's docstring.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from src.grid_generator import GRIDROUTE_NL_ANSWER_SUFFIX

ALPHAMAZE_LOCAL_PATH = "./data/models/alphamaze-v0.2-1.5b"

# ── model presets ──────────────────────────────────────────────────
# fits_6gb is about your laptop specifically; Kaggle's 16 GB T4 covers
# everything except gemma4-e4b's training (marginal -- see module docstring).
MODEL_PRESETS: dict[str, dict] = {
    # ── confirmed FEASIBLE on 6 GB (bitsandbytes 4-bit) ──
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
    "alphamaze": {
        "model_id": ALPHAMAZE_LOCAL_PATH,
        "fits_6gb": True,
    },
    # ── marginal on 6 GB (fine on Kaggle's 16 GB T4) ──
    "qwen2.5-3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "fits_6gb": "marginal",
    },
    # ── primary target model. Needs Unsloth (see hf_models.load_trainable_model);
    #    E2B LoRA measured ~8-10GB elsewhere, comfortable on a 16GB T4, won't
    #    fit a 6GB laptop. ──
    "gemma4-e2b": {
        "model_id": "google/gemma-4-E2B-it",
        "fits_6gb": False,
        "fits_kaggle_t4": True,
    },
    # ── E4B LoRA needs ~17GB elsewhere measured -- at/over the free T4's 16GB
    #    ceiling. Run check_finetune_feasibility.py with this model before
    #    committing to a full training run; may need to drop --lora_r or
    #    --num_generations, or may simply not fit. ──
    "gemma4-e4b": {
        "model_id": "google/gemma-4-E4B-it",
        "fits_6gb": False,
        "fits_kaggle_t4": "marginal",
    },
}


def build_dataset(condition: str, n_tasks: int, grid_size: int = 5, seed: int = 42):
    """Build the GRPO training set for the given condition.

    single:      one NL row per task.
    mixed/consistency: two rows per task (NL + token), same underlying grid.
    consistency additionally carries partner_prompt/partner_fmt per row,
    pointing at its sibling's prompt -- used by the consistency reward to
    score cross-format agreement without depending on both rows landing in
    the same training batch (see make_reward_fn).
    """
    from datasets import Dataset
    from src.grid_generator import generate_gridroute_maps, gridroute_defaults
    from src import token_maze

    obs_size, n_obs = gridroute_defaults(grid_size)
    tasks = generate_gridroute_maps(
        size=grid_size, obstacle_size=obs_size, num_obstacles=n_obs,
        num_maps=100, pairs_per_map=5, seed=seed,
    )
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(tasks), size=min(n_tasks, len(tasks)), replace=False))

    cols = {"prompt": [], "fmt": [], "start": [], "goal": [], "grid_json": [], "optimal_length": []}
    if condition == "consistency":
        cols["partner_prompt"] = []
        cols["partner_fmt"] = []

    for i in idx:
        t = tasks[i]
        grid_json = json.dumps(t.grid.tolist())
        nl_prompt = t.nl_variants["direct"] + GRIDROUTE_NL_ANSWER_SUFFIX
        token_prompt = token_maze.TASK_INSTRUCTION + "\n\n" + token_maze.grid_to_token_maze(
            np.array(t.grid), t.start, t.goal)

        rows = [("nl", nl_prompt)] if condition == "single" else [("nl", nl_prompt), ("token", token_prompt)]
        partner_of = {"nl": token_prompt, "token": nl_prompt}

        for fmt, prompt in rows:
            cols["prompt"].append(prompt)
            cols["fmt"].append(fmt)
            cols["start"].append(list(t.start))
            cols["goal"].append(list(t.goal))
            cols["grid_json"].append(grid_json)
            cols["optimal_length"].append(t.optimal_length)
            if condition == "consistency":
                cols["partner_prompt"].append(partner_of[fmt])
                cols["partner_fmt"].append("token" if fmt == "nl" else "nl")

    return Dataset.from_dict(cols)


def _decode_and_score(text, fmt, start, goal, grid, finished=True):
    """Score a completion, reading only its own reported final answer (never
    mining coordinates/moves out of a thinking trace it didn't present as its
    answer) -- same policy as eval.py, applied to the training reward signal
    too, since a reward function that mines thinking tokens would directly
    train the model to rely on exactly the sloppy behavior we don't want."""
    from hf_models import parse_path_response, extract_reported_answer
    from src import token_maze
    from src.evaluation import _is_collision_free, _is_in_bounds, _is_valid_steps

    answer = extract_reported_answer(text, finished=finished, require_marker=(fmt != "token"))
    if answer is None:
        return None, False

    if fmt == "token":
        moves = token_maze.parse_move_tokens(answer)
        path = token_maze.moves_to_path(moves, start, grid, goal=goal) if moves else None
    else:
        path = parse_path_response(answer, start, goal)
    if path is None:
        return None, False
    valid = _is_in_bounds(path, grid.shape) and _is_collision_free(path, grid) and _is_valid_steps(path)
    return path, valid


def _base_reward(path, valid, opt_len):
    if path is None:
        return -0.5  # no parseable path at all
    if not valid:
        return 0.0   # parsed but invalid (collision/OOB/bad step)
    length = len(path) - 1
    return 1.0 if length == opt_len else 0.5  # valid: optimal vs suboptimal


def make_reward_fn(condition: str, model=None, tokenizer=None, max_completion_length: int = 512):
    def approx_finished(text: str) -> bool:
        """GRPOTrainer's reward interface gives decoded completion strings,
        not per-completion truncation metadata, so this re-tokenizes to
        approximate whether the completion hit max_completion_length (likely
        cut off mid-thought) versus ended naturally well within budget. Not
        exact (GRPOTrainer's own tokenization may differ slightly), but a
        completion that used up (almost) its whole budget with no closing
        marker is not something to trust as complete either way."""
        return len(tokenizer.encode(text)) < max_completion_length - 4

    def generate_partner_completion(partner_prompt: str) -> str:
        """Greedy-decode the model's OWN current completion for the paired
        prompt in the other format, purely to score cross-format agreement --
        never used as a training target itself. Generated fresh per reward
        call rather than relying on the paired row landing in the same
        training batch, which would be a fragile assumption about GRPOTrainer's
        internal batching/shuffling order."""
        was_training = model.training
        model.eval()
        try:
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": partner_prompt}],
                add_generation_prompt=True, tokenize=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(input_ids=inputs, max_new_tokens=max_completion_length,
                                      do_sample=False, pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        finally:
            if was_training:
                model.train()

    def reward_fn(completions, fmt, start, goal, grid_json, optimal_length,
                  partner_prompt=None, partner_fmt=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else str(completion)
            s, g = tuple(start[i]), tuple(goal[i])
            grid = np.array(json.loads(grid_json[i]))
            opt_len = optimal_length[i]

            path, valid = _decode_and_score(text, fmt[i], s, g, grid, finished=approx_finished(text))
            optimal = valid and (len(path) - 1 == opt_len)
            reward = _base_reward(path, valid, opt_len)

            if condition == "consistency" and partner_prompt is not None:
                p_text = generate_partner_completion(partner_prompt[i])
                p_path, p_valid = _decode_and_score(p_text, partner_fmt[i], s, g, grid,
                                                     finished=approx_finished(p_text))
                p_optimal = p_valid and (len(p_path) - 1 == opt_len) if p_path else False
                # Bonus for cross-format agreement, on top of (not instead of)
                # per-format correctness above -- so two confidently-wrong-but-
                # matching answers can't earn it, only two independently-
                # correct ones. Two tiers: full agreement on the optimal
                # solution, partial credit for agreeing on mere validity.
                if optimal and p_optimal:
                    reward += 0.5
                elif valid and p_valid:
                    reward += 0.25

            rewards.append(reward)
        return rewards

    return reward_fn


def main():
    from hf_models import configure_quiet_logging
    configure_quiet_logging()

    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning for on-device spatial reasoning on GridRoute + MazeBench.")
    parser.add_argument("--model", type=str, default="deepseek-r1-distill-qwen-1.5b",
                         help="Model shorthand (%s) or full HuggingFace model ID."
                              % ", ".join(MODEL_PRESETS))
    parser.add_argument("--condition", choices=["single", "mixed", "consistency"], default="single",
                         help="Training condition -- see module docstring.")
    parser.add_argument("--grid_size", type=int, default=5,
                         help="GridRoute grid size (default 5, matching MazeBench's 5x5 for a fair "
                              "cross-format comparison)")
    parser.add_argument("--n_tasks", type=int, default=50,
                         help="Number of distinct GridRoute problems in the training set "
                              "(mixed/consistency conditions use 2x this many rows: NL + token per problem)")
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
    parser.add_argument("--max_completion_length", type=int, default=4096,
                         help="Maximum completion length for GRPO generations (default: 4096 -- "
                              "generous on purpose so thinking isn't cut off mid-thought the way "
                              "512 was found to truncate MazeBench-style reasoning in Phase 1; this "
                              "is also the single biggest lever on wall-clock cost since it applies "
                              "per completion x num_generations, so run the timing test below before "
                              "committing to a full step count, and lower it if Kaggle quota is tight)")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = f"./results/grpo_{args.model}_{args.condition}"

    # Resolve model: shorthand → HF ID (or local path for alphamaze), or use raw ID if unrecognized.
    preset = MODEL_PRESETS.get(args.model)
    if preset:
        model_id = preset["model_id"]
        fits = preset["fits_6gb"]
    else:
        model_id = args.model
        fits = None  # unknown — user's responsibility

    # Warn if model is known-infeasible on 6 GB.
    if fits is False:
        note = " (E4B LoRA needs ~17GB -- run check_finetune_feasibility.py first, may not fit)" \
            if preset and preset.get("fits_kaggle_t4") == "marginal" else " (fine on Kaggle's 16GB T4)"
        print(f"⚠️  NOTE: '{args.model}' needs ~9+GB and won't fit a 6GB laptop GPU.{note}")
    elif fits == "marginal":
        print(f"⚠️  NOTE: '{args.model}' is tight on 6 GB.  Consider reducing --num_generations or --max_seq_length, "
              "or run on Kaggle's 16 GB T4.")
    if fits is not False and not args.load_in_4bit:
        print("⚠️  NOTE: --no_4bit is set.  Full-precision weights may OOM on 6 GB GPUs.")

    from trl import GRPOConfig, GRPOTrainer
    from hf_models import load_trainable_model

    print(f"Loading {model_id} (4bit={args.load_in_4bit}, seq_len={args.max_seq_length})...")
    t0 = time.time()
    model, tokenizer, backend = load_trainable_model(
        model_id, load_in_4bit=args.load_in_4bit, max_seq_length=args.max_seq_length,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    )
    print(f"Model + LoRA ready in {time.time()-t0:.1f}s (backend={backend})")

    print(f"Building '{args.condition}' dataset ({args.n_tasks} GridRoute {args.grid_size}x{args.grid_size} "
          f"problems)...")
    dataset = build_dataset(args.condition, args.n_tasks, grid_size=args.grid_size)
    print(f"  {len(dataset)} training rows.")

    reward_fn = make_reward_fn(
        args.condition, model=model, tokenizer=tokenizer,
        max_completion_length=args.max_completion_length,
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,  # must be divisible by num_generations
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

    print(f"\nStarting GRPO: condition={args.condition}, {args.max_steps} steps, "
          f"group size {args.num_generations}, {len(dataset)} training rows.")
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
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
