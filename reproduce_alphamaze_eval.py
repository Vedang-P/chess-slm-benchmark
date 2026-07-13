"""Phase 1 reproduction check #1: does OUR eval harness correctly score
AlphaMaze's own public checkpoint on their own public eval data?

Downloads homebrewltd/AlphaMaze-v0.2-1.5B (Apache 2.0) and Maze-Bench-v0.2
(their eval set), runs the checkpoint through it, and reports valid/optimal
rates. If this doesn't land close to their reported numbers, the problem is
in our evaluation code, not in anything about Gemma 4 -- fix this before
trusting any of our own training results.

Cheap and fast: no training, one model, their own data. Not yet run on real
hardware.
"""

import argparse
import json
import re
import time

import numpy as np


def parse_alphamaze_response(content: str) -> list:
    """AlphaMaze's format uses movement tokens/directions, not raw (x,y)
    coordinates like GridRoute -- adjust extraction if their actual output
    format differs once we see real responses. Falls back to coordinate
    extraction in case the model still emits (x,y) pairs."""
    patterns = [r"\((\d+),\s*(\d+)\)", r"\[(\d+),\s*(\d+)\]"]
    coords = []
    for pat in patterns:
        for x, y in re.findall(pat, content):
            coords.append((int(x), int(y)))
    return coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tasks", type=int, default=100,
                         help="Max eval tasks (Maze-Bench-v0.2 is ~100 total per AlphaMaze's paper)")
    parser.add_argument("--output_dir", type=str, default="./results/reproduce_alphamaze")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    from datasets import load_dataset
    from unsloth import FastLanguageModel
    import torch

    print("Loading Maze-Bench-v0.2...")
    eval_data = load_dataset("homebrewltd/Maze-Bench-v0.2", split="train")
    print(f"Loaded {len(eval_data)} eval examples. First example keys: {list(eval_data[0].keys())}")
    print(f"First example: {json.dumps({k: str(v)[:200] for k, v in eval_data[0].items()}, indent=2)}")

    n = min(args.n_tasks, len(eval_data))
    eval_data = eval_data.select(range(n))

    print(f"\nLoading homebrewltd/AlphaMaze-v0.2-1.5B via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="homebrewltd/AlphaMaze-v0.2-1.5B", max_seq_length=2048,
        dtype=None, load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    results = []
    t0 = time.time()
    for i, row in enumerate(eval_data):
        # NOTE: exact column names/prompt format need confirming against the
        # actual dataset schema once loaded -- this assumes a "prompt" or
        # "text" column; adjust based on what the printout above shows.
        prompt = row.get("prompt") or row.get("text") or row.get("input") or str(row)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        content = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:],
                                    skip_special_tokens=True)
        results.append({"index": i, "raw_output": content[:1000]})

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}")

    elapsed = time.time() - t0
    print(f"\nDone: {n} examples in {elapsed:.1f}s ({elapsed/n:.2f}s/example)")

    with open(f"{args.output_dir}/raw_outputs.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw outputs saved to {args.output_dir}/raw_outputs.json")
    print("\nNEXT: inspect raw_outputs.json to confirm the response format, "
          "then score against Maze-Bench-v0.2's ground truth (format not yet "
          "confirmed from this side -- check the printed dataset schema above).")


if __name__ == "__main__":
    main()
