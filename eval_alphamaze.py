"""Evaluate AlphaMaze-v0.2-1.5B on GridRoute benchmark."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent))

def main():
    model_path = "./data/models/alphamaze-v0.2-1.5b"
    n_tasks = 10

    print(f"Loading AlphaMaze from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Load GridRoute tasks
    from src.grid_generator import generate_gridroute_maps
    tasks = generate_gridroute_maps(
        size=10, obstacle_size=3, num_obstacles=2,
        num_maps=100, pairs_per_map=5, seed=42,
    )
    rng = np.random.RandomState(42)
    idx = sorted(rng.choice(len(tasks), size=n_tasks, replace=False))
    print(f"Loaded {n_tasks} GridRoute 10x10 tasks")

    from hf_models import parse_path_response
    from src.evaluation import _is_collision_free, _is_in_bounds, _is_valid_steps

    valid, optimal, total = 0, 0, 0
    for i, task_idx in enumerate(idx):
        t = tasks[task_idx]
        prompt = t.nl_variants["direct"]

        # AlphaMaze expects a specific chat format
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, temperature=0.0, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        path = parse_path_response(text, tuple(t.start), tuple(t.goal))
        grid = np.array(t.grid)
        total += 1

        if path and _is_in_bounds(path, grid.shape) and _is_collision_free(path, grid) and _is_valid_steps(path):
            valid += 1
            if len(path) - 1 == t.optimal_length:
                optimal += 1
            status = "✅" if len(path)-1 == t.optimal_length else "⚠️ (suboptimal)"
        else:
            status = "❌ (invalid)"
        print(f"  [{i+1}/{n_tasks}] {status} path={'VALID' if path and status != '❌' else 'INVALID'}")

    print(f"\n{'='*50}")
    print(f"AlphaMaze-v0.2-1.5B on GridRoute 10x10 (n={total})")
    print(f"  Valid paths:  {valid}/{total} ({100*valid/total:.1f}%)")
    print(f"  Optimal:      {optimal}/{total} ({100*optimal/total:.1f}%)")
    print(f"{'='*50}")

    with open("./results/alphamaze_gridroute.json", "w") as f:
        json.dump({"valid": valid, "optimal": optimal, "total": total, "rate": valid/total}, f)
    print("Saved to ./results/alphamaze_gridroute.json")


if __name__ == "__main__":
    main()
