"""Clean baseline eval — excludes prompt coords from extraction."""
import json, os, sys, time, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent))

def extract_path(text: str, exclude_coords: set) -> list | None:
    """Extract coordinate path from text, excluding known prompt coords."""
    coords = re.findall(r'\((\d+),\s*(\d+)\)', text)
    path = [(int(x), int(y)) for x, y in coords if (int(x), int(y)) not in exclude_coords]
    return path if len(path) >= 2 else None

def eval_model(model_path: str, label: str, n: int = 50):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb, device_map="auto", trust_remote_code=True)

    from src.grid_generator import generate_gridroute_maps
    from src.evaluation import _is_collision_free, _is_in_bounds, _is_valid_steps

    tasks = generate_gridroute_maps(size=10, obstacle_size=3, num_obstacles=2, num_maps=100, pairs_per_map=5, seed=42)
    rng = np.random.RandomState(42)
    idx = sorted(rng.choice(len(tasks), size=n, replace=False))

    valid, optimal, no_path, collisions, oob, bad_steps = 0, 0, 0, 0, 0, 0
    for i, ti in enumerate(idx):
        t = tasks[ti]
        # Build exclude set from prompt coords (start, goal, obstacles)
        exclude = {tuple(t.start), tuple(t.goal)}
        for ox, oy, _, _ in t.obstacles_rect:
            exclude.add((ox, oy))
        prompt = t.nl_variants["direct"]
        messages = [{"role": "user", "content": prompt + "\n\nOutput a Python list of (row,col) tuples representing the path."}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only from response (after the prompt)
        response = text
        if "Output a Python list" in text:
            response = text.split("Output a Python list")[-1]

        path = extract_path(response, exclude)
        grid = np.array(t.grid)

        if path is None or len(path) < 2:
            no_path += 1
            if i < 3: print(f"  [{i+1}] ❌ no parseable path")
            continue

        in_bounds = _is_in_bounds(path, grid.shape)
        collision_free = _is_collision_free(path, grid)
        valid_steps = _is_valid_steps(path)

        if not in_bounds: oob += 1
        elif not collision_free: collisions += 1
        elif not valid_steps: bad_steps += 1
        else:
            valid += 1
            if len(path) - 1 == t.optimal_length: optimal += 1

        if i < 5:
            vmark = "✅" if (in_bounds and collision_free and valid_steps) else "❌"
            print(f"  [{i+1}] {vmark} len={len(path)-1} opt={t.optimal_length} "
                  f"({('OOB' if not in_bounds else 'collision' if not collision_free else 'bad_step' if not valid_steps else 'VALID')})")

    print(f"\n  Valid: {valid}/{n} ({100*valid/n:.1f}%)  Optimal: {optimal}/{n} ({100*optimal/n:.1f}%)")
    print(f"  Failures: no_path={no_path} oob={oob} collision={collisions} bad_steps={bad_steps}")
    del model; torch.cuda.empty_cache()
    return {"valid": valid, "optimal": optimal, "total": n, "no_path": no_path, "oob": oob, "collisions": collisions, "bad_steps": bad_steps}

if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    results = {}
    results["untrained"] = eval_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "UNTRAINED DeepSeek-R1-Qwen-1.5B", 50)
    results["alphamaze"] = eval_model("./data/models/alphamaze-v0.2-1.5b", "AlphaMaze-v0.2-1.5B (TRAINED)", 50)
    with open("./results/baseline_clean.json", "w") as f: json.dump(results, f, indent=2)
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Untrained: {results['untrained']['valid']}/50  AlphaMaze: {results['alphamaze']['valid']}/50")
    print(f"Saved to ./results/baseline_clean.json")
