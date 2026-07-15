#!/usr/bin/env python3
"""Full AlphaMaze replication with detailed results + raw outputs saved."""
import torch, re, time, numpy as np, sys, json, os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

sys.path.insert(0, '.')
from src.grid_generator import generate_gridroute_maps
from src.evaluation import _is_collision_free, _is_in_bounds, _is_valid_steps

OUT_DIR = Path('./results/alphamaze_replication')
OUT_DIR.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained('./data/models/alphamaze-v0.2-1.5b')
if tok.pad_token is None: tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    './data/models/alphamaze-v0.2-1.5b', quantization_config=bnb,
    device_map='auto', torch_dtype=torch.bfloat16)

SYS = (
    "You are a helpful assistant that solves mazes. You will be given a maze represented by "
    "a series of tokens. The tokens represent: "
    "- Coordinates: <|row-col|> (e.g., <|0-0|>, <|2-4|>) "
    "- Walls: <|no_wall|>, <|up_wall|>, <|down_wall|>, <|left_wall|>, <|right_wall|>, <|up_down_wall|>, etc. "
    "- Origin: <|origin|> "
    "- Target: <|target|> "
    "- Movement: <|up|>, <|down|>, <|left|>, <|right|>, <|blank|> "
    "Your task is to output the sequence of movements "
    "(<|up|>, <|down|>, <|left|>, <|right|>) required to navigate "
    "from the origin to the target, based on the provided maze representation. "
    "Think step by step. At each step, predict only the next movement token. "
    "Output only the move tokens, separated by spaces."
)

all_results = {"mazebench": [], "gridroute": []}

# ═══════ MAZEBENCH ═══════
ds = load_dataset('Menlo/Maze-Bench-v0.2', split='test')
rng = np.random.RandomState(42); idx = sorted(rng.choice(len(ds), 20, replace=False))

print(f'MAZEBENCH: {len(idx)} mazes...')
correct = 0
for i, j in enumerate(idx):
    row = ds[int(j)]
    prompt = f'{SYS} MAZE: {row["Prompt"]}'
    msgs = [{'role': 'user', 'content': prompt}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_tensors='pt')
    inp_ids = inp['input_ids'].to(model.device)
    t0 = time.time()
    out = model.generate(input_ids=inp_ids, max_new_tokens=8192, temperature=0.8,
                         repetition_penalty=1.1, do_sample=True, eos_token_id=tok.eos_token_id)
    elapsed = time.time() - t0
    ntokens = out.shape[-1] - inp_ids.shape[-1]
    resp = tok.decode(out[0][inp_ids.shape[-1]:], skip_special_tokens=True)
    gt = re.findall(r'<\|(up|down|left|right)\|>', row['Response'])
    pred = re.findall(r'<\|(up|down|left|right)\|>', resp)
    ok = pred == gt
    if ok: correct += 1

    all_results["mazebench"].append({
        "idx": int(j), "level": row["Level"], "correct": ok,
        "gt_moves": gt, "pred_moves": pred,
        "raw_output": resp, "prompt": prompt[:500],
        "ntokens": ntokens, "time_s": round(elapsed, 1),
        "finished": ntokens < 8192,
    })
    print(f'  [{i+1}/20] {"✅" if ok else "❌"} L={row["Level"]} GT={gt} Pred={pred} ({ntokens}tok)')

all_results["mazebench_summary"] = {"correct": correct, "total": len(idx), "accuracy_pct": round(100*correct/len(idx), 1)}

# ═══════ GRIDROUTE 5x5 ═══════
GR_SUFFIX = (
    "\n\nThink step by step about the path. After your thinking, output your final answer "
    "on a new line starting with exactly 'FINAL ANSWER:' followed by a Python list of "
    "(row,col) tuples, like: FINAL ANSWER: [(0,0), (0,1), (1,1)]"
)
tasks = generate_gridroute_maps(size=5, obstacle_size=1, num_obstacles=1,
                                 num_maps=30, pairs_per_map=1, seed=99)
rng2 = np.random.RandomState(99); idx2 = sorted(rng2.choice(len(tasks), 20, replace=False))

print(f'\nGRIDROUTE 5x5: {len(idx2)} tasks...')
valid, optimal = 0, 0
for i, j in enumerate(idx2):
    t = tasks[j]
    prompt = t.nl_variants['direct'] + GR_SUFFIX
    msgs = [{'role': 'user', 'content': prompt}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_tensors='pt')
    inp_ids = inp['input_ids'].to(model.device)
    t0 = time.time()
    out = model.generate(input_ids=inp_ids, max_new_tokens=4096, temperature=0.8,
                         repetition_penalty=1.1, do_sample=True, eos_token_id=tok.eos_token_id)
    elapsed = time.time() - t0
    ntokens = out.shape[-1] - inp_ids.shape[-1]
    resp = tok.decode(out[0][inp_ids.shape[-1]:], skip_special_tokens=True)

    # Only parse FINAL ANSWER section
    m = re.search(r'FINAL\s*ANSWER\s*:\s*(.+)', resp, re.IGNORECASE | re.DOTALL)
    answer_section = m.group(1).strip() if m else resp
    coords = re.findall(r'\((\d+),\s*(\d+)\)', answer_section)
    path = [(int(x), int(y)) for x, y in coords]

    grid = np.array(t.grid)
    ib = _is_in_bounds(path, grid.shape) if path else False
    cf = _is_collision_free(path, grid) if path and ib else False
    vs = _is_valid_steps(path) if path and cf else False
    is_valid = ib and cf and vs
    is_optimal = is_valid and len(path) - 1 == t.optimal_length
    if is_valid: valid += 1
    if is_optimal: optimal += 1

    status = 'OPTIMAL' if is_optimal else ('VALID' if is_valid else 'INVALID')
    all_results["gridroute"].append({
        "idx": int(j), "start": list(t.start), "goal": list(t.goal),
        "optimal_length": t.optimal_length, "status": status,
        "parsed_path": path, "path_length": len(path) - 1 if path else 0,
        "raw_output": resp, "prompt": prompt[:500],
        "ntokens": ntokens, "time_s": round(elapsed, 1),
        "finished": ntokens < 4096, "has_final_answer": m is not None,
    })
    print(f'  [{i+1}/20] {status:8s} Start={t.start} Goal={t.goal} Opt={t.optimal_length} Got={len(path)-1 if path else 0}')

all_results["gridroute_summary"] = {"valid": valid, "optimal": optimal, "total": len(idx2),
                                     "valid_pct": round(100*valid/len(idx2), 1),
                                     "optimal_pct": round(100*optimal/len(idx2), 1)}

# Save
ts = time.strftime('%Y%m%d_%H%M%S')
out_path = OUT_DIR / f'alphamaze_replication_{ts}.json'
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f'\nSaved: {out_path}')
print(f'MazeBench: {correct}/{len(idx)} ({100*correct/len(idx):.1f}%)')
print(f'GridRoute: Valid={valid}/{len(idx2)} ({100*valid/len(idx2):.1f}%) Optimal={optimal}/{len(idx2)} ({100*optimal/len(idx2):.1f}%)')
del model; torch.cuda.empty_cache()
