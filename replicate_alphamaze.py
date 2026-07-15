#!/usr/bin/env python3
"""Replicate AlphaMaze exactly — correct prompt, sampling, full thinking, no cutoff."""
import torch, re, time, numpy as np, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

sys.path.insert(0, '.')
from src.grid_generator import generate_gridroute_maps
from src.evaluation import _is_collision_free, _is_in_bounds, _is_valid_steps

tok = AutoTokenizer.from_pretrained('./data/models/alphamaze-v0.2-1.5b')
if tok.pad_token is None: tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    './data/models/alphamaze-v0.2-1.5b', quantization_config=bnb,
    device_map='auto', torch_dtype=torch.bfloat16)
print(f'Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB')

# ── EXACT AlphaMaze system prompt (from their inference.py) ──
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

# ═══════════════════════════════════════════════════════════════════
# MAZEBENCH — Exact replication
# ═══════════════════════════════════════════════════════════════════
ds = load_dataset('Menlo/Maze-Bench-v0.2', split='test')
rng = np.random.RandomState(42)
idx = sorted(rng.choice(len(ds), 10, replace=False))

print(f'\n{"="*60}')
print('MAZEBENCH — AlphaMaze exact replication')
print(f'{"="*60}')

correct, finished = 0, 0
for i, j in enumerate(idx):
    row = ds[int(j)]
    prompt = f'{SYS} MAZE: {row["Prompt"]}'
    msgs = [{'role': 'user', 'content': prompt}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_tensors='pt')
    inp_ids = inp['input_ids'].to(model.device)

    t0 = time.time()
    out = model.generate(
        input_ids=inp_ids, max_new_tokens=8192, temperature=0.8,
        repetition_penalty=1.1, do_sample=True, eos_token_id=tok.eos_token_id)
    elapsed = time.time() - t0
    ntokens = out.shape[-1] - inp_ids.shape[-1]
    did_finish = ntokens < 8192
    if did_finish: finished += 1

    resp = tok.decode(out[0][inp_ids.shape[-1]:], skip_special_tokens=True)
    gt = re.findall(r'<\|(up|down|left|right)\|>', row['Response'])
    pred = re.findall(r'<\|(up|down|left|right)\|>', resp)

    ok = pred == gt
    if ok: correct += 1

    status = '✅' if ok else '❌'
    cutoff = '' if did_finish else ' [CUT OFF]'
    print(f'  [{i+1}/10] {status} GT={gt} Pred={pred}{cutoff} ({ntokens}tok, {elapsed:.0f}s)')
    if i < 3:
        print(f'    Output: {resp[:250]}...')
        print()

print(f'\n  Accuracy: {correct}/10 ({100*correct/10:.0f}%)  Finished: {finished}/10')

# ═══════════════════════════════════════════════════════════════════
# GRIDROUTE 5x5 — Structured final answer
# ═══════════════════════════════════════════════════════════════════
tasks = generate_gridroute_maps(size=5, obstacle_size=1, num_obstacles=1,
                                 num_maps=20, pairs_per_map=1, seed=99)
rng2 = np.random.RandomState(99)
idx2 = sorted(rng2.choice(len(tasks), 10, replace=False))

print(f'\n{"="*60}')
print('GRIDROUTE 5x5 — Structured FINAL ANSWER')
print(f'{"="*60}')

GR_PROMPT_SUFFIX = (
    "\n\nThink step by step about the path. After your thinking, output your final answer "
    "on a new line starting with exactly 'FINAL ANSWER:' followed by a Python list of "
    "(row,col) tuples, like: FINAL ANSWER: [(0,0), (0,1), (1,1)]"
)

valid, optimal = 0, 0
for i, j in enumerate(idx2):
    t = tasks[j]
    prompt = t.nl_variants['direct'] + GR_PROMPT_SUFFIX
    msgs = [{'role': 'user', 'content': prompt}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_tensors='pt')
    inp_ids = inp['input_ids'].to(model.device)

    t0 = time.time()
    out = model.generate(
        input_ids=inp_ids, max_new_tokens=4096, temperature=0.8,
        repetition_penalty=1.1, do_sample=True, eos_token_id=tok.eos_token_id)
    elapsed = time.time() - t0
    ntokens = out.shape[-1] - inp_ids.shape[-1]
    did_finish = ntokens < 4096

    resp = tok.decode(out[0][inp_ids.shape[-1]:], skip_special_tokens=True)

    # Only parse FINAL ANSWER section — not thinking
    m = re.search(r'FINAL\s*ANSWER\s*:\s*(.+)', resp, re.IGNORECASE | re.DOTALL)
    answer_section = m.group(1).strip() if m else resp
    coords = re.findall(r'\((\d+),\s*(\d+)\)', answer_section)
    path = [(int(x), int(y)) for x, y in coords]

    grid = np.array(t.grid)
    ib = _is_in_bounds(path, grid.shape) if path else False
    cf = _is_collision_free(path, grid) if path and ib else False
    vs = _is_valid_steps(path) if path and cf else False
    if ib and cf and vs:
        valid += 1
        if len(path) - 1 == t.optimal_length: optimal += 1

    s = '✅OPT' if (ib and cf and vs and len(path)-1 == t.optimal_length) else \
        ('✅VAL' if (ib and cf and vs) else '❌')
    cutoff = ' [CUT OFF]' if not did_finish else ''
    no_final = ' [NO FINAL ANSWER]' if not m else ''
    print(f'  [{i+1}/10] {s} Start={t.start} Goal={t.goal} Opt={t.optimal_length} '
          f'Got={len(path)-1 if path else 0}{cutoff}{no_final}')
    if i < 3:
        print(f'    Path: {path[:8]}...')
        print(f'    Output: {resp[:300]}...')
        print()

print(f'\n  Valid: {valid}/10  Optimal: {optimal}/10')
print(f'\n=== DONE ===')
del model; torch.cuda.empty_cache()
