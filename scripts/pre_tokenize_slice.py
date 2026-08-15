"""Pre-tokenize the noexplain slice dataset so the Kaggle kernel does NOT
spend 36+ minutes calling apply_chat_template per row on a shared CPU.

Outputs <dir>/train_pretok.jsonl and <dir>/eval_pretok.jsonl — one JSON
object per line with input_ids + labels (assistant-mask via prefix
difference, identical logic to train_mate_lora.to_ids). The trainer loads
these directly when present.

    python3 scripts/pre_tokenize_slice.py \
        --train data/positions/noexplain-slice/train.jsonl \
        --eval data/positions/noexplain-slice/eval.jsonl \
        --out data/positions/noexplain-slice
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoProcessor

MAX_SEQ = 2048


def tokenize_rows(processor, rows_path: Path) -> list[dict]:
    out = []
    with open(rows_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            msgs = d["messages"]
            full = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt",
                enable_thinking=False)["input_ids"][0]
            prompt = processor.apply_chat_template(
                msgs[:1], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
                enable_thinking=False)["input_ids"][0]
            if not full[:len(prompt)].tolist() == prompt.tolist():
                raise RuntimeError("prompt not a prefix -- mask would be wrong")
            iids = full.tolist()[:MAX_SEQ]
            labels = ([-100] * len(prompt) + iids[len(prompt):])[:MAX_SEQ]
            out.append({"input_ids": iids, "labels": labels})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")

    t0 = time.time()
    print("tokenizing train...", flush=True)
    train = tokenize_rows(processor, Path(args.train))
    print(f"  {len(train)} rows in {(time.time()-t0)/60:.1f}min", flush=True)
    with open(out / "train_pretok.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")

    t0 = time.time()
    print("tokenizing eval...", flush=True)
    ev = tokenize_rows(processor, Path(args.eval))
    print(f"  {len(ev)} rows in {(time.time()-t0)/60:.1f}min", flush=True)
    with open(out / "eval_pretok.jsonl", "w") as f:
        for r in ev:
            f.write(json.dumps(r) + "\n")
    print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
