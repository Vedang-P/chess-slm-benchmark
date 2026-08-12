"""Evaluate a trained MATE text transformer on the shared testset.

Uses the SAME 1k-position noexplain testset as the campaign
(data/positions/mate-selection-test-noexplain.json) so numbers are
directly comparable to gemma 61.1% / deepseek 92.2% / MATE LLaMA 89.7%.

    python3 scripts/eval_mate_text.py \
        --model results/mate-text-joint/best.pt \
        [--task-file data/positions/mate-selection-test-noexplain.json]

Reports overall accuracy + per-subset breakdown (the testset carries
`theme` per row) + a candidate-order sanity split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mate_text.model import MateTextConfig, MateTextTransformer  # noqa: E402
from src.mate_text.tokenizer import MateTokenizer  # noqa: E402


def load_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    tok = MateTokenizer()
    cfg = MateTextConfig(**ckpt["cfg"])
    model = MateTextTransformer(cfg).to(device)
    if device == "cuda":
        model = model.bfloat16()
        state = {k: v.bfloat16() for k, v in ckpt["model"].items()}
    else:
        state = ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    return model, tok


@torch.no_grad()
def eval_task(model, tok, task_path: str, device: str,
              sample: int | None = None) -> dict:
    rows = json.load(open(task_path))
    if sample:
        rows = rows[:sample]
    by_theme: dict[str, list[int]] = {}
    by_order: dict[str, list[int]] = {"A_truth": [], "B_truth": []}
    total = correct = 0
    for r in rows:
        te = r.get("task_extra", {})
        board = chess.Board(r["fen"])
        bid = tok.board_ids(board)
        tid = tok.text_ids(te["candidate_a"], te["candidate_b"])
        toks = torch.tensor([bid + tid]).to(device)
        types = torch.zeros(1, len(toks), dtype=torch.long).to(device)
        logits, _ = model(toks, types)
        pred = "A" if logits.argmax(-1).item() == 0 else "B"
        label = te["truth_label"]
        ok = pred == label
        total += 1
        correct += ok
        theme = te.get("theme", "unknown")
        by_theme.setdefault(theme, []).append(int(ok))
        by_order[f"{label}_truth"].append(int(ok))
    return {
        "n": total,
        "accuracy": correct / total,
        "by_theme": {k: (sum(v) / len(v), len(v)) for k, v in by_theme.items()},
        "by_order": {k: (sum(v) / len(v), len(v)) for k, v in by_order.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task-file",
                    default="data/positions/mate-selection-test-noexplain.json")
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, device)
    res = eval_task(model, tok, args.task_file, device, args.sample)
    print(f"model: {args.model}")
    print(f"task: {args.task_file} ({res['n']} positions)")
    print(f"accuracy: {res['accuracy']:.4f}")
    for theme, (acc, n) in res["by_theme"].items():
        print(f"  theme {theme}: {acc:.4f} ({n})")
    for order, (acc, n) in res["by_order"].items():
        print(f"  {order}: {acc:.4f} ({n})")


if __name__ == "__main__":
    main()
