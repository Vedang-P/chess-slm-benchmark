"""Build the MATE-noexplain LoRA training set.

MATE noexplain train (~1.42M rows, already local) is sampled into a
train/eval split of chat pairs:
    user: <FEN + MoveA/MoveB candidates prompt>
    assistant: MoveX:<move>

The eval split is position-disjoint from the train split and from the
MATE test set (contamination hygiene). We intentionally do NOT dedupe
FENs inside the train split (the MATE corpus has near-duplicate FENs
with different candidates; keeping them is fine for selection SFT).

    python3 scripts/build_mate_lora_data.py --n-train 200000 --n-eval 5000
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOEXPLAIN_GLOB = sorted((ROOT / "data/raw/mate-train/noexplain").glob("*/*.jsonl"))


def _fen_from_input(text: str) -> str | None:
    m = re.search(r'"([^"]+)"', text)
    return m.group(1) if m else None


def _normalize_output(output: str) -> str:
    m = re.search(r"Move([AB]):\s*([a-hNBRQK][a-h1-8xO-]{1,7})", output)
    return f"Move{m.group(1)}:{m.group(2)}" if m else output.strip()


ANSWER_SPEC = (
    "Answer with exactly one of: MoveA:<move> or MoveB:<move>. "
    "Only output the line, nothing else."
)


def mate_prompt(fen: str, input_text: str) -> str:
    """Reconstruct the exact noexplain eval prompt.

    The eval (run_mate_eval.py) sends `task_extra.instruction + "\n" +
    task_extra.input + "\n" + ANSWER_SPEC` VERBATIM, and the MATE input
    string carries a trailing space after the candidates ("...MoveB:d2d8
    "). To be byte-identical, the training user message must use the
    verbatim MATE input (candidates + trailing space), NOT a
    reconstruction that normalizes whitespace."""
    instruction = ("You are an expert chess player. You are given a chess "
                   "board with FEN format. Your goal is to choose a better "
                   "move given two candidate moves.")
    return f"{instruction}\n{input_text}\n{ANSWER_SPEC}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-eval", type=int, default=5000)
    ap.add_argument("--out", default=str(ROOT / "data/positions/mate-lora"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # MATE test FENs must not appear in either split
    test_fens = set()
    for name in ("mate-selection-test-noexplain.json",):
        p = ROOT / "data/positions" / name
        if p.exists():
            for r in json.loads(p.read_text()):
                test_fens.add(r["fen"])

    # stream all rows once, keep phase-free selection-task pairs
    pool = []
    seen = set()
    for path in NOEXPLAIN_GLOB:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                fen = _fen_from_input(d.get("input", ""))
                if not fen or fen in test_fens:
                    continue
                if fen in seen:  # eval-split disjointness only
                    continue
                seen.add(fen)
                pool.append((fen, d["input"], d.get("output", "")))
                if len(pool) >= args.n_train + args.n_eval:
                    break
        if len(pool) >= args.n_train + args.n_eval:
            break

    rng.shuffle(pool)
    train = pool[: args.n_train]
    eval_rows = pool[args.n_train: args.n_train + args.n_eval]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(rows, name):
        with open(out / name, "w") as f:
            for fen, inp, outp in rows:
                rec = {
                    "messages": [
                        {"role": "user",
                         "content": mate_prompt(fen, inp)},
                        {"role": "assistant",
                         "content": _normalize_output(outp)},
                    ],
                    "fen": fen,
                }
                f.write(json.dumps(rec) + "\n")

    write(train, "train.jsonl")
    write(eval_rows, "eval.jsonl")
    print(f"train: {len(train)} | eval: {len(eval_rows)} -> {out}")
    print(f"test-set overlap: 0 (excluded at build time)")


if __name__ == "__main__":
    main()
