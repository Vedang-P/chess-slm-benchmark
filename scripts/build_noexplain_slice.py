"""Build the noexplain slice SFT dataset: phase-natural labels for the
noexplain-first vertical slice.

Reads the MATE noexplain train pool (data/raw/mate-train/noexplain/*.jsonl),
excludes every FEN in the four MATE test sets, samples n_train + n_eval
positions uniformly (phase-natural — the pool is already ~91/6/3
opening/middlegame/endgame, matching the eval distribution), and writes
chat-pair JSONL exactly like build_mate_lora_data.py (byte-identical eval
prompt: candidates + trailing space + ANSWER_SPEC).

    python3 scripts/build_noexplain_slice.py --n-train 600000 --n-eval 5000
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

ROOT = Path(__file__).resolve().parent.parent
NOEXPLAIN_GLOB = sorted((ROOT / "data/raw/mate-train/noexplain").glob("*/*.jsonl"))

ANSWER_SPEC = (
    "Answer with exactly one of: MoveA:<move> or MoveB:<move>. "
    "Only output the line, nothing else."
)

INSTRUCTION = ("You are an expert chess player. You are given a chess board "
               "with FEN format. Your goal is to choose a better move given "
               "two candidate moves.")


def _fen_from_input(text: str) -> str | None:
    m = re.search(r'"([^"]+)"', text)
    return m.group(1) if m else None


def _normalize_output(output: str) -> str:
    m = re.search(r"Move([AB]):\s*([a-hNBRQK][a-h1-8xO-]{1,7})", output)
    return f"Move{m.group(1)}:{m.group(2)}" if m else output.strip()


def mate_prompt(fen: str, input_text: str) -> str:
    return f"{INSTRUCTION}\n{input_text}\n{ANSWER_SPEC}"


def load_test_fens() -> set[str]:
    test = set()
    for name in ("mate-selection-test.json", "mate-selection-test-noexplain.json",
                 "mate-selection-test-tactic.json", "mate-selection-test-both.json"):
        p = ROOT / "data/positions" / name
        if p.exists():
            for r in json.loads(p.read_text()):
                test.add(r["fen"])
    return test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=600000)
    ap.add_argument("--n-eval", type=int, default=5000)
    ap.add_argument("--out", default=str(ROOT / "data/positions/noexplain-slice"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    test_fens = load_test_fens()
    print(f"test FENs excluded: {len(test_fens)}", flush=True)

    # stream once; keep the first (fen, input, output) per FEN
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
                if not fen or fen in test_fens or fen in seen:
                    continue
                seen.add(fen)
                pool.append((fen, d.get("input", ""), d.get("output", "")))
                if len(pool) >= args.n_train + args.n_eval:
                    break
        if len(pool) >= args.n_train + args.n_eval:
            break
    print(f"pool: {len(pool)} positions", flush=True)

    rng.shuffle(pool)
    train_rows, eval_rows = pool[:args.n_train], pool[args.n_train:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(rows, name):
        with open(out / name, "w") as f:
            for fen, inp, outp in rows:
                rec = {
                    "messages": [
                        {"role": "user", "content": mate_prompt(fen, inp)},
                        {"role": "assistant", "content": _normalize_output(outp)},
                    ],
                    "fen": fen,
                }
                f.write(json.dumps(rec) + "\n")
        return len(rows)

    n_train = write(train_rows, "train.jsonl")
    n_eval = write(eval_rows, "eval.jsonl")

    # manifest: counts + phase-natural sanity (sampled pool phase split)
    from build_phase_dataset import classify_fen
    phase = Counter()
    for fen, _, _ in train_rows[:20000]:
        p = classify_fen(fen)
        phase["opening" if p == "opening" else ("middlegame" if p == "middlegame"
                                                else ("endgame" if p == "endgame"
                                                      else "sparse"))] += 1
    manifest = {
        "classifier": "build_phase_dataset.v1",
        "subset": "noexplain",
        "seed": args.seed,
        "n_train": n_train,
        "n_eval": n_eval,
        "test_overlap": 0,
        "sampling": "phase-natural (uniform over pool; pool is ~91/6/3)",
        "train_phase_sample_20k": {k: v for k, v in phase.items()},
        "answer_spec": ANSWER_SPEC,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"train: {n_train} | eval: {n_eval} -> {out}", flush=True)
    print("phase sample:", dict(phase), flush=True)


if __name__ == "__main__":
    main()
