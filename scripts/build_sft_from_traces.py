"""Build the caveman-SFT chat-pair dataset from verified traces.

    python3 scripts/build_sft_from_traces.py \
        --traces results/caveman/traces-2000.jsonl \
        --out data/positions/caveman-sft

User prompt is byte-identical to the run_mate_eval protocol prompt
(copied verbatim from data/positions/noexplain-slice/train.jsonl rows,
verified 2026-08-18 — trailing space after the MoveB candidate and the
literal ANSWER_SPEC included). Assistant = verified trace + the
engine-given answer appended deterministically:

    <caveman explanation>
    MoveX:<uci>

Only rows that PASS verification AND whose engine_preferred answer
agrees with the expert truth_label are kept — the model is never taught
a wrong answer (1995/2000 agree in lines-2000; the 5 disagreements drop).

Split is 90/10, position-disjoint (seed 42), exactly like the earlier
noexplain-slice split.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ANSWER_SPEC = ("Answer with exactly one of: MoveA:<move> or MoveB:<move>. "
               "Only output the line, nothing else.")


def build_user_prompt(fen: str, candidate_a: str, candidate_b: str) -> str:
    return (f"You are an expert chess player. You are given a chess board "
            f"with FEN format. Your goal is to choose a better move given "
            f"two candidate moves.\n"
            f"The FEN of the given chess board is \"{fen}\". "
            f"Which move is better? MoveA:{candidate_a} MoveB:{candidate_b} "
            f"\n{ANSWER_SPEC}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True,
                    help="synthesize_caveman_traces.py output (verified rows)")
    ap.add_argument("--out", default="data/positions/caveman-sft",
                    help="dir for train.jsonl + eval.jsonl")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.traces).read_text().splitlines()
            if l.strip()]
    kept = []
    skipped = {"unverified": 0, "disagreement": 0, "no_trace": 0}
    for r in rows:
        if not r.get("verified"):
            skipped["unverified"] += 1
            continue
        trace = (r.get("trace") or "").strip()
        if not trace:
            skipped["no_trace"] += 1
            continue
        label = r.get("engine_preferred") or r.get("choice_label")
        if label not in ("A", "B"):
            skipped["unverified"] += 1
            continue
        if r.get("truth_label") and label != r["truth_label"]:
            skipped["disagreement"] += 1
            continue
        uci = r.get("answer") or f"Move{label}:{r[f'candidate_{label.lower()}']}"
        msgs = [
            {"role": "user",
             "content": build_user_prompt(r["fen"], r["candidate_a"],
                                          r["candidate_b"])},
            {"role": "assistant",
             "content": f"{trace}\n{uci}"},
        ]
        kept.append({"fen": r["fen"], "messages": msgs})
    print(f"traces: {len(rows)} -> kept {len(kept)} "
          f"(skipped {skipped})", flush=True)
    if not kept:
        raise SystemExit("nothing to split")

    rng = random.Random(args.seed)
    fens = sorted({r["fen"] for r in kept})
    rng.shuffle(fens)
    n_eval = max(1, int(len(fens) * args.eval_frac))
    eval_fens = set(fens[:n_eval])
    train, eval_rows = [], []
    for r in kept:
        (eval_rows if r["fen"] in eval_fens else train).append(r)
    print(f"split: {len(train)} train / {len(eval_rows)} eval "
          f"({len(eval_fens)} eval positions)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("eval", eval_rows)):
        with (out / f"{name}.jsonl").open("w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    print(f"written: {out}/train.jsonl ({len(train)}), "
          f"{out}/eval.jsonl ({len(eval_rows)})", flush=True)


if __name__ == "__main__":
    main()
