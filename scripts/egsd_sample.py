"""EGSD Phase 0/1/2 sampling + engine grading (egsd-plan.md).

Sample NL reasoning traces from the current model on pool positions,
grade each trace with Stockfish d12 (outcome + process + parseable), and
emit the accepted set for SFT. Multi-round: point --model at the previous
round's adapter to self-improve.

Grading (identical semantics to train_mate_grpo rewards):
- outcome: parsed MoveA/MoveB label == oracle best label at the position
- process: every UCI claim in the trace is legal AND eval-stable
  (|delta eval| <= 100cp, Stockfish d12), checked in sequence
- parse: completion contains a parseable MoveA:/MoveB: answer
Accepted = parse AND outcome AND process == 1.0 (all-pass).

Usage:
    python3 scripts/egsd_sample.py \
        --pool results/rlvr-pool/train-5k.jsonl \
        --model google/gemma-4-E2B-it [--adapter results/r1-adapter] \
        --out results/egsd-r1-accepted.jsonl \
        --n-positions 1000 --n-rollouts 4 \
        --max-tokens 1024 --temperature 0.8 \
        --stockfish /opt/homebrew/bin/stockfish --depth 12
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# gated model (google/gemma-4-E2B-it): read the write token from .env so
# from_pretrained authenticates (same pattern as train_mate_grpo)
if not os.environ.get("HF_TOKEN"):
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_WRITE_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                if tok:
                    os.environ["HF_TOKEN"] = tok
                break

from train_mate_grpo import (ANSWER_SPEC_FORCED, Oracle,  # noqa: E402
                             parse_choice)


def build_prompt(fen: str, candidate_a: str, candidate_b: str) -> str:
    instruction = ("You are an expert chess player. You are given a chess "
                   "board with FEN format. Your goal is to choose a better "
                   "move given two candidate moves.")
    board_text = (f'The FEN of the given chess board is "{fen}". Which move '
                  f"is better? MoveA:{candidate_a} MoveB:{candidate_b} ")
    return instruction + "\n" + board_text + "\n" + ANSWER_SPEC_FORCED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--adapter", default="",
                    help="LoRA adapter dir from the previous round (SFT "
                         "path); empty = base model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-positions", type=int, default=1000)
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="SAMPLING cap only (rollout budget for data "
                         "generation); the EGSD loop trains no cap")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        print("[egsd] WARNING: no CUDA; CPU smoke mode (tiny --n-positions "
              "expected)", flush=True)
        device = "cpu"
    else:
        device = "cuda"

    # ---- proven 4-bit P100 stack (same shims as train_mate_grpo) ----
    import transformers.utils.quantization_config as _qc
    if not hasattr(_qc, "torch"):
        _qc.torch = torch
    if not hasattr(torch.nn.Module, "set_submodule"):
        def _set_submodule(self, target, module):
            atoms = target.split(".")
            parent = self
            for atom in atoms[:-1]:
                parent = getattr(parent, atom)
            setattr(parent, atoms[-1], module)
        torch.nn.Module.set_submodule = _set_submodule

    processor = AutoProcessor.from_pretrained(args.model,
                                              trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        from transformers import BitsAndBytesConfig
        cap = torch.cuda.get_device_capability(0)
        compute_dtype = torch.bfloat16 if cap >= (7, 5) else torch.float16
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, quantization_config=quant, device_map={"": 0},
            dtype=compute_dtype)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, device_map="cpu", dtype=torch.float32,
            low_cpu_mem_usage=True)

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"[egsd] adapter merged: {args.adapter}", flush=True)

    model.eval()
    oracle = Oracle("stockfish", depth=args.depth,
                    stockfish=args.stockfish)

    rows = [json.loads(l) for l in open(args.pool)]
    if args.n_positions > 0:
        rows = rows[:args.n_positions]
    print(f"[egsd] {len(rows)} positions x {args.n_rollouts} rollouts "
          f"| max_tokens={args.max_tokens} temp={args.temperature}",
          flush=True)

    accepted = []
    rejected = 0
    t0 = time.time()
    for pi, row in enumerate(rows):
        fen, ca, cb, truth = (row["fen"], row["candidate_a"],
                              row["candidate_b"], row["truth_label"])
        prompt = build_prompt(fen, ca, cb)
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True)
        inputs = processor(text=text, return_tensors="pt",
                           add_special_tokens=False)
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        best_label = oracle.best_label(fen, ca, cb, truth)

        for ri in range(args.n_rollouts):
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=[1, 106],
                    pad_token_id=tokenizer.pad_token_id)
            completion = processor.decode(
                gen[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False)
            text_c = completion

            label, move = parse_choice(text_c, ca, cb)
            parsed = label is not None
            ok = (best_label is not None and label == best_label
                  or (move and move == oracle.best_move(fen)))
            outcome = 1.0 if ok else 0.0

            # process: legality + eval-stability of every UCI claim
            uci_re = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?")
            moves = uci_re.findall(text_c)
            process = 0.0
            if moves:
                import chess as _ch
                stable = 0
                board = _ch.Board(fen)
                try:
                    for m in moves:
                        mv = _ch.Move.from_uci(m)
                        if mv not in board.legal_moves:
                            break
                        ea = oracle.eval_cp(board.fen())
                        board.push(mv)
                        eb = oracle.eval_cp(board.fen())
                        if (ea is not None and eb is not None
                                and abs(eb - ea) <= 100.0):
                            stable += 1
                        else:
                            break
                except Exception:
                    pass
                process = stable / len(moves)

            record = {
                "fen": fen, "candidate_a": ca, "candidate_b": cb,
                "truth": truth, "best_label": best_label,
                "completion": text_c,
                "tokens": len(gen[0]) - inputs["input_ids"].shape[1],
                "label": label, "move": move,
                "outcome": outcome, "process": round(process, 4),
                "parsed": parsed,
                "accept": bool(parsed and outcome == 1.0 and process == 1.0),
            }
            if record["accept"]:
                accepted.append(record)
            else:
                rejected += 1

        if (pi + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[egsd] {pi+1}/{len(rows)} positions | "
                  f"accepted={len(accepted)} rejected={rejected} "
                  f"| {el:.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Also emit the SFT-ready form (train_mate_lora schema: messages +
    # fen) so the accepted set drops straight into the SFT trainer.
    sft_out = out.with_name(out.stem + "_sft.jsonl")
    with sft_out.open("w") as f:
        for r in accepted:
            f.write(json.dumps({
                "fen": r["fen"],
                "messages": [
                    {"role": "user",
                     "content": build_prompt(r["fen"], r["candidate_a"],
                                             r["candidate_b"])},
                    {"role": "assistant", "content": r["completion"]},
                ],
            }, ensure_ascii=False) + "\n")
    print(f"[egsd] SFT-ready -> {sft_out}", flush=True)
    print(f"[egsd] DONE: {len(accepted)} accepted, {rejected} rejected "
          f"-> {out}", flush=True)
    print(f"[egsd] accept rate: "
          f"{len(accepted)/(len(accepted)+rejected)*100:.1f}%", flush=True)
    oracle.close()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[egsd] FAILED: {e}", flush=True)
        sys.exit(1)
