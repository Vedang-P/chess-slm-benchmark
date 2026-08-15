"""Research-grade metrics for the SFT run, logged to wandb.

Everything the paper needs beyond train/eval loss:

- live_accuracy: real-task accuracy on a sample of the ACTUAL noexplain
  test set (byte-identical eval prompt, thinking off for speed) computed
  during training -> the "accuracy vs training step" curve
- live_accuracy_*_phase: per-phase breakdown (opening/middlegame/endgame)
- tokens_per_correct: output tokens per correct answer (the paper's
  efficiency headline metric), tracked live
- parse_rate, samples/sec, tokens/sec, GPU mem (GB)
- config: full hyperparams + data provenance + git commit

Used by train_mate_lora.py via a TrainerCallback.
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import torch

ANSWER_RE = re.compile(
    r"\bMove\s*([AB])\b\s*[:.\-]?\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)?", re.I)


def phase_of_fen(fen: str) -> str:
    import sys
    from pathlib import Path as _P
    _scripts = _P(__file__).resolve().parent.parent / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from build_phase_dataset import classify_fen
    p = classify_fen(fen)
    return "opening" if p == "opening" else ("middlegame" if p == "middlegame"
                                             else "endgame")


def parse_choice(text: str, candidate_a: str, candidate_b: str):
    matches = list(ANSWER_RE.finditer(text))
    if matches:
        m = matches[-1]
        label = m.group(1).upper()
        move = m.group(2) or (candidate_a if label == "A" else candidate_b)
        return label, move
    return None, None


class LiveEvalCallback:
    """Compute real-task metrics on a test-set sample every N steps and
    log them to wandb. Uses the exact run_mate_eval prompt so the curve is
    comparable to the final numbers."""

    def __init__(self, model, processor, test_path: str,
                 n: int = 100, every_steps: int = 500,
                 seed: int = 42, max_new_tokens: int = 64,
                 tag: str = "live"):
        self.model = model
        self.processor = processor
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens
        self.tag = tag
        rows = json.loads(Path(test_path).read_text())
        rng = random.Random(seed)
        self.rows = rng.sample(rows, min(n, len(rows)))
        self.last_step = -10**9
        self.t0 = time.time()
        self.steps_done = 0
        self.tokens_gen = 0
        self._cached_prompts = None

    def _build_prompts(self) -> list[dict]:
        if self._cached_prompts is not None:
            return self._cached_prompts
        out = []
        for r in self.rows:
            te = r.get("task_extra", {})
            prompt = (te.get("instruction", "") + "\n" + te.get("input", "") +
                      "\nAnswer with exactly one of: MoveA:<move> or "
                      "MoveB:<move>. Only output the line, nothing else.")
            out.append({"prompt": prompt,
                        "candidate_a": te.get("candidate_a", ""),
                        "candidate_b": te.get("candidate_b", ""),
                        "truth_label": te.get("truth_label"),
                        "phase": phase_of_fen(r["fen"])})
        self._cached_prompts = out
        return out

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step - self.last_step < self.every_steps:
            return
        self.last_step = step
        self.steps_done += 1
        self._run_eval(step)

    def on_train_end(self, args, state, control, **kwargs):
        self._run_eval(state.global_step, final=True)

    def _run_eval(self, step: int, final: bool = False):
        import wandb

        prompts = self._build_prompts()
        self.model.eval()
        n_correct = 0
        n_parsed = 0
        n_total = 0
        tokens_used = 0
        per_phase = {"opening": [0, 0], "middlegame": [0, 0], "endgame": [0, 0]}
        t_start = time.time()
        with torch.no_grad():
            for p in prompts:
                msg = [{"role": "user", "content": p["prompt"]}]
                try:
                    inputs = self.processor.apply_chat_template(
                        msg, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt",
                        enable_thinking=False).to(self.model.device)
                    out = self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens,
                        do_sample=False, pad_token_id=self.processor.tokenizer.eos_token_id)
                    input_len = inputs["input_ids"].shape[-1]
                    content = self.processor.decode(
                        out[0][input_len:], skip_special_tokens=True)
                    used = int(out[0][input_len:].shape[-1])
                except Exception as e:
                    print(f"[live-eval] row failed: {e}", flush=True)
                    continue
                n_total += 1
                tokens_used += used
                self.tokens_gen += used
                label, move = parse_choice(content, p["candidate_a"], p["candidate_b"])
                if label is None:
                    continue
                n_parsed += 1
                if label == p["truth_label"]:
                    n_correct += 1
                    per_phase[p["phase"]][0] += 1
                per_phase[p["phase"]][1] += 1
        self.model.train()
        dt = time.time() - t_start
        acc = n_correct / n_total if n_total else 0.0
        parse_rate = n_parsed / n_total if n_total else 0.0
        tok_per_correct = (tokens_used / n_correct) if n_correct else None
        log = {
            f"{self.tag}/accuracy": acc,
            f"{self.tag}/parse_rate": parse_rate,
            f"{self.tag}/n_evaluated": n_total,
            f"{self.tag}/samples_per_sec": n_total / dt if dt else 0,
            f"{self.tag}/tokens_per_correct": tok_per_correct,
            f"{self.tag}/total_tokens_used": tokens_used,
            "system/gpu_mem_gb": _gpu_mem_gb(),
        }
        for ph, (c, t) in per_phase.items():
            if t:
                log[f"{self.tag}/acc_{ph}"] = c / t
                log[f"{self.tag}/n_{ph}"] = t
        wandb.log(log, step=step)
        print(f"[live-eval] step={step} acc={acc:.3f} parse={parse_rate:.3f} "
              f"tok/correct={tok_per_correct if tok_per_correct else 'NA'} "
              f"({n_total} rows)", flush=True)


def _gpu_mem_gb() -> float:
    if torch.cuda.is_available():
        try:
            return torch.cuda.memory_allocated() / 1024**3
        except Exception:
            return -1.0
    return -1.0


def log_run_config(wandb_run, args, extra: dict | None = None):
    """Log the full training config + data provenance to wandb.config."""
    cfg = {
        "base_model": args.base,
        "train_path": args.train,
        "eval_path": args.eval,
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank,
        "lora_alpha": args.alpha,
        "batch_size": args.batch,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch * args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "train_tag": args.train_tag,
        "hf_repo": args.hf_repo,
        "hf_upload_every_s": args.hf_upload_every,
        "lora_target_modules": "all-linear",
        "quantization": "4bit-nf4-double-quant",
        "git_commit": _git_commit(),
        "data_provenance": "MATE noexplain train, phase-natural, "
                           "test-FEN-excluded, seed 42",
    }
    if extra:
        cfg.update(extra)
    wandb_run.config.update(cfg, allow_val_change=True)


def _git_commit() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parent.parent)
                              ).stdout.strip()
    except Exception:
        return "unknown"
