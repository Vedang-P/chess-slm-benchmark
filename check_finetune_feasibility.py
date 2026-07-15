"""Feasibility check: can each candidate model be loaded, LoRA-attached, and
survive a forward+backward pass on available VRAM?

Routes through hf_models.load_trainable_model() -- the same loader
train_sft.py/train_grpo.py use, bitsandbytes+peft for every model including
Gemma 4 (peft>=0.19.0, pinned in requirements.txt, ships its own
Gemma4ClippableLinear support) -- so this measures the actual path each
model will really train with, not a generic approximation that might not
even be able to attach LoRA to begin with.

Run this on the actual training hardware (laptop or Kaggle) before
committing to a full run -- especially for gemma4-e4b, whose LoRA VRAM
footprint has been reported elsewhere as ~17GB, right at/over a free-tier
Kaggle T4's 16GB ceiling. Don't trust that number until this script confirms
it on the real hardware you'll actually train on.
"""

import traceback

import torch

# ── candidates ────────────────────────────────────────────────────
CANDIDATES = [
    ("deepseek-r1-distill-qwen-1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("gemma4-e2b", "unsloth/gemma-4-E2B-it"),
    ("gemma4-e4b", "unsloth/gemma-4-E4B-it"),
]


def check(key: str, model_id: str) -> dict:
    """Load, LoRA-attach, forward+backward, measure VRAM.

    Cleanup (del model/tokenizer + empty_cache) always runs, success or
    failure -- a candidate that OOMs or crashes mid-forward-pass previously
    left its weights resident in VRAM, so the NEXT candidate in CANDIDATES
    would fail for an unrelated reason (leftover memory from the dead model),
    not its own actual footprint. On failure, returns the full traceback
    string (not just the exception message) so a downstream caller/log has
    enough to diagnose without re-running.
    """
    from hf_models import load_trainable_model

    print(f"\n{'='*60}")
    print(f"Testing: {key}  ({model_id})")
    print(f"{'='*60}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1e9

    model, tokenizer = None, None
    try:
        print("  Loading + attaching LoRA (r=16)...")
        model, tokenizer, backend = load_trainable_model(model_id, load_in_4bit=True, lora_r=16, lora_alpha=16)
        vram_after_lora = torch.cuda.memory_allocated() / 1e9
        print(f"  Backend: {backend}")

        # ── forward + backward ──
        print("  Running forward+backward (seq_len<=256, batch=1)...")
        model.train()
        # text= as an explicit keyword, not a bare positional arg: Gemma 4's
        # processor is a multimodal ProcessorMixin (__call__(images=, text=,
        # videos=, ...)) -- confirmed via a real traceback (when this ran
        # through Unsloth's patched __call__ wrapper specifically, which
        # reads `text` by keyword) that a positional string here can
        # silently become `text=None` instead of erroring, crashing later on
        # `text[0]` with the unhelpful 'NoneType' object is not subscriptable.
        # Passing it as a keyword is correct and unambiguous either way.
        dummy_input = tokenizer(
            text="Find the shortest path from (0,0) to (9,9) avoiding obstacles.",
            return_tensors="pt", truncation=True, max_length=256,
        )
        # .to(model.device), not a bare .cuda() -- explicit about landing on
        # whichever device the model actually loaded onto, rather than relying
        # on "current default CUDA device" (normally cuda:0, but don't assume it).
        target_device = next(model.parameters()).device
        dummy_input = {k: v.to(target_device) for k, v in dummy_input.items()}
        dummy_labels = dummy_input["input_ids"].clone()

        outputs = model(**dummy_input, labels=dummy_labels)
        loss = outputs.loss
        loss.backward()

        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        vram_after_bwd = torch.cuda.memory_allocated() / 1e9

        # ── report ──
        peak_vram = max(vram_after_lora, vram_peak, vram_after_bwd)

        print(f"\n  VRAM breakdown:")
        print(f"    Before:       {vram_before:.2f} GB")
        print(f"    After LoRA:   {vram_after_lora:.2f} GB  (+{vram_after_lora - vram_before:.2f})")
        print(f"    After fwd+bwd:{vram_after_bwd:.2f} GB")
        print(f"    Peak:         {peak_vram:.2f} GB")

        # Estimate GRPO: peak * 1.4 (overhead for reference logits + generation KV cache).
        # This ratio was empirically measured on DeepSeek/SmolLM2 -- treat it as a
        # rough estimate for Gemma 4 specifically, not a confirmed number, until an
        # actual train_grpo.py timing test (--max_steps 20) has run.
        grpo_est = peak_vram * 1.4
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        if grpo_est < total * 0.85:
            verdict = "✅ FEASIBLE"
        elif grpo_est < total * 0.95:
            verdict = "⚠️  MARGINAL"
        else:
            verdict = "❌ INFEASIBLE"

        print(f"    Est. GRPO:    {grpo_est:.2f} GB  → {verdict}  (rough estimate, see docstring)")

        return {
            "key": key,
            "backend": backend,
            "load_lora_gb": round(vram_after_lora - vram_before, 2),
            "peak_gb": round(peak_vram, 2),
            "est_grpo_gb": round(grpo_est, 2),
            "verdict": verdict,
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n  ❌ FAILED: {e}\n{tb}")
        return {"key": key, "backend": "?", "load_lora_gb": 0, "peak_gb": 0,
                "est_grpo_gb": 0, "verdict": f"ERROR: {e}", "traceback": tb}
    finally:
        del model, tokenizer
        # gc.collect() before empty_cache(): a bare `del` drops this scope's
        # references, but PEFT wrapper objects can hold internal
        # reference cycles that only actually get freed on a GC pass --
        # without this, empty_cache() sometimes has nothing to reclaim yet,
        # which showed up as one candidate's crash leaving enough VRAM
        # "reserved but unallocated" to OOM the next candidate's load.
        import gc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    from hf_models import configure_quiet_logging
    configure_quiet_logging()

    if not torch.cuda.is_available():
        print("❌ CUDA not available — cannot run feasibility check.")
        exit(1)

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_gb:.2f} GB")
    print(f"CUDA allocated before tests: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    results = [check(key, model_id) for key, model_id in CANDIDATES]

    # ── summary ──
    print(f"\n{'='*80}")
    print(f"{'MODEL':<24} {'BACKEND':<10} {'LOAD+LORA':>10} {'PEAK':>6} {'GRPO':>6}  VERDICT")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['key']:<24} {r['backend']:<10} {r['load_lora_gb']:>9.1f}G {r['peak_gb']:>5.1f}G "
              f"{r['est_grpo_gb']:>5.1f}G  {r['verdict']}")
    print(f"{'='*80}")
