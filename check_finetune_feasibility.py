"""Feasibility check: can each candidate model be loaded at 4-bit,
LoRA-attached, and survive a forward+backward pass on available VRAM?

Uses bitsandbytes + peft (no Unsloth dependency) since Qwen2 and
Gemma 3 are standard architectures.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── candidates ────────────────────────────────────────────────────
CANDIDATES = [
    ("deepseek-r1-distill-qwen-1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
]


def check(key: str, model_id: str) -> dict:
    """Load, LoRA-attach, forward+backward, measure VRAM."""
    print(f"\n{'='*60}")
    print(f"Testing: {key}  ({model_id})")
    print(f"{'='*60}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1e9

    # ── load tokenizer ──
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 4-bit config ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ── load model ──
    print("  Loading model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    vram_after_load = torch.cuda.memory_allocated() / 1e9

    # ── prepare for k-bit training ──
    model = prepare_model_for_kbit_training(model)

    # ── attach LoRA ──
    print("  Attaching LoRA (r=16)...")
    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    vram_after_lora = torch.cuda.memory_allocated() / 1e9

    # ── forward + backward ──
    print("  Running forward+backward (seq_len=256, batch=1)...")
    model.train()
    dummy_input = tokenizer(
        "Find the shortest path from (0,0) to (9,9) avoiding obstacles.",
        return_tensors="pt", truncation=True, max_length=256,
    )
    dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
    dummy_labels = dummy_input["input_ids"].clone()

    outputs = model(**dummy_input, labels=dummy_labels)
    loss = outputs.loss
    loss.backward()

    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    vram_after_bwd = torch.cuda.memory_allocated() / 1e9

    # ── report ──
    peak_vram = max(vram_after_load, vram_after_lora, vram_peak, vram_after_bwd)

    print(f"\n  VRAM breakdown:")
    print(f"    Before:       {vram_before:.2f} GB")
    print(f"    After load:   {vram_after_load:.2f} GB  (+{vram_after_load - vram_before:.2f})")
    print(f"    After LoRA:   {vram_after_lora:.2f} GB  (+{vram_after_lora - vram_after_load:.2f})")
    print(f"    After fwd+bwd:{vram_after_bwd:.2f} GB")
    print(f"    Peak:         {peak_vram:.2f} GB")

    # Estimate GRPO: peak * 1.4 (overhead for reference logits + generation KV cache)
    grpo_est = peak_vram * 1.4
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    if grpo_est < total * 0.85:
        verdict = "✅ FEASIBLE"
    elif grpo_est < total * 0.95:
        verdict = "⚠️  MARGINAL"
    else:
        verdict = "❌ INFEASIBLE"

    print(f"    Est. GRPO:    {grpo_est:.2f} GB  → {verdict}")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "key": key,
        "load_gb": round(vram_after_load - vram_before, 2),
        "peak_gb": round(peak_vram, 2),
        "est_grpo_gb": round(grpo_est, 2),
        "verdict": verdict,
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available — cannot run feasibility check.")
        exit(1)

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_gb:.2f} GB")
    print(f"CUDA allocated before tests: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    results = []
    for key, model_id in CANDIDATES:
        try:
            results.append(check(key, model_id))
        except Exception as e:
            print(f"\n  ❌ FAILED: {e}")
            results.append({"key": key, "load_gb": 0, "peak_gb": 0, "est_grpo_gb": 0, "verdict": f"ERROR: {e}"})

    # ── summary ──
    print(f"\n{'='*70}")
    print(f"{'MODEL':<40} {'LOAD':>6} {'PEAK':>6} {'GRPO':>6}  VERDICT")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['key']:<40} {r['load_gb']:>5.1f}G {r['peak_gb']:>5.1f}G {r['est_grpo_gb']:>5.1f}G  {r['verdict']}")
    print(f"{'='*70}")
