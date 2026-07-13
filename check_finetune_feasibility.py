"""Feasibility check: can each candidate model be loaded and LoRA-attached
for fine-tuning via Unsloth on available VRAM? Run before building the full
GRPO pipeline. Uses Unsloth specifically because plain transformers+peft
cannot attach LoRA to Gemma 4 (Gemma4ClippableLinear isn't a recognized
peft target module type -- confirmed by direct test on this hardware).
"""

import torch
from unsloth import FastLanguageModel

MODELS = {
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
}


def check(key: str, model_id: str) -> dict:
    print(f"\n{'='*60}\n{key} ({model_id})\n{'='*60}")
    torch.cuda.reset_peak_memory_stats()
    result = {"model": key, "load_ok": False, "lora_ok": False, "backward_ok": False}

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=2048, dtype=None, load_in_4bit=False,
        )
        print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        result["load_ok"] = True

        model = FastLanguageModel.get_peft_model(
            model, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=42,
        )
        print(f"VRAM after LoRA attach: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        result["lora_ok"] = True

        prompt = "Navigate from (0,0) to (2,2) on a 5x5 grid. Move up/down/left/right."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()
        result["backward_ok"] = True
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"Loss: {outputs.loss.item():.4f} | Peak VRAM: {result['peak_vram_gb']:.2f} GB")

        del model, outputs
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        result["error"] = f"{type(e).__name__}: {e}"
        torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    results = [check(k, v) for k, v in MODELS.items()]

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for r in results:
        status = "OK" if r["backward_ok"] else "FAILED"
        peak = f", peak {r.get('peak_vram_gb', 0):.2f}GB" if r["backward_ok"] else f" -- {r.get('error', '')}"
        print(f"  {r['model']}: {status}{peak}")
