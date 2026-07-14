"""Unified model inference wrapper for the Gemma 4 spatial-reasoning project.
Three backends behind one .load()/.generate() interface so train.py doesn't
need per-model or per-backend branching:
  - UnslothModel: primary path. Handles architecture quirks (e.g. Gemma 4's
    custom Gemma4ClippableLinear layers, which plain HF transformers + peft
    LoRA cannot attach to) that raw transformers can't.
  - HFModel: plain HF transformers, inference-only (no LoRA). Useful as a
    lighter-weight fallback for pure evaluation runs.
  - OllamaModel: local Ollama server, quantized models. For low-VRAM machines
    or quick eval without loading full-precision weights.
"""

import re
import time
from typing import Optional


MODEL_IDS = {
    # ── ✅ confirmed trainable on 6 GB with 4-bit LoRA GRPO ──
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # ── ⚠️ marginal (tight, reduce generations) ──
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    # ── inference-only (GRPO needs >6 GB VRAM) ──
    "gemma4-e2b": "google/gemma-4-E2B-it",
    # ── gated (needs HF access approval) ──
    # "gemma-3-1b": "google/gemma-3-1b-it",
}

OLLAMA_MODEL_TAGS = {
    "deepseek-r1-distill-qwen-1.5b": "deepseek-r1:1.5b",
    "smollm2-1.7b": "smollm2:1.7b",
    "qwen2.5-3b": "qwen2.5:3b",
    "gemma4-e2b": "gemma4:e2b",
}


def strip_gemma_thoughts(text: str) -> str:
    """Gemma 3 & 4 emit thinking content before a final-answer marker.
    Extract everything after the last such marker if present."""
    for marker in ("<channel|>", "<end_of_thought>"):
        parts = text.split(marker)
        if len(parts) > 1:
            return parts[-1].strip()
    return text.strip()


class HFModel:
    """Loads a HF causal LM once, reused across all generations in a run."""

    def __init__(self, model_key: str, dtype="bfloat16", smoke_test: bool = False):
        self.model_key = model_key
        self.smoke_test = smoke_test
        self.model = None
        self.tokenizer = None
        self._dtype_name = dtype

    def load(self):
        if self.smoke_test:
            # No real model load in smoke-test mode -- keep preflight fast and CPU-only.
            return self

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = MODEL_IDS[self.model_key]
        dtype = getattr(torch, self._dtype_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto"
        )
        self.model.eval()
        return self

    def generate(self, prompt: str, max_new_tokens: int = 1536, temperature: float = 0.0) -> dict:
        """Returns dict(content, input_tokens, output_tokens, latency_ms)."""
        t0 = time.time()

        if self.smoke_test:
            # Deterministic stub response so the parsing/scoring pipeline is
            # exercised end-to-end without loading a multi-GB model.
            content = "(0,0) (0,1) (0,2)"
            return {
                "content": content,
                "input_tokens": len(prompt.split()),
                "output_tokens": len(content.split()),
                "latency_ms": (time.time() - t0) * 1000,
            }

        import torch

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            do_sample = temperature > 0.0
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
            outputs = self.model.generate(**gen_kwargs)

        output_ids = outputs[0][input_len:]
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if "gemma" in self.model_key.lower():
            content = strip_gemma_thoughts(content)

        return {
            "content": content,
            "input_tokens": int(input_len),
            "output_tokens": int(output_ids.shape[-1]),
            "latency_ms": (time.time() - t0) * 1000,
        }


class UnslothModel:
    """Primary backend. Loads via Unsloth's FastLanguageModel, which patches
    around architecture-specific quirks plain transformers+peft chokes on
    (confirmed: Gemma 4's Gemma4ClippableLinear layers aren't a recognized
    peft target module type; Unsloth's loader handles this).

    Supports both plain inference (this class) and LoRA attachment for
    fine-tuning (see attach_lora()) through the same loaded model/tokenizer.
    """

    def __init__(self, model_key: str, max_seq_length: int = 2048,
                 load_in_4bit: bool = True, smoke_test: bool = False):
        self.model_key = model_key
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.smoke_test = smoke_test
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.smoke_test:
            return self

        from unsloth import FastLanguageModel

        model_id = MODEL_IDS[self.model_key]
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=self.max_seq_length,
            dtype=None,  # auto-detect
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        return self

    def attach_lora(self, r: int = 16, lora_alpha: int = 16):
        """Attach a LoRA adapter for fine-tuning. Call after load(), before
        training. Returns self for chaining."""
        from unsloth import FastLanguageModel

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        return self

    def generate(self, prompt: str, max_new_tokens: int = 1536, temperature: float = 0.0) -> dict:
        t0 = time.time()

        if self.smoke_test:
            content = "(0,0) (0,1) (0,2)"
            return {
                "content": content,
                "input_tokens": len(prompt.split()),
                "output_tokens": len(content.split()),
                "latency_ms": (time.time() - t0) * 1000,
            }

        import torch

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            do_sample = temperature > 0.0
            gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            if do_sample:
                gen_kwargs["temperature"] = temperature
            outputs = self.model.generate(**gen_kwargs)

        output_ids = outputs[0][input_len:]
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if "gemma" in self.model_key.lower():
            content = strip_gemma_thoughts(content)

        return {
            "content": content,
            "input_tokens": int(input_len),
            "output_tokens": int(output_ids.shape[-1]),
            "latency_ms": (time.time() - t0) * 1000,
        }


class OllamaModel:
    """Same .load()/.generate() interface as HFModel, backed by a local
    Ollama server instead of HF transformers. For running on a machine
    (e.g. an RTX 4050 laptop) that already has Ollama + the quantized
    models set up -- avoids the VRAM cost of loading full bf16 weights.
    """

    def __init__(self, model_key: str, base_url: str = "http://localhost:11434",
                 smoke_test: bool = False):
        self.model_key = model_key
        self.smoke_test = smoke_test
        self.base_url = base_url
        self.env = None

    def load(self):
        if self.smoke_test:
            return self
        from src.ollama_env import OllamaEnv
        self.env = OllamaEnv(model=OLLAMA_MODEL_TAGS[self.model_key],
                              base_url=self.base_url, temperature=0.0)
        return self

    def generate(self, prompt: str, max_new_tokens: int = 1536, temperature: float = 0.0) -> dict:
        if self.smoke_test:
            content = "(0,0) (0,1) (0,2)"
            return {
                "content": content,
                "input_tokens": len(prompt.split()),
                "output_tokens": len(content.split()),
                "latency_ms": 0.0,
            }
        result = self.env.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens, temperature=temperature,
        )
        return {
            "content": result.get("content", ""),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "latency_ms": result.get("latency_ms", 0.0),
        }


def extract_coords_from_text(text: str) -> list:
    """Language-agnostic coordinate extraction: numbers don't need
    translation, so this works regardless of instruction language."""
    patterns = [r"\((\d+),\s*(\d+)\)", r"\[(\d+),\s*(\d+)\]"]
    coords = []
    for pat in patterns:
        for x, y in re.findall(pat, text):
            coords.append((int(x), int(y)))
    return coords


def parse_path_response(content: str, start: tuple, goal: tuple) -> Optional[list]:
    """Extract a start->goal coordinate sequence from free-form model output,
    independent of the instruction/response language."""
    coords = extract_coords_from_text(content)
    if len(coords) < 2:
        return None

    if coords[0] == start and coords[-1] == goal:
        return coords

    # Find the longest start->goal subsequence (handles preamble/rambling).
    best = None
    for i, c in enumerate(coords):
        if c != start:
            continue
        for j in range(len(coords) - 1, i, -1):
            if coords[j] != goal:
                continue
            candidate = coords[i:j + 1]
            if best is None or len(candidate) > len(best):
                best = candidate
    return best
