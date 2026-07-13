"""Unified HuggingFace transformers inference wrapper for the cross-lingual
navigation experiment. Covers Gemma 4 E2B and the Qwen2.5 comparison models
with one interface so train.py doesn't need per-model branching.
"""

import re
import time
from typing import Optional


MODEL_IDS = {
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
}

OLLAMA_MODEL_TAGS = {
    "gemma4-e2b": "gemma4-e2b:q3_k_s",
    "qwen2.5-1.5b": "qwen2.5:1.5b",
    "qwen2.5-3b": "qwen2.5:3b",
}


def strip_gemma_thoughts(text: str) -> str:
    """Gemma 4 emits thinking content before a final-answer marker.
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
        if self.model_key == "gemma4-e2b":
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
