"""Model loading for the chess benchmark: 4-bit HF inference + optional Ollama.

Only what the benchmark needs: the registry, quiet-logging setup, the HF
loader (4-bit bitsandbytes + chat template), and the Ollama backend.
"""
from __future__ import annotations

import os
import re
import time
from typing import Dict, Optional

MODEL_IDS = {
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
}


def configure_quiet_logging() -> None:
    """Silence HF/datasets/tokenizers before any heavy import; disable wandb
    (Kaggle's preinstalled wandb crashes trl's import check) and enable
    expandable-segments allocator to reduce OOM risk."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import logging

    logging.getLogger().setLevel(logging.ERROR)


class HFModel:
    """4-bit HF model with chat-template generation (greedy by default)."""

    def __init__(self, model_key: str, smoke_test: bool = False,
                 system_prompt: str = ""):
        self.model_key = model_key
        self.model_id = MODEL_IDS.get(model_key, model_key)
        self.smoke_test = smoke_test
        self.system_prompt = system_prompt
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_gemma4 = model_key in ("gemma4-e2b", "gemma4-e4b")

    def load(self):
        if self.smoke_test:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self.is_gemma4:
            # Gemma 4 E2B/E4B are multimodal Gemma4ForConditionalGeneration
            # checkpoints: the model card's documented path is AutoProcessor +
            # AutoModelForImageTextToText. Load in bf16 WITHOUT quantization
            # (E2B ~4.3GB, E4B ~8.6GB — both fit the T4 16GB alone) to avoid
            # bitsandbytes-vs-VLM incompatibilities, and disable thinking so
            # the model answers directly instead of burning the budget on
            # <|channel>thought reasoning (its default behavior).
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, device_map={"": 0}, torch_dtype=torch.bfloat16,
            )
        else:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=quant, device_map={"": 0},
                torch_dtype=torch.bfloat16,
            )
        self.model.eval()

    def render_chat(self, prompt: str) -> str:
        """The exact string the model will see (for debugging)."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        if self.is_gemma4:
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0, top_p: float = 1.0,
                 repetition_penalty: float = 1.0, stream: bool = False) -> dict:
        """Returns {content, input_tokens, output_tokens, latency_ms, finished}.
        With stream=True, tokens are printed live to stdout as they are
        generated (chain-of-thought visibility in notebook cells)."""
        t0 = time.time()
        if self.smoke_test:
            return {
                "content": "MOVE: a1a2",
                "input_tokens": len(prompt.split()),
                "output_tokens": 8,
                "latency_ms": (time.time() - t0) * 1000,
                "finished": True,
            }
        import torch

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        if self.is_gemma4:
            # Gemma 4: render with the processor's chat template and
            # THINKING DISABLED — otherwise E2B/E4B spend the entire token
            # budget on <|channel>thought reasoning and never emit the
            # answer (observed: parse_rate 0.0 at 1024 tokens).
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=False,
            ).to(self.model.device)
            decode_fn = self.processor.decode
            pad_token_id = self.processor.tokenizer.eos_token_id
            stream_tok = self.processor.tokenizer
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(self.model.device)
            decode_fn = self.tokenizer.decode
            pad_token_id = self.tokenizer.eos_token_id
            stream_tok = self.tokenizer
        input_len = inputs["input_ids"].shape[-1]
        do_sample = temperature > 0.0
        gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens,
                          do_sample=do_sample,
                          pad_token_id=pad_token_id)
        if stream:
            from transformers import TextStreamer

            gen_kwargs["streamer"] = TextStreamer(
                stream_tok, skip_prompt=True, skip_special_tokens=False,
            )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["repetition_penalty"] = repetition_penalty
        with torch.no_grad():
            out = self.model.generate(**gen_kwargs)
        output_ids = out[0][input_len:]
        content = decode_fn(output_ids, skip_special_tokens=True)
        return {
            "content": content,
            "input_tokens": input_len,
            "output_tokens": int(output_ids.shape[-1]),
            "latency_ms": (time.time() - t0) * 1000,
            "finished": bool(output_ids.shape[-1] < max_new_tokens),
        }


class OllamaModel:
    """Local Ollama backend (/api/generate). Used only for quick local checks."""

    def __init__(self, model_key: str, base_url: str = "http://localhost:11434",
                 smoke_test: bool = False):
        self.model_id = {
            "deepseek-r1-distill-qwen-1.5b": "deepseek-r1:1.5b",
            "smollm2-1.7b": "smollm2:1.7b",
            "qwen2.5-3b": "qwen2.5:3b",
            "gemma4-e2b": "gemma4:e2b",
        }.get(model_key, model_key)
        self.base_url = base_url
        self.smoke_test = smoke_test

    def load(self):
        pass

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0, **_) -> dict:
        import requests

        t0 = time.time()
        payload = {"model": self.model_id, "prompt": prompt,
                   "stream": False, "options": {"num_predict": max_new_tokens,
                                                 "temperature": temperature}}
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=600)
            data = resp.json()
            content = data.get("response", "")
            return {"content": content,
                    "input_tokens": data.get("prompt_eval_count", 0),
                    "output_tokens": data.get("eval_count", 0),
                    "latency_ms": (time.time() - t0) * 1000,
                    "finished": True}
        except Exception as e:
            return {"content": f"ERROR: {e}", "input_tokens": 0, "output_tokens": 0,
                    "latency_ms": (time.time() - t0) * 1000, "finished": True}
