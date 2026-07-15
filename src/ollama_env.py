"""Ollama model wrapper -- alternative backend for hf_models.OllamaModel,
for machines that already have Ollama + a quantized model set up locally
(e.g. Gemma 4 E2B on a low-VRAM laptop) instead of loading full/4-bit
weights via transformers.
"""

import time
from typing import Optional

import requests


def strip_gemma_thoughts(text: str) -> str:
    """Gemma 4 outputs thinking/reasoning first, then the actual answer after
    a <channel|> marker. Extract everything after the LAST such marker."""
    parts = text.split("<channel|>")
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()


class OllamaEnv:
    """Talks to a local Ollama server's /api/generate endpoint."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        top_p: float = 0.95,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.is_gemma = "gemma4" in model or "gemma-4" in model

    def load(self):
        return self

    def _chat_to_prompt(self, messages: list) -> str:
        """Convert chat messages to a plain prompt string for /api/generate."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def generate(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> dict:
        t0 = time.time()
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        payload = {
            "model": self.model,
            "prompt": self._chat_to_prompt(messages),
            "stream": False,
            "options": {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens},
        }

        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {
                "content": f"Error: {e}",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": (time.time() - t0) * 1000,
            }

        content = data.get("response", "")
        if self.is_gemma:
            content = strip_gemma_thoughts(content)

        return {
            "content": content,
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "latency_ms": (time.time() - t0) * 1000,
        }
