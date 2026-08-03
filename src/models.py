"""Model loading for the chess benchmark: 4-bit HF inference + optional
opencode-go gateway (DeepSeek V4 Flash) API backend.

Only what the benchmark needs: the registry, quiet-logging setup, the HF
loader (4-bit bitsandbytes + chat template), and the gateway client.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

MODEL_IDS = {
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
}

# opencode-go gateway (OpenAI-compatible). The key is NEVER committed: it
# comes from OPENCODE_API_KEY env, then a local gitignored .env file, then
# the Kaggle secret with the same name (injected as an env var at kernel
# start). Model id is the bare id; the provider prefix is not accepted.
OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
DEEPSEEK_V4_FLASH = "deepseek-v4-flash"


def resolve_api_key() -> Optional[str]:
    for name in ("OPENCODE_API_KEY", "DEEPSEEK_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENCODE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


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
            # checkpoints; the model card's documented path is AutoProcessor +
            # AutoModelForImageTextToText. Load 4-BIT: E4B in bf16 exceeds the
            # T4's 16GB (observed CUDA OOM at load: ~14.5GB used, still at
            # 68% of shards) — 4-bit E2B ~3GB, E4B ~6GB.
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )

            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, quantization_config=quant, device_map={"": 0},
                dtype=torch.bfloat16,
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
        elif self.is_gemma4:
            # Gemma 4 degenerates into repetition loops under greedy decoding
            # ("Black's pieces are: are: Rb8, Rb8, ..." cycling forever) and
            # burns the whole budget without reaching EOS or an answer. A mild
            # penalty breaks the loop even in greedy mode.
            gen_kwargs["repetition_penalty"] = 1.15
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


class OpenCodeGoModel:
    """OpenAI-compatible client for the opencode-go gateway (DeepSeek V4
    Flash). Stateless, key from env/.env. Used for the API-backed frontier
    model in the study. Never raises on network errors — returns an ERROR
    string so the sweep records the failure instead of dying."""

    MODEL = DEEPSEEK_V4_FLASH
    MIN_INTERVAL_S = 1.0  # polite rate limit; the gateway is cheap but shared

    def __init__(self, model_key: str, smoke_test: bool = False,
                 base_url: str = OPENCODE_GO_BASE_URL):
        self.model_key = model_key
        self.base_url = base_url.rstrip("/")
        self.smoke_test = smoke_test
        self._last_call = 0.0
        self.key = None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def load(self) -> None:
        self.key = resolve_api_key()
        if not self.key:
            raise RuntimeError(
                "no OPENCODE_API_KEY — set it as a Kaggle secret (Add secret, "
                "save, Restart & Run All) or in a local .env file")

    def render_chat(self, prompt: str) -> str:
        """The gateway takes plain text; this mirrors HFModel's interface."""
        return prompt

    def _post(self, payload: dict, stream: bool = False):
        wait = self.MIN_INTERVAL_S - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()
        if stream:
            payload = {**payload, "stream": True}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json",
                     # the gateway sits behind Cloudflare, which 403s (1010)
                     # the default urllib user-agent
                     "User-Agent": "openai-python/1.0 chess-benchmark"},
            method="POST")
        return urllib.request.urlopen(req, timeout=3600)

    def _sse_chunks(self, resp):
        """Yield parsed JSON chunks from a stream=true SSE response."""
        buf = ""
        for raw in resp:
            buf += raw.decode("utf-8", "ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    try:
                        yield json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        pass

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0, stream: bool = False,
                 on_chunk=None) -> dict:
        """Generate via the gateway. Thinking is ENABLED and UNBOUNDED: V4
        Flash reasons long on chess positions, and the study wants that
        thinking visible — we just wait for the final answer (max_tokens
        set high enough that reasoning + answer both fit).

        With on_chunk=callable(partial_dict), the client calls back with
        {content, reasoning, phase} during generation so the runner can
        write live.json snapshots (the website shows thinking at ~1 min
        lag). Returns {content, reasoning, tokens, latency_ms, finished,
        cache_hit, cache_miss} — the gateway's automatic prompt caching
        (DeepSeek-style) is surfaced so we can track cache utilization."""
        t0 = time.time()
        if self.smoke_test:
            return {"content": "MOVE: a1a2", "reasoning": "",
                    "input_tokens": 0, "output_tokens": 8,
                    "latency_ms": 1, "finished": True,
                    "cache_hit": None, "cache_miss": None}
        payload = {"model": self.MODEL, "max_tokens": max_new_tokens,
                   "temperature": temperature,
                   "thinking": {"type": "enabled"},
                   "messages": [{"role": "user", "content": prompt}]}
        try:
            if stream:
                content, reasoning = "", ""
                n_tokens = 0
                cache_hit = cache_miss = None
                last_chunk_at = time.time()
                with self._post(payload, stream=True) as resp:
                    for chunk in self._sse_chunks(resp):
                        ch = chunk.get("choices") or [{}]
                        if not ch:
                            continue
                        delta = ch[0].get("delta") or {}
                        content += delta.get("content") or ""
                        reasoning += delta.get("reasoning_content") or ""
                        usage = chunk.get("usage") or {}
                        n_tokens = usage.get("completion_tokens", n_tokens)
                        if "prompt_cache_hit_tokens" in usage:
                            cache_hit = usage.get("prompt_cache_hit_tokens")
                            cache_miss = usage.get("prompt_cache_miss_tokens")
                        finish = ch[0].get("finish_reason")
                        if on_chunk and (time.time() - last_chunk_at >= 5
                                         or finish):
                            last_chunk_at = time.time()
                            on_chunk({
                                "content": content, "reasoning": reasoning,
                                "phase": "reasoning" if not content else "answering",
                                "finished": bool(finish),
                            })
                data = {"choices": [{"message": {"content": content,
                                                 "reasoning_content": reasoning},
                                     "finish_reason": "stop" if content else "length"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": n_tokens,
                                  "prompt_cache_hit_tokens": cache_hit,
                                  "prompt_cache_miss_tokens": cache_miss}}
                if on_chunk:
                    on_chunk({"content": content, "reasoning": reasoning,
                              "phase": "done", "finished": True})
            else:
                with self._post(payload) as resp:
                    data = json.load(resp)
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            usage = data.get("usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            return {
                "content": content,
                "reasoning": reasoning,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "latency_ms": (time.time() - t0) * 1000,
                "finished": data["choices"][0].get("finish_reason") in ("stop", None),
                "cache_hit": usage.get("prompt_cache_hit_tokens"),
                "cache_miss": usage.get("prompt_cache_miss_tokens"),
            }
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            return {"content": f"ERROR HTTP {e.code}: {body}", "reasoning": "",
                    "input_tokens": 0, "output_tokens": 0,
                    "latency_ms": (time.time() - t0) * 1000, "finished": True,
                    "cache_hit": None, "cache_miss": None}
        except Exception as e:
            return {"content": f"ERROR {type(e).__name__}: {e}", "reasoning": "",
                    "input_tokens": 0, "output_tokens": 0,
                    "latency_ms": (time.time() - t0) * 1000, "finished": True,
                    "cache_hit": None, "cache_miss": None}

    def force_answer(self, prompt: str, max_new_tokens: int = 256,
                     temperature: float = 0.0) -> dict:
        """Second-chance call used when the thinking pass produced no answer:
        thinking is disabled and the model is ordered to commit to a move,
        any move. 'No answer' is never acceptable for the benchmark's
        comparison baseline — if this also fails, the runner falls back to
        a legal move extracted from the reasoning (or a random legal move),
        always flagged as a fallback."""
        t0 = time.time()
        commit = (prompt
                  + "\n\nFINAL ANSWER REQUIRED: output exactly one line, "
                    "MOVE: <from><to>. If you are not sure, still output your "
                    "best guess — you must output a move.")
        payload = {"model": self.MODEL, "max_tokens": max_new_tokens,
                   "temperature": temperature,
                   "thinking": {"type": "disabled"},
                   "messages": [{"role": "user", "content": commit}]}
        try:
            with self._post(payload) as resp:
                data = json.load(resp)
            msg = data["choices"][0]["message"]
            usage = data.get("usage", {})
            return {
                "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "latency_ms": (time.time() - t0) * 1000,
                "finished": True,
                "cache_hit": usage.get("prompt_cache_hit_tokens"),
                "cache_miss": usage.get("prompt_cache_miss_tokens"),
                "forced": True,
            }
        except Exception as e:
            return {"content": f"ERROR {type(e).__name__}: {e}", "reasoning": "",
                    "input_tokens": 0, "output_tokens": 0,
                    "latency_ms": (time.time() - t0) * 1000, "finished": True,
                    "cache_hit": None, "cache_miss": None, "forced": True}


def make_model(model_key: str, smoke_test: bool = False):
    """Registry: local 4-bit HF models + the gateway API model."""
    if model_key == DEEPSEEK_V4_FLASH:
        return OpenCodeGoModel(model_key, smoke_test=smoke_test)
    return HFModel(model_key, smoke_test=smoke_test)
