"""Model loading for the chess benchmark: 4-bit HF inference + optional
opencode-go gateway (DeepSeek V4 Flash) API backend.

Only what the benchmark needs: the registry, quiet-logging setup, the HF
loader (4-bit bitsandbytes + chat template), and the gateway client.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from src.token_usage import from_provider_usage, local_usage, with_rates

MODEL_IDS = {
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
}

# opencode-go gateway (OpenAI-compatible). The key is NEVER committed: it
# comes from OPENCODE_API_KEY env, then a local gitignored .env file, then
# the Kaggle secret with the same name (injected as an env var at kernel
# start). Model id is the bare id; the provider prefix is not accepted.
OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
# Free tier of the same gateway: model deepseek-v4-flash-free served at the
# zen/v1 endpoint with the literal Bearer key "public" (the opencode CLI
# itself uses this when no API key is configured; paid models at zen/go/v1
# return CreditsError once an account's balance is exhausted).
OPENCODE_GO_FREE_BASE_URL = "https://opencode.ai/zen/v1"
DEEPSEEK_V4_FLASH = "deepseek-v4-flash"
DEEPSEEK_V4_FLASH_FREE = "deepseek-v4-flash-free"


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


def _attach_usage(result: dict, usage: dict) -> dict:
    """Keep flat legacy fields while adding the normalized token schema."""
    normalized = usage if "generation_seconds" in usage else with_rates(
        usage, result.get("latency_ms"))
    result["token_usage"] = normalized
    result["input_tokens"] = normalized.get("input_tokens")
    result["output_tokens"] = normalized.get("output_tokens")
    result["reasoning_tokens"] = normalized.get("reasoning_tokens")
    result["total_tokens"] = normalized.get("total_tokens")
    result["cache_hit"] = normalized.get("cache_hit_tokens")
    result["cache_miss"] = normalized.get("cache_miss_tokens")
    result["reasoning_chars"] = len(result.get("reasoning") or "")
    result["answer_chars"] = len(result.get("content") or "")
    return result


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


class _LocalStreamer:
    """transformers-compatible streamer that mirrors the gateway's on_chunk
    contract for local models: prints to stdout (optional) and calls
    on_chunk({content, reasoning, phase, finished}) at most every 2s so the
    runner can write live.json without slowing decoding.

    With `split_thinking`, the raw stream is decoded WITHOUT stripping the
    special tokens and re-split on every chunk: text up to the first
    `<|channel>thought` marker is dropped, the thinking block becomes
    `reasoning`, and everything after `<channel|>` becomes `content`. The
    gateway arm streams reasoning_content separately; this makes the local
    Gemma 4 E2B thinking arm behave the same way on the dashboard instead of
    dumping the whole thought blob into the answer box."""

    THINK_OPEN = "<|channel>thought"
    THINK_CLOSE = "<channel|>"

    def __init__(self, tokenizer, to_stdout: bool = False, on_chunk=None,
                 min_interval_s: float = 2.0, split_thinking: bool = False):
        self.tokenizer = tokenizer
        self.to_stdout = to_stdout
        self.on_chunk = on_chunk
        self.min_interval_s = min_interval_s
        self.split_thinking = split_thinking
        self.text = ""
        self.reasoning = ""
        self.content = ""
        self._last_emit = 0.0
        self._prompt_seen = False

    def _split(self) -> tuple:
        """Recompute (reasoning, content) from the full decoded text. The
        marker may arrive split across tokens, so the whole text is rescanned
        each chunk (the authoritative split after generation is still
        processor.parse_response in generate())."""
        i = self.text.find(self.THINK_OPEN)
        if i == -1:
            return "", self.text
        rest = self.text[i + len(self.THINK_OPEN):]
        j = rest.find(self.THINK_CLOSE)
        if j == -1:
            return rest.lstrip("\n"), ""
        return rest[:j].lstrip("\n"), rest[j + len(self.THINK_CLOSE):]

    def _emit(self, phase: str, finished: bool) -> None:
        if not self.on_chunk:
            return
        if self.split_thinking:
            self.reasoning, self.content = self._split()
            if phase != "done":
                phase = ("answering" if self.content
                         else "reasoning" if self.reasoning else "answering")
            self.on_chunk({"content": self.content, "reasoning": self.reasoning,
                           "phase": phase, "finished": finished})
        else:
            self.on_chunk({"content": self.text, "reasoning": "",
                           "phase": phase, "finished": finished})

    def put(self, value) -> None:
        # transformers feeds the prompt ids first, then one token at a time
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        ids = value[0] if hasattr(value, "shape") and len(value.shape) > 1 else value
        piece = self.tokenizer.decode(
            ids, skip_special_tokens=not self.split_thinking)
        self.text += piece
        if self.to_stdout:
            print(piece, end="", flush=True)
        now = time.time()
        if now - self._last_emit >= self.min_interval_s:
            self._last_emit = now
            self._emit("answering", False)

    def end(self) -> None:
        if self.to_stdout:
            print("", flush=True)
        self._emit("done", True)


def _split_gemma4_thought(raw: str, processor, prefix_text: str) -> tuple:
    """Split a Gemma 4 generation into (thinking, answer).

    Primary: the tokenizer's response_template parser
    (processor.parse_response), which understands the
    `<|channel>thought ... <channel|>` channel structure documented on the
    model card and shipped in the checkpoint's tokenizer_config.json.

    Fallback: the exact channel-marker split of that same documented format,
    so a template drift or parse failure can never crash a run — the answer
    text is still the model's own output, never a reconstruction.
    """
    try:
        parsed = processor.parse_response(raw, prefix=prefix_text)
        if isinstance(parsed, dict):
            thinking = parsed.get("thinking") or ""
            content = parsed.get("content")
            if content is not None:
                return thinking, content
    except Exception:
        pass
    think_open = "<|channel>thought\n"
    think_close = "<channel|>"
    if think_open in raw:
        rest = raw.split(think_open, 1)[1]
        if think_close in rest:
            thinking, tail = rest.split(think_close, 1)
            return thinking.rstrip(), _strip_turn_markers(tail)
        return rest.rstrip(), ""
    return "", _strip_turn_markers(raw)


def _strip_turn_markers(text: str) -> str:
    """Remove chat-template scaffolding that survives decode(..., skip_
    special_tokens=False) in the marker-split fallback: the assistant turn
    header the template pre-writes and the turn/eos close markers. The
    canonical parse_response path handles these via its start_anchor."""
    text = text.strip()
    for header in ("<|turn>model\n", "<|turn>model", "<|start|>assistant\n"):
        if text.startswith(header):
            text = text[len(header):].strip()
            break
    for closer in ("<|turn|>", "<eos>", "</s>", "<|endoftext|>"):
        if text.endswith(closer):
            text = text[:-len(closer)].rstrip()
    return text


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

            # Kaggle's free tier hands out P100 (sm_60) OR T4 (sm_75). P100
            # has no bf16 support; T4 does. Compute dtype must match the
            # device, or generation fails with "no kernel image is available
            # for execution on the device".
            cap = torch.cuda.get_device_capability(0)
            compute_dtype = (torch.bfloat16 if cap >= (7, 5)
                             else torch.float16)
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.tokenizer = self.processor.tokenizer
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id, quantization_config=quant,
                    device_map={"": 0}, dtype=compute_dtype,
                )
            except Exception:
                # bitsandbytes has no kernels for some older GPUs (P100:
                # sm_60). Fall back to unquantized fp16 — E2B is 5.1B params
                # (~10GB fp16) and fits a 16GB card; the 4-bit path stays the
                # default on T4.
                print("4-bit load failed; falling back to fp16 (no "
                      "quantization)", flush=True)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id, device_map={"": 0}, dtype=compute_dtype,
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
                 repetition_penalty: float = 1.0, stream: bool = False,
                 on_chunk=None, thinking_budget: Optional[int] = None,
                 thinking_disabled: bool = False,
                 local_thinking: bool = False) -> dict:
        """Returns {content, input_tokens, output_tokens, latency_ms, finished}.
        With stream=True, tokens are printed live to stdout as they are
        generated (chain-of-thought visibility in notebook cells).

        `on_chunk`, `thinking_budget` and `thinking_disabled` exist so the
        runners can call every backend with one signature. Local Gemma
        renders with the thought channel only when `local_thinking` is set
        (thinking is a deliberate per-arm choice, see run_mate_eval
        --local-thinking); the channel is parsed back into separate
        `reasoning` (thinking block) and `content` (final answer) fields
        with the thinking tokens accounted in token_usage.reasoning_tokens,
        mirroring the gateway arm's schema. `thinking_budget` is not
        applicable to local decoding (one budget: max_new_tokens) and
        `thinking_disabled` is recorded but not silently re-enabled.
        `on_chunk` receives partial text during generation."""
        t0 = time.time()
        if self.smoke_test:
            return _attach_usage({
                "content": "MOVE: a1a2",
                "output_tokens": 8,
                "latency_ms": (time.time() - t0) * 1000,
                "finished": True,
                "reasoning": "",
            }, local_usage(len(prompt.split()), 8, source="smoke"))
        import torch

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        if self.is_gemma4:
            # Gemma 4: render with the processor's chat template. Thinking
            # (the <|channel>thought ... <channel|> block) is ON only when
            # local_thinking is set — otherwise E2B/E4B spend the entire
            # token budget on reasoning and never emit the answer (observed:
            # parse_rate 0.0 at 1024 tokens with thinking enabled).
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=local_thinking,
            ).to(self.model.device)
            if local_thinking:
                prefix_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True,
                )
            else:
                prefix_text = None
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
        if stream or on_chunk is not None:
            gen_kwargs["streamer"] = _LocalStreamer(
                stream_tok, to_stdout=stream, on_chunk=on_chunk,
                split_thinking=self.is_gemma4 and local_thinking)
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
        total_gen = int(output_ids.shape[-1])
        if self.is_gemma4 and local_thinking:
            # thinking block + final answer are separated BEFORE any
            # truncation logic sees the text: a budget-cut generation keeps
            # its partial thinking in `reasoning` and an empty `content`,
            # which the runner records honestly as a truncated no_answer.
            raw = decode_fn(output_ids, skip_special_tokens=False)
            thinking, content = _split_gemma4_thought(raw, self.processor, prefix_text)
            reasoning_tokens = None
            if thinking:
                reasoning_tokens = len(
                    self.processor.tokenizer.encode(thinking))
            return _attach_usage({
                "content": content,
                "reasoning": thinking,
                "latency_ms": (time.time() - t0) * 1000,
                "finished": bool(total_gen < max_new_tokens),
            }, local_usage(input_len, total_gen, source="hf_transformers",
                           reasoning_tokens=reasoning_tokens))
        content = decode_fn(output_ids, skip_special_tokens=True)
        return _attach_usage({
            "content": content,
            "latency_ms": (time.time() - t0) * 1000,
            "finished": bool(total_gen < max_new_tokens),
            "reasoning": "",
        }, local_usage(input_len, total_gen, source="hf_transformers"))


class OpenCodeGoModel:
    """OpenAI-compatible client for the opencode-go gateway (DeepSeek V4
    Flash). Stateless, key from env/.env. Used for the API-backed frontier
    model in the study. Never raises on network errors — returns an ERROR
    string so the sweep records the failure instead of dying."""

    MODEL = DEEPSEEK_V4_FLASH
    MIN_INTERVAL_S = 1.0  # polite rate limit; the gateway is cheap but shared
    # a stream that opens, then closes with zero delta events and no
    # finish_reason, is a stalled transport, not a model answer (measured
    # 2026-08-04: mate-sel-00543 failed this exact way 3x on Kaggle, then
    # converged correctly in 30.7s on an uncontended retry) -- retried like a
    # transport error, capped so a genuinely silent gateway still surfaces
    # honestly rather than retrying forever
    MAX_SILENT_STREAM_RETRIES = 2
    # a stream that DELIVERED tokens but closed without a finish_reason is
    # ALSO a transport cut, not a model outcome -- and it used to be recorded
    # as a fake "truncated" no_answer with zero usage data. Evidence from the
    # 1000-position run: all 42 truncated samples had token_usage=None while
    # every completed sample had usage, and completed streams reached 305k
    # chars / 1026s without hitting any cap -- so there is no gateway size or
    # time limit being hit; those were pure mid-stream connection cuts. The
    # old loop broke out on `stream_events > 0` ("any token = real outcome"),
    # so a connection that died mid-reasoning produced a permanent fake
    # truncation. Now a cut stream is retried with a FRESH request (same
    # prompt, temp 0 -> deterministic) until it lands an explicit
    # finish_reason, exactly like the zero-token case.
    MAX_MIDSTREAM_RETRIES = 3
    HEARTBEAT_INTERVAL_S = 5.0  # dashboard liveness ping while a stream is silent

    def __init__(self, model_key: str, smoke_test: bool = False,
                 base_url: str = OPENCODE_GO_BASE_URL):
        self.model_key = model_key
        self.is_free = model_key == DEEPSEEK_V4_FLASH_FREE
        self.model_id = (DEEPSEEK_V4_FLASH_FREE if self.is_free
                         else DEEPSEEK_V4_FLASH)
        self.base_url = (OPENCODE_GO_FREE_BASE_URL if self.is_free
                         else base_url).rstrip("/")
        self.smoke_test = smoke_test
        self._last_call = 0.0
        self.key = "public" if self.is_free else None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def load(self) -> None:
        if self.is_free:
            # free gateway tier authenticates with the literal key "public";
            # no account key required
            return
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
        headers = {"Authorization": f"Bearer {self.key}",
                   "Content-Type": "application/json",
                   # the gateway sits behind Cloudflare, which 403s (1010)
                   # the default urllib user-agent
                   "User-Agent": "openai-python/1.0 chess-benchmark"}
        if stream:
            # we ask for an SSE stream (payload["stream"]=True) but never told
            # any intermediary (the gateway sits behind Cloudflare, confirmed
            # via response headers) that we're an SSE consumer -- some proxies
            # decide whether to flush chunks incrementally or buffer the whole
            # response based on this header. Measured 2026-08-04: mate-sel-02999
            # repeatedly opened a connection, ran long, and delivered ZERO bytes
            # before closing -- consistent with something buffering a slow
            # response instead of streaming it, then giving up on the buffer
            # rather than the model. Missing header, not a model problem.
            headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST")
        return urllib.request.urlopen(req, timeout=3600)

    def _post_with_retry(self, payload: dict, stream: bool = False,
                         max_attempts: int = 3):
        """Transport-level retry with backoff. NOT a model fallback: a
        transport failure (gateway 5xx / timeout / connection reset) produced
        no model output, so re-sending the IDENTICAL request cannot
        contaminate the answer. Client errors (4xx) are not retried — a bad
        request will not succeed twice. Returns (response, attempts)."""
        last = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._post(payload, stream=stream), attempt
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    OSError) as e:
                last = e
                if attempt < max_attempts:
                    time.sleep(2 ** attempt)  # 2s, 4s
            except urllib.error.HTTPError as e:
                if e.code in (429, 408) or e.code >= 500:
                    last = e
                    if attempt < max_attempts:
                        time.sleep(2 ** attempt)
                else:
                    raise
        raise last

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
                 on_chunk=None, thinking_budget: Optional[int] = None,
                 thinking_disabled: bool = False,
                 local_thinking: bool = False) -> dict:
        """Generate via the gateway. Thinking is ENABLED and UNBOUNDED: V4
        Flash reasons long on chess positions, and the study wants that
        thinking visible — we just wait for the final answer (max_tokens
        set high enough that reasoning + answer both fit).

        `local_thinking` exists only so every backend shares one call
        signature; the gateway ignores it (it thinks unless thinking is
        disabled).

        With on_chunk=callable(partial_dict), the client calls back with
        {content, reasoning, phase} during generation so the runner can
        write live.json snapshots (the website shows thinking at ~1 min
        lag). Returns {content, reasoning, tokens, latency_ms, finished,
        cache_hit, cache_miss} — the gateway's automatic prompt caching
        (DeepSeek-style) is surfaced so we can track cache utilization."""
        t0 = time.time()
        if self.smoke_test:
            return _attach_usage({
                "content": "MOVE: a1a2", "reasoning": "",
                "latency_ms": 1, "finished": True,
            }, local_usage(0, 8, source="smoke"))
        if thinking_disabled:
            thinking = {"type": "disabled"}
        elif thinking_budget is not None:
            thinking = {"type": "enabled", "budget_tokens": thinking_budget}
        else:
            thinking = {"type": "enabled"}
        payload = {"model": self.model_id, "max_tokens": max_new_tokens,
                   "temperature": temperature,
                   "thinking": thinking,
                   "messages": [{"role": "user", "content": prompt}]}
        try:
            if stream:
                stream_payload = {**payload, "stream_options": {"include_usage": True}}
                attempts = 0
                # A stream that opens (HTTP 200, real headers) and then closes
                # having delivered ZERO delta events and no finish_reason is not
                # a model outcome -- measured directly: mate-sel-00543 failed
                # this way three times on Kaggle (stream_events=0, connection
                # open 92-147s, silence, close) then converged correctly in
                # 30.7s/4.1k tokens on an isolated retry with nothing else
                # competing for the gateway. That signature is a transport/
                # gateway stall wearing a no_answer costume, not the model
                # struggling, so it gets the same treatment as a raised
                # transport error: retry a FRESH request. A real outcome --
                # any token at all, or an explicit finish_reason (including a
                # genuine length cutoff) -- is accepted immediately and never
                # retried, so a real truncation is still recorded honestly.
                max_stream_attempts = (self.MAX_SILENT_STREAM_RETRIES
                                       + self.MAX_MIDSTREAM_RETRIES + 1)
                for silent_attempt in range(max_stream_attempts):
                    content, reasoning = "", ""
                    final_usage = {}
                    stream_events = 0
                    first_token_at = None
                    finish_reason = None
                    last_chunk_at = time.time()
                    # A stream can legitimately sit connected and silent for
                    # 5+ minutes before delivering anything -- measured
                    # directly on mate-sel-02999 (2026-08-04): the connection
                    # stayed open 322.96s with periodic empty keep-alive
                    # chunks, then delivered 106k reasoning chars in one burst
                    # and a correct finish_reason=stop. That is NOT a hang; a
                    # closed connection with zero content (the silent-retry
                    # case below) is the actual failure signature. Without
                    # this heartbeat, distinguishing "slow but working" from
                    # "stuck" required a bespoke diagnostic script and several
                    # rounds of manually killing a process that was fine --
                    # so this prints directly to the run's own log (not just
                    # the dashboard) every HEARTBEAT_INTERVAL_S, unconditionally,
                    # so the answer is visible in any run's stdout with no
                    # extra tooling. Never touches `content`/`reasoning`, so
                    # it cannot contaminate the scored answer either way.
                    heartbeat_stop = threading.Event()
                    attempt_no = silent_attempt + 1
                    attempt_start = time.time()

                    def _heartbeat() -> None:
                        while not heartbeat_stop.wait(self.HEARTBEAT_INTERVAL_S):
                            elapsed = time.time() - attempt_start
                            print(f"[opencode_go] attempt {attempt_no}/"
                                  f"{self.MAX_SILENT_STREAM_RETRIES + self.MAX_MIDSTREAM_RETRIES + 1}: still "
                                  f"connected, {elapsed:.0f}s elapsed, no tokens "
                                  f"yet -- known to sometimes take 5+ minutes "
                                  f"before the first token; not necessarily stuck",
                                  flush=True)
                            if on_chunk:
                                on_chunk({"content": "", "reasoning": "",
                                         "phase": "reasoning", "finished": False})

                    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
                    hb_thread.start()
                    try:
                        resp, connect_attempts = self._post_with_retry(stream_payload, stream=True)
                        attempts += connect_attempts
                        with resp:
                            for chunk in self._sse_chunks(resp):
                                ch = chunk.get("choices") or [{}]
                                if chunk.get("usage"):
                                    final_usage = chunk["usage"]
                                if not ch:
                                    continue
                                delta = ch[0].get("delta") or {}
                                delta_content = delta.get("content") or ""
                                delta_reasoning = delta.get("reasoning_content") or ""
                                if delta_content or delta_reasoning:
                                    if stream_events == 0:
                                        print(f"[opencode_go] attempt {attempt_no}: "
                                              f"first token after "
                                              f"{time.time() - attempt_start:.1f}s "
                                              f"-- real generation confirmed",
                                              flush=True)
                                    heartbeat_stop.set()  # real data now; stop the heartbeat
                                    stream_events += 1
                                    if first_token_at is None:
                                        first_token_at = time.time()
                                content += delta_content
                                reasoning += delta_reasoning
                                finish = ch[0].get("finish_reason")
                                finish_reason = finish or finish_reason
                                if on_chunk and (time.time() - last_chunk_at >= 2
                                                 or finish):
                                    last_chunk_at = time.time()
                                    on_chunk({
                                        "content": content, "reasoning": reasoning,
                                        "phase": "reasoning" if not content else "answering",
                                        "finished": bool(finish),
                                    })
                    finally:
                        heartbeat_stop.set()
                        if hb_thread:
                            hb_thread.join(timeout=1)
                    # ONLY an explicit finish_reason (stop / length / tool
                    # calls) is a real model outcome. A stream that closed
                    # without one -- whether it delivered zero tokens or
                    # 100k reasoning chars then died -- is a transport cut
                    # and gets a fresh request. The old `stream_events > 0
                    # or finish_reason` break accepted token-delivering but
                    # unfinished streams as outcomes, which turned gateway
                    # connection drops into permanent fake "truncated"
                    # no_answers with no usage data (measured: 42/42 of the
                    # 1000-run truncations had token_usage=None).
                    if finish_reason is not None:
                        break
                    midstream = stream_events > 0
                    retries_left = (self.MAX_MIDSTREAM_RETRIES if midstream
                                    else self.MAX_SILENT_STREAM_RETRIES)
                    if silent_attempt < retries_left:
                        if midstream:
                            print(f"[opencode_go] attempt {attempt_no}: stream "
                                  f"delivered {stream_events} chunk(s) then closed "
                                  f"without a finish_reason -- mid-stream transport "
                                  f"cut, retrying a FRESH request", flush=True)
                        time.sleep(2 ** (silent_attempt + 1))  # 2s, 4s, 8s
                        continue
                    break  # retries exhausted -- record it honestly
                data = {"choices": [{"message": {"content": content,
                                                 "reasoning_content": reasoning},
                                     "finish_reason": finish_reason or
                                     ("stop" if content else "length")}],
                        "usage": final_usage}
                if on_chunk:
                    on_chunk({"content": content, "reasoning": reasoning,
                              "phase": "done", "finished": True})
            else:
                resp, attempts = self._post_with_retry(payload)
                with resp:
                    data = json.load(resp)
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            usage = data.get("usage", {})
            self.prompt_tokens += usage.get("prompt_tokens") or 0
            self.completion_tokens += usage.get("completion_tokens") or 0
            usage_record = from_provider_usage(
                usage,
                source="opencode_go",
                stream_events=stream_events if stream else None,
                time_to_first_token_ms=(
                    (first_token_at - t0) * 1000
                    if stream and first_token_at is not None else None
                ),
            )
            return _attach_usage({
                "content": content,
                "reasoning": reasoning,
                "attempts": attempts,
                "latency_ms": (time.time() - t0) * 1000,
                "finished": data["choices"][0].get("finish_reason") in ("stop", None),
            }, usage_record)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            return self._error_result(f"HTTP {e.code}: {body}", t0)
        except Exception as e:
            return self._error_result(f"{type(e).__name__}: {e}", t0)

    @staticmethod
    def _error_result(detail: str, t0: float) -> dict:
        """A transport/gateway failure is NOT model output. `content` stays
        empty so the answer parser can never turn an error body into a move
        (an error body containing e.g. '"code":"e4"' used to parse as the SAN
        move e3e4 and be scored as a legal answer); the detail is carried in a
        dedicated `error` field and the sample is recorded as api_error, which
        is excluded from every accuracy denominator."""
        return _attach_usage({
            "content": "", "reasoning": "", "error": detail,
            "latency_ms": (time.time() - t0) * 1000, "finished": False,
        }, local_usage(None, None, source="error"))

def make_model(model_key: str, smoke_test: bool = False):
    """Registry: local 4-bit HF models + the gateway API model."""
    if model_key in (DEEPSEEK_V4_FLASH, DEEPSEEK_V4_FLASH_FREE):
        return OpenCodeGoModel(model_key, smoke_test=smoke_test)
    return HFModel(model_key, smoke_test=smoke_test)
