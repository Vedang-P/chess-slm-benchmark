"""Ollama model wrapper for on-device inference.

Provides the same interface as Gemma4Env for drop-in replacement.
Supports qwen2.5-coder:7b, gemma4-e2b:q3_k_s, etc.
"""

import time
import json
import re
from typing import Optional

import requests


def strip_gemma_thoughts(text: str) -> str:
    """Strip Gemma 4 E2B thinking and return the content after the answer marker.

    Gemma 4 E2B outputs thinking/reasoning first, then the actual answer after
    a <channel|> marker. We extract everything after the LAST <channel|>.
    """
    parts = text.split("<channel|>")
    if len(parts) > 1:
        return parts[-1].strip()
    # If no marker, try to find the section after "Thinking Process:"
    # by looking for the path at the end
    return text.strip()


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract the first JSON object from text."""
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass
    return None


class OllamaEnv:
    """Ollama inference backend using Ollama's native API."""

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
        self.loaded = True

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

    def _call_chat(self, messages: list, tools: Optional[list], max_tokens: int,
                    temperature: float, top_p: float) -> dict:
        """Call /api/chat endpoint (for models with tool support)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return None

        msg = data.get("message", {})
        content = msg.get("content", "")
        tool_calls_raw = msg.get("tool_calls", [])

        tool_calls = []
        for tc in tool_calls_raw:
            if "function" in tc:
                raw_args = tc["function"].get("arguments", "{}")
                if isinstance(raw_args, dict):
                    raw_args = json.dumps(raw_args)
                tool_calls.append({
                    "type": "function",
                    "function": {
                        "name": tc["function"].get("name", ""),
                        "arguments": raw_args,
                    }
                })

        return {
            "content": content,
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "tool_calls": tool_calls,
        }

    def _call_generate(self, messages: list, max_tokens: int,
                        temperature: float, top_p: float) -> dict:
        """Call /api/generate endpoint (preserves thinking tags)."""
        prompt = self._chat_to_prompt(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {
                "content": f"Error: {e}",
                "input_tokens": 0,
                "output_tokens": 0,
                "tool_calls": [],
            }

        content = data.get("response", "")

        if self.is_gemma:
            content = strip_gemma_thoughts(content)

        return {
            "content": content,
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "tool_calls": [],
        }

    def generate(
        self,
        messages: list,
        tools: Optional[list] = None,
        max_tokens: int = 512,
        temperature: float = None,
        top_p: float = None,
    ) -> dict:
        t0 = time.time()

        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        # Gemma 4 E2B always produces thinking tags via chat API (stripped → empty).
        # For tool calls: use chat API directly (returns proper tool calls).
        # For non-tool: use generate API directly to get full output, strip thoughts.
        if self.is_gemma:
            if tools:
                result = self._call_chat(messages, tools, max_tokens, temperature, top_p)
                if result and result.get("tool_calls"):
                    result["latency_ms"] = (time.time() - t0) * 1000
                    return result
                # fallback: try generate + JSON extraction
                result = self._call_generate(messages, max_tokens, temperature, top_p)
                if result and result["content"]:
                    json_obj = extract_json_from_text(result["content"])
                    if json_obj:
                        result["tool_calls"] = [{
                            "type": "function",
                            "function": {
                                "name": (tools[0].get("function", {})
                                          .get("name", "extract")),
                                "arguments": json.dumps(json_obj),
                            }
                        }]
                result["latency_ms"] = (time.time() - t0) * 1000
                return result
            else:
                result = self._call_generate(messages, max_tokens, temperature, top_p)
                result["latency_ms"] = (time.time() - t0) * 1000
                return result

        # Non-Gemma models: try chat API first, fall back to generate
        result = self._call_chat(messages, tools, max_tokens, temperature, top_p)

        if result is not None:
            has_tool_calls = bool(result.get("tool_calls"))
            content_empty = result["content"] == ""
            has_output_tokens = result["output_tokens"] > 0

            if tools and has_tool_calls:
                pass
            elif content_empty and has_output_tokens:
                result2 = self._call_generate(messages, max_tokens, temperature, top_p)
                if result2 and result2["content"]:
                    result = result2
            elif tools and not has_tool_calls and has_output_tokens:
                result2 = self._call_generate(messages, max_tokens, temperature, top_p)
                if result2 and result2["content"]:
                    json_obj = extract_json_from_text(result2["content"])
                    if json_obj:
                        result = {
                            "content": result2["content"],
                            "input_tokens": result2["input_tokens"],
                            "output_tokens": result2["output_tokens"],
                            "tool_calls": [{
                                "type": "function",
                                "function": {
                                    "name": (tools[0].get("function", {})
                                              .get("name", "extract")),
                                    "arguments": json.dumps(json_obj),
                                }
                            }],
                        }
                    else:
                        result["content"] = result2["content"]

            result["latency_ms"] = (time.time() - t0) * 1000
            return result

        result = self._call_generate(messages, max_tokens, temperature, top_p)
        result["latency_ms"] = (time.time() - t0) * 1000

        if tools and not result["tool_calls"] and result["content"]:
            json_obj = extract_json_from_text(result["content"])
            if json_obj:
                result["tool_calls"] = [{
                    "type": "function",
                    "function": {
                        "name": (tools[0].get("function", {})
                                  .get("name", "extract")),
                        "arguments": json.dumps(json_obj),
                    }
                }]

        return result
