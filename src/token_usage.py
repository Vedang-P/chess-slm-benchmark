"""Normalized token and latency accounting for every model backend."""
from __future__ import annotations

from typing import Any


TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
)


def _number(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def from_provider_usage(usage: dict | None, *, source: str = "provider",
                        stream_events: int | None = None,
                        time_to_first_token_ms: float | None = None) -> dict:
    """Normalize OpenAI-compatible usage variants without inventing values."""
    usage = usage or {}
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    input_tokens = _number(usage.get("prompt_tokens"))
    output_tokens = _number(usage.get("completion_tokens"))
    reasoning_tokens = _number(completion_details.get("reasoning_tokens"))
    cache_hit = _number(usage.get("prompt_cache_hit_tokens"))
    cache_miss = _number(usage.get("prompt_cache_miss_tokens"))
    if cache_hit is None:
        cache_hit = _number(prompt_details.get("cached_tokens"))
    if cache_miss is None and input_tokens is not None and cache_hit is not None:
        cache_miss = max(0, input_tokens - cache_hit)
    total_tokens = _number(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    result = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": cache_hit,
        "cache_write_tokens": _number(usage.get("cache_write_tokens")),
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
        "usage_source": source,
        "usage_complete": input_tokens is not None and output_tokens is not None,
        "stream_events": stream_events,
        "time_to_first_token_ms": time_to_first_token_ms,
    }
    return result


def local_usage(input_tokens: int | None, output_tokens: int | None, *,
                source: str, reasoning_tokens: int | None = None,
                stream_events: int | None = None,
                time_to_first_token_ms: float | None = None) -> dict:
    """Usage record for local HF inference or a smoke backend."""
    total = None
    if input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "cache_hit_tokens": None,
        "cache_miss_tokens": None,
        "usage_source": source,
        "usage_complete": input_tokens is not None and output_tokens is not None,
        "stream_events": stream_events,
        "time_to_first_token_ms": time_to_first_token_ms,
    }


def with_rates(usage: dict, latency_ms: float | None) -> dict:
    """Add rates while preserving missing/unknown usage as null."""
    result = dict(usage)
    result["generation_seconds"] = latency_ms / 1000 if latency_ms is not None else None
    seconds = result["generation_seconds"]
    result["output_tokens_per_second"] = (
        result["output_tokens"] / seconds
        if seconds and result.get("output_tokens") is not None else None
    )
    result["reasoning_tokens_per_second"] = (
        result["reasoning_tokens"] / seconds
        if seconds and result.get("reasoning_tokens") is not None else None
    )
    return result


def merge_usage(first: dict | None, second: dict | None,
                latency_ms: float | None = None) -> dict:
    """Combine a primary generation and a forced-answer retry."""
    first = first or {}
    second = second or {}
    result = {}
    for field in TOKEN_FIELDS:
        values = [v for v in (first.get(field), second.get(field)) if v is not None]
        result[field] = sum(values) if values else None
    result["usage_source"] = "combined"
    result["usage_complete"] = bool(first.get("usage_complete")) and bool(second.get("usage_complete"))
    result["stream_events"] = sum(v for v in (first.get("stream_events"), second.get("stream_events")) if v is not None) or None
    result["time_to_first_token_ms"] = first.get("time_to_first_token_ms")
    return with_rates(result, latency_ms)
