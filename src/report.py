"""Result writing: per-sample JSONL, per-run summary JSON, comparison CSV.

The comparison table is the paper's main artifact: rows = (model, task,
representation, condition), columns = correctness/legality plus normalized
token, cache, latency, and fallback aggregates.
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional


class ResultWriter:
    def __init__(self, output_dir: Path, run_name: str, meta: dict):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.meta = meta
        self.samples_path = self.output_dir / f"{run_name}.samples.jsonl"
        self.summary_path = self.output_dir / f"{run_name}.summary.json"
        self.samples: List[dict] = []

    def add(self, sample: dict) -> None:
        self.samples.append(sample)
        with open(self.samples_path, "a") as f:
            f.write(json.dumps(sample) + "\n")

    def finish(self, metrics: dict) -> dict:
        summary = {
            "run": self.run_name,
            "schema": 3,  # bumped when the scorer/parser or loader changes such
                          # that old summaries must be re-scored. v3: standard
                          # chess via python-chess (castling/ep/double-step
                          # legal) — earlier results used a simplified variant
            "meta": self.meta,
            "metrics": metrics,
            "n_samples": len(self.samples),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.summary_path.write_text(json.dumps(summary, indent=1))
        return summary


def aggregate_samples(samples: List[dict]) -> dict:
    """Per-condition metrics over samples. Samples carry: position_id,
    condition, status, compliance (True/False/None), move."""
    conds = {}
    for s in samples:
        c = s["condition"]
        if c not in conds:
            conds[c] = {"n": 0, "no_answer": 0, "parse_error": 0, "illegal": 0,
                        "legal": 0, "compliant": 0, "noncompliant": 0,
                        "undefined": 0, "compliance_of_legal": None}
        d = conds[c]
        d["n"] += 1
        status = s["status"]
        d[status] += 1
        if status == "legal":
            if s.get("compliance") is True:
                d["compliant"] += 1
            elif s.get("compliance") is False:
                d["noncompliant"] += 1
            else:
                d["undefined"] += 1
    for d in conds.values():
        legal = d["legal"]
        decided = d["compliant"] + d["noncompliant"]
        d["parse_rate"] = round((d["legal"] + d["illegal"]) / d["n"], 4) if d["n"] else 0.0
        d["legal_rate"] = round(d["legal"] / d["n"], 4) if d["n"] else 0.0
        d["compliance_of_legal"] = (
            round(d["compliant"] / decided, 4) if decided else None
        )
        # compliance of ALL samples (treat no-answer/parse/illegal as
        # non-compliant -- the strict policy used in the paper)
        d["compliance_strict"] = round(d["compliant"] / d["n"], 4) if d["n"] else 0.0
    return {"conditions": conds, "token_usage": aggregate_token_usage(samples)}


def aggregate_token_usage(samples: List[dict]) -> dict:
    """Aggregate provider/local usage without treating missing values as 0.

    This is deliberately provider-neutral: the paper can compare API
    reasoning tokens, local generated tokens, cache reads, and latency while
    retaining the number of samples for which a provider did not report an
    exact field.
    """
    fields = (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
        "generation_seconds",
        "output_tokens_per_second",
        "reasoning_tokens_per_second",
        "time_to_first_token_ms",
    )
    values = {field: [] for field in fields}
    fallback_counts: Dict[str, int] = {}
    for sample in samples:
        usage = sample.get("token_usage") or {}
        for field in fields:
            value = usage.get(field)
            if isinstance(value, (int, float)):
                values[field].append(float(value))
        fallback = sample.get("fallback")
        if fallback:
            fallback_counts[fallback] = fallback_counts.get(fallback, 0) + 1

    result = {
        "sample_count": len(samples),
        "fallback_counts": fallback_counts,
        "missing_exact_usage": sum(
            1 for sample in samples
            if not (sample.get("token_usage") or {}).get("usage_complete", False)
        ),
    }
    for field, nums in values.items():
        result[f"{field}_n"] = len(nums)
        result[f"{field}_total"] = round(sum(nums), 4) if nums else None
        result[f"{field}_mean"] = round(statistics.mean(nums), 4) if nums else None
        result[f"{field}_median"] = round(statistics.median(nums), 4) if nums else None
        result[f"{field}_p95"] = round(statistics.quantiles(nums, n=20)[18], 4) if len(nums) >= 2 else (round(nums[0], 4) if nums else None)
    return result


def divergence_rate(samples: List[dict]) -> Optional[float]:
    """Fraction of positions where the model's move differs between WIN and
    LOSE conditions (both legal). This is the within-model control."""
    by_pos: Dict[str, dict] = {}
    for s in samples:
        p = s["position_id"]
        if s["status"] == "legal":
            by_pos.setdefault(p, {})[s["condition"]] = s["move"]
    paired = {p: m for p, m in by_pos.items() if "win" in m and "lose" in m}
    if not paired:
        return None
    same = sum(1 for m in paired.values() if m["win"] == m["lose"])
    return round(1.0 - same / len(paired), 4)


def write_comparison_csv(output_dir: Path, rows: List[dict]) -> Path:
    """rows: list of dicts with run/meta/metrics already flattened."""
    path = Path(output_dir) / "comparison_table.csv"
    cols = ["model", "task", "variant", "condition", "n", "parse_rate", "legal_rate",
            "compliance_of_legal", "compliance_strict", "undefined",
            "avg_input_tokens", "avg_output_tokens", "avg_reasoning_tokens",
            "avg_total_tokens", "avg_generation_seconds",
            "avg_output_tokens_per_second", "cache_read_tokens_total",
            "cache_miss_tokens_total", "fallback_count"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            usage = r.get("token_usage") or {}
            fallback_counts = usage.get("fallback_counts") or {}
            values = {
                **r,
                "avg_input_tokens": usage.get("input_tokens_mean"),
                "avg_output_tokens": usage.get("output_tokens_mean"),
                "avg_reasoning_tokens": usage.get("reasoning_tokens_mean"),
                "avg_total_tokens": usage.get("total_tokens_mean"),
                "avg_generation_seconds": usage.get("generation_seconds_mean"),
                "avg_output_tokens_per_second": usage.get("output_tokens_per_second_mean"),
                "cache_read_tokens_total": usage.get("cache_read_tokens_total"),
                "cache_miss_tokens_total": usage.get("cache_miss_tokens_total"),
                "fallback_count": sum(fallback_counts.values()),
            }
            w.writerow({c: values.get(c, "") for c in cols})
    return path
