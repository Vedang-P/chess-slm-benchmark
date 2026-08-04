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


SCHEMA_VERSION = 4
# 4: mate-in-2 gets its own prompt (was reusing the mate-in-1 "mate in exactly
#    one move" objective); transport failures are `api_error`, not parse_error;
#    SAN extraction takes the LAST legal token. Any summary at schema < 4 was
#    produced by a different scorer/prompt and must be re-run, not merged.
# 3: standard chess via python-chess (castling/ep/double-step legal).


class ResultWriter:
    """Writes one cell's samples + summary.

    `resume=False` (the default) TRUNCATES the samples file: a re-run of a
    cell must not append onto the previous attempt's rows. Appending is how a
    re-scored/crashed cell used to leave duplicated and mixed-schema samples
    in the JSONL that feeds the paper figures, while the summary — computed
    from memory — looked clean.

    `prior_samples` is the count of rows already in the file that this process
    is deliberately keeping (resume), so `n_samples` reports the true total
    rather than only what this process wrote.
    """

    def __init__(self, output_dir: Path, run_name: str, meta: dict,
                 resume: bool = False, prior_samples: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.meta = meta
        self.samples_path = self.output_dir / f"{run_name}.samples.jsonl"
        self.summary_path = self.output_dir / f"{run_name}.summary.json"
        self.samples: List[dict] = []
        self.prior_samples = prior_samples if resume else 0
        if not resume and self.samples_path.exists():
            self.samples_path.unlink()

    def add(self, sample: dict) -> None:
        self.samples.append(sample)
        with open(self.samples_path, "a") as f:
            f.write(json.dumps(sample) + "\n")

    def finish(self, metrics: dict) -> dict:
        summary = {
            "run": self.run_name,
            "schema": SCHEMA_VERSION,
            "meta": self.meta,
            "metrics": metrics,
            "n_samples": self.prior_samples + len(self.samples),
            "n_samples_this_process": len(self.samples),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.summary_path.write_text(json.dumps(summary, indent=1))
        return summary


def aggregate_samples(samples: List[dict]) -> dict:
    """Per-condition metrics over samples. Samples carry: position_id,
    condition, status, compliance (True/False/None), move.

    `n` is the number of SCORED samples: api_error rows (gateway 5xx,
    timeouts, rate limits) are counted separately in `api_error` and kept out
    of every rate, because a transport failure is not a model failure. `n_attempted`
    reports the raw row count so the two can never be silently confused.
    """
    conds = {}
    for s in samples:
        c = s["condition"]
        if c not in conds:
            conds[c] = {"n": 0, "n_attempted": 0, "api_error": 0,
                        "no_answer": 0, "parse_error": 0, "illegal": 0,
                        "legal": 0, "compliant": 0, "noncompliant": 0,
                        "undefined": 0, "compliance_of_legal": None}
        d = conds[c]
        d["n_attempted"] += 1
        status = s["status"]
        if status not in d:
            d[status] = 0
        d[status] += 1
        if status == "api_error":
            continue  # not a scored sample
        d["n"] += 1
        if status == "legal":
            if s.get("compliance") is True:
                d["compliant"] += 1
            elif s.get("compliance") is False:
                d["noncompliant"] += 1
            else:
                d["undefined"] += 1
    for d in conds.values():
        decided = d["compliant"] + d["noncompliant"]
        d["parse_rate"] = round((d["legal"] + d["illegal"]) / d["n"], 4) if d["n"] else 0.0
        d["legal_rate"] = round(d["legal"] / d["n"], 4) if d["n"] else 0.0
        d["compliance_of_legal"] = (
            round(d["compliant"] / decided, 4) if decided else None
        )
        # compliance of ALL scored samples (no-answer/parse/illegal count as
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


def write_comparison_csv(output_dir: Path, rows: List[dict]) -> Path:
    """rows: list of dicts with run/meta/metrics already flattened."""
    path = Path(output_dir) / "comparison_table.csv"
    cols = ["model", "task", "variant", "condition", "n", "n_attempted",
            "api_error", "parse_rate", "legal_rate",
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
