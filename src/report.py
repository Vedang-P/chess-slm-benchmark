"""Result writing: per-sample JSONL, per-run summary JSON, comparison CSV.

The comparison table is the paper's main artifact: rows = (model, task,
condition), columns = n, parse_rate, legal_rate, compliance, plus
per-position divergence for paired conditions.
"""
from __future__ import annotations

import csv
import json
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
            d["legal"] += 1
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
    return {"conditions": conds}


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
    cols = ["model", "task", "condition", "n", "parse_rate", "legal_rate",
            "compliance_of_legal", "compliance_strict", "undefined"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path
