"""Generate the three paper figure families from raw sample JSONL.

Outputs under ``analysis/``:

1. ``performance_vs_tokens``: strict accuracy vs mean reported total tokens
   per sample; thinking-enabled models are marked separately.
2. ``response_breakdown``: stacked outcome bars overall and by task category.
3. ``accuracy_heatmap``: strict accuracy by category and model.

The script never imputes missing provider token counts. It writes a complete
sample table and explicitly records missing-usage counts so figures can be
reproduced or withheld when accounting is incomplete.

Usage:
    python scripts/analyze_paper_figures.py --results-dir results/chess
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


CATEGORY_BY_TASK = {
    "mate1-lichess": "Short Tactics",
    "mate2-lichess": "Short Tactics",
    "bestmove-8x8": "Position Judgment",
    "mate-selection-test": "Short Tactics",
}

STATUS_ORDER = ("correct", "legal_wrong", "illegal", "parse_error", "no_answer")
STATUS_LABELS = {
    "correct": "Correct",
    "legal_wrong": "Legal, wrong oracle",
    "illegal": "Illegal",
    "parse_error": "Parse error",
    "no_answer": "No answer",
}
PROVIDER_COLORS = {
    "deepseek": "#f4a261",
    "google": "#e76f51",
    "unknown": "#6c757d",
}


def _number(value):
    return value if isinstance(value, (int, float)) else None


def _provider(model: str) -> str:
    model = model.lower()
    if "deepseek" in model:
        return "deepseek"
    if "gemma" in model:
        return "google"
    return "unknown"


def _category(task: str, sample: dict) -> str:
    return sample.get("task_category") or CATEGORY_BY_TASK.get(task, "Uncategorized")


def _status_group(sample: dict) -> str:
    if sample.get("status") == "legal" and sample.get("compliance") is True:
        return "correct"
    if sample.get("status") == "legal":
        return "legal_wrong"
    if sample.get("status") == "illegal":
        return "illegal"
    if sample.get("status") == "parse_error":
        return "parse_error"
    return "no_answer"


def _legacy_usage(sample: dict) -> dict:
    usage = sample.get("token_usage") or {}
    if usage:
        return usage
    input_tokens = _number(sample.get("input_tokens"))
    output_tokens = _number(sample.get("output_tokens"))
    total = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": None,
        "total_tokens": total,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "cache_hit_tokens": sample.get("cache_hit"),
        "cache_miss_tokens": sample.get("cache_miss"),
        "generation_seconds": (_number(sample.get("latency_ms")) or 0) / 1000 or None,
        "output_tokens_per_second": None,
        "reasoning_tokens_per_second": None,
        "time_to_first_token_ms": None,
        "usage_source": "legacy_flat" if total is not None else "missing",
        "usage_complete": total is not None,
    }


def _load_samples(results_dir: Path) -> list[dict]:
    rows = []
    for samples_path in sorted(results_dir.rglob("*.samples.jsonl")):
        summary_path = samples_path.with_name(samples_path.name.replace(".samples.jsonl", ".summary.json"))
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        meta = summary.get("meta", {})
        match = re.match(r"(.+?)_(mate1-lichess|mate2-lichess|bestmove-8x8|cap-legal-8x8|mob-8x8|mate1-8x8)_(.+)$", samples_path.stem.replace(".samples", ""))
        fallback_model = match.group(1) if match else "unknown"
        fallback_task = match.group(2) if match else "unknown"
        fallback_variant = match.group(3) if match else "unknown"
        for line in samples_path.read_text().splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            model = sample.get("model") or meta.get("model") or fallback_model
            task = sample.get("task") or meta.get("task") or fallback_task
            usage = _legacy_usage(sample)
            metadata = sample.get("position_metadata") or {}
            extra = metadata.get("task_extra") or {}
            total_tokens = usage.get("total_tokens")
            rows.append({
                "model": model,
                "provider": _provider(model),
                "task": task,
                "category": _category(task, sample),
                "representation": sample.get("representation") or meta.get("prompt_variant") or fallback_variant,
                "position_id": sample.get("position_id"),
                "status": sample.get("status"),
                "status_group": _status_group(sample),
                "strict_correct": int(sample.get("compliance") is True),
                "legal": int(sample.get("status") == "legal"),
                "format": sample.get("format"),
                "fallback": sample.get("fallback"),
                "thinking_enabled": sample.get("thinking_enabled", meta.get("thinking_enabled")),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "total_tokens": total_tokens,
                "cache_read_tokens": usage.get("cache_read_tokens"),
                "cache_miss_tokens": usage.get("cache_miss_tokens"),
                "generation_seconds": usage.get("generation_seconds"),
                "output_tokens_per_second": usage.get("output_tokens_per_second"),
                "reasoning_tokens_per_second": usage.get("reasoning_tokens_per_second"),
                "time_to_first_token_ms": usage.get("time_to_first_token_ms"),
                "usage_complete": usage.get("usage_complete", False),
                "rating": metadata.get("rating"),
                "rating_deviation": metadata.get("rating_deviation"),
                "popularity": metadata.get("popularity"),
                "nb_plays": metadata.get("nb_plays"),
                "themes": extra.get("themes"),
                "opening_tags": extra.get("opening_tags"),
                "cp": extra.get("cp"),
                "depth": extra.get("depth"),
                "knodes": extra.get("knodes"),
            })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model", "task", "category"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def _cell_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["task"], row["category"], row["representation"])].append(row)
    result = []
    for (model, task, category, representation), group in sorted(groups.items()):
        result.append({
            "model": model,
            "task": task,
            "category": category,
            "representation": representation,
            "n": len(group),
            "legal_rate": _mean([r["legal"] for r in group]),
            "strict_accuracy": _mean([r["strict_correct"] for r in group]),
            "avg_input_tokens": _mean([r["input_tokens"] for r in group]),
            "avg_output_tokens": _mean([r["output_tokens"] for r in group]),
            "avg_reasoning_tokens": _mean([r["reasoning_tokens"] for r in group]),
            "avg_total_tokens": _mean([r["total_tokens"] for r in group]),
            "avg_generation_seconds": _mean([r["generation_seconds"] for r in group]),
            "avg_output_tokens_per_second": _mean([r["output_tokens_per_second"] for r in group]),
            "cache_read_tokens": sum(r["cache_read_tokens"] or 0 for r in group) if any(r["cache_read_tokens"] is not None for r in group) else None,
            "cache_miss_tokens": sum(r["cache_miss_tokens"] or 0 for r in group) if any(r["cache_miss_tokens"] is not None for r in group) else None,
            "fallback_count": sum(bool(r["fallback"]) for r in group),
            "usage_complete_n": sum(bool(r["usage_complete"]) for r in group),
        })
    return result


def _save_fig(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")


def make_figures(rows: list[dict], cells: list[dict], out_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # Figure 1: performance versus token budget.
    token_cells = [c for c in cells if c["avg_total_tokens"] is not None]
    if token_cells:
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in token_cells:
            thinking = any(r["model"] == c["model"] and r["thinking_enabled"] for r in rows)
            marker = "*" if thinking else "o"
            ax.scatter(c["avg_total_tokens"], c["strict_accuracy"],
                       s=80, marker=marker, color=PROVIDER_COLORS.get(_provider(c["model"]), "#6c757d"),
                       edgecolors="black", linewidths=0.35)
            ax.annotate(f"{c['model']}\n{c['task']}",
                        (c["avg_total_tokens"], c["strict_accuracy"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Average reported total tokens per sample")
        ax.set_ylabel("Strict accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.18)
        _save_fig(fig, out_dir, "performance_vs_tokens")
        plt.close(fig)
        figures.append("performance_vs_tokens")

    # Figure 2: stacked outcome bars, overall plus categories.
    categories = ["All tasks"] + sorted({c["category"] for c in cells})
    models = sorted({c["model"] for c in rows})
    fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), max(3.5, 0.55 * len(models))), sharey=True)
    axes = [axes] if len(categories) == 1 else list(axes)
    status_colors = {"correct": "#59b99b", "legal_wrong": "#ff8765", "illegal": "#f3cd3b", "parse_error": "#8c9bc4", "no_answer": "#b9b9b9"}
    for ax, category in zip(axes, categories):
        selected = [r for r in rows if category == "All tasks" or r["category"] == category]
        for idx, model in enumerate(models):
            group = [r for r in selected if r["model"] == model]
            denom = len(group) or 1
            left = 0
            for status in STATUS_ORDER:
                value = sum(r["status_group"] == status for r in group) / denom
                ax.barh(idx, value, left=left, color=status_colors[status], label=STATUS_LABELS[status] if idx == 0 else None)
                left += value
        ax.set_title(category)
        ax.set_xlim(0, 1)
        ax.set_yticks(range(len(models)), models)
        ax.set_xlabel("Fraction of samples")
        ax.grid(axis="x", alpha=0.15)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.28), ncol=3, fontsize=8)
    _save_fig(fig, out_dir, "response_breakdown")
    plt.close(fig)
    figures.append("response_breakdown")

    # Figure 3: category x model heatmap of strict accuracy.
    heat_categories = sorted({c["category"] for c in cells})
    matrix = np.full((len(heat_categories), len(models)), np.nan)
    for i, category in enumerate(heat_categories):
        for j, model in enumerate(models):
            group = [r for r in rows if r["category"] == category and r["model"] == model]
            if group:
                matrix[i, j] = _mean([r["strict_correct"] for r in group])
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(models)), max(3, 0.8 * len(heat_categories))))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)), models, rotation=35, ha="right")
    ax.set_yticks(range(len(heat_categories)), heat_categories)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center",
                        color="white" if matrix[i, j] > 0.55 else "#18212b", fontsize=9)
    fig.colorbar(im, ax=ax, label="Strict accuracy")
    ax.set_title("Strict accuracy by task category and model")
    _save_fig(fig, out_dir, "accuracy_heatmap")
    plt.close(fig)
    figures.append("accuracy_heatmap")
    return figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/chess")
    parser.add_argument("--out-dir", default="analysis")
    args = parser.parse_args()
    rows = _load_samples(Path(args.results_dir))
    if not rows:
        raise SystemExit(f"no samples found in {args.results_dir}")
    out = Path(args.out_dir)
    cells = _cell_rows(rows)
    _write_csv(out / "tables" / "paper_samples.csv", rows)
    _write_csv(out / "tables" / "paper_cells.csv", cells)
    figures = make_figures(rows, cells, out / "figures")
    (out / "figure_data.json").write_text(json.dumps({
        "sample_count": len(rows),
        "cell_count": len(cells),
        "figures": figures,
        "token_usage_complete_samples": sum(bool(r["usage_complete"]) for r in rows),
        "token_usage_missing_samples": sum(not r["usage_complete"] for r in rows),
    }, indent=1))
    print(f"samples={len(rows)} cells={len(cells)} figures={figures}")


if __name__ == "__main__":
    main()
