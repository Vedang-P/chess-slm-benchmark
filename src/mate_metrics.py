"""MATE selection scoreboard math, shared by every producer of it.

`compute_mate_metrics` turns raw per-sample rows into the dashboard's MATE
scoreboard shape. `combine_mate_metrics` recombines N such dicts (e.g. one
per parallel Kaggle worker plus a fixed baseline) back into that same shape.

The combine step sums the RAW counts (correct, n, truth_a, ...) and only
then recomputes rates -- it never averages a rate that was itself already
an average. Averaging pre-rounded per-worker accuracies would silently
drift from the true combined accuracy whenever workers finish unequal
numbers of positions, which they always will (they start at different
times and positions take wildly different amounts of time to reason
through). Given the whole point of this study is trustworthy numbers,
that drift is not acceptable, so every combinable field here is a count,
never a rate, until the final round().
"""
from __future__ import annotations


def compute_mate_metrics(rows: list[dict], done_here: int = 0,
                         elapsed_h: float | None = None) -> dict:
    """Everything the dashboard needs about a MATE selection run, computed
    from raw sample rows. `done_here`/`elapsed_h` are this process's own
    throughput window (None/0 for a static baseline that isn't running)."""
    scored = [s for s in rows if s.get("status") != "api_error"]
    answered = [s for s in scored if s.get("status") in ("correct", "wrong")]
    n = len(scored)
    truth = lambda s: ((s.get("position_metadata") or {}).get("task_extra") or {}).get("truth_label")
    by_truth = {}
    for label in ("A", "B"):
        group = [s for s in scored if truth(s) == label]
        correct_in_group = sum(bool(s["compliance"]) for s in group)
        by_truth[label] = {
            "n": len(group),
            "correct": correct_in_group,  # raw count: what combine_mate_metrics sums
            "accuracy": round(correct_in_group / len(group), 4) if group else None,
        }
    reasons = {}
    for s in scored:
        reason = s.get("no_answer_reason")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1

    def _mean(values):
        values = [v for v in values if isinstance(v, (int, float))]
        return round(sum(values) / len(values), 2) if values else None

    positions_per_hour = None
    if done_here and elapsed_h:
        positions_per_hour = round(done_here / elapsed_h, 1)

    return {
        "n": n,
        "n_attempted": len(rows),
        "answered": len(answered),
        "correct": sum(bool(s["compliance"]) for s in scored),
        "wrong": sum(s.get("status") == "wrong" for s in scored),
        "no_answer": sum(s.get("status") == "no_answer" for s in scored),
        "parse_error": sum(s.get("status") == "parse_error" for s in scored),
        "api_error": sum(s.get("status") == "api_error" for s in rows),
        "accuracy": round(sum(bool(s["compliance"]) for s in scored) / n, 4) if n else None,
        "accuracy_of_answered": (
            round(sum(bool(s["compliance"]) for s in answered) / len(answered), 4)
            if answered else None),
        "answer_rate": round(len(answered) / n, 4) if n else None,
        # the run's actual signal: does the model just always say B?
        "picked_a": sum(s.get("label") == "A" for s in scored),
        "picked_b": sum(s.get("label") == "B" for s in scored),
        "truth_a": by_truth["A"]["n"],
        "truth_b": by_truth["B"]["n"],
        "correct_truth_a": by_truth["A"]["correct"],
        "correct_truth_b": by_truth["B"]["correct"],
        "accuracy_truth_a": by_truth["A"]["accuracy"],
        "accuracy_truth_b": by_truth["B"]["accuracy"],
        "no_answer_reasons": reasons,
        "mean_latency_s": _mean([(s.get("latency_ms") or 0) / 1000 for s in scored
                                 if s.get("latency_ms") is not None]),
        "mean_output_tokens": _mean([(s.get("token_usage") or {}).get("output_tokens")
                                     for s in scored]),
        "mean_reasoning_tokens": _mean([(s.get("token_usage") or {}).get("reasoning_tokens")
                                        for s in scored]),
        "positions_per_hour": positions_per_hour,
    }


def combine_mate_metrics(parts: list[dict]) -> dict:
    """Recombine N compute_mate_metrics() dicts into one, e.g. a fixed
    100-position baseline plus 6 live parallel workers.

    Every count-like field sums exactly. `mean_*` fields have no raw
    sum/count stored on the part (only the pre-averaged mean), so they're
    recombined as an n-weighted average of means -- a documented
    approximation for secondary display stats, never used for the
    headline accuracy number, which is always an exact sum-of-counts.
    positions_per_hour sums directly: workers run concurrently, so their
    throughputs add.
    """
    def total(key):
        return sum(p.get(key) or 0 for p in parts)

    n = total("n")
    answered = total("answered")
    correct = total("correct")
    truth_a = total("truth_a")
    truth_b = total("truth_b")
    correct_truth_a = total("correct_truth_a")
    correct_truth_b = total("correct_truth_b")

    reasons: dict = {}
    for p in parts:
        for reason, count in (p.get("no_answer_reasons") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + count

    def _weighted_mean(key):
        pairs = [(p.get(key), p.get("n") or 0) for p in parts
                 if p.get(key) is not None and (p.get("n") or 0) > 0]
        weight = sum(w for _, w in pairs)
        return round(sum(v * w for v, w in pairs) / weight, 2) if weight else None

    rates = [p.get("positions_per_hour") for p in parts if p.get("positions_per_hour")]
    positions_per_hour = round(sum(rates), 1) if rates else None

    return {
        "n": n,
        "n_attempted": total("n_attempted"),
        "answered": answered,
        "correct": correct,
        "wrong": total("wrong"),
        "no_answer": total("no_answer"),
        "parse_error": total("parse_error"),
        "api_error": total("api_error"),
        "accuracy": round(correct / n, 4) if n else None,
        "accuracy_of_answered": round(correct / answered, 4) if answered else None,
        "answer_rate": round(answered / n, 4) if n else None,
        "picked_a": total("picked_a"),
        "picked_b": total("picked_b"),
        "truth_a": truth_a,
        "truth_b": truth_b,
        "correct_truth_a": correct_truth_a,
        "correct_truth_b": correct_truth_b,
        "accuracy_truth_a": round(correct_truth_a / truth_a, 4) if truth_a else None,
        "accuracy_truth_b": round(correct_truth_b / truth_b, 4) if truth_b else None,
        "no_answer_reasons": reasons,
        "mean_latency_s": _weighted_mean("mean_latency_s"),
        "mean_output_tokens": _weighted_mean("mean_output_tokens"),
        "mean_reasoning_tokens": _weighted_mean("mean_reasoning_tokens"),
        "positions_per_hour": positions_per_hour,
    }
