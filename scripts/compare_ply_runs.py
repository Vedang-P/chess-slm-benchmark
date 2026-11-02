"""Paired validation comparison of the later-ply control and treatment runs.

Both runs start from the same main-run checkpoint and differ only in ``--ply-groups``.
This reads their per-puzzle validation results (``--val-detail``), applies the
decision rule recorded in PROJECT-STATUS.md before the runs were launched, and prints
the verdict. It never reads the official 10,000-puzzle evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The pre-registered rule (PROJECT-STATUS.md, "Later-ply roots: decision rule").
RULE = {"min_gain": 20,          # solved-puzzle gain of the treatment at the final step
        "alpha": 0.05,           # two-sided exact McNemar p at the final step
        "min_mean_gain": 15,     # mean gain over the last three evaluations
        "later_ratio": 0.90,     # treatment later-ply failures <= 90% of the control's
        "first_ratio": 1.05,     # treatment first-decision failures <= 105% of the control's
        "dev_ratio": 1.01,       # treatment broad dev loss <= 101% of the control's
        "reject_gain": 8}        # at or below this gain the treatment is rejected
EVAL_STEPS = (10_000, 12_500, 15_000)


def mcnemar_exact(only_treatment: int, only_control: int) -> float:
    """Two-sided exact p-value of the discordant pairs under a fair coin."""
    n = only_treatment + only_control
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_treatment, only_control) + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def failures(correct: np.ndarray, fail_ply: list) -> dict:
    """Unsolved puzzles by where the model first went wrong (ply 1 = first decision)."""
    by_ply: dict[int, int] = {}
    for ok, ply in zip(correct, fail_ply):
        if not ok and ply is not None:
            by_ply[int(ply)] = by_ply.get(int(ply), 0) + 1
    first = by_ply.get(1, 0)
    return {"first": first, "later": sum(by_ply.values()) - first,
            "by_ply": {str(k): v for k, v in sorted(by_ply.items())}}


def paired(control: dict, treatment: dict) -> dict:
    """Paired summary of two runs' results on the same validation puzzles at one step."""
    if control["ids"] != treatment["ids"]:
        raise ValueError("the two runs were validated on different puzzles")
    c, t = np.asarray(control["correct"], dtype=bool), np.asarray(treatment["correct"], dtype=bool)
    only_t, only_c = int((t & ~c).sum()), int((c & ~t).sum())
    return {"total": int(len(c)), "solved": {"control": int(c.sum()), "treatment": int(t.sum())},
            "gain": int(t.sum() - c.sum()), "only_treatment": only_t, "only_control": only_c,
            "mcnemar_p": mcnemar_exact(only_t, only_c),
            "gain_se": math.sqrt(only_t + only_c),
            "failures": {"control": failures(c, control["fail_ply"]),
                         "treatment": failures(t, treatment["fail_ply"])}}


def verdict(per_step: dict[int, dict], dev_loss: dict[str, float], rule: dict = RULE) -> dict:
    """ADOPT / REJECT / INCONCLUSIVE from the paired summaries of the last evaluations."""
    final = per_step[max(per_step)]
    gains = [per_step[s]["gain"] for s in sorted(per_step)]
    fc, ft = final["failures"]["control"], final["failures"]["treatment"]
    checks = {
        "effect": final["gain"] >= rule["min_gain"] and final["mcnemar_p"] < rule["alpha"],
        "consistent": float(np.mean(gains)) >= rule["min_mean_gain"],
        "mechanism": ft["later"] <= rule["later_ratio"] * fc["later"]
                     and ft["first"] <= rule["first_ratio"] * fc["first"],
        "broad_loss": dev_loss["treatment"] <= rule["dev_ratio"] * dev_loss["control"]}
    if all(checks.values()):
        outcome = "ADOPT"
    elif final["gain"] <= rule["reject_gain"] or ft["later"] >= fc["later"]:
        outcome = "REJECT"
    else:
        outcome = "INCONCLUSIVE"
    return {"outcome": outcome, "checks": checks, "gains": dict(zip(sorted(per_step), gains)),
            "rule": rule}


def fetch_results(run: str, step: int, repo: str, client) -> tuple[dict, dict]:
    from huggingface_hub import hf_hub_download
    folder = f"{run}/checkpoint-{step}"
    results = json.loads(Path(hf_hub_download(
        repo, f"{folder}/val-results.json", repo_type="dataset", token=client.token)).read_text())
    metrics = json.loads(Path(hf_hub_download(
        repo, f"{folder}/metrics.json", repo_type="dataset", token=client.token)).read_text())
    return results, metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--control", required=True)
    p.add_argument("--treatment", required=True)
    p.add_argument("--steps", type=int, nargs="+", default=list(EVAL_STEPS))
    p.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    args = p.parse_args()
    from scripts.kaggle_checkpoint import api as make_api
    client = make_api(ROOT)
    per_step, dev = {}, {}
    for step in sorted(args.steps):
        rc, mc = fetch_results(args.control, step, args.hf_repo, client)
        rt, mt = fetch_results(args.treatment, step, args.hf_repo, client)
        per_step[step] = paired(rc, rt)
        dev = {"control": mc["dev_loss"], "treatment": mt["dev_loss"]}
        s = per_step[step]
        print(f"step {step}: control {s['solved']['control']}/{s['total']}  treatment "
              f"{s['solved']['treatment']}/{s['total']}  gain {s['gain']:+d} (se {s['gain_se']:.1f}, "
              f"treatment-only {s['only_treatment']}, control-only {s['only_control']}, "
              f"McNemar p={s['mcnemar_p']:.4f})  failures first/later: control "
              f"{s['failures']['control']['first']}/{s['failures']['control']['later']}, treatment "
              f"{s['failures']['treatment']['first']}/{s['failures']['treatment']['later']}", flush=True)
    print(f"broad dev loss at step {max(args.steps)}: {dev}", flush=True)
    print(json.dumps(verdict(per_step, dev), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
