"""Run the model x task x prompt-variant matrix and produce the comparison table.

With --monitor, the suite publishes progress to the repo's `live` branch
(monitor/state.json + monitor/history.jsonl) so the dashboard can render it.
Monitoring pushes never break the sweep: every git step is failure-tolerant.

Usage:
    python scripts/run_suite.py                # full sweep (paper data)
    python scripts/run_suite.py --check        # tiny sanity sweep
    python scripts/run_suite.py --smoke        # stub models, no GPU
    python scripts/run_suite.py --monitor      # publish progress to 'live' branch
    python scripts/run_suite.py --models deepseek-r1-distill-qwen-1.5b smollm2-1.7b
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.report import write_comparison_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MONITOR_BRANCH = "live"
MONITOR_DIR = ROOT / "monitor"
HISTORY_CAP = 500
PUBLIC_LIVE_REPO = "Vedang-P/chess-bench-live"  # public repo: the dashboard reads from here
PUBLIC_LIVE_BRANCH = "main"


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Monitor:
    def __init__(self, interval_s: int = 120, output_dir: str = "results/chess"):
        self.interval = interval_s
        self.last_push = 0.0
        self.cells_done = 0
        self.cells_total = 0
        self.started_at = _ts()
        self.rows = []
        self.meta = {}
        self.current = None
        self.output_dir = Path(output_dir)
        self.uploaded = set()  # remote paths already uploaded (tracked in-memory)

    def set_meta(self, **kw) -> None:
        self.meta.update(kw)

    def cell_done(self, row_parts: list) -> None:
        self.cells_done += 1
        for r in row_parts:
            self.rows.append(r)

    def set_current(self, model: str = None, task: str = None, variant: str = None) -> None:
        self.current = (model, task, variant) if model else None

    def maybe_push(self, force: bool = False, last_error: str = None) -> None:
        if not force and time.time() - self.last_push < self.interval:
            return
        self._write_state(last_error)
        self._push()
        self.last_push = time.time()

    # ------------------------------------------------------------------ #
    def _state(self, last_error: str = None) -> dict:
        # group rows into per-cell objects
        cells = []
        by_key = {}
        for r in self.rows:
            key = (r.get("model"), r.get("task"), r.get("variant"))
            by_key.setdefault(key, {})[r.get("condition")] = r
        for (model, task, variant), conds in sorted(by_key.items()):
            cell = {"model": model, "task": task, "variant": variant,
                    "done": True, "n": None,
                    "win": {}, "lose": {}, "divergence": None}
            for cond, r in conds.items():
                if cond == "divergence":
                    cell["divergence"] = r.get("compliance_of_legal")
                elif cond == "game":
                    cell["game"] = {k: v for k, v in r.items()
                                    if k not in ("model", "task", "variant", "condition")}
                else:
                    cell[cond] = {
                        "n": r.get("n"), "parse_rate": r.get("parse_rate"),
                        "legal_rate": r.get("legal_rate"),
                        "compliance_of_legal": r.get("compliance_of_legal"),
                        "compliance_strict": r.get("compliance_strict"),
                    }
                    cell["n"] = r.get("n")
            cells.append(cell)
        done = self.cells_done
        total = self.cells_total
        elapsed_s = time.time() - time.mktime(time.strptime(self.started_at, "%Y-%m-%dT%H:%M:%S"))
        eta_min = None
        if done > 0 and elapsed_s > 0:
            eta_min = int(elapsed_s / done * (total - done) / 60)
        return {
            "repo": "Vedang-P/neuro-symbolic-pathfinding",
            "mode": self.meta.get("mode"),
            "stage": "sweep",
            "started_at": self.started_at,
            "updated_at": _ts(),
            "progress": {"cells_done": done, "cells_total": total,
                         "fraction": round(done / total, 4) if total else 0.0},
            "eta_min": eta_min,
            "models": self.meta.get("models", []),
            "current": {"model": self.current[0], "task": self.current[1],
                        "variant": self.current[2]} if self.current else None,
            "last_error": last_error,
            "cells": cells,
        }

    def _write_state(self, last_error: str = None) -> None:
        MONITOR_DIR.mkdir(exist_ok=True)
        state = self._state(last_error)
        (MONITOR_DIR / "state.json").write_text(json.dumps(state, indent=1))
        hist = MONITOR_DIR / "history.jsonl"
        lines = hist.read_text().splitlines() if hist.exists() else []
        lines.append(json.dumps({
            "ts": state["updated_at"], "cells_done": state["progress"]["cells_done"],
            "fraction": state["progress"]["fraction"], "eta_min": state["eta_min"],
            "legal_avg": _avg_legal(state["cells"]),
            "last_error": last_error,
        }))
        lines = lines[-HISTORY_CAP:]
        hist.write_text("\n".join(lines) + "\n")

    def _push(self) -> None:
        """Contents-API upload to the PUBLIC live repo (works from any machine
        with GITHUB_TOKEN/GH_TOKEN); falls back to a git push of the private
        `live` branch for debugging. Never raises."""
        from src.live_push import resolve_token

        token = resolve_token()
        if token:
            try:
                self._push_via_api(token)
                return
            except Exception as e:
                print(f"monitor: contents-API push failed ({e}) — retrying once", flush=True)
                time.sleep(2)
                try:
                    self._push_via_api(token)
                    return
                except Exception as e2:
                    print(f"monitor: contents-API retry failed ({e2}) — git fallback", flush=True)
        if not (ROOT / ".git").exists():
            print("monitor: not a git repo — push skipped", flush=True)
            return
        steps = [
            ["git", "add", "-f", "monitor/state.json", "monitor/history.jsonl"],
            # identity-free commit: Kaggle containers have no git user configured
            ["git", "-c", "user.name=chess-monitor",
             "-c", "user.email=chess-monitor@users.noreply.github.com",
             "commit", "--quiet", "-m", f"monitor {_ts()}"],
            ["git", "push", "--force", "origin", f"HEAD:{MONITOR_BRANCH}"],
        ]
        for cmd in steps:
            env = {**os.environ,
                   "GIT_AUTHOR_NAME": "chess-monitor",
                   "GIT_AUTHOR_EMAIL": "chess-monitor@users.noreply.github.com",
                   "GIT_COMMITTER_NAME": "chess-monitor",
                   "GIT_COMMITTER_EMAIL": "chess-monitor@users.noreply.github.com"}
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
            if r.returncode != 0:
                if cmd[1] == "commit" and "nothing to commit" in r.stderr:
                    continue  # no change since last push — still push (harmless)
                print(f"monitor: git {cmd[1]} failed: {r.stderr[-200:]}", flush=True)
                return

    def _push_via_api(self, token: str) -> None:
        from src.live_push import upload_file

        # collect files to upload: monitor state/history + completed cell
        # summaries + running comparison CSV + results index + live.json
        uploads = []
        for fname in ("state.json", "history.jsonl", "live.json"):
            path = MONITOR_DIR / fname
            if path.exists():
                uploads.append((f"monitor/{fname}", path))
        if self.output_dir.exists():
            for summary in sorted(self.output_dir.glob("*.summary.json")):
                remote = f"results/chess/{summary.name}"
                if remote not in self.uploaded:
                    uploads.append((remote, summary))
            csv_path = self.output_dir / "comparison_table.csv"
            if csv_path.exists():
                uploads.append(("results/comparison_table.csv", csv_path))
            # index of all completed summaries (for the recovery flow)
            index = {
                "files": sorted(p.name for p in self.output_dir.glob("*.summary.json")),
                "updated_at": _ts(),
            }
            uploads.append(("results/index.json",
                            _BytesFile(json.dumps(index, indent=1).encode())))
        for remote, local in uploads:
            data = local.data if isinstance(local, _BytesFile) else local.read_bytes()
            if not upload_file(token, remote, data):
                print(f"monitor: upload failed for {remote}", flush=True)
                continue
            if isinstance(local, Path):
                self.uploaded.add(remote)
        print("monitor: pushed state + results to public live repo", flush=True)


class _BytesFile:
    """In-memory stand-in for a Path upload."""
    __slots__ = ("data",)

    def __init__(self, data: bytes):
        self.data = data


def _avg_legal(cells: list) -> float:
    vals = [c["win"].get("legal_rate") for c in cells
            if c["win"].get("legal_rate") is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _rows_from_summary(model: str, task: str, variant: str, summary: dict) -> list:
    rows = []
    metrics = summary.get("metrics", {})
    if "games" in metrics:
        g = metrics["games"]
        rows.append({
            "model": model, "task": task, "variant": variant, "condition": "game",
            "n": g["n"], "parse_rate": "", "legal_rate": g.get("legal_rate"),
            "compliance_of_legal": g.get("win_rate"),
            "compliance_strict": "", "undefined": "",
            "game": {k: v for k, v in g.items()},
        })
        return rows
    for cond, m in metrics.get("conditions", {}).items():
        rows.append({
            "model": model, "task": task, "variant": variant, "condition": cond,
            "n": m["n"], "parse_rate": m["parse_rate"],
            "legal_rate": m["legal_rate"],
            "compliance_of_legal": m["compliance_of_legal"],
            "compliance_strict": m["compliance_strict"],
            "undefined": m["undefined"],
        })
    div = summary["metrics"].get("divergence_rate")
    if div is not None:
        rows.append({
            "model": model, "task": task, "variant": variant,
            "condition": "divergence", "n": "", "parse_rate": "",
            "legal_rate": "", "compliance_of_legal": div,
            "compliance_strict": "", "undefined": "",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--config", default="configs/suite.yaml",
                    help="sweep definition (default configs/suite.yaml)")
    ap.add_argument("--output_dir", default="results/chess")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose summary.json already exists (loads them "
                         "into the comparison table instead of re-running)")
    ap.add_argument("--monitor", action="store_true", help="publish to the public live repo")
    ap.add_argument("--monitor-interval", type=int, default=120)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    mode = cfg["check"] if args.check else cfg["full"]
    models = args.models or (mode["models"] if args.check else cfg["models"])
    tasks = args.tasks or list(cfg["tasks"])
    max_tokens = mode.get("max_new_tokens", 512)
    game_tokens = mode.get("game_max_new_tokens", max_tokens)

    cells = []
    for task in tasks:
        t = cfg["tasks"][task]
        for variant in t.get("variants", ["grid"]):
            cells.append((task, variant))

    monitor = Monitor(interval_s=args.monitor_interval,
                      output_dir=args.output_dir) if args.monitor else None
    if monitor:
        monitor.set_meta(mode="check" if args.check else "full", models=models)
        monitor.cells_total = len(models) * len(cells)
        monitor.maybe_push(force=True, last_error=None)

    print(f"suite: {len(models)} models x {len(cells)} task-variant cells "
          f"({'CHECK' if args.check else 'FULL'} mode"
          f"{', resume' if args.resume else ''})", flush=True)
    rows = []
    t0 = time.time()
    last_error = None
    skipped = 0
    for model in models:
        for task, variant in cells:
            n = cfg["tasks"][task]["check_n"] if args.check else cfg["tasks"][task]["full_n"]
            summary_path = ROOT / args.output_dir / f"{model}_{task}_{variant}.summary.json"
            if args.resume and summary_path.exists():
                summary = json.loads(summary_path.read_text())
                cell_rows = _rows_from_summary(model, task, variant, summary)
                rows.extend(cell_rows)
                skipped += 1
                print(f"  resume: {model} x {task}:{variant} already done — "
                      f"loaded {len(cell_rows)} rows", flush=True)
                if monitor:
                    monitor.cell_done(cell_rows)
                continue
            is_game = bool(cfg["tasks"][task].get("game", False))
            mt = game_tokens if is_game else max_tokens
            cmd = [
                sys.executable, str(ROOT / "scripts" / "run_chess.py"),
                "--model", model, "--task", task, "--prompt-variant", variant,
                "--n", str(n), "--max_new_tokens", str(mt),
                "--output_dir", str(ROOT / args.output_dir),
            ]
            conds = mode.get("conditions")
            if conds and not is_game:
                cmd += ["--conditions"] + conds
            if args.smoke:
                cmd.append("--smoke")
            print(f"\n>>> {model} x {task}:{variant} (n={n}, tokens={mt})", flush=True)
            if monitor:
                monitor.set_current(model, task, variant)
                monitor.maybe_push(last_error=last_error)
            t = time.time()
            res = subprocess.run(cmd, cwd=ROOT)
            cell_rows = []
            if res.returncode != 0:
                print(f"!!! {model} x {task}:{variant} FAILED rc={res.returncode}", flush=True)
                last_error = f"{model} x {task}:{variant} failed rc={res.returncode}"
            else:
                summary = json.loads(summary_path.read_text())
                cell_rows = _rows_from_summary(model, task, variant, summary)
                rows.extend(cell_rows)
            if monitor:
                monitor.cell_done(cell_rows)
                monitor.set_current()
                monitor.maybe_push(last_error=last_error)
            print(f"    {time.time() - t:.0f}s", flush=True)
    csv_path = write_comparison_csv(ROOT / args.output_dir, rows)
    print(f"\ncomparison table: {csv_path} ({len(rows)} rows, {skipped} resumed cells) "
          f"total {time.time() - t0:.0f}s", flush=True)
    if monitor:
        monitor.maybe_push(force=True, last_error=last_error)


if __name__ == "__main__":
    main()
