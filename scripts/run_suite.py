"""Run the model x task x prompt-variant matrix and produce the comparison table.

With --monitor, the suite publishes progress to the repo's `live` branch
(monitor/state.json + monitor/history.jsonl) so the dashboard can render it.
Monitoring pushes never break the sweep: every git step is failure-tolerant.

Usage:
    python scripts/run_suite.py                # full sweep (paper data)
    python scripts/run_suite.py --check        # tiny sanity sweep
    python scripts/run_suite.py --smoke        # stub models, no GPU
    python scripts/run_suite.py --monitor      # publish progress to 'live' branch
    python scripts/run_suite.py --models deepseek-v4-flash
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.report import write_comparison_csv  # noqa: E402

SCHEMA = 3  # must match src/report.py ResultWriter.finish; bump both when
            # scorer/parser/loader changes so stale summaries re-run

ROOT = Path(__file__).resolve().parent.parent
MONITOR_BRANCH = "live"
MONITOR_DIR = ROOT / "monitor"
HISTORY_CAP = 500
PUBLIC_LIVE_REPO = "Vedang-P/chess-bench-live"  # public repo: the dashboard reads from here
PUBLIC_LIVE_BRANCH = "main"


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Monitor:
    def __init__(self, interval_s: int = 120, output_dir: str = "results/chess",
                 remote_prefix: str = "results/chess"):
        self.interval = interval_s
        self.last_push = 0.0
        self.cells_done = 0
        self.cells_failed = 0
        self.cells_total = 0
        self.started_at = _ts()
        self.started_epoch = time.time()
        self.rows = []
        self.meta = {}
        self.current = None
        self.output_dir = Path(output_dir)
        self.remote_prefix = remote_prefix  # results/chess (full) or results_check/chess (check)
        self.uploaded = set()  # remote paths already uploaded (tracked in-memory)
        self.hf_uploaded = set()  # cell names already persisted to HF

    def _push_hf(self) -> None:
        """Persist every completed cell (report.json with the full thinking
        chains + samples + summary) to the free HF dataset repo. Runs on
        every tick; cells already uploaded are skipped."""
        try:
            from src.hf_push import resolve_hf_token, upload_cell

            if not resolve_hf_token():
                return  # no token configured — local runs still work
            for summary_path in sorted(self.output_dir.glob("*.summary.json")):
                cell = summary_path.stem
                if cell in self.hf_uploaded:
                    continue
                upload_cell(self.output_dir, self.started_at, summary_path)
                self.hf_uploaded.add(cell)
        except Exception as e:
            print(f"hf: push failed ({type(e).__name__}: {e})", flush=True)

    def set_meta(self, **kw) -> None:
        self.meta.update(kw)

    def cell_done(self, row_parts: list) -> None:
        if not row_parts:
            return
        self.cells_done += 1
        for r in row_parts:
            self.rows.append(r)

    def cell_failed(self) -> None:
        self.cells_failed += 1

    def set_current(self, model: str = None, task: str = None, variant: str = None) -> None:
        self.current = (model, task, variant) if model else None

    def maybe_push(self, force: bool = False, last_error: str = None) -> None:
        if not force and time.time() - self.last_push < self.interval:
            return
        self._write_state(last_error)
        self._push()
        self._push_hf()
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
                    game = r.get("game")
                    if not isinstance(game, dict):
                        game = {k: v for k, v in r.items()
                                if k not in ("model", "task", "variant", "condition", "game")}
                    cell["game"] = game
                    cell["n"] = game.get("n", r.get("n"))
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
        elapsed_s = time.time() - self.started_epoch
        eta_min = None
        if done > 0 and elapsed_s > 0:
            eta_min = int(elapsed_s / done * (total - done) / 60)
        return {
            "repo": "Vedang-P/chess-slm-benchmark",
            "mode": self.meta.get("mode"),
            "run_id": self.started_at,
            "stage": "complete" if total > 0 and done + self.cells_failed >= total else "sweep",
            "started_at": self.started_at,
            "updated_at": _ts(),
            "progress": {"cells_done": done, "cells_failed": self.cells_failed,
                         "cells_attempted": done + self.cells_failed,
                         "cells_total": total,
                         "fraction": round(done / total, 4) if total else 0.0},
            "eta_min": eta_min,
            "models": self.meta.get("models", []),
            "cot_requested": bool(self.meta.get("cot", False)),
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
            "run_id": state["run_id"],
            "ts": state["updated_at"], "cells_done": state["progress"]["cells_done"],
            "fraction": state["progress"]["fraction"], "eta_min": state["eta_min"],
            "legal_avg": _avg_legal(state["cells"]),
            "last_error": last_error,
        }))
        lines = lines[-HISTORY_CAP:]
        hist.write_text("\n".join(lines) + "\n")

    def _log_push_error(self, reason: str) -> None:
        """Persist API failure reasons so they can be inspected after the run
        (the notebook zips monitor/, which includes this log)."""
        try:
            MONITOR_DIR.mkdir(exist_ok=True)
            with open(MONITOR_DIR / "push_errors.log", "a") as f:
                f.write(f"{_ts()} {reason}\n")
        except Exception:
            pass

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
                reason = f"{type(e).__name__}: {e}"
                print(f"monitor: contents-API push failed ({reason}) — retrying once", flush=True)
                self._log_push_error(reason)
                time.sleep(2)
                try:
                    self._push_via_api(token)
                    return
                except Exception as e2:
                    reason2 = f"{type(e2).__name__}: {e2}"
                    print(f"monitor: contents-API retry failed ({reason2}) — git fallback", flush=True)
                    self._log_push_error(reason2)
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
                remote = f"{self.remote_prefix}/{summary.name}"
                if remote not in self.uploaded:
                    uploads.append((remote, summary))
                    # samples JSONL travels with its summary so recovery can
                    # restore raw outputs, not just aggregates
                    samples = summary.with_name(summary.name.replace(".summary.json", ".samples.jsonl"))
                    if samples.exists():
                        uploads.append((f"{self.remote_prefix}/{samples.name}", samples))
            csv_path = self.output_dir / "comparison_table.csv"
            if csv_path.exists():
                uploads.append((f"{self.remote_prefix.rsplit('/', 1)[0]}/comparison_table.csv", csv_path))
            # index of all completed summaries (for the recovery flow)
            index = {
                "files": sorted(p.name for p in self.output_dir.glob("*.summary.json")),
                "updated_at": _ts(),
            }
            uploads.append((f"{self.remote_prefix.rsplit('/', 1)[0]}/index.json",
                            _BytesFile(json.dumps(index, indent=1).encode())))
        for remote, local in uploads:
            data = local.data if isinstance(local, _BytesFile) else local.read_bytes()
            err = upload_file(token, remote, data)
            if err:
                print(f"monitor: upload failed for {remote}: {err}", flush=True)
                self._log_push_error(f"{remote}: {err}")
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
            "token_usage": metrics.get("token_usage", {}),
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
            "token_usage": metrics.get("token_usage", {}),
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
    ap.add_argument("--n", type=int, default=None,
                    help="override per-cell sample count (demos: small n, full run: unset)")
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
    ap.add_argument("--monitor-interval", type=int, default=60)
    ap.add_argument("--verbose", action="store_true",
                    help="forward to run_chess: print the exact prompt per sample")
    ap.add_argument("--live-push", action="store_true",
                    help="forward to run_chess: push live.json on every snapshot "
                         "(minimal website lag for local runs)")
    ap.add_argument("--stream", action="store_true",
                    help="forward to run_chess: stream model output live")
    ap.add_argument("--cot", action="store_true",
                    help="forward to run_chess: add 'think step by step' to prompts")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    mode = cfg["check"] if args.check else cfg["full"]
    models = args.models or cfg["models"]
    tasks = args.tasks or list(cfg["tasks"])
    max_tokens = mode.get("max_new_tokens", 512)
    game_tokens = mode.get("game_max_new_tokens", max_tokens)

    cells = []
    for task in tasks:
        t = cfg["tasks"][task]
        for variant in t.get("variants", ["grid"]):
            cells.append((task, variant))

    monitor = Monitor(interval_s=args.monitor_interval,
                      output_dir=args.output_dir,
                      remote_prefix=args.output_dir) if args.monitor else None
    if monitor:
        # push on a timer, not just between cells: with thinking models a
        # single cell can run 10+ minutes and the website must keep
        # showing live reasoning during it
        import threading

        def _pusher() -> None:
            while True:
                time.sleep(max(monitor.interval, 10))
                try:
                    monitor.maybe_push(last_error=None)
                except Exception:
                    pass

        threading.Thread(target=_pusher, daemon=True).start()
    if monitor:
        monitor.set_meta(mode="check" if args.check else "full", models=models,
                         cot=args.cot)
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
            n = (args.n or (cfg["tasks"][task]["check_n"] if args.check
                            else cfg["tasks"][task]["full_n"]))
            summary_path = ROOT / args.output_dir / f"{model}_{task}_{variant}.summary.json"
            if args.resume and summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if (summary.get("schema") != SCHEMA
                        or summary.get("n_samples", 0) < n):
                    print(f"  resume: {model} x {task}:{variant} schema v{summary.get('schema')} "
                          f"n={summary.get('n_samples')} < n={n} (stale/partial) — re-running",
                          flush=True)
                else:
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
            if args.verbose:
                cmd.append("--verbose")
            if args.stream:
                cmd.append("--stream")
            if args.cot:
                cmd.append("--cot")
            if args.smoke:
                cmd.append("--smoke")
            if args.live_push:
                cmd.append("--live-push")
            print(f"\n>>> {model} x {task}:{variant} (n={n}, tokens={mt})", flush=True)
            if monitor:
                monitor.set_current(model, task, variant)
                monitor.maybe_push(last_error=last_error)
            t = time.time()
            child_env = os.environ.copy()
            child_env["BENCH_RUN_ID"] = monitor.started_at if monitor else _ts()
            res = subprocess.run(cmd, cwd=ROOT, env=child_env)
            cell_rows = []
            if res.returncode != 0:
                print(f"!!! {model} x {task}:{variant} FAILED rc={res.returncode}", flush=True)
                last_error = f"{model} x {task}:{variant} failed rc={res.returncode}"
                if monitor:
                    monitor.cell_failed()
            else:
                summary = json.loads(summary_path.read_text())
                cell_rows = _rows_from_summary(model, task, variant, summary)
                rows.extend(cell_rows)
            if monitor:
                if cell_rows:
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
