"""Analyze base gemma's MATE eval samples: how much of the ~40% error is
illegal moves vs legal-but-wrong candidates.

Reads the raw per-sample records from the HF archive
(vedangfake/chess-bench-results, runs/gemma1000-w*/...strategy.samples.jsonl),
checks each recorded move for LEGALITY on the position's FEN with
python-chess, and classifies every answer:

  correct          -> move matches MATE expert truth
  wrong_illegal    -> wrong AND the model's claimed move is ILLEGAL on the FEN
  wrong_legal_other-> wrong AND legal, but not one of the two candidates
  wrong_legal_cand -> wrong AND legal AND one of the candidates (chose bad candidate)

Outputs a JSON report + a per-sample annotated JSONL. Run on a Kaggle CPU
kernel (fast; ~1000 rows), results uploaded to HF + repo.

    python3 scripts/analyze_gemma_legality.py [--workers w1,w2]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_REPO = "vedangfake/chess-bench-results"
SAMPLE_GLOB = "runs/gemma1000-{w}/gemma4-e2b_mate-selection-test_strategy.samples.jsonl"
UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?")


def resolve_token() -> str:
    for name in ("HF_WRITE_TOKEN", "HF_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_WRITE_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("no HF token")


def load_samples(workers: list[str]) -> list[dict]:
    from huggingface_hub import hf_hub_download

    token = resolve_token()
    rows = []
    for w in workers:
        path = hf_hub_download(repo_id=HF_REPO,
                               filename=SAMPLE_GLOB.format(w=w),
                               repo_type="dataset", token=token)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def candidates_from_input(inp: str) -> tuple[str, str]:
    ucis = UCI_RE.findall(inp)
    return (ucis[0], ucis[1]) if len(ucis) >= 2 else ("", "")


def classify(rows: list[dict]) -> tuple[list[dict], dict]:
    out = []
    stats = Counter()
    for r in rows:
        fen = (r.get("position_metadata") or {}).get("fen", "")
        move = (r.get("move") or "").lower()
        status = r.get("status")
        truth = r.get("correct") or {}
        truth_move = str(truth.get("move", "")).lower()
        ca, cb = candidates_from_input(r.get("model_input") or r.get("prompt") or "")
        rec = {"position_id": r.get("position_id"), "fen": fen,
               "move": move, "status": status,
               "truth": truth_move, "ca": ca, "cb": cb,
               "run_id": r.get("run_id")}

        if status == "api_error":
            rec["bucket"] = "api_error"
            stats["api_error"] += 1
            out.append(rec)
            continue
        if status == "no_answer":
            rec["bucket"] = "no_answer"
            stats["no_answer"] += 1
            out.append(rec)
            continue
        if status == "parse_error":
            rec["bucket"] = "parse_error"
            stats["parse_error"] += 1
            out.append(rec)
            continue

        # status in ("correct", "wrong") -> we have a move
        board = chess.Board(fen) if fen else None
        legal = False
        if board is not None and move:
            try:
                legal = board.is_legal(chess.Move.from_uci(move))
            except ValueError:
                legal = False
        rec["legal"] = legal

        if status == "correct":
            rec["bucket"] = "correct"
            stats["correct"] += 1
        else:  # wrong
            if not legal:
                rec["bucket"] = "wrong_illegal"
            elif move not in (ca.lower(), cb.lower()):
                rec["bucket"] = "wrong_legal_other"
            else:
                rec["bucket"] = "wrong_legal_cand"
            stats[rec["bucket"]] += 1
        out.append(rec)
    return out, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="w1,w2")
    ap.add_argument("--out", default=str(ROOT / "results/gemma-legality"))
    ap.add_argument("--upload", action="store_true",
                    help="upload report to HF dataset repo")
    args = ap.parse_args()

    rows = load_samples([w.strip() for w in args.workers.split(",")])
    print(f"loaded {len(rows)} samples", flush=True)

    annotated, stats = classify(rows)
    n = len(rows)
    wrong = (stats.get("wrong_illegal", 0) + stats.get("wrong_legal_other", 0)
             + stats.get("wrong_legal_cand", 0))
    illegal = stats.get("wrong_illegal", 0)
    legal_wrong = wrong - illegal
    legal_total = stats.get("correct", 0) + legal_wrong

    report = {
        "source": "HF archive runs/gemma1000-w1,w2 (strategy, thinking ON, forced prompt)",
        "n": n,
        "stats": stats,
        "accuracy": round(stats.get("correct", 0) / n, 4) if n else None,
        "wrong_total": wrong,
        "wrong_illegal": illegal,
        "wrong_illegal_pct_of_wrong": round(illegal / wrong, 4) if wrong else None,
        "wrong_legal_pct_of_wrong": round(legal_wrong / wrong, 4) if wrong else None,
        "legal_move_rate_overall": round(legal_total / n, 4) if n else None,
        "legal_move_rate_of_answers": round(
            legal_total / (n - stats.get("api_error", 0)), 4) if n else None,
        "note": ("wrong_illegal = model's claimed move is ILLEGAL on the FEN "
                 "(not a chess move); wrong_legal_cand = legal but the wrong "
                 "of the two candidates; wrong_legal_other = legal but not "
                 "one of the candidates"),
    }
    print(json.dumps(report, indent=1), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "legality_report.json").write_text(json.dumps(report, indent=1))
    with open(out / "annotated_samples.jsonl", "w") as f:
        for rec in annotated:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {out}/legality_report.json + annotated_samples.jsonl",
          flush=True)

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=resolve_token())
        for fname in ("legality_report.json", "annotated_samples.jsonl"):
            api.upload_file(path_or_fileobj=str(out / fname),
                            path_in_repo=f"gemma-legality/{fname}",
                            repo_id=HF_REPO, repo_type="dataset",
                            commit_message=f"gemma legality {fname}")
        print("uploaded to HF", flush=True)


if __name__ == "__main__":
    main()
