"""Generate verified lucid traces on noexplain positions (resume-safe).

The MATE expert answer is the oracle — we always trust it (eval scores
against it). No Stockfish anywhere in this pipeline.

  Stage A (--stage select):  sample K noexplain train positions
        (phase-natural, test-FEN-excluded). Writes {out}/selected.jsonl.
  Stage B (--stage traces):  for each selected position, deepseek-v4-flash
        writes a compressed lucid trace + final choice. Verified iff the
        trace's FINAL choice (last MoveA/B mention) matches the MATE expert
        truth and every mentioned move is legal. Append-only + resume-safe.
  Stage C (--stage merge):   merge verified traces into the label corpus
        (assistant = lucid trace + MoveX answer), tagged "Verified: yes".

    python3 scripts/build_lucid_traces.py --stage select --k 3000 \
        --out data/positions/noexplain-slice/traces
    python3 scripts/build_lucid_traces.py --stage traces --offset 0 \
        --count 1000 --out data/positions/noexplain-slice/traces
    python3 scripts/build_lucid_traces.py --stage merge \
        --out data/positions/noexplain-slice/traces

Stage B is sharded: one kernel per (--offset, --count) slice, one API key
each; 3 kernels x 1000 positions finishes ~3k in one overnight run.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LUCID_INSTRUCTION = (
    "You are an expert chess player analyzing a position to choose between "
    "two candidate moves. Think in a compressed, telegraphic style: short "
    "fragments, no prose. State only the key tactical/positional facts, the "
    "line you trust, and your choice. Finish with exactly one of:\n"
    "MoveA:<move> or MoveB:<move>\n"
    "Only output the line for your choice, nothing else after it."
)
MAX_TRACE_TOKENS = 4096
ANSWER_RE = re.compile(
    r"\bMove\s*([AB])\b\s*[:.\-]?\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)?", re.I)
UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?")


def fen_of(text: str) -> str | None:
    m = re.search(r'"([^"]+)"', text)
    return m.group(1) if m else None


def parse_final_choice(text: str):
    """Last MoveA/B mention wins, like run_mate_eval.parse_choice."""
    matches = list(ANSWER_RE.finditer(text))
    if matches:
        m = matches[-1]
        return m.group(1).upper(), (m.group(2) or "").lower()
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "traces", "merge"])
    ap.add_argument("--k", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT / "data/positions/noexplain-slice/traces"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.stage == "select":
        rng = random.Random(args.seed)
        test_fens = set()
        for name in ("mate-selection-test.json", "mate-selection-test-noexplain.json",
                     "mate-selection-test-tactic.json", "mate-selection-test-both.json"):
            p = ROOT / "data/positions" / name
            if p.exists():
                for r in json.loads(p.read_text()):
                    test_fens.add(r["fen"])
        pool = []
        seen = set()
        for path in sorted((ROOT / "data/raw/mate-train/noexplain").glob("*/*.jsonl")):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    fen = fen_of(d.get("input", ""))
                    if not fen or fen in test_fens or fen in seen:
                        continue
                    seen.add(fen)
                    m = re.search(r"Move([AB]):\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)",
                                  d.get("output", ""))
                    if not m:
                        continue
                    pool.append({"fen": fen, "input": d.get("input", ""),
                                 "truth": m.group(2).lower(), "truth_label": m.group(1)})
                    if len(pool) >= args.k:
                        break
            if len(pool) >= args.k:
                break
        rng.shuffle(pool)
        sel = pool[: args.k]
        (out / "selected.jsonl").write_text(
            "\n".join(json.dumps(r) for r in sel) + "\n")
        print(f"selected {len(sel)} -> {out}/selected.jsonl", flush=True)

    elif args.stage == "traces":
        from src.models import make_model
        sel = [json.loads(l) for l in
               (out / "selected.jsonl").read_text().splitlines() if l.strip()]
        if args.count > 0:
            sel = sel[args.offset: args.offset + args.count]
        else:
            sel = sel[args.offset:]
        done_path = out / f"traces-{args.offset}.jsonl"
        done = set()
        if done_path.exists():
            for line in done_path.read_text().splitlines():
                if line.strip():
                    done.add(json.loads(line)["fen"])
        todo = [s for s in sel if s["fen"] not in done]
        print(f"shard offset={args.offset} count={args.count}: "
              f"{len(todo)} remaining", flush=True)
        if not todo:
            return
        model = make_model("deepseek-v4-flash")
        ok = fail = api_err = 0
        with open(done_path, "a") as fout:
            for i, s in enumerate(todo):
                prompt = (LUCID_INSTRUCTION + "\n\nThe FEN of the given chess "
                          f"board is \"{s['fen']}\". Which move is better? "
                          f"MoveA:{s.get('move_a') or _cand(s,'A')} "
                          f"MoveB:{s.get('move_b') or _cand(s,'B')} ")
                # NOTE: candidates come from the input string; build prompt
                # directly from the stored input to stay byte-faithful.
                prompt = (LUCID_INSTRUCTION + "\n\n" + s["input"])
                res = model.generate(prompt, max_new_tokens=MAX_TRACE_TOKENS)
                if res.get("error"):
                    api_err += 1
                    print(f"  [{i}] API error: {res['error'][:90]}", flush=True)
                    time.sleep(10)
                    continue
                content = res.get("content") or ""
                reasoning = res.get("reasoning") or ""
                trace = (reasoning + "\n" + content).strip()
                label, choice_uci = parse_final_choice(content)
                # trust the expert: final choice must equal MATE truth
                choice_ok = (label is not None and
                             choice_uci == s["truth"])
                # all mentioned moves legal on the position
                board = chess.Board(s["fen"])
                legal = True
                for uci in set(UCI_RE.findall(trace)):
                    try:
                        mv = chess.Move.from_uci(uci)
                    except ValueError:
                        continue
                    if not board.is_legal(mv):
                        legal = False
                        break
                grounded = choice_ok and legal
                rec = {"fen": s["fen"], "input": s["input"], "trace": trace,
                       "choice_label": label, "choice_uci": choice_uci,
                       "truth": s["truth"], "verified": grounded,
                       "legal": legal,
                       "reasoning_tokens": res.get("reasoning_tokens")}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                ok += grounded
                fail += (not grounded)
                if (i + 1) % 25 == 0:
                    print(f"  [{i + 1}/{len(todo)}] grounded={ok} dropped={fail} "
                          f"api_err={api_err}", flush=True)
        print(f"shard done: grounded={ok} dropped={fail} api_err={api_err} "
              f"-> {done_path}", flush=True)

    elif args.stage == "merge":
        traces = []
        for p in sorted(out.glob("traces-*.jsonl")):
            for line in p.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    if d.get("verified"):
                        traces.append(d)
        print(f"verified traces: {len(traces)}", flush=True)
        label_path = ROOT / "data/positions/noexplain-slice/train.jsonl"
        merged = ROOT / "data/positions/noexplain-slice/train_with_traces.jsonl"
        trace_by_fen = {}
        for t in traces:
            trace_by_fen.setdefault(t["fen"], []).append(t)
        n_written = 0
        with open(merged, "w") as fout:
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                tlist = trace_by_fen.get(rec.get("fen"), [])
                if tlist:
                    for t in tlist:
                        rec2 = dict(rec)
                        rec2["messages"] = [
                            rec["messages"][0],
                            {"role": "assistant",
                             "content": t["trace"].rstrip() + "\n\n" +
                                        rec["messages"][1]["content"]},
                        ]
                        rec2["verified"] = True
                        fout.write(json.dumps(rec2) + "\n")
                        n_written += 1
                else:
                    fout.write(line + "\n")
        print(f"merged {n_written} trace rows into {merged}", flush=True)


def _cand(s: dict, label: str) -> str:
    m = re.search(rf"Move{label}:\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)", s.get("input", ""))
    return m.group(1) if m else ""


if __name__ == "__main__":
    main()
