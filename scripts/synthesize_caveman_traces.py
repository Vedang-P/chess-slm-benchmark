"""Stage B of the caveman pipeline: deepseek explains engine lines in
caveman style, then every trace is verified before it can be trained on.

The chess content is engine-given (build_caveman_lines.py): deepseek only
writes the WORDS. But it can still distort a line (claim "wins queen"
when the line wins a pawn), so each trace is checked, not trusted:

  1. final choice (last MoveX mention) == engine_preferred candidate
  2. every UCI mentioned is a legal move from the root position
  3. every UCI mentioned appears in the engine lines we supplied
     (candidate moves + both continuations) — the trace may only explain
     the given lines, never invent a move (user-approved design 2026-08-18)
  4. length band: 60-200 tokens measured with the TRAINING tokenizer
     (gemma-4-E2B-it), so the band is the real train-time cost
  5. no hedging/filler words (caveman style is telegraphic, never hedged)

Generation is UNLIMITED (max 32768, the bench's normal budget): bounding
the budget only cut deepseek off mid-think and produced empty content,
every time it was tried — the model finishes its thinking and then writes
the answer; we wait for it. The private CoT (reasoning channel) is NOT the
training target — the trace is content only.

An example that fails ANY check is dropped. Resume-safe + append-only.

    python3 scripts/synthesize_caveman_traces.py \
        --lines results/caveman-pilot/lines.jsonl \
        --out results/caveman-pilot/traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANSWER_RE = re.compile(
    r"\bMove\s*([AB])\b\s*[:.\-]?\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)?", re.I)
UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?")
FORBIDDEN = ("maybe", "i think", "perhaps", "probably", "seems",
             "let's", "however", "therefore", "as a result", "hmm",
             "consider", "could be", "might be", "looks like")

MIN_TRACE_TOKENS = 60
MAX_TRACE_TOKENS = 200
# Effectively no budget cap for generation: any cap cut deepseek off
# mid-think with empty content (measured at 600/800/4096). The model
# finishes its thinking, then writes the caveman answer; we wait.
MAX_GEN_TOKENS = 32768

CAVEMAN_SPEC = """\
You are an expert chess player explaining why one of two candidate moves is
better. You are given the position and BOTH candidates with their engine
lines and evaluations. Explain the comparison in CAVEMAN style: short,
clipped, telegraphic thinking.

RULES (every one is mandatory):
1. One claim per line. Maximum 12 words per line.
2. Short words only. No paragraphs, no prose, no connectors
   ("therefore", "however", "as a result", "so then").
3. Never hedge. Forbidden words: maybe, perhaps, probably, seems, could
   be, might be, i think, let's, hmm, consider, looks like.
4. Never restate the position or the prompt.
5. You MUST compare BOTH candidates: at least one line on why the losing
   move fails, and one line on why the winning move works.
6. THE MOST IMPORTANT RULE: every move you mention must come EXACTLY
   from the engine lines given below — the two candidate moves and their
   listed continuations. Never invent, extend, alter, or guess a move.
   If a move is not in the lines below, do not mention it. If a fact is
   not visible in the lines below, do not claim it.
7. Every move you mention MUST be in UCI notation: start square + end
   square, e.g. "c3e4" or "g5h7" — never piece names.
8. The last line MUST be exactly one of: MoveA:<uci> or MoveB:<uci>.

STYLE SAMPLE (a good caveman trace looks exactly like this):

c3e4 trades knights.
f3g5 runs into h7h6.
h6 chases knight away.
white loses tempo.
c3d5 takes knight first.
black recaptures b7d5.
c3e4 keeps knight safe.
better for white.
MoveA:c3e4

Now do the same for the position below."""


def parse_final_choice(text: str):
    """Last MoveA/B mention wins, like run_mate_eval.parse_choice."""
    matches = list(ANSWER_RE.finditer(text))
    if matches:
        m = matches[-1]
        return m.group(1).upper(), (m.group(2) or "").lower()
    return None, None


def brief_of(row: dict) -> str:
    a = f"MoveA:{row['candidate_a']} (engine line: "
    a += " ".join(row.get("line_a") or []) + f", eval {row['eval_a']}cp)"
    b = f"MoveB:{row['candidate_b']} (engine line: "
    b += " ".join(row.get("line_b") or []) + f", eval {row['eval_b']}cp)"
    return (f"POSITION (FEN): {row['fen']}\n"
            f"Candidate A: {a}\n"
            f"Candidate B: {b}\n\n"
            "Explain why the better move is better. CAVEMAN style.")


def _lines_legal(row: dict) -> bool:
    """Each supplied candidate line must be a fully legal sequence from the
    root position (the candidate move, then the engine continuation with
    alternating sides). This validates the input data itself — an illegal
    move anywhere disqualifies the position."""
    import chess
    ok = True
    for cand, line in ((row["candidate_a"], row.get("line_a") or []),
                       (row["candidate_b"], row.get("line_b") or [])):
        board = chess.Board(row["fen"])
        for uci in [cand] + list(line):
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                ok = False
                break
            board.push(mv)
    return ok


def _verify_grounding(trace: str, row: dict) -> bool:
    """User-approved checking (2026-08-18) for comparison traces: every
    UCI mentioned must be ONE OF the moves we supplied — the two
    candidates plus their engine continuations. The trace may only
    explain the given lines, never invent a move. (Legality of the lines
    themselves is _lines_legal; an eval-stability walk was tried and
    dropped — it plays two alternative lines as one sequence and falsely
    rejects them.)"""
    moves = UCI_RE.findall(trace)
    if not moves:
        return False
    allowed = {row["candidate_a"], row["candidate_b"]}
    allowed |= set(row.get("line_a") or [])
    allowed |= set(row.get("line_b") or [])
    return all(uci in allowed for uci in moves)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True,
                    help="output of build_caveman_lines.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-v4-flash",
                    help="gateway model id. NOTE (measured 2026-08-18): the "
                         "gateway silently routes unknown ids to "
                         "deepseek-v4-flash — verify the echoed model field "
                         "before trusting a run")
    ap.add_argument("--tokenizer", default="google/gemma-4-E2B-it",
                    help="tokenizer used to measure the length band "
                         "(must be the training tokenizer)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark",
                    help="HF dataset repo for periodic trace uploads")
    ap.add_argument("--hf-path", default="caveman-traces/traces.jsonl",
                    help="path_in_repo for the shard (overwritten each "
                         "upload — the archive's canonical trace file)")
    ap.add_argument("--hf-upload-every", type=int, default=25,
                    help="re-upload the shard to HF every N rows — "
                         "checkpointing progress so a killed kernel never "
                         "strands it (AGENTS.md rule)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from src.models import make_model

    rows = [json.loads(l) for l in Path(args.lines).read_text().splitlines()
            if l.strip()]
    if args.count > 0:
        rows = rows[args.offset: args.offset + args.count]
    else:
        rows = rows[args.offset:]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["fen"])
    todo = [r for r in rows if r["fen"] not in done]
    print(f"positions: {len(rows)} (resume: {len(done)} done, "
          f"{len(todo)} to synthesize)", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = make_model(args.model)
    model.load()

    api = None
    if args.hf_upload_every > 0:
        from scripts.train_mate_grpo import _hf_api
        api = _hf_api()

    def _upload_checkpoint(rows_written: int, out: Path):
        if api is None:
            return
        try:
            api.upload_file(path_or_fileobj=str(out), path_in_repo=args.hf_path,
                            repo_id=args.hf_repo, repo_type="dataset",
                            commit_message=f"caveman traces checkpoint ({rows_written} rows)")
            print(f"hf: checkpoint upload after {rows_written} rows -> "
                  f"{args.hf_repo}/{args.hf_path}", flush=True)
        except Exception as e:
            print(f"hf: checkpoint upload failed (will retry later): {e}",
                  flush=True)

    ok = fail = api_err = 0
    written = 0
    with out.open("a") as fout:
        for i, row in enumerate(todo):
            prompt = CAVEMAN_SPEC + "\n\n" + brief_of(row)
            # STREAMING is mandatory: the non-stream path cuts the model off
            # mid-think and returns empty content on positions that need
            # >~12k reasoning tokens (measured). Streaming waits for the
            # real finish_reason and delivers reasoning + content in full.
            res = model.generate(prompt, max_new_tokens=MAX_GEN_TOKENS,
                                 stream=True)
            if res.get("error"):
                api_err += 1
                print(f"  [{i}] API error: {res['error'][:90]}", flush=True)
                time.sleep(10)
                continue
            trace = (res.get("content") or "").strip()
            label, uci = parse_final_choice(trace)
            checks = {
                "choice": (label is not None and label == row["engine_preferred"]
                           and uci == row[f"candidate_{label.lower()}"].lower()),
                "legal": True,
                "grounded": True,
                "length": False,
                "words": True,
                "empty": not trace,
            }
            n_tokens = len(tok.encode(trace))
            checks["length"] = MIN_TRACE_TOKENS <= n_tokens <= MAX_TRACE_TOKENS
            low = trace.lower()
            checks["words"] = not any(w in low for w in FORBIDDEN)
            if checks["choice"] and not checks["empty"]:
                checks["legal"] = _lines_legal(row)
                checks["grounded"] = _verify_grounding(trace, row)
            verified = all(checks.values())
            rec = {**row, "trace": trace, "choice_label": label,
                   "choice_uci": uci, "n_tokens": n_tokens,
                   "verified": verified, "checks": checks,
                   # the FULL deepseek thinking is archived with the answer
                   # (user-requested: useful for later analysis), plus the
                   # honest finish/usage evidence for yield auditing
                   "reasoning": res.get("reasoning") or "",
                   "reasoning_chars": len(res.get("reasoning") or ""),
                   "finished": res.get("finished"),
                   "attempts": res.get("attempts"),
                   "usage": res.get("token_usage")}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            written += 1
            ok += verified
            fail += (not verified)
            if (i + 1) % 10 == 0 or not verified:
                print(f"  [{i + 1}/{len(todo)}] verified={ok} dropped={fail} "
                      f"api_err={api_err} | {row['fen']} tokens={n_tokens} "
                      f"checks={checks}", flush=True)
            if args.hf_upload_every > 0 and written % args.hf_upload_every == 0:
                _upload_checkpoint(written, out)
    if api is not None:
        _upload_checkpoint(written, out)
    print(f"shard done: verified={ok} dropped={fail} api_err={api_err} "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
