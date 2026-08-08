"""Build the MATE+C1-style fine-tuning dataset (Arm B of the campaign).

The Arm A baseline is the vanilla MATE recipe: 50k expert-annotated
move-selection records, one packed epoch, 4-bit QLoRA on Gemma 4 E2B. This
script builds the Arm B dataset that replaces/augments it following the
Master Distillation pipeline (Tang et al., arXiv:2603.20510) and the
Best-Line result of Dionisopoulos et al. (ICML 2026, arXiv:2604.05134):

    1. pool      — MATE *train* positions (never the held-out testset),
                   FEN-deduped, testset FENs excluded, stratified per
                   subset (noexplain optionally weighted).
    2. engine    — local Stockfish pass at --depth: best move + PV + eval
                   per position; `agree` = engine best move == the
                   expert-annotated truth (the engine/expert agreement
                   filter; disagreement is measured and the position is
                   EXCLUDED from training so the teacher never rationalizes
                   a wrong solution).
    3. teacher   — DeepSeek V4 Flash (opencode-go gateway) verbalizes the
                   engine line into a 4-10 sentence chain-of-thought via
                   Feigned Discovery prompting (teacher pretends not to
                   know the solution); outputs are validated against the
                   engine move and failed samples dropped.
    4. assemble  — emits the student-format training set in the EXACT MATE
                   noexplain format (FEN + MoveA/MoveB candidates) with the
                   verbalized CoT prepended to the expert answer: the model
                   must reproduce the reasoning the noexplain format lacks.
                   Token counts are reported so Arm B can be matched to
                   Arm A's token budget.

Decisions locked 2026-08-08 (session): pilot = 150 positions (stop if
engine-expert agreement > 95%); subset = noexplain only; teacher = free
tier (deepseek-v4-flash-free); training format = MATE only; full run =
10k positions on Kaggle CPU kernels (vedangpandeyy account).

Every stage is resumable and idempotent: it writes its own file under
data/raw/mate-c1/ and skips existing work unless --force. The teacher
stage is the only paid/rate-limited one; the pilot should run it with a
few hundred positions per subset and measure agreement rate, tokens per
sample and cost before scaling.

Outputs (gitignored, data/raw/):
    data/raw/mate-c1/pool.jsonl       parsed MATE train positions
    data/raw/mate-c1/engine.jsonl     pool + Stockfish info + agree flag
    data/raw/mate-c1/teacher.jsonl    agreeing positions + CoT + validation
    data/raw/mate-c1/train.jsonl      final student-format training set
    data/raw/mate-c1/stats.json       agreement rates, token budget, cost
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "mate-c1"
TRAIN_DIR = ROOT / "data" / "raw" / "mate-train"
TEST_FILES = [
    "data/positions/mate-selection-test.json",
    "data/positions/mate-selection-test-noexplain.json",
    "data/positions/mate-selection-test-tactic.json",
    "data/positions/mate-selection-test-both.json",
]
SEED = 2026

HF_BASE = "https://huggingface.co/datasets/OutFlankShu/MATE_DATASET/resolve/main"
TRAIN_ZIPS = {
    "strategy": "strategy.zip",
    "tactic": "tactic.zip",
    "noexplain": "no_explain.zip",
    "both": "both.zip",
}

FEN_RE = re.compile(r'FEN of the given chess board is "([^"]+)"')
MOVE_RE = re.compile(r"Move([AB]):([a-h][1-8][a-h][1-8](?:[qrbnQRBN])?)")

# --------------------------------------------------------------------------
# Stage 1: download + parse the MATE TRAIN split (held-out testset excluded)
# --------------------------------------------------------------------------

def _download_zip(url: str, dest: Path) -> None:
    print(f"downloading {url}", flush=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "chess-bench"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    shutil.move(str(tmp), dest)


def download(args: argparse.Namespace) -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    for subset, zip_name in TRAIN_ZIPS.items():
        dest = TRAIN_DIR / zip_name
        if dest.exists() and not args.force:
            print(f"skip {zip_name} (exists)", flush=True)
            continue
        _download_zip(f"{HF_BASE}/{zip_name}", dest)
        with zipfile.ZipFile(dest) as z:
            z.extractall(TRAIN_DIR / subset)
        print(f"{subset}: extracted to {TRAIN_DIR / subset}", flush=True)


def _parse_train_record(line: str, idx: int, subset: str) -> dict | None:
    d = json.loads(line)
    fen_m = FEN_RE.search(d.get("input") or "")
    if not fen_m:
        return None
    moves = MOVE_RE.findall(d.get("input") or "")
    cand = {label: uci for label, uci in moves}
    if "A" not in cand or "B" not in cand:
        return None
    truth_m = re.match(r"Move([AB]):", d.get("output") or "")
    if not truth_m:
        return None
    return {
        "id": f"mate-c1-{subset}-{idx:06d}",
        "subset": subset,
        "fen": fen_m.group(1),
        "candidate_a": cand["A"],
        "candidate_b": cand["B"],
        "truth_label": truth_m.group(1),
        "truth_move": cand[truth_m.group(1)],
        "mate_instruction": d.get("instruction") or "",
        "mate_input": d.get("input") or "",
        "mate_output": d.get("output") or "",
    }


def _testset_fens() -> set[str]:
    fens = set()
    for name in TEST_FILES:
        path = ROOT / name
        if not path.exists():
            continue
        for rec in json.loads(path.read_text()):
            if rec.get("fen"):
                fens.add(rec["fen"])
    return fens


def pool(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pool.jsonl"
    if out.exists() and not args.force:
        print("skip pool (exists)", flush=True)
        return
    excluded = _testset_fens()
    print(f"testset FENs to exclude: {len(excluded)}", flush=True)
    rng = random.Random(SEED)
    records: list[dict] = []
    # noexplain first: the same position appears in several MATE subsets,
    # and the noexplain variant is our special target, so it wins dedupe.
    for subset in sorted(args.subsets, key=lambda s: s != "noexplain"):
        src_dir = TRAIN_DIR / subset
        jsonl = sorted(p for p in src_dir.rglob("*.jsonl")
                       if not p.name.startswith("._"))
        if not jsonl:
            print(f"WARNING: no train jsonl for {subset} in {src_dir} "
                  f"(run `download` first)", flush=True)
            continue
        seen: set[str] = set()
        subset_records = []
        for path in jsonl:
            with path.open() as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    rec = _parse_train_record(line, idx, subset)
                    if rec is None:
                        continue
                    if rec["fen"] in seen or rec["fen"] in excluded:
                        continue
                    seen.add(rec["fen"])
                    subset_records.append(rec)
        rng.shuffle(subset_records)
        n = int(args.n * (args.noexplain_weight
                          if subset == "noexplain" else 1))
        picked = subset_records[:n]
        records.extend(picked)
        print(f"{subset}: {len(subset_records)} unique non-test positions, "
              f"kept {len(picked)}", flush=True)
    rng.shuffle(records)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(records)} pool records -> {out}", flush=True)

# --------------------------------------------------------------------------
# Stage 2: local Stockfish pass (best move, PV, eval, agreement filter)
# --------------------------------------------------------------------------

def _stockfish_info(depth: int, movetime: int) -> dict:
    sf = shutil.which("stockfish")
    if not sf:
        raise RuntimeError("stockfish binary not found on PATH")
    import chess.engine
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    return {"eng": eng, "depth": depth, "movetime": movetime}


def _analyze(info: dict, fen: str) -> dict | None:
    """One-shot analyze via python-chess's UCI wrapper. A hand-rolled UCI
    reader blocked forever here: `ucinewgame`/`position` produce no reply,
    so readline() sat on the pipe while the deadline could never fire. The
    library reads in a background thread and bounds work per call."""
    import chess
    import chess.engine
    board = chess.Board(fen)
    limit = chess.engine.Limit(depth=info["depth"], time=info["movetime"] / 1000)
    try:
        result = info["eng"].analyse(board, limit=limit)
    except Exception:
        return None
    if "pv" not in result or not result["pv"]:
        return None
    score = result.get("score")
    return {
        "bestmove": result["pv"][0].uci(),
        "pv": [m.uci() for m in result["pv"]],
        "score_type": (score.pov(board.turn).is_mate() and "mate" or "cp")
                      if score is not None else "",
        "score": (abs(score.pov(board.turn).mate()) if score is not None
                  and score.pov(board.turn).is_mate()
                  else score.pov(board.turn).score() if score is not None
                  else None),
    }


def engine(args: argparse.Namespace) -> None:
    pool_path = OUT_DIR / "pool.jsonl"
    out = OUT_DIR / "engine.jsonl"
    if out.exists() and not args.force:
        print("skip engine (exists)", flush=True)
        return
    if not pool_path.exists():
        print("no pool.jsonl (run `pool` first)", flush=True)
        sys.exit(1)
    info = _stockfish_info(args.depth, args.movetime)
    n_ok = n_agree = n_seen = 0
    with pool_path.open() as fin, out.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            eng = _analyze(info, rec["fen"])
            if eng is None:
                rec["engine_error"] = True
            else:
                rec["engine_error"] = False
                rec["engine_bestmove"] = eng["bestmove"]
                rec["engine_pv"] = eng["pv"]
                rec["engine_score_type"] = eng["score_type"]
                rec["engine_score"] = eng["score"]
                n_ok += 1
                rec["agree"] = eng["bestmove"] == rec["truth_move"]
                if rec["agree"]:
                    n_agree += 1
            fout.write(json.dumps(rec) + "\n")
            n_seen += 1
            if n_seen % 25 == 0:
                print(f"engine: {n_seen} analyzed, {n_agree} agree so far "
                      f"({100.0 * n_agree / max(n_ok, 1):.1f}%)", flush=True)
    info["eng"].quit()
    print(f"engine pass done: {n_ok} analyzed, {n_agree} agree "
          f"({100.0 * n_agree / max(n_ok, 1):.1f}%) -> {out}", flush=True)


# --------------------------------------------------------------------------
# Stage 3: teacher verbalization (opencode-go gateway, DeepSeek V4 Flash)
# --------------------------------------------------------------------------

TEACHER_SYSTEM = (
    "You are a chess grandmaster generating training data for teaching small "
    "LLMs to solve chess positions. CRITICAL: The small model only sees the "
    "FEN string. Your analysis must show how to go from FEN -> piece "
    "positions -> tactical relationships -> solution. Ground everything in "
    "explicit square names."
)


def _pieces_text(fen: str) -> str:
    import chess
    board = chess.Board(fen)
    w = [f"{p.symbol().upper()}{chess.square_name(s)}"
         for s in chess.SQUARES
         if (p := board.piece_at(s)) and p.color == chess.WHITE]
    b = [f"{p.symbol().upper()}{chess.square_name(s)}"
         for s in chess.SQUARES
         if (p := board.piece_at(s)) and p.color == chess.BLACK]
    return (f"White: {', '.join(w) or 'none'}\n"
            f"Black: {', '.join(b) or 'none'}")


def _legal_moves_text(fen: str) -> str:
    import chess
    board = chess.Board(fen)
    return ", ".join(sorted(board.uci(m) for m in board.legal_moves))


def _teacher_prompt(rec: dict) -> str:
    pieces = _pieces_text(rec["fen"])
    legal = _legal_moves_text(rec["fen"])
    pv = " ".join(rec.get("engine_pv") or [])
    solution = rec["engine_bestmove"]
    return "\n".join([
        TEACHER_SYSTEM,
        "",
        "Context:",
        f"FEN: {rec['fen']}",
        f"Pieces: {pieces}",
        f"Side to Move: {rec['fen'].split()[1]}",
        f"Solution: {solution}",
        f"PVs: {pv}",
        f"Legal Moves: {legal}",
        "",
        "TASK: Write natural chain-of-thought analysis arriving at "
        f"{solution}. Your analysis should cover (in natural prose, vary "
        "the structure):",
        "* Where key pieces are (use square names: 'the queen on h5', "
        "'king on g1')",
        "* What tactical relationship exists (attacks, pins, weak squares, "
        "defender counts)",
        "* Why the move works (what it threatens, why the opponent cannot "
        "respond adequately)",
        "",
        "Style:",
        "* Objective voice, no 'I see/notice'",
        "* Standard notation with brief clarification when helpful: "
        "'Qxh7+ (queen takes h7 with check)'",
        "* Never mention engine scores, ratings, or that you were given the "
        "solution",
        "* 4-10 sentences, scaled to complexity",
        "",
        f"End with exactly: MOVE: {solution}",
    ])


def teacher(args: argparse.Namespace) -> None:
    engine_path = OUT_DIR / "engine.jsonl"
    out = OUT_DIR / "teacher.jsonl"
    if out.exists() and not args.force:
        print("skip teacher (exists)", flush=True)
        return
    if not engine_path.exists():
        print("no engine.jsonl (run `engine` first)", flush=True)
        sys.exit(1)
    sys.path.insert(0, str(ROOT))
    from src.models import OpenCodeGoModel

    model = OpenCodeGoModel(args.model)
    model.load()
    n_in = n_out = n_fail = 0
    total_prompt_tok = 0
    total_comp_tok = 0
    with engine_path.open() as fin, out.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec.get("engine_error") or not rec.get("agree"):
                continue
            n_in += 1
            prompt = _teacher_prompt(rec)
            result = None
            for attempt in range(1, args.max_attempts + 1):
                result = model.generate(
                    prompt, max_new_tokens=args.max_tokens,
                    temperature=args.temperature, thinking_disabled=True)
                content = (result.get("content") or "").strip()
                m = re.search(r"MOVE:\s*([a-h][1-8][a-h][1-8](?:[qrbnQRBN])?)",
                              content, re.IGNORECASE)
                if m and m.group(1) == rec["engine_bestmove"]:
                    rec["cot"] = content
                    rec["teacher_model"] = model.model_id
                    rec["teacher_attempts"] = attempt
                    rec["teacher_usage"] = result.get("token_usage")
                    total_prompt_tok += result.get("input_tokens") or 0
                    total_comp_tok += result.get("output_tokens") or 0
                    n_out += 1
                    break
                print(f"{rec['id']} attempt {attempt}: "
                      f"{'no MOVE:' if not m else 'wrong move'} "
                      f"(len {len(content)}), retrying", flush=True)
            else:
                rec["teacher_error"] = True
                n_fail += 1
                print(f"{rec['id']} FAILED after {args.max_attempts} attempts",
                      flush=True)
                continue
            fout.write(json.dumps(rec) + "\n")
            if n_out % 50 == 0:
                print(f"teacher: {n_out} ok / {n_in} processed", flush=True)
    print(f"teacher done: {n_in} positions, {n_out} valid "
          f"({100.0 * n_out / max(n_in, 1):.1f}%), {n_fail} failed; "
          f"~{total_prompt_tok} prompt + ~{total_comp_tok} completion tokens "
          f"-> {out}", flush=True)

# --------------------------------------------------------------------------
# Stage 4: assemble the student-format training set (MATE format only)
# --------------------------------------------------------------------------

def _tokens_estimate(text: str, tokenizer) -> int | None:
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return None


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
        return tok
    except Exception:
        return None


def assemble(args: argparse.Namespace) -> None:
    teacher_path = OUT_DIR / "teacher.jsonl"
    out = OUT_DIR / "train.jsonl"
    if out.exists() and not args.force:
        print("skip assemble (exists)", flush=True)
        return
    if not teacher_path.exists():
        print("no teacher.jsonl (run `teacher` first)", flush=True)
        sys.exit(1)
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        print("WARNING: gemma tokenizer unavailable; token counts use "
              "chars/4 estimate", flush=True)
    rows = []
    with teacher_path.open() as f:
        for line in f:
            rec = json.loads(line)
            # MATE format: the EXACT MATE noexplain prompt (FEN + MoveA/
            # MoveB candidates, no explanations) with the verbalized CoT
            # prepended to the expert answer. The CoT supplies the reasoning
            # the noexplain format lacks — the special target of Arm B.
            prompt = rec["mate_input"]
            target = rec["cot"] + "\n" + f"Move{rec['truth_label']}:{rec['truth_move']}"
            row = {
                "id": f"{rec['id']}-mate",
                "subset": rec["subset"],
                "format": "mate",
                "fen": rec["fen"],
                "truth_move": rec["engine_bestmove"],
                "engine_pv": rec.get("engine_pv", []),
                "engine_score_type": rec.get("engine_score_type", ""),
                "engine_score": rec.get("engine_score"),
                "prompt": prompt,
                "target": target,
            }
            row["prompt_tokens"] = _tokens_estimate(
                prompt, tokenizer) or max(1, len(prompt) // 4)
            row["target_tokens"] = _tokens_estimate(
                target, tokenizer) or max(1, len(target) // 4)
            rows.append(row)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    total_tok = sum(r["prompt_tokens"] + r["target_tokens"] for r in rows)
    by_subset = {}
    for r in rows:
        by_subset[r["subset"]] = by_subset.get(r["subset"], 0) + 1
    stats = {
        "n_total": len(rows),
        "by_subset": by_subset,
        "total_tokens": total_tok,
        "mean_tokens_per_sample": total_tok / max(len(rows), 1),
        "note": ("Arm A (vanilla MATE, 50k samples) is roughly 6-10M tokens; "
                 "match Arm B by choosing the per-subset n so total_tokens "
                 "lands in the same band"),
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)
    print(f"wrote {len(rows)} training rows -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("download", help="fetch MATE train zips from HF")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("pool", help="parse, dedupe, exclude testset, sample")
    p.add_argument("--subsets", nargs="+", default=["noexplain"])
    p.add_argument("--n", type=int, default=500,
                   help="max positions per subset (before noexplain weight)")
    p.add_argument("--noexplain-weight", type=float, default=2.0,
                   help="multiplier for the noexplain quota (our weakest "
                        "eval arm; special target)")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("engine", help="Stockfish pass + agreement filter")
    p.add_argument("--depth", type=int, default=24)
    p.add_argument("--movetime", type=int, default=8000,
                   help="hard per-position time cap in ms (bounds worst case)")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("teacher", help="DeepSeek verbalization of engine lines")
    p.add_argument("--model", default="deepseek-v4-flash-free",
                   help="gateway model id (free tier by default)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("assemble", help="emit student-format training set")
    p.add_argument("--force", action="store_true")

    args = ap.parse_args()
    if args.stage == "download":
        download(args)
    elif args.stage == "pool":
        pool(args)
    elif args.stage == "engine":
        engine(args)
    elif args.stage == "teacher":
        teacher(args)
    elif args.stage == "assemble":
        assemble(args)


if __name__ == "__main__":
    main()
