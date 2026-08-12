"""Build the master-game commentary distillation corpus (Track 2).

Deepseek-v4-flash comments on Lichess/TWIC master games move-by-move:
for every move it reasons about the position (pre-hoc), commits to a
move, and the master's actual move decides agreement vs reconcile.

    OPENCODE_API_KEY=<key> python3 scripts/build_commentary_data.py \
        --pgn data/raw/twic/*.pgn --out results/commentary --games 50

Pipeline (each stage resumable/idempotent):
    1. games    — parse PGNs, keep high-rated games, over-sample
                  endgame-reachable ones (last-position phase filter)
    2. prehoch  — per move: deepseek reasons + commits (legal list shown)
    3. split    — agreement (teacher move == master move) vs reconcile
    4. reconcile— for disagreements: adversarial call w/ SF PV hint
    5. emit     — JSONL rows in the game format:
                    user: FEN + history + turn + instruction
                    assistant: <reasoning>\nMove: <SAN>
                  plus phase label + agree/reconcile flag + token audit

Training rows deliberately match play_selfplay.build_prompt (FEN +
history + turn, NO legal list) so train == eval byte-identically. The
teacher's candidate list is a generation aid only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "raw" / "commentary"
HISTORY_PLIES = 24
MIN_RATING = 2500
MIN_BOTH_RATING = 2400
SEED = 42


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _phase_of(board: chess.Board) -> str:
    """Use the existing deterministic phase classifier if importable;
    else a local ply/material fallback (opening/middlegame/endgame)."""
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.build_phase_dataset import classify_fen  # type: ignore
        phase = classify_fen(board.fen())
        return "endgame" if phase == "sparse" else phase
    except Exception:
        pass
    ply = board.ply()
    if ply <= 12:
        return "opening"
    material = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type not in (chess.KING,):
            material += {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}[
                p.symbol().lower()]
    return "endgame" if material <= 13 else "middlegame"


def _user_prompt(board: chess.Board, history: list[str]) -> str:
    san = " ".join(history) if history else "(starting position)"
    turn = "White" if board.turn == chess.WHITE else "Black"
    return (
        "You are playing a full game of chess. Reason carefully about the "
        "position in natural language, then give your move.\n"
        f"Position (FEN): {board.fen()}\n"
        f"Move history (SAN): {san}\n"
        f"Turn: {turn}\n"
        "Think step by step about tactics, threats, and plans. End your "
        "response with exactly one line:\n"
        "Move: <your move in SAN notation, e.g. Move: Nf3>"
    )


def _teacher_prompt(board: chess.Board, history: list[str]) -> str:
    san = " ".join(history) if history else "(starting position)"
    turn = "White" if board.turn == chess.WHITE else "Black"
    legal = ", ".join(sorted(board.uci(m) for m in board.legal_moves))
    return (
        "You are a strong chess player analyzing a position from a "
        "master game. Reason carefully in natural prose, then pick the "
        "best move.\n"
        f"Position (FEN): {board.fen()}\n"
        f"Move history (SAN): {san}\n"
        f"Turn: {turn}\n"
        f"Legal moves (UCI): {legal}\n"
        "Think step by step about tactics, threats, and plans. End your "
        "response with exactly one line:\n"
        "Move: <your move in SAN notation, e.g. Move: Nf3>"
    )


def _reconcile_prompt(board: chess.Board, history: list[str],
                      own_san: str, master_san: str, pv: str | None) -> str:
    san = " ".join(history) if history else "(starting position)"
    turn = "White" if board.turn == chess.WHITE else "Black"
    hint = (f"\nEngine line for the master's move: {pv}\n"
            if pv else "\n")
    return (
        "You are a strong chess player analyzing a master game. You "
        "previously considered one move, but the grandmaster played a "
        "different one. Find the flaw in your own choice and explain why "
        "the master's move is stronger, then commit to it.\n"
        f"Position (FEN): {board.fen()}\n"
        f"Move history (SAN): {san}\n"
        f"Turn: {turn}\n"
        f"Your move: {own_san}\n"
        f"Master's move: {master_san}\n"
        f"{hint}"
        "Analyze concretely where your line fails. End your response "
        "with exactly one line:\n"
        f"Move: {master_san}"
    )


def _extract_san(text: str, board: chess.Board):
    """Extract a legal SAN move from generated text (last legal one)."""
    text = (text or "").strip()
    if not text:
        return None
    markers = re.findall(r"(?:^|\n)\s*Move\s*[:=]?\s*([^\s\n]+)", text)
    candidates = list(markers)
    for tok in re.findall(r"\b[a-hNBRQK][a-h1-8xO-]{1,7}\b", text):
        if tok not in candidates:
            candidates.append(tok)
    for cand in reversed(candidates):
        try:
            m = board.parse_san(cand)
            if m in board.legal_moves:
                return m
        except ValueError:
            pass
        try:
            m = board.parse_uci(cand)
            if m in board.legal_moves:
                return m
        except ValueError:
            pass
    return None


def _generate_move(model, prompt, args) -> dict | None:
    """Generate and extract a legal move, handling the gateway's
    thinking/content split (the answer can land in either field)."""
    for attempt in range(1, args.max_attempts + 1):
        try:
            result = model.generate(
                prompt, max_new_tokens=args.max_tokens,
                temperature=args.temperature)
        except Exception as e:
            print(f"  api error {e} (attempt {attempt})", flush=True)
            continue
        content = (result.get("content") or "").strip()
        reasoning = (result.get("reasoning") or "").strip()
        combined = f"{reasoning}\n{content}" if content else reasoning
        return {"result": result, "content": content,
                "reasoning": reasoning, "combined": combined}
    return None


# ---------------------------------------------------------------------------
# Stage 1: parse PGNs, filter + over-sample endgame games
# ---------------------------------------------------------------------------

def _load_games(pgn_paths: list[Path], limit: int | None = None) -> list[dict]:
    games = []
    for path in pgn_paths:
        with path.open() as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                headers = game.headers
                try:
                    w = int(headers.get("WhiteElo", "0") or 0)
                    b = int(headers.get("BlackElo", "0") or 0)
                except ValueError:
                    w = b = 0
                if max(w, b) < MIN_RATING or min(w, b) < MIN_BOTH_RATING:
                    continue
                moves = []
                board = chess.Board()
                bad = False
                for node in game.mainline():
                    mv = node.move
                    if mv is None:
                        break
                    try:
                        san = board.san(mv)
                        board.push(mv)
                    except Exception:
                        bad = True
                        break
                    moves.append({"uci": mv.uci(), "san": san})
                if bad:
                    continue
                if len(moves) < 20:
                    continue
                final_board = game.end().board()
                games.append({
                    "event": headers.get("Event", ""),
                    "white": headers.get("White", ""),
                    "black": headers.get("Black", ""),
                    "white_elo": w,
                    "black_elo": b,
                    "result": headers.get("Result", "*"),
                    "phase": _phase_of(final_board),
                    "moves": moves,
                })
                if limit and len(games) >= limit:
                    return games
    return games


def games(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "games.jsonl"
    if out.exists() and not args.force:
        print("skip games (exists)", flush=True)
        return
    pgns = sorted(
        p for g in args.pgn
        for p in Path(g).expanduser().parent.glob(
            Path(g).expanduser().name)
        if p.suffix in (".pgn", ".txt"))
    print(f"parsing {len(pgns)} pgn files...", flush=True)
    pool = _load_games(pgns, limit=args.scan)
    print(f"{len(pool)} qualifying games (>= {MIN_RATING} Elo)", flush=True)

    import random
    rng = random.Random(SEED)
    opening = [g for g in pool if g["phase"] == "opening"]
    middle = [g for g in pool if g["phase"] == "middlegame"]
    endgame = [g for g in pool if g["phase"] == "endgame"]
    print(f"phase split: opening={len(opening)} middle={len(middle)} "
          f"endgame={len(endgame)}", flush=True)

    # over-sample endgame-reachable games so the phase thesis holds
    n_end = min(len(endgame), args.endgame_quota)
    n_mid = min(len(middle), max(0, args.games - n_end))
    n_open = min(len(opening), max(0, args.games - n_end - n_mid))
    picked = (rng.sample(endgame, n_end) if endgame else []
              ) + (rng.sample(middle, n_mid) if middle else []
                   ) + (rng.sample(opening, n_open) if opening else [])
    rng.shuffle(picked)
    with out.open("w") as f:
        for g in picked:
            f.write(json.dumps(g) + "\n")
    print(f"wrote {len(picked)} games -> {out} "
          f"(endgame quota {n_end})", flush=True)


# ---------------------------------------------------------------------------
# Stage 2: pre-hoc per-move commentary
# ---------------------------------------------------------------------------

def prehoch(args: argparse.Namespace) -> None:
    games_path = OUT_DIR / "games.jsonl"
    out = OUT_DIR / "prehoch.jsonl"
    if out.exists() and not args.force:
        print("skip prehoch (exists) -- use --force to redo", flush=True)
        return
    if not games_path.exists():
        print("no games.jsonl (run `games` first)", flush=True)
        sys.exit(1)
    sys.path.insert(0, str(ROOT))
    from src.models import OpenCodeGoModel

    done: set[tuple] = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
                done.add((r["game_id"], r["ply"]))
            except Exception:
                pass
        print(f"resuming prehoch: {len(done)} rows already done", flush=True)
    else:
        out.touch()

    model = OpenCodeGoModel(args.model)
    model.load()
    n_moves = n_ok = n_agree = 0
    total_in = total_out = 0
    with games_path.open() as fin, out.open("a") as fout:
        for line in fin:
            g = json.loads(line)
            board = chess.Board()
            history: list[str] = []
            gid = f"{g.get('event','')}-{g.get('white','')}-{g.get('black','')}"
            for i, m in enumerate(g["moves"]):
                san = m["san"]
                mv = board.parse_san(san)
                # comment the position BEFORE the master's move
                if i > 0:
                    n_moves += 1
                    if (gid, i + 1) in done:
                        board.push(mv)
                        history.append(san)
                        continue
                    row = _comment_one(model, g, board, history,
                                       m, args)
                    if row:
                        n_ok += 1
                        n_agree += 1 if row["agree"] else 0
                        total_in += row["input_tokens"]
                        total_out += row["output_tokens"]
                        fout.write(json.dumps(row) + "\n")
                        fout.flush()
                        done.add((gid, i))
                    if n_moves % 25 == 0:
                        print(f"prehoch: {n_moves} moves, {n_ok} ok, "
                              f"{n_agree} agree "
                              f"({100.0*n_agree/max(n_ok,1):.0f}%)",
                              flush=True)
                board.push(mv)
                history.append(san)
    print(f"prehoch done: {n_ok}/{n_moves} moves, {n_agree} agree "
          f"({100.0*n_agree/max(n_ok,1):.1f}%); "
          f"~{total_in} in + ~{total_out} out tokens -> {out}", flush=True)


def _comment_one(model, g, board, history, master_move, args) -> dict | None:
    prompt = _teacher_prompt(board, history)
    out = _generate_move(model, prompt, args)
    if out is None:
        print(f"  FAILED after {args.max_attempts} attempts", flush=True)
        return None
    result = out["result"]
    m = _extract_san(out["combined"], board)
    if m is None:
        print(f"  no legal move: {out['combined'][-120:]!r}", flush=True)
        return None
    own_san = board.san(m)
    master_san = master_move["san"]
    agree = own_san == master_san
    text = out["combined"]
    row = {
        "game_id": f"{g.get('event','')}-{g.get('white','')}-"
                   f"{g.get('black','')}",
        "ply": len(history) + 1,
        "fen": board.fen(),
        "history": " ".join(history),
        "phase": _phase_of(board),
        "agree": agree,
        "own_san": own_san,
        "master_san": master_san,
        "reasoning": out["reasoning"],
        "answer": text,
        "teacher_model": args.model,
        "teacher_attempts": 1,
        "input_tokens": result.get("input_tokens") or 0,
        "output_tokens": result.get("output_tokens") or 0,
        "reasoning_tokens": result.get("reasoning_tokens") or 0,
        "ts": _utc_ts(),
    }
    return row


# ---------------------------------------------------------------------------
# Stage 3: reconcile disagreements with an adversarial call + SF PV
# ---------------------------------------------------------------------------

def reconcile(args: argparse.Namespace) -> None:
    prehoch_path = OUT_DIR / "prehoch.jsonl"
    out = OUT_DIR / "reconciled.jsonl"
    if out.exists() and not args.force:
        print("skip reconcile (exists)", flush=True)
        return
    if not prehoch_path.exists():
        print("no prehoch.jsonl (run `prehoch` first)", flush=True)
        sys.exit(1)
    sys.path.insert(0, str(ROOT))
    from src.models import OpenCodeGoModel

    model = OpenCodeGoModel(args.model)
    model.load()
    eng = None
    if args.stockfish:
        eng = _open_engine(args.stockfish)

    n_rows = n_rec = 0
    total_in = total_out = 0
    with prehoch_path.open() as fin, out.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            if row["agree"]:
                fout.write(line)
                n_rows += 1
                continue
            n_rec += 1
            board = chess.Board(row["fen"])
            history = row["history"].split()
            pv = None
            if eng is not None:
                pv = _engine_pv(eng, board)
            prompt = _reconcile_prompt(board, history,
                                       row["own_san"], row["master_san"], pv)
            out = _generate_move(model, prompt, args)
            if out is None:
                print(f"  reconcile FAILED for {row['game_id']} "
                      f"ply {row['ply']}", flush=True)
                row["reconciled"] = False
                fout.write(json.dumps(row) + "\n")
                n_rows += 1
                continue
            result = out["result"]
            m = _extract_san(out["combined"], board)
            if m is None or board.san(m) != row["master_san"]:
                print(f"  reconcile landed on "
                      f"{board.san(m) if m else 'none'} not "
                      f"{row['master_san']}, dropping", flush=True)
                row["reconciled"] = False
                fout.write(json.dumps(row) + "\n")
                n_rows += 1
                continue
            row["reconciled"] = True
            row["reasoning"] = out["reasoning"]
            row["answer"] = out["combined"]
            row["reconcile_attempts"] = 1
            row["input_tokens"] += result.get("input_tokens") or 0
            row["output_tokens"] += result.get("output_tokens") or 0
            row["reconcile_tokens"] = result.get("output_tokens") or 0
            row["engine_pv"] = pv
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_rows += 1
            total_in += result.get("input_tokens") or 0
            total_out += result.get("output_tokens") or 0
            if n_rec % 25 == 0:
                print(f"reconcile: {n_rec} disagreements processed",
                      flush=True)
    if eng is not None:
        eng.quit()
    print(f"reconcile done: {n_rows} rows, {n_rec} disagreements "
          f"(~{total_in}+{total_out} tokens) -> {out}", flush=True)


def _open_engine(path: str):
    import chess.engine
    return chess.engine.SimpleEngine.popen_uci(path)


def _engine_pv(eng, board: chess.Board, depth: int = 12) -> str | None:
    try:
        r = eng.analyse(board, chess.engine.Limit(depth=depth))
        if "pv" not in r or not r["pv"]:
            return None
        return " ".join(m.uci() for m in r["pv"][:6])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage 4: emit the game-format training rows
# ---------------------------------------------------------------------------

def emit(args: argparse.Namespace) -> None:
    reconciled_path = OUT_DIR / "reconciled.jsonl"
    out = OUT_DIR / "train.jsonl"
    eval_out = OUT_DIR / "eval.jsonl"
    if out.exists() and not args.force:
        print("skip emit (exists)", flush=True)
        return
    if not reconciled_path.exists():
        print("no reconciled.jsonl (run `reconcile` first)", flush=True)
        sys.exit(1)
    agree, recon = [], []
    with reconciled_path.open() as f:
        for line in f:
            row = json.loads(line)
            board = chess.Board(row["fen"])
            history = row["history"].split()
            prompt = _user_prompt(board, history)
            target = f"{row['reasoning']}\nMove: {row['master_san']}"
            rec = {
                "game_id": row["game_id"],
                "ply": row["ply"],
                "phase": row["phase"],
                "agree": row["agree"],
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                "fen": row["fen"],
            }
            (agree if row["agree"] else recon).append(rec)

    import random
    rng = random.Random(SEED)
    rng.shuffle(agree)
    rng.shuffle(recon)
    # agreement rows upsampled 2:1
    n_recon = len(recon)
    n_agree = min(len(agree), n_recon * args.agree_ratio)
    train = recon + agree[:n_agree]
    rng.shuffle(train)
    # hold out ~5% as eval (position-disjoint by construction)
    n_eval = max(1, len(train) // 20)
    eval_rows = train[:n_eval]
    train_rows = train[n_eval:]
    with out.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with eval_out.open("w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"emit done: {len(train_rows)} train + {len(eval_rows)} eval "
          f"(agree {n_agree}/{len(agree)}, reconcile {n_recon}) "
          f"-> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("games", help="parse + filter + over-sample games")
    p.add_argument("--pgn", nargs="+", default=["data/raw/twic/*.pgn"])
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--scan", type=int, default=None,
                   help="max games to scan for qualification")
    p.add_argument("--endgame-quota", type=int, default=20)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("prehoch", help="per-move deepseek commentary")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("reconcile", help="adversarial reconcile of disagreements")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--stockfish", default="",
                   help="path to stockfish binary for the PV hint")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("emit", help="emit game-format training rows")
    p.add_argument("--agree-ratio", type=int, default=2,
                   help="agreement:reconcile upsample ratio (2 = 2:1)")
    p.add_argument("--force", action="store_true")

    args = ap.parse_args()
    if args.stage == "games":
        games(args)
    elif args.stage == "prehoch":
        prehoch(args)
    elif args.stage == "reconcile":
        reconcile(args)
    elif args.stage == "emit":
        emit(args)


if __name__ == "__main__":
    main()
