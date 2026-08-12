"""Build the master-game lucid-commentary distillation corpus (Track 2).

For every move of a sample of master games (both sides), deepseek
explains WHY the master's move works — in the compressed "lucid"
reasoning style. One API call per move. The raw thinking trace is saved
(provenance) but only the lucid commentary is used for training.

    OPENCODE_API_KEY=... python3 scripts/build_commentary_data.py \
        --pgn "data/raw/twic/*.pgn" --games 100 --out results/commentary

Pipeline (each stage resumable/idempotent):
    1. games      — parse PGNs, keep high-rated games, over-sample
                    endgame-reachable games, sample N
    2. commentate — per move: lucid explanation of the master's move;
                    saves reasoning (thinking trace) + content (lucid)
    3. emit       — training rows: user = FEN + history + turn +
                    instruction; assistant = <lucid>\nMove: <SAN>

Training rows are byte-identical to the eval game loop prompt
(FEN + history + turn -> reason -> Move: <SAN>).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "commentary"
MIN_RATING = 2500
MIN_BOTH_RATING = 2400
SEED = 42


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _phase_of(board: chess.Board) -> str:
    """Existing deterministic phase classifier (published artifact)."""
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


def _lucid_prompt(board: chess.Board, history: list[str],
                  move_san: str) -> str:
    san = " ".join(history) if history else "(starting position)"
    turn = "White" if board.turn == chess.WHITE else "Black"
    return (
        "You are a chess analyst explaining a move from a master game. "
        "The grandmaster played "
        f"{move_san}.\n"
        f"Position (FEN): {board.fen()}\n"
        f"Move history (SAN): {san}\n"
        f"Turn: {turn}\n"
        "Explain why this move works in a CONCISE, compressed reasoning "
        "style: short phrases, key squares, the tactical/positional idea, "
        "no filler, no 'I see' or 'we notice'. 2-6 sentences. Ground every "
        "claim in concrete squares and pieces.\n"
        "End with exactly one line:\n"
        f"Move: {move_san}"
    )


def _extract_san(text: str, board: chess.Board):
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


def _generate(model, prompt, args) -> dict | None:
    for attempt in range(1, args.max_attempts + 1):
        try:
            result = model.generate(
                prompt, max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                thinking_disabled=not args.thinking)
        except Exception as e:
            print(f"  api error {e} (attempt {attempt})", flush=True)
            continue
        content = (result.get("content") or "").strip()
        reasoning = (result.get("reasoning") or "").strip()
        return {"result": result, "content": content,
                "reasoning": reasoning}
    return None


# ---------------------------------------------------------------------------
# Stage 1: parse PGNs, filter + over-sample, sample N
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
                if bad or len(moves) < 20:
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
    print(f"{len(pool)} qualifying games (>= {MIN_RATING} Elo both >= "
          f"{MIN_BOTH_RATING})", flush=True)

    rng = random.Random(SEED)
    opening = [g for g in pool if g["phase"] == "opening"]
    middle = [g for g in pool if g["phase"] == "middlegame"]
    endgame = [g for g in pool if g["phase"] == "endgame"]
    print(f"phase split: opening={len(opening)} middle={len(middle)} "
          f"endgame={len(endgame)}", flush=True)

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
    print(f"wrote {len(picked)} games -> {out} (endgame quota {n_end})",
          flush=True)


# ---------------------------------------------------------------------------
# Stage 2: per-move lucid commentary (one call per move, both sides)
# ---------------------------------------------------------------------------

def commentate(args: argparse.Namespace) -> None:
    games_path = OUT_DIR / "games.jsonl"
    out = OUT_DIR / "commentary.jsonl"
    if not games_path.exists():
        print("no games.jsonl (run `games` first)", flush=True)
        sys.exit(1)

    done: set[tuple] = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
                done.add((r["game_id"], r["ply"]))
            except Exception:
                pass
        print(f"resuming: {len(done)} rows already done", flush=True)
    else:
        out.touch()

    sys.path.insert(0, str(ROOT))
    from src.models import OpenCodeGoModel

    model = OpenCodeGoModel(args.model)
    model.load()
    n_moves = n_ok = 0
    total_in = total_out = 0
    with games_path.open() as fin, out.open("a") as fout:
        for line in fin:
            g = json.loads(line)
            gid = f"{g.get('event','')}-{g.get('white','')}-{g.get('black','')}"
            board = chess.Board()
            history: list[str] = []
            for i, m in enumerate(g["moves"]):
                san = m["san"]
                mv = board.parse_san(san)
                # comment on the position BEFORE the move was played
                if i > 0:
                    n_moves += 1
                    if (gid, i + 1) in done:
                        board.push(mv)
                        history.append(san)
                        continue
                    row = _comment_one(model, g, gid, i, board,
                                       history, m, args)
                    if row:
                        n_ok += 1
                        total_in += row["input_tokens"]
                        total_out += row["output_tokens"]
                        fout.write(json.dumps(row) + "\n")
                        fout.flush()
                        done.add((gid, i + 1))
                    if n_moves % 25 == 0:
                        print(f"commentate: {n_moves} moves, {n_ok} ok "
                              f"({100.0*n_ok/max(n_moves,1):.0f}%)",
                              flush=True)
                board.push(mv)
                history.append(san)
    print(f"commentate done: {n_ok}/{n_moves} moves; "
          f"~{total_in} in + ~{total_out} out tokens -> {out}", flush=True)


def _comment_one(model, g, gid, i, board, history, master_move,
                 args) -> dict | None:
    master_san = master_move["san"]
    prompt = _lucid_prompt(board, history, master_san)
    out = _generate(model, prompt, args)
    if out is None:
        print(f"  FAILED after {args.max_attempts} attempts", flush=True)
        return None
    result = out["result"]
    m = _extract_san(out["content"] or out["reasoning"], board)
    if m is None:
        # keep the row anyway; the lucid text is the training signal
        pass
    row = {
        "game_id": gid,
        "ply": i + 1,
        "fen": board.fen(),
        "history": " ".join(history),
        "phase": _phase_of(board),
        "master_san": master_san,
        "thinking": out["reasoning"],     # raw trace (saved, not trained)
        "lucid": out["content"],          # training signal
        "teacher_model": args.model,
        "input_tokens": result.get("input_tokens") or 0,
        "output_tokens": result.get("output_tokens") or 0,
        "reasoning_tokens": result.get("reasoning_tokens") or 0,
        "ts": _utc_ts(),
    }
    return row


# ---------------------------------------------------------------------------
# Stage 3: emit training rows (lucid answer only, not the thinking trace)
# ---------------------------------------------------------------------------

def emit(args: argparse.Namespace) -> None:
    commentary_path = OUT_DIR / "commentary.jsonl"
    out = OUT_DIR / "train.jsonl"
    eval_out = OUT_DIR / "eval.jsonl"
    if out.exists() and not args.force:
        print("skip emit (exists)", flush=True)
        return
    if not commentary_path.exists():
        print("no commentary.jsonl (run `commentate` first)", flush=True)
        sys.exit(1)

    rows = []
    for line in commentary_path.open():
        row = json.loads(line)
        board = chess.Board(row["fen"])
        history = row["history"].split()
        prompt = _user_prompt(board, history)
        target = f"{row['lucid']}\nMove: {row['master_san']}"
        rows.append({
            "game_id": row["game_id"],
            "ply": row["ply"],
            "phase": row["phase"],
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            "fen": row["fen"],
        })

    rng = random.Random(SEED)
    rng.shuffle(rows)
    n_eval = max(1, len(rows) // 20)
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]
    with out.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with eval_out.open("w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"emit done: {len(train_rows)} train + {len(eval_rows)} eval "
          f"-> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("games", help="parse + filter + over-sample games")
    p.add_argument("--pgn", nargs="+", default=["data/raw/twic/*.pgn"])
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--scan", type=int, default=None)
    p.add_argument("--endgame-quota", type=int, default=30)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("commentate", help="per-move lucid commentary")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--thinking", action="store_true",
                   help="keep deepseek thinking ON (slower, richer trace); "
                        "default OFF because the lucid answer is the "
                        "training signal and thinking burns ~4k tokens/move")

    p = sub.add_parser("emit", help="emit game-format training rows")
    p.add_argument("--force", action="store_true")

    args = ap.parse_args()
    if args.stage == "games":
        games(args)
    elif args.stage == "commentate":
        commentate(args)
    elif args.stage == "emit":
        emit(args)


if __name__ == "__main__":
    main()
