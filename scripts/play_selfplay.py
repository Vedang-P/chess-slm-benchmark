"""One deepseek-vs-deepseek full game — the data generator for full-game
trace distillation.

    OPENCODE_API_KEY=<key> python3 scripts/play_selfplay.py \
        --game-id g0001 --out results/selfplay

Division of labour (single-writer design — no per-move network I/O here):
  - This script owns ONLY the local files:
      {out}/{id}.json        full game record (all plies + FULL thinking),
                             written after every ply (also the resume checkpoint)
      {out}/live-{id}.json   cheap live snapshot (current fen, side, thinking
                             tail) rewritten every few seconds while thinking
      {out}/{id}.traces.jsonl  training-data rows (one per ply, full prompt +
                             full thinking + answer + legality) — this is the
                             distillation dataset
  - A SINGLE campaign supervisor (run_selfplay_campaign.py) aggregates the
    live-*.json files into one throttled GitHub feed and uploads completed
    games + traces to HF. No GitHub/HF calls from inside this loop.

Game loop: python-chess legality; illegal output retried (fresh generation)
up to 3x then resign; ends at mate/stalemate/50-move/threefold/
insufficient material/ply cap. Thinking is unbounded.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import make_model  # noqa: E402

MAX_NEW_TOKENS = 131072
ILLEGAL_RETRIES = 3
PLY_CAP = 160
LIVE_SNAPSHOT_INTERVAL_S = 4.0
THINKING_TAIL_CHARS = 4000

OPENING_LINES = [
    ["e4", "e5", "Nf3", "Nc6", "Bb5"],                  # Ruy Lopez
    ["e4", "e5", "Nf3", "Nc6", "Bc4"],                  # Italian
    ["e4", "c5", "Nf3", "d6", "d4"],                    # Sicilian
    ["d4", "d5", "c4"],                                 # QGD
    ["d4", "Nf6", "c4", "g6"],                          # KID
    ["d4", "Nf6", "c4", "e6"],                          # Nimzo/QID
    ["c4"],                                              # English
    ["Nf3", "d5", "g3"],                                # Réti
    ["e4", "c6"],                                       # Caro-Kann
    ["e4", "e6", "d4", "d5"],                           # French
    ["d4", "f5"],                                       # Dutch
    ["e4", "d6", "d4", "Nf6", "Nc3"],                   # Pirc
    ["c4", "e5", "Nc3", "Nf6"],                         # English opening
    ["Nf3", "Nf6", "g3", "g6", "Bg2"],                  # Fianchetto
    ["e4", "Nf6"],                                      # Alekhine
    ["e4", "c5", "c3"],                                 # Alapin
    ["d4", "d5", "Nf3", "Nf6", "c4"],                   # QG exchange-ish
    ["e4", "e5", "f4"],                                 # King's Gambit
    ["d4", "d5", "e4"],                                 # Blackmar-Diemer
    ["e4", "e5", "Nf3", "Nc6", "d4"],                   # Scotch
]

FIRST_PLIES = ["e4", "d4", "c4", "Nf3", "g3", "b3"]


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_move(text: str, board: chess.Board):
    text = (text or "").strip()
    if not text:
        return None
    markers = re.findall(r"(?:^|\n)\s*Move\s*[:=]?\s*([^\s\n]+)", text)
    candidates = list(markers)
    for tok in re.findall(r"\b[a-hNBRQK][a-h1-8xO-]{1,7}\b", text):
        if tok not in candidates:
            candidates.append(tok)
    for cand in candidates:
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


def _game_over_reason(board: chess.Board) -> str | None:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_fifty_moves():
        return "fifty_moves"
    if board.is_repetition(3):
        return "threefold"
    return None


def _result_of(board: chess.Board) -> str:
    if board.is_checkmate():
        return "1-0" if board.turn == chess.BLACK else "0-1"
    return "1/2-1/2"


def build_prompt(board: chess.Board, history: list[str]) -> str:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", default="g0001")
    ap.add_argument("--out", default="results/selfplay")
    ap.add_argument("--opening", type=int, default=None)
    ap.add_argument("--from-move-one", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{args.game_id}.json"
    live_path = out_dir / f"live-{args.game_id}.json"
    traces_path = out_dir / f"{args.game_id}.traces.jsonl"

    game: dict = {}
    if checkpoint.exists():
        game = json.loads(checkpoint.read_text())
        print(f"resuming {args.game_id} at ply "
              f"{len(game.get('plies', []))}", flush=True)

    if not game:
        board = chess.Board()
        history: list[str] = []
        plies: list[dict] = []
        opening = args.opening if args.opening is not None else \
            random.randrange(len(OPENING_LINES))
        line = OPENING_LINES[opening]
        if args.from_move_one:
            line = [random.choice(FIRST_PLIES)]
        for san in line:
            mv = board.parse_san(san)
            board.push(mv)
            history.append(san)
            plies.append({
                "n": board.fullmove_number,
                "by": "w" if board.turn == chess.BLACK else "b",
                "san": san, "uci": mv.uci(), "fen": board.fen(),
                "book": True, "thinking": "", "answer": "",
                "tokens": 0, "latency_ms": 0, "status": "book",
                "ts": _utc_ts(),
            })
        game = {
            "game_id": args.game_id,
            "status": "running", "result": "*",
            "opening": {"index": opening, "name": f"opening-{opening}"},
            "white": {"model": "deepseek-v4-flash", "thinking": "",
                      "thinking_tokens": 0},
            "black": {"model": "deepseek-v4-flash", "thinking": "",
                      "thinking_tokens": 0},
            "plies": plies, "reason": None,
            "started_at": _utc_ts(), "updated_at": _utc_ts(),
        }
    else:
        board = chess.Board(game["plies"][-1]["fen"])
        history = [p["san"] for p in game["plies"]]
        plies = game["plies"]

    model = make_model("deepseek-v4-flash")
    model.load()

    def write_live(thinking: str, by: str) -> None:
        """Live snapshot for the website. Carries BOTH sides' thinking:
        the active side's in-progress tail plus the other side's last
        completed trace, so the dashboard always shows both players."""
        w_done = game["white"]["thinking"]
        b_done = game["black"]["thinking"]
        live_path.write_text(json.dumps({
            "game_id": args.game_id,
            "status": game["status"],
            "plies": len(plies),
            "by": by,
            "fen": board.fen(),
            "turn": "w" if board.turn == chess.WHITE else "b",
            "last_san": plies[-1]["san"] if plies else None,
            "history": [p["san"] for p in plies],
            "thinking": {
                "w": (thinking if by == "w" else w_done)[-THINKING_TAIL_CHARS:],
                "b": (thinking if by == "b" else b_done)[-THINKING_TAIL_CHARS:],
            },
            "updated_at": _utc_ts(),
        }))

    def write_trace(ply: dict, prompt: str) -> None:
        row = {
            "game_id": args.game_id,
            "ply": len(plies),
            "by": ply["by"],
            "fen": ply["fen"],
            "history": " ".join(history),
            "prompt": prompt,
            "thinking": ply["thinking"],
            "answer": ply["answer"],
            "move_san": ply["san"],
            "move_uci": ply["uci"],
            "tokens": ply["tokens"],
            "latency_ms": ply["latency_ms"],
            "status": ply["status"],
            "ts": ply["ts"],
        }
        with open(traces_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    while len(plies) < PLY_CAP:
        over = _game_over_reason(board)
        if over:
            game["status"] = over
            game["result"] = _result_of(board)
            game["reason"] = over
            break

        by = "w" if board.turn == chess.WHITE else "b"
        sn = "white" if by == "w" else "black"
        prompt = build_prompt(board, history)
        first_ts = time.time()
        last_live = 0.0

        def _on_chunk(partial: dict) -> None:
            nonlocal last_live
            if partial.get("phase") == "done":
                return
            now = time.time()
            if now - last_live >= LIVE_SNAPSHOT_INTERVAL_S:
                last_live = now
                write_live(partial.get("reasoning", "") or
                           partial.get("content", ""), by)

        move = None
        attempts = 0
        answer = ""
        reasoning = ""
        while attempts < ILLEGAL_RETRIES + 1:
            out = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                stream=False,
                on_chunk=_on_chunk,
            )
            answer = out.get("content", "")
            reasoning = out.get("reasoning", "") or reasoning
            if out.get("error"):
                print(f"    api error: {out['error'][:120]}", flush=True)
                attempts += 1
                if attempts > ILLEGAL_RETRIES:
                    game["status"] = "resigned"
                    game["result"] = "0-1" if board.turn == chess.WHITE else "1-0"
                    game["reason"] = "api_error"
                    break
                continue
            move = _parse_move(answer, board)
            if move:
                break
            attempts += 1
            print(f"    illegal/no move on attempt {attempts}: "
                  f"{answer[-200:]!r}", flush=True)
        else:
            game["status"] = "resigned"
            game["result"] = "0-1" if board.turn == chess.WHITE else "1-0"
            game["reason"] = "illegal"
            break

        if game["status"] != "running":
            break

        san = board.san(move)
        board.push(move)
        history.append(san)
        ply = {
            "n": board.fullmove_number,
            "by": by, "san": san, "uci": move.uci(), "fen": board.fen(),
            "book": False, "thinking": reasoning,
            "answer": answer.strip()[-2000:],
            "tokens": (out.get("output_tokens") or 0),
            "latency_ms": int((time.time() - first_ts) * 1000),
            "status": "legal", "ts": _utc_ts(),
        }
        plies.append(ply)
        game[sn]["thinking"] = reasoning
        game[sn]["thinking_tokens"] += ply["tokens"]
        game["updated_at"] = _utc_ts()
        checkpoint.write_text(json.dumps(game, indent=1))
        write_trace(ply, prompt)
        write_live(reasoning, by)
        print(f"  ply {len(plies)}: {san} "
              f"({ply['tokens']} tok, {ply['latency_ms'] // 1000}s)", flush=True)

    if game["status"] == "running" and len(plies) >= PLY_CAP:
        game["status"] = "ply_cap"
        game["result"] = "1/2-1/2"
        game["reason"] = "ply_cap"
    checkpoint.write_text(json.dumps(game, indent=1))
    print(f"DONE {args.game_id}: {game['status']} {game['result']} "
          f"({len(plies)} plies)", flush=True)


if __name__ == "__main__":
    main()
