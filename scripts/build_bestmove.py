"""Build the best-move task set (Stockfish ground truth).

Ground truth comes from Lichess's Stockfish evaluations:
- --cloud: the lichess cloud-eval API, applied to the cap-legal position FENs
  (same base positions as the legality task; small, committed dataset).
- --stream: the 21GB lichess eval DB (CC0) stream-filtered — for regenerating
  at scale on Kaggle; needs zstandard + a fast connection.

Filters: eval depth >= 20, Stockfish's best move is LEGAL in our engine
variant (no double-step/castling best moves), >= 2 legal moves total.

Usage:
    python scripts/build_bestmove.py --cloud    # default; builds data/positions/bestmove-8x8.json
    python scripts/build_bestmove.py --stream --n 200
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.games.fen import parse_fen  # noqa: E402

CLOUD_URL = "https://lichess.org/api/cloud-eval?fen={fen}"
EVAL_DB_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
OUT = Path("data/positions/bestmove-8x8.json")
MIN_DEPTH = 20
MIN_LEGAL = 2


def _best_move(fen: str) -> dict:
    """(best_move_uci, cp, mate, depth) from cloud-eval, or None."""
    url = CLOUD_URL.format(fen=urllib.parse.quote(fen))
    req = urllib.request.Request(url, headers={"User-Agent": "chess-bench"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    evals = d.get("evals") or [d]
    best = max(evals, key=lambda e: e.get("depth", 0))
    if best.get("depth", 0) < MIN_DEPTH:
        return None
    pv = best["pvs"][0]
    line = pv.get("line") or pv.get("moves") or ""
    if not line:
        return None
    return {"best": line.split()[0], "cp": pv.get("cp"), "mate": pv.get("mate"),
            "depth": best["depth"]}


def _fits_variant_ours(fen: str, best_uci: str) -> bool:
    """Stockfish's best move must be legal under our variant rules: no
    double-step pawn pushes, no castling (checked via our own engine)."""
    from src.benchmarks.games.rules import Board, algebraic_to_sq

    b = parse_fen(fen)
    fr, to = algebraic_to_sq(best_uci[:2]), algebraic_to_sq(best_uci[2:4])
    if fr is None or to is None:
        return False
    moves = {m.uci: m for m in b.legal_moves()}
    if best_uci not in moves:
        return False
    m = moves[best_uci]
    if m.piece == "P" and abs(to[0] - fr[0]) == 2:
        return False
    if m.piece == "K" and abs(to[1] - fr[1]) == 2:
        return False
    return True


def _record(fen: str, info: dict, idx: int) -> dict:
    from src.benchmarks.games.rules import Board

    b = parse_fen(fen)
    return {
        "id": f"bm-{idx:04d}",
        "source": "lichess-stockfish",
        "n": 8,
        "turn": b.turn,
        "value": "cap",
        "fen": fen,
        "pieces": [
            {"sq": f"{chr(ord('a') + c)}{r + 1}", "color": color, "kind": kind}
            for (r, c), (color, kind) in sorted(b.pieces.items())
        ],
        "win_moves": [info["best"]],
        "lose_moves": [],
        "over_budget": False,
        "task_extra": {"best_move": info["best"], "cp": info["cp"], "mate": info["mate"],
                       "depth": info["depth"]},
    }


def build_cloud(n: int, fens: list) -> None:
    out = []
    t0 = time.time()
    for i, fen in enumerate(fens):
        if time.time() - t0 > 480:
            break  # hard cap: keep what we have
        try:
            info = _best_move(fen)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15)  # rate limit: back off and retry once
                try:
                    info = _best_move(fen)
                except Exception:
                    info = None
            else:
                info = None
        except Exception:
            info = None
        if info and _fits_variant_ours(fen, info["best"]):
            if len(parse_fen(fen).legal_moves()) >= MIN_LEGAL:
                out.append(_record(fen, info, len(out)))
        if i % 10 == 0:
            print(f"  cloud-eval {i + 1}/{len(fens)}, kept {len(out)}", flush=True)
        if len(out) >= n:
            break
        time.sleep(0.3)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"built {len(out)} best-move positions -> {OUT}")


def build_stream(n: int) -> None:
    import io
    import zstandard

    dctx = zstandard.ZstdDecompressor()
    out = []
    with urllib.request.urlopen(EVAL_DB_URL) as resp:
        for line in io.TextIOWrapper(dctx.stream_reader(resp), encoding="utf-8"):
            if len(out) >= n:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            evals = d.get("evals") or []
            if not evals:
                continue
            best = max(evals, key=lambda e: e.get("depth", 0))
            if best.get("depth", 0) < MIN_DEPTH:
                continue
            pv = best["pvs"][0]
            line_moves = pv.get("line") or ""
            if not line_moves:
                continue
            best_uci = line_moves.split()[0]
            if not _fits_variant_ours(d["fen"], best_uci):
                continue
            if len(parse_fen(d["fen"]).legal_moves()) < MIN_LEGAL:
                continue
            out.append(_record(d["fen"], {"best": best_uci, "cp": pv.get("cp"),
                                          "mate": pv.get("mate"), "depth": best["depth"]},
                               len(out)))
    OUT.write_text(json.dumps(out, indent=1))
    print(f"built {len(out)} best-move positions -> {OUT}")


def build_local(n: int, fens: list, depth: int = 15) -> None:
    """Compute best moves with a local Stockfish (python-chess.uci). Fast,
    deterministic, unlimited — the primary path (cloud-eval is rate-limited)."""
    import chess
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    try:
        out = []
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
            except Exception:
                continue
            pv = info.get("pv")
            score = info.get("score")
            if not pv:
                continue
            best_uci = pv[0].uci()
            cp = None
            mate = None
            if score and not score.is_mate():
                cp = score.pov(chess.WHITE).score()
            elif score and score.is_mate():
                mate = score.pov(chess.WHITE).mate()
            if not _fits_variant_ours(fen, best_uci):
                continue
            if len(parse_fen(fen).legal_moves()) < MIN_LEGAL:
                continue
            out.append(_record(fen, {"best": best_uci, "cp": cp, "mate": mate,
                                     "depth": info.get("depth")}, len(out)))
            if len(out) >= n:
                break
            if i % 10 == 0:
                print(f"  stockfish {i + 1}/{len(fens)}, kept {len(out)}", flush=True)
    finally:
        engine.quit()
    OUT.write_text(json.dumps(out, indent=1))
    print(f"built {len(out)} best-move positions -> {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="local Stockfish (default)")
    ap.add_argument("--cloud", action="store_true", help="use the cloud-eval API")
    ap.add_argument("--stream", action="store_true", help="stream the 21GB eval DB")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--fens-from", default="data/positions/cap-legal-8x8.json")
    args = ap.parse_args()

    if args.stream:
        build_stream(args.n)
        return
    if args.cloud:
        src = Path(args.fens_from)
        if src.suffix == ".csv":
            import csv

            rows = list(csv.DictReader(open(src)))
            fens = [r["FEN"] for r in rows]
        else:
            fens = [r["fen"] for r in json.loads(src.read_text())]
        build_cloud(args.n, fens)
        return
    src = Path(args.fens_from)
    if src.suffix == ".csv":
        import csv

        rows = list(csv.DictReader(open(src)))
        fens = [r["FEN"] for r in rows]
    else:
        fens = [r["fen"] for r in json.loads(src.read_text())]
    build_local(args.n, fens, depth=args.depth)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
