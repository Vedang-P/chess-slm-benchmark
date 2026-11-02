"""Batched, exact-semantics puzzle and MATE evaluation.

The scoring rules mirror scripts/eval_gavn.py: legal moves are ordered by the
official action ids (ties are observable), claimable repetitions score 0.5,
the policy is the argmax of the expected return over the legal moves, and a
puzzle is solved when every model move matches the stored solution or an
alternative move delivers checkmate. Many positions are scored per forward
pass instead of one; that changes speed, not results.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np


def ordered_legal_moves(board, move_to_action) -> list:
    return sorted(board.legal_moves, key=lambda m: move_to_action[m.uci()])


class BatchedScorer:
    """`scorer(boards) -> [(ordered_legal_moves, values)]` in fp32."""

    def __init__(self, torch, model, utils, device, chunk_rows: int = 8192):
        self.torch, self.model, self.utils = torch, model, utils
        self.device, self.chunk_rows = device, chunk_rows
        self.bucket_values = torch.as_tensor(
            np.asarray(utils.get_uniform_buckets_edges_values(128)[1],
                       dtype=np.float32), device=device)

    def __call__(self, boards):
        from scripts.eval_gavn import tokenize_fen

        torch = self.torch
        legal = [ordered_legal_moves(b, self.utils.MOVE_TO_ACTION) for b in boards]
        lens = [len(m) for m in legal]
        total = sum(lens)
        values = np.empty(total, dtype=np.float32)
        if total:
            tokens = np.concatenate([
                np.repeat(tokenize_fen(b.fen())[None, :], n, axis=0)
                for b, n in zip(boards, lens) if n])
            actions = np.fromiter(
                (self.utils.MOVE_TO_ACTION[m.uci()] for ms in legal for m in ms),
                dtype=np.int64, count=total)
            with torch.inference_mode():
                for start in range(0, total, self.chunk_rows):
                    sl = slice(start, start + self.chunk_rows)
                    logits = self.model(
                        torch.as_tensor(tokens[sl], dtype=torch.long, device=self.device),
                        torch.as_tensor(actions[sl], dtype=torch.long, device=self.device))
                    out = torch.softmax(logits.float(), -1) @ self.bucket_values
                    values[sl] = out.detach().cpu().numpy()
        results, offset = [], 0
        for board, moves, n in zip(boards, legal, lens):
            v = values[offset: offset + n].copy()
            offset += n
            # Match ActionValueEngine's rule-based score for claimable repetitions.
            for i, move in enumerate(moves):
                board.push(move)
                if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                    v[i] = 0.5
                board.pop()
            results.append((moves, v))
        return results


def evaluate_puzzles(scorer, puzzles: list[dict], chunk: int = 256) -> list[dict]:
    """Score puzzles in lockstep. Each puzzle is {id, fen, moves: [uci, ...]}.

    Index 0 is the opponent's setup move; the model plays every odd index.
    """
    import chess

    out: list[dict] = []
    for start in range(0, len(puzzles), chunk):
        batch = puzzles[start: start + chunk]
        boards = [chess.Board(p["fen"]) for p in batch]
        moves = [[chess.Move.from_uci(u) for u in p["moves"]] for p in batch]
        j = [0] * len(batch)
        result: list[dict | None] = [None] * len(batch)
        active = list(range(len(batch)))

        def push_official(i: int) -> None:
            if moves[i][j[i]] not in boards[i].legal_moves:
                raise ValueError(f"puzzle {batch[i]['id']}: illegal official move "
                                 f"{moves[i][j[i]].uci()}")
            boards[i].push(moves[i][j[i]])
            j[i] += 1

        while active:
            need = []
            for i in list(active):
                while j[i] < len(moves[i]) and j[i] % 2 == 0:
                    push_official(i)
                if j[i] >= len(moves[i]):
                    result[i] = {"id": batch[i]["id"], "correct": True, "fail_ply": None}
                    active.remove(i)
                else:
                    need.append(i)
            if not need:
                break
            for i, (legal, values) in zip(need, scorer([boards[i] for i in need])):
                predicted = legal[int(np.argmax(values))]
                if predicted != moves[i][j[i]]:
                    boards[i].push(predicted)
                    result[i] = {"id": batch[i]["id"], "correct": bool(boards[i].is_checkmate()),
                                 "fail_ply": j[i]}
                    active.remove(i)
                else:
                    push_official(i)
        out.extend(r for r in result if r is not None)
    return out


def mate_candidates(row: dict) -> tuple[str, str, str]:
    extra = row.get("task_extra") or {}
    return (row.get("candidate_a") or row.get("move_a") or extra.get("candidate_a"),
            row.get("candidate_b") or row.get("move_b") or extra.get("candidate_b"),
            row.get("truth_label") or row.get("label") or extra.get("truth_label"))


def evaluate_mate(scorer, rows: list[dict], chunk: int = 256) -> list[dict]:
    import chess

    out: list[dict] = []
    for start in range(0, len(rows), chunk):
        batch = rows[start: start + chunk]
        boards, picks = [], []
        for r in batch:
            fen = r.get("fen") or r.get("position")
            ca, cb, truth = mate_candidates(r)
            if not (fen and ca and cb and truth in ("A", "B")):
                raise ValueError("malformed MATE row")
            boards.append(chess.Board(fen))
            picks.append((ca, cb, truth))
        for (ca, cb, truth), (legal, values) in zip(picks, scorer(boards)):
            by_uci = {m.uci(): float(v) for m, v in zip(legal, values)}
            if ca not in by_uci or cb not in by_uci:
                raise ValueError("MATE candidate is not legal")
            pred = "A" if by_uci[ca] > by_uci[cb] else "B"
            out.append({"correct": pred == truth, "pred": pred, "truth": truth})
    return out


# ------------------------------------------------------------------ loaders
def lichess_puzzles(rows: list[dict]) -> list[dict]:
    """Rows of the Lichess export (PuzzleId, FEN, Moves): FEN precedes Moves[0]."""
    return [{"id": r["PuzzleId"], "fen": r["FEN"], "moves": r["Moves"].split(),
             "rating": int(r.get("Rating") or 0)} for r in rows]


def official_puzzles(csv_path: Path, limit: int = 10_000) -> list[dict]:
    """The official ChessBench file: the start board is the end of its PGN."""
    import chess.pgn
    import pandas as pd

    table = pd.read_csv(csv_path, nrows=limit)
    if len(table) != limit:
        raise ValueError(f"puzzles.csv has {len(table)} rows, expected {limit}")
    puzzles = []
    for i, row in table.iterrows():
        game = chess.pgn.read_game(io.StringIO(row["PGN"]))
        if game is None or game.errors:
            raise ValueError(f"puzzle row {i}: invalid PGN")
        try:
            rating = int(row.get("Rating", 0))
        except (TypeError, ValueError):
            rating = 0
        puzzles.append({"id": str(row.get("PuzzleId", i)), "fen": game.end().board().fen(),
                        "moves": str(row["Moves"]).split(), "rating": rating})
    return puzzles


def mate_rows(data_dir: Path, names=("mate-selection-test.json",
                                      "mate-selection-test-noexplain.json",
                                      "mate-selection-test-tactic.json",
                                      "mate-selection-test-both.json")) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        payload = json.loads((Path(data_dir) / name).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{name}: MATE file must contain a JSON list")
        rows.extend(payload)
    return rows


def summarize(results: list[dict], puzzles: list[dict] | None = None) -> dict:
    total = len(results)
    solved = sum(bool(r["correct"]) for r in results)
    summary = {"solved": solved, "total": total, "pct": 100.0 * solved / max(1, total)}
    if puzzles is not None and len(puzzles) == total:
        bins: dict[int, list[int]] = {}
        for p, r in zip(puzzles, results):
            b = (p.get("rating", 0) // 400) * 400
            bins.setdefault(b, [0, 0])
            bins[b][0] += int(bool(r["correct"]))
            bins[b][1] += 1
        summary["by_rating"] = {f"{b}-{b + 399}": {"solved": s, "total": t}
                                for b, (s, t) in sorted(bins.items())}
        plies = [r["fail_ply"] for r in results if r.get("fail_ply") is not None
                 and not r["correct"]]
        summary["failed_at_first_decision"] = int(sum(1 for p in plies if p == 1))
        summary["failed_later"] = int(sum(1 for p in plies if p != 1))
    return summary
