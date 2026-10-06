"""Evaluate a GAVN / CC-GAVN checkpoint on MATE and the official puzzle protocol.

Exact-set semantics (2026-10-06 review fixes):
  - every requested MATE row and puzzle is scored in full; a malformed row or
    a wrong total aborts the run with a nonzero exit code instead of silently
    reporting accuracy over fewer rows,
  - per-example records are appended to ``<examples-out>/mate.jsonl`` and
    ``<examples-out>/puzzles.jsonl`` so a killed session resumes instead of
    rescoring, and so downstream analysis can cluster by position,
  - ``--summary-out`` writes a machine-readable summary the eval kernels gate
    DONE on (returncode, exact totals, completion).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_gavn import GAVN, action_tables, relation_types  # noqa: E402


_CHARS = {c: i for i, c in enumerate([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'p', 'n', 'r', 'k', 'q', 'P', 'B', 'N', 'R', 'Q',
    'K', 'w', '.'])}


def tokenize_fen(fen: str) -> np.ndarray:
    """Dependency-free equivalent of the official 77-token tokenizer."""
    board, side, castling, en_passant, halfmoves, fullmoves = fen.split(' ')
    chars = [side] + list(board.replace('/', ''))
    expanded = []
    for char in chars:
        if char in '12345678':
            expanded.extend(['.'] * int(char))
        else:
            expanded.append(char)
    expanded += ['.'] * 4 if castling == '-' else list(castling) + ['.'] * (4 - len(castling))
    expanded += ['.', '.'] if en_passant == '-' else list(en_passant)
    expanded += list((halfmoves + '...')[:3])
    expanded += list((fullmoves + '...')[:3])
    if len(expanded) != 77:
        raise ValueError(f"tokenizer produced {len(expanded)} tokens for {fen}")
    return np.asarray([_CHARS[x] for x in expanded], dtype=np.int64)


def position_key(fen: str) -> str:
    """Exact piece-placement + side + castling + ep: the clustering key."""
    return " ".join(fen.split()[:4])


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(cp: Path, cfg: dict, architecture: str,
                        model_version: str, score_mode: str) -> dict:
    """Everything that determines which checkpoint produced a score."""
    return {
        "checkpoint": cp.name,
        "state_sha256": _sha256_file(cp / "state.pt"),
        "architecture": architecture,
        "model_version": model_version,
        "score": score_mode,
        "source_commit": cfg.get("source_commit"),
        "dim": cfg.get("dim"), "layers": cfg.get("layers"), "heads": cfg.get("heads"),
    }


def dataset_identity(mate_files: list[Path], puzzles: Path | None) -> dict:
    """Content identity of the evaluation sets, in scoring order."""
    mate = hashlib.sha256()
    for path in mate_files:
        mate.update(path.name.encode())
        mate.update(b"\0")
        mate.update(path.read_bytes())
    identity = {"mate_files": [p.name for p in mate_files],
                "mate_sha256": mate.hexdigest(), "puzzles": None}
    if puzzles is not None:
        identity["puzzles"] = {"name": puzzles.name,
                               "sha256": _sha256_file(puzzles)}
    return identity


def validate_examples_identity(examples_dir: Path, identity: dict) -> None:
    """Refuse to append to per-example files that belong to a different
    checkpoint, dataset, or scoring mode. The 2026-10-06 review found the eval
    kernel reused one examples dir across checkpoints; stale rows were then
    skipped by key and the old scores could be re-published under the new
    checkpoint."""
    examples_dir.mkdir(parents=True, exist_ok=True)
    marker = examples_dir / "identity.json"
    existing_rows = [p for name in ("mate.jsonl", "puzzles.jsonl")
                     if (p := examples_dir / name).exists() and p.stat().st_size]
    if marker.exists():
        stored = json.loads(marker.read_text(encoding="utf-8"))
        if stored != identity:
            raise ValueError(
                "identity mismatch for saved eval examples: refusing to reuse "
                "rows scored for a different checkpoint, dataset, or score mode")
    elif existing_rows:
        raise ValueError(
            "saved eval examples exist without identity.json; refusing to reuse")
    marker.write_text(json.dumps(identity, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _row_candidates(row: dict) -> tuple[str, str, str]:
    extra = row.get("task_extra") or {}
    return (row.get("candidate_a") or row.get("move_a") or extra.get("candidate_a"),
            row.get("candidate_b") or row.get("move_b") or extra.get("candidate_b"),
            row.get("truth_label") or row.get("label") or extra.get("truth_label"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="checkpoint-N directory")
    p.add_argument("--sl-repo", default=os.environ.get("SL_REPO", "/kaggle/working/searchless_chess"))
    p.add_argument("--eval", default="", help="comma-separated MATE JSON files")
    p.add_argument("--max-rows", type=int, default=0,
                   help="smoke only: score a prefix and mark the run partial")
    p.add_argument("--puzzles", default="", help="official puzzles.csv path")
    p.add_argument("--num-puzzles", type=int, default=10000)
    p.add_argument("--score", choices=["auto", "q", "dist"], default="auto",
                   help="Decision score. auto/dist uses the trained return\n"
                        "distribution expectation; q is only a diagnostic\n"
                        "for checkpoints trained with a nonzero --w-q.")
    p.add_argument("--examples-out", default="",
                   help="directory for resumable per-example mate.jsonl/puzzles.jsonl")
    p.add_argument("--summary-out", default="",
                   help="write the machine-readable summary JSON here")
    p.add_argument("--expect-mate-rows", type=int, default=4000)
    return p.parse_args()


def main() -> int:
    import chess
    import chess.pgn
    import pandas as pd
    import torch

    args = parse_args()
    cp = Path(args.checkpoint)
    cfg = json.loads((cp / "config.json").read_text(encoding="utf-8"))
    src, dst, promo, _ = action_tables(Path(args.sl_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_version = str(cfg.get("model_version", "v1"))
    if cfg.get("architecture") == "cc-gavn-v1":
        from scripts.train_ccgavn import CCGAVN, candidate_relation_types
        model = CCGAVN(torch, int(cfg["dim"]), int(cfg["layers"]), int(cfg["heads"]),
                       src, dst, promo, candidate_relation_types(),
                       model_version=model_version).to(device)
        architecture = "cc-gavn-v1"
    else:
        legacy_relations = (cfg.get("relation_schema") is None
                            or str(cfg.get("relation_schema")).startswith("legacy"))
        model = GAVN(torch, int(cfg["dim"]), int(cfg["layers"]), int(cfg["heads"]),
                     src, dst, promo, relation_types(legacy=legacy_relations),
                     bias_mode=cfg.get("bias_mode", "both")).to(device)
        architecture = "gavn"
    state = torch.load(cp / "state.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    sys.path.insert(0, str(Path(args.sl_repo).parent))
    from searchless_chess.src import utils  # type: ignore
    from searchless_chess.src.engines import engine as engine_lib  # type: ignore
    bucket_values = torch.as_tensor(
        np.asarray(utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32),
        device=device)

    score_mode = "dist" if args.score == "auto" else args.score
    if score_mode == "q" and not cfg.get("q_head_trained", cfg.get("w_q", 0.0)):
        raise ValueError(
            "This checkpoint's q_head was not trained (w_q=0). "
            "Use --score dist, which is the canonical action-value output.")
    if architecture == "cc-gavn-v1" and score_mode == "q":
        raise ValueError("CC-GAVN has no scalar q_head; use --score dist")
    print(f"[gavn] architecture={architecture} model={model_version} "
          f"decision score={score_mode}", flush=True)

    def scores(board):
        # The official engine specifies its own stable ordering.  Ordering only
        # affects ties, but exact puzzle sequences make those ties observable.
        moves = engine_lib.get_ordered_legal_moves(board)
        action_ids = [utils.MOVE_TO_ACTION[m.uci()] for m in moves]
        tokens = np.repeat(tokenize_fen(board.fen())[None, :], len(moves), axis=0)
        with torch.inference_mode():
            outputs = model(torch.as_tensor(tokens, dtype=torch.long, device=device),
                            torch.as_tensor(action_ids, dtype=torch.long, device=device))
            if architecture == "cc-gavn-v1":
                logits, q = outputs, None
            else:
                logits, q = outputs
            if score_mode == "q":
                values = q
            else:
                values = torch.softmax(logits, -1) @ bucket_values
        values = values.detach().cpu().numpy()
        # Match ActionValueEngine's rule-based score for claimable repetitions.
        for i, move in enumerate(moves):
            board.push(move)
            if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                values[i] = 0.5
            board.pop()
        return moves, values

    examples_dir = Path(args.examples_out) if args.examples_out else None
    if examples_dir is not None:
        mate_files = ([Path(f.strip()) for f in args.eval.split(",")]
                      if args.eval else [])
        validate_examples_identity(examples_dir, {
            "checkpoint": checkpoint_identity(cp, cfg, architecture,
                                              model_version, score_mode),
            "dataset": dataset_identity(
                mate_files, Path(args.puzzles) if args.puzzles else None),
        })
    mate_jsonl = examples_dir / "mate.jsonl" if examples_dir else None
    puz_jsonl = examples_dir / "puzzles.jsonl" if examples_dir else None
    summary = {"checkpoint": str(cp), "architecture": architecture,
               "model_version": model_version, "score": score_mode,
               "returncode": 1, "complete": False, "partial": bool(args.max_rows),
               "mate": None, "puzzles": None, "errors": []}

    if args.eval:
        done = {(r["file"], r["row"]) for r in _load_jsonl(mate_jsonl)} if mate_jsonl else set()
        files = [Path(f.strip()) for f in args.eval.split(",")]
        rows = []
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f"MATE file missing: {f}")
            payload = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"{f}: MATE file must contain a JSON list")
            rows.extend((str(f), i, row) for i, row in enumerate(payload))
        if args.max_rows:
            rows = rows[:args.max_rows]
        elif len(rows) != args.expect_mate_rows:
            raise ValueError(f"MATE rows {len(rows)} != required {args.expect_mate_rows}")
        scored_correct = scored_total = 0
        for file_name, i, row in rows:
            if (file_name, i) in done:
                continue
            fen = row.get("fen") or row.get("position")
            ca, cb, truth = _row_candidates(row)
            if not (fen and ca and cb and truth in ("A", "B")):
                raise ValueError(f"{file_name} row {i}: malformed MATE row")
            board = chess.Board(fen)
            moves, values = scores(board)
            by_uci = {m.uci(): float(v) for m, v in zip(moves, values)}
            if ca not in by_uci or cb not in by_uci:
                raise ValueError(f"{file_name} row {i}: candidate not legal in {fen}")
            pred = "A" if by_uci[ca] > by_uci[cb] else "B"
            record = {"file": file_name, "row": i, "position": position_key(fen),
                      "truth": truth, "pred": pred, "correct": pred == truth,
                      "score_a": by_uci[ca], "score_b": by_uci[cb],
                      "candidate_a": ca, "candidate_b": cb}
            if mate_jsonl:
                _append_jsonl(mate_jsonl, record)
            scored_correct += int(record["correct"])
            scored_total += 1
            done.add((file_name, i))
        if mate_jsonl:
            records = _load_jsonl(mate_jsonl)
            correct = sum(bool(r["correct"]) for r in records)
            total = len(records)
        else:
            records = []
            correct, total = scored_correct, scored_total
        expected = min(args.expect_mate_rows, len(rows) if args.max_rows else args.expect_mate_rows)
        summary["mate"] = {"correct": correct, "total": total, "expected": expected,
                           "unique_positions": len({r["position"] for r in records})}
        print(f"[gavn] MATE: {correct}/{total} = {100*correct/max(1,total):.2f}% "
              f"(unique positions {summary['mate']['unique_positions']})", flush=True)
        if total != expected:
            summary["errors"].append(f"MATE scored {total} != expected {expected}")
            _finish(summary, args)
            return 1

    if args.puzzles:
        puz = pd.read_csv(args.puzzles, nrows=args.num_puzzles)
        if not args.max_rows and len(puz) != args.num_puzzles:
            raise ValueError(f"puzzles.csv has {len(puz)} rows, expected {args.num_puzzles}")
        done = {r["row"] for r in _load_jsonl(puz_jsonl)} if puz_jsonl else set()
        solved_new = total_new = 0
        for i, puzzle in puz.iterrows():
            if i in done:
                continue
            game = chess.pgn.read_game(io.StringIO(puzzle["PGN"]))
            if game is None or game.errors:
                raise ValueError(f"puzzle row {i}: invalid PGN")
            board = game.end().board()
            moves_uci = str(puzzle["Moves"]).split()
            ok = True
            for j, uci in enumerate(moves_uci):
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise ValueError(f"puzzle row {i}: illegal official move {uci}")
                if j % 2 == 1:
                    legal, values = scores(board)
                    predicted = legal[int(np.argmax(values))].uci()
                    if predicted != uci:
                        board.push(chess.Move.from_uci(predicted))
                        ok = board.is_checkmate()
                        break
                board.push(move)
            if puz_jsonl:
                _append_jsonl(puz_jsonl, {"row": int(i), "correct": bool(ok),
                                          "puzzle_id": str(puzzle.get("PuzzleId", i))})
            solved_new += int(ok)
            total_new += 1
            done.add(i)
            if (int(i) + 1) % 100 == 0:
                print(f"[gavn] puzzle progress row={int(i)+1} solved={solved_new}", flush=True)
        if puz_jsonl:
            records = _load_jsonl(puz_jsonl)
            solved = sum(bool(r["correct"]) for r in records)
            total = len(records)
        else:
            solved, total = solved_new, total_new
        expected = min(args.num_puzzles, len(puz) if args.max_rows else args.num_puzzles)
        summary["puzzles"] = {"solved": solved, "total": total, "expected": expected}
        print(f"[gavn] puzzles: {solved}/{total} = {100*solved/max(1,total):.2f}%", flush=True)
        if total != expected:
            summary["errors"].append(f"puzzles scored {total} != expected {expected}")
            _finish(summary, args)
            return 1

    if args.max_rows:
        summary["errors"].append("partial run (--max-rows) is not a complete evaluation")
        _finish(summary, args)
        return 0
    if not args.eval and not args.puzzles:
        summary["errors"].append("nothing requested")
        _finish(summary, args)
        return 1
    summary["returncode"] = 0
    summary["complete"] = True
    _finish(summary, args)
    return 0


def _finish(summary: dict, args) -> None:
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[gavn] SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    sys.exit(main())
