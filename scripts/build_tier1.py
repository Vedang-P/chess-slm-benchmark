"""Build a Tier-1 (board literacy) corpus shard.

    python scripts/build_tier1.py --n 200000 --out data/corpus/tier1/train-000.jsonl
    python scripts/build_tier1.py --n 5000 --split heldout_phrasing \
        --out data/corpus/tier1/heldout-phrasing.jsonl

Positions come from the Lichess puzzle CSV when available (human-plausible
distribution) and otherwise from random playouts, so the go/no-go gate is
runnable with no downloads.

Leakage control (review item R5): every example records the board-only FEN of
its position, and ``--exclude-fens`` drops any position appearing in an
evaluation set. Splitting by game id is not enough -- Lichess puzzles are
derived from Lichess games, so the same position can reach training and
evaluation by two different routes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chessreasoner import serialize  # noqa: E402
from src.chessreasoner.generators import tier1  # noqa: E402
from src.chessreasoner.tokenizer import SEG_ANSWER, SEG_BOARD, ChessTokenizer  # noqa: E402

DEFAULT_PUZZLE_CSV = Path("data/raw/lichess_db_puzzle.csv")


def load_excluded_fens(paths: list[str]) -> set[str]:
    """Board-only FENs to keep out of the corpus (evaluation positions)."""
    excluded: set[str] = set()
    for path in paths:
        text = Path(path).read_text()
        try:
            payload = json.loads(text)
            records = payload if isinstance(payload, list) else payload.get("records", [])
            fens = [r["fen"] for r in records if isinstance(r, dict) and "fen" in r]
        except json.JSONDecodeError:
            fens = [line.strip() for line in text.splitlines() if line.strip()]
        for fen in fens:
            try:
                excluded.add(chess.Board(fen).board_fen())
            except ValueError:
                continue
    return excluded


def position_source(args, rng: random.Random):
    if args.source == "playout":
        return tier1.random_playout_positions(rng)
    csv_path = Path(args.puzzle_csv)
    if not csv_path.exists():
        raise SystemExit(f"puzzle CSV not found at {csv_path}; use --source playout")
    return tier1.puzzle_csv_positions(str(csv_path), rng)


def filtered(source, excluded: set[str], stats: Counter):
    for board in source:
        if excluded and board.board_fen() in excluded:
            stats["excluded_positions"] += 1
            continue
        yield board


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "heldout_phrasing"], default="train")
    ap.add_argument("--source", choices=["puzzles", "playout"], default="puzzles")
    ap.add_argument("--puzzle-csv", default=str(DEFAULT_PUZZLE_CSV))
    ap.add_argument("--exclude-fens", nargs="*", default=[],
                    help="JSON task sets or plain FEN lists to keep out of the corpus")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pack", type=int, nargs=2, metavar=("MIN", "MAX"), default=None,
                    help="pack MIN..MAX questions onto each board. The 72-token board "
                         "plane is 82%% of an unpacked example, so packing 6-10 questions "
                         "raises the supervised answer share from 8%% to ~32%%.")
    ap.add_argument("--fit-tokenizer", type=Path, default=None,
                    help="also fit the prose BPE on this shard and save it here")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats: Counter = Counter()
    excluded = load_excluded_fens(args.exclude_fens)
    if excluded:
        print(f"excluding {len(excluded)} evaluation positions")

    source = filtered(position_source(args, rng), excluded, stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    prose_corpus: list[str] = []
    fens: set[str] = set()
    n_questions = 0

    def collect_prose(parts):
        if args.fit_tokenizer is not None:
            prose_corpus.extend(
                p for p in parts if not (p.startswith("<") and p.endswith(">")))

    with args.out.open("w") as fh:
        if args.pack:
            stream = tier1.generate_packed(source, args.n, rng, split=args.split,
                                           questions_per_board=tuple(args.pack))
            for packed in stream:
                fen = serialize.parts_to_board(packed.board_parts).board_fen()
                fens.add(fen)
                n_questions += len(packed.qa_pairs)
                for task in packed.tasks:
                    stats[task] += 1
                fh.write(json.dumps({
                    "tier": 1, "split": args.split, "packed": True, "board_fen": fen,
                    "board_parts": packed.board_parts,
                    "qa_pairs": [[q, a] for q, a in packed.qa_pairs],
                    "tasks": packed.tasks, "metas": packed.metas,
                }) + "\n")
                collect_prose(packed.board_parts)
                for question, answer in packed.qa_pairs:
                    collect_prose(list(question) + list(answer))
        else:
            for example in tier1.generate(source, args.n, rng, split=args.split):
                fen = serialize.parts_to_board(
                    example.prompt_parts[:serialize.BOARD_SPAN_LEN]).board_fen()
                fens.add(fen)
                n_questions += 1
                stats[example.task] += 1
                fh.write(json.dumps({
                    "tier": 1, "split": args.split, "packed": False, "board_fen": fen,
                    "task": example.task,
                    "prompt_parts": example.prompt_parts,
                    "answer_parts": example.answer_parts,
                    "meta": example.meta,
                }) + "\n")
                collect_prose(example.prompt_parts + example.answer_parts)

    unit = "packed boards" if args.pack else "examples"
    print(f"\nwrote {args.n} {unit} ({n_questions} questions) to {args.out}")
    print(f"distinct positions: {len(fens)}")
    if stats["excluded_positions"]:
        print(f"positions skipped for evaluation overlap: {stats['excluded_positions']}")
    print("task distribution:")
    for task, count in sorted(stats.items(), key=lambda kv: -kv[1]):
        if task == "excluded_positions":
            continue
        print(f"  {task:16s} {count:8d}  ({count / max(n_questions, 1):5.1%})")

    if args.fit_tokenizer is not None:
        print(f"\nfitting prose BPE on {len(prose_corpus)} prose runs ...")
        tok = ChessTokenizer.fit_prose(prose_corpus)
        tok.save(args.fit_tokenizer)
        print(f"saved tokenizer to {args.fit_tokenizer} (vocab {tok.vocab_size})")

        # token accounting on a sample -- this is what sets the corpus budget
        sample = [json.loads(line) for line in args.out.open().readlines()[:2000]]
        total = board = answer = 0
        for record in sample:
            if record.get("packed"):
                enc = tok.encode_packed(record["board_parts"],
                                        [(q, a) for q, a in record["qa_pairs"]])
            else:
                enc = tok.encode_example(record["prompt_parts"], record["answer_parts"])
            total += len(enc["ids"])
            board += sum(1 for s in enc["segments"] if s == SEG_BOARD)
            answer += sum(1 for s in enc["segments"] if s == SEG_ANSWER)
        print(f"\ntokens/example: {total / len(sample):.1f}"
              f"   board {board / total:.1%}"
              f"   answer {answer / total:.1%}"
              f"   question {(total - board - answer) / total:.1%}")
        print(f"projected shard size: {total / len(sample) * args.n / 1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
