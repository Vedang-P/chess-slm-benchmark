"""Correctness gate for the ChessReasoner tokenizer and Tier-1 generator.

Run: ``python scripts/test_chessreasoner.py``

The round-trip and predicate tests are the ones that matter -- a silent
serialization bug would poison the entire corpus and only show up as a model
that mysteriously will not learn.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chessreasoner import serialize, vocab  # noqa: E402
from src.chessreasoner.generators import tier1  # noqa: E402
from src.chessreasoner.tokenizer import (  # noqa: E402
    SEG_ANSWER, SEG_BOARD, SEG_PROMPT, ChessTokenizer,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def sample_boards(n: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    out = []
    src = tier1.random_playout_positions(rng)
    for _ in range(n):
        out.append(next(src))
    return out


# ---------------------------------------------------------------------------
print("\n== vocabulary ==")

check("chess surface forms are unique",
      len(set(vocab.CHESS_TOKENS)) == len(vocab.CHESS_TOKENS))
check("64 square tokens in a1->h8 order",
      vocab.SQUARE_TOKENS[0] == "<a1>" and vocab.SQUARE_TOKENS[63] == "<h8>"
      and vocab.SQUARE_TOKENS[chess.E4] == "<e4>")
check("13 board-content classes for the board head",
      len(vocab.BOARD_CONTENT_TOKENS) == 13)
check("castling flags do not collide with piece tokens",
      vocab.YES not in vocab.PIECE_TOKENS and vocab.NO not in vocab.PIECE_TOKENS)
check("move vocabulary is derived and has the expected size",
      vocab.MOVE_VOCAB_SIZE == 1968, f"got {vocab.MOVE_VOCAB_SIZE}")
check(f"chess vocabulary fits well under 8192 (is {vocab.CHESS_VOCAB_SIZE})",
      vocab.CHESS_VOCAB_SIZE < 200)

# every legal move of a real game must be indexable by the policy head
rng = random.Random(1)
missing = []
for board in sample_boards(200, seed=3):
    for move in board.legal_moves:
        try:
            vocab.policy_index(move)
        except KeyError:
            missing.append(move.uci())
check("every legal move maps into the policy head", not missing, str(missing[:5]))


# ---------------------------------------------------------------------------
print("\n== board serialization ==")

boards = sample_boards(400, seed=7)
check("board span is exactly 72 tokens",
      all(len(serialize.board_to_parts(b)) == serialize.BOARD_SPAN_LEN for b in boards))

roundtrip_bad = []
for b in boards:
    back = serialize.parts_to_board(serialize.board_to_parts(b))
    # halfmove/fullmove clocks are deliberately not carried in the span
    if back.board_fen() != b.board_fen() or back.turn != b.turn \
            or back.castling_rights != b.castling_rights or back.ep_square != b.ep_square:
        roundtrip_bad.append(b.fen())
check("board round-trips (occupancy, stm, castling, ep)",
      not roundtrip_bad, f"{len(roundtrip_bad)} failed, e.g. {roundtrip_bad[:1]}")

legal_bad = []
for b in boards:
    back = serialize.parts_to_board(serialize.board_to_parts(b))
    if set(m.uci() for m in back.legal_moves) != set(m.uci() for m in b.legal_moves):
        legal_bad.append(b.fen())
check("legal-move set survives the round-trip", not legal_bad,
      f"{len(legal_bad)} failed, e.g. {legal_bad[:1]}")

# positions that actually exercise castling and en passant
special = [
    chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"),
    chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
    chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1"),
]
check("castling / en-passant / bare-kings round-trip",
      all(serialize.parts_to_board(serialize.board_to_parts(b)).fen().rsplit(" ", 2)[0]
          == b.fen().rsplit(" ", 2)[0] for b in special))

promo = chess.Move.from_uci("a7a8q")
check("promotion move round-trips",
      serialize.parts_to_move(serialize.move_to_parts(promo)) == promo)
check("wrapped move round-trips",
      serialize.parts_to_move(serialize.move_to_parts(promo, wrap=True)) == promo)
check("a move is 2 tokens, never confusable with one square",
      serialize.move_to_parts(chess.Move.from_uci("f3g5")) == ["<f3>", "<g5>"])


# ---------------------------------------------------------------------------
print("\n== tier-1 generator ==")

rng = random.Random(11)
examples = list(tier1.generate(tier1.random_playout_positions(rng), 4000, rng, split="train"))
check("generated the requested number of examples", len(examples) == 4000)
check("all examples are unique", len({e.key() for e in examples}) == len(examples))

by_task: dict[str, int] = {}
for e in examples:
    by_task[e.task] = by_task.get(e.task, 0) + 1
check("every task family is represented",
      set(by_task) == set(tier1.TASKS), f"missing {set(tier1.TASKS) - set(by_task)}")
print("        distribution:", dict(sorted(by_task.items(), key=lambda kv: -kv[1])))

check("prose never contains angle brackets",
      not [p for e in examples for p in e.prompt_parts + e.answer_parts
           if p not in vocab.CHESS_TOKEN_TO_ID and ("<" in p or ">" in p)])
check("every prompt begins with a board span",
      all(e.prompt_parts[:1] == [vocab.FEN_BEGIN] for e in examples))

# --- the important one: are the ANSWERS actually true? ---
print("\n  verifying answers against python-chess:")


def squares_in(parts: list[str]) -> list[str]:
    return [chess.square_name(vocab.TOKEN_TO_SQUARE[p])
            for p in parts if p in vocab.TOKEN_TO_SQUARE]


def by_index(squares) -> list[str]:
    """Canonical order is square *index* (a1,b1,..,h1,a2,..), matching the
    board raster -- not alphabetical square names, which would interleave files
    and ranks."""
    return [chess.square_name(s) for s in sorted(squares)]


wrong: dict[str, int] = {}
for e in examples:
    board = serialize.parts_to_board(e.prompt_parts[:serialize.BOARD_SPAN_LEN])
    ok = True
    if e.task == "piece_at":
        piece = board.piece_at(chess.parse_square(e.meta["square"]))
        ok = e.answer_parts == [vocab.piece_token(piece)]
    elif e.task == "find_pieces":
        ok = squares_in(e.answer_parts) == e.meta["squares"]
    elif e.task == "apply_move":
        after = board.copy()
        after.push(chess.Move.from_uci(e.meta["move"]))
        ok = e.answer_parts == [vocab.piece_token(
            after.piece_at(chess.parse_square(e.meta["square"])))]
    elif e.task == "legal_moves_of":
        want = by_index({m.to_square for m in board.legal_moves
                         if m.from_square == chess.parse_square(e.meta["from"])})
        ok = squares_in(e.answer_parts) == want
    elif e.task == "in_check":
        ok = (bool(board.checkers()) == e.meta["in_check"]
              and squares_in(e.answer_parts) == by_index(board.checkers()))
    elif e.task == "attackers_of":
        color = chess.WHITE if e.meta["color"] == "white" else chess.BLACK
        want = by_index(board.attackers(color, chess.parse_square(e.meta["square"])))
        ok = squares_in(e.answer_parts) == want
    elif e.task == "hanging":
        color = chess.WHITE if e.meta["color"] == "white" else chess.BLACK
        want = by_index(sq for sq, p in board.piece_map().items()
                        if p.color == color and p.piece_type != chess.KING
                        and board.attackers(not color, sq)
                        and not board.attackers(color, sq))
        ok = squares_in(e.answer_parts) == want
    elif e.task == "piece_list":
        ok = len(squares_in(e.answer_parts)) == e.meta["n_pieces"]
    if not ok:
        wrong[e.task] = wrong.get(e.task, 0) + 1

check("every generated answer verifies against python-chess", not wrong, str(wrong))
check("destination lists contain no duplicates (promotions collapse to one square)",
      all(len(set(squares_in(e.answer_parts))) == len(squares_in(e.answer_parts))
          for e in examples if e.task == "legal_moves_of"))

# --- R7: held-out phrasing pool is genuinely disjoint ---
train_t = {t for fam in (tier1.T_PIECE_AT, tier1.T_FIND_PIECES, tier1.T_APPLY_MOVE,
                         tier1.T_LEGAL_MOVES, tier1.T_IN_CHECK, tier1.T_ATTACKERS,
                         tier1.T_HANGING, tier1.T_PIECE_LIST)
           for t in tier1._templates(fam, "train")}
held_t = {t for fam in (tier1.T_PIECE_AT, tier1.T_FIND_PIECES, tier1.T_APPLY_MOVE,
                        tier1.T_LEGAL_MOVES, tier1.T_IN_CHECK, tier1.T_ATTACKERS,
                        tier1.T_HANGING, tier1.T_PIECE_LIST)
          for t in tier1._templates(fam, "heldout_phrasing")}
check("train and held-out phrasings are disjoint", not (train_t & held_t))
check("held-out pool is non-empty", len(held_t) == 8 * tier1.N_HELDOUT_TEMPLATES)

rng2 = random.Random(11)
held = list(tier1.generate(tier1.random_playout_positions(rng2), 300, rng2,
                           split="heldout_phrasing"))
held_prose = {p for e in held for p in e.prompt_parts
              if p not in vocab.CHESS_TOKEN_TO_ID}
train_prose = {p for e in examples for p in e.prompt_parts
               if p not in vocab.CHESS_TOKEN_TO_ID}
leaked = held_prose & train_prose
# separators and answer scaffolding are shared by design; only question stems matter
leaked = {s for s in leaked if len(s.split()) > 2}
check("held-out question stems never appear in the train split", not leaked,
      str(sorted(leaked)[:3]))


# ---------------------------------------------------------------------------
print("\n== tokenizer ==")

prose_corpus = [p for e in examples for p in e.prompt_parts + e.answer_parts
                if p not in vocab.CHESS_TOKEN_TO_ID]
tok = ChessTokenizer.fit_prose(prose_corpus, vocab_size=8192 - vocab.CHESS_VOCAB_SIZE)
check(f"total vocabulary is within 8192 (is {tok.vocab_size})", tok.vocab_size <= 8192)

ex = examples[0]
enc = tok.encode_example(ex.prompt_parts, ex.answer_parts)
check("ids and segments are the same length", len(enc["ids"]) == len(enc["segments"]))
check("chess ids stay below the prose offset",
      all(i < vocab.CHESS_VOCAB_SIZE for i, p in zip(tok.encode(ex.prompt_parts[:72]),
                                                     ex.prompt_parts[:72])))

board_ids = [i for i, s in zip(enc["ids"], enc["segments"]) if s == SEG_BOARD]
check("board span occupies exactly 72 tokens of the encoding",
      len(board_ids) == 72, f"got {len(board_ids)}")
check("all three segments are present",
      set(enc["segments"]) == {SEG_BOARD, SEG_PROMPT, SEG_ANSWER})

weights = tok.loss_weights(enc["segments"])
check("prompt prose is unsupervised, answers are fully supervised",
      all(w == 0.0 for w, s in zip(weights, enc["segments"]) if s == SEG_PROMPT)
      and all(w == 1.0 for w, s in zip(weights, enc["segments"]) if s == SEG_ANSWER))

sq_ids = tok.encode(["<e4>", "<f3>"])
check("square tokens survive as single atomic ids", len(sq_ids) == 2)
check("a move is 2 ids, not a subword sequence",
      len(tok.encode(serialize.move_to_parts(chess.Move.from_uci("f3g5")))) == 2)
check("decode recovers chess tokens verbatim",
      "<e4>" in tok.decode(sq_ids) and "<f3>" in tok.decode(sq_ids))

try:
    tok.encode(["this has <angle> brackets"])
    check("prose with markup is rejected", False)
except ValueError:
    check("prose with markup is rejected", True)

import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as d:
    tok.save(d)
    tok2 = ChessTokenizer.load(d)
    check("tokenizer round-trips through save/load",
          tok2.encode(ex.prompt_parts) == tok.encode(ex.prompt_parts))

lengths = [len(tok.encode_example(e.prompt_parts, e.answer_parts)["ids"]) for e in examples]
print(f"\n        example length: mean {sum(lengths)/len(lengths):.1f} tokens, "
      f"max {max(lengths)}, min {min(lengths)}")
supervised = sum(
    sum(1 for s in tok.encode_example(e.prompt_parts, e.answer_parts)["segments"]
        if s == SEG_ANSWER) for e in examples[:500])
total = sum(len(tok.encode_example(e.prompt_parts, e.answer_parts)["ids"])
            for e in examples[:500])
print(f"        answer tokens are {supervised/total:.1%} of the stream "
      f"(the rest is board plane + question)")


# ---------------------------------------------------------------------------
print("\n== packed examples ==")

rng3 = random.Random(23)
packed = list(tier1.generate_packed(tier1.random_playout_positions(rng3), 400, rng3,
                                    split="train", questions_per_board=(6, 10)))
check("produced the requested number of packed boards", len(packed) == 400)
check("each packed example carries one board span",
      all(len(pk.board_parts) == serialize.BOARD_SPAN_LEN for pk in packed))
check("question count stays within the requested range",
      all(1 <= len(pk.qa_pairs) <= 10 for pk in packed))
check("tasks/metas line up with qa pairs",
      all(len(pk.tasks) == len(pk.qa_pairs) == len(pk.metas) for pk in packed))
check("no question is repeated within a board",
      all(len({(t, tuple(q)) for t, (q, _) in zip(pk.tasks, pk.qa_pairs)}) == len(pk.qa_pairs)
          for pk in packed))
check("packed questions carry no board span of their own",
      not [q for pk in packed for q, _ in pk.qa_pairs if vocab.FEN_BEGIN in q])

# answers must still verify -- packing must not corrupt the board/answer pairing
pack_wrong = 0
for pk in packed:
    b = serialize.parts_to_board(pk.board_parts)
    for task, meta, (_q, ans) in zip(pk.tasks, pk.metas, pk.qa_pairs):
        if task == "piece_at":
            if ans != [vocab.piece_token(b.piece_at(chess.parse_square(meta["square"])))]:
                pack_wrong += 1
        elif task == "attackers_of":
            color = chess.WHITE if meta["color"] == "white" else chess.BLACK
            if squares_in(ans) != by_index(b.attackers(color, chess.parse_square(meta["square"]))):
                pack_wrong += 1
        elif task == "legal_moves_of":
            want = by_index({m.to_square for m in b.legal_moves
                             if m.from_square == chess.parse_square(meta["from"])})
            if squares_in(ans) != want:
                pack_wrong += 1
check("packed answers still verify against python-chess", pack_wrong == 0, str(pack_wrong))

enc_p = tok.encode_packed(packed[0].board_parts, packed[0].qa_pairs)
check("packed ids and segments align", len(enc_p["ids"]) == len(enc_p["segments"]))
check("packed example holds exactly one 72-token board plane",
      sum(1 for s in enc_p["segments"] if s == SEG_BOARD) == 72)

tot = sum(len(tok.encode_packed(pk.board_parts, pk.qa_pairs)["ids"]) for pk in packed)
ans = sum(sum(1 for s in tok.encode_packed(pk.board_parts, pk.qa_pairs)["segments"]
              if s == SEG_ANSWER) for pk in packed)
share = ans / tot
check(f"packing lifts the answer share above 20% (is {share:.1%})", share > 0.20)
print(f"        packed: {tot/len(packed):.0f} tokens/example, answer share {share:.1%}")


# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("All checks passed.")
