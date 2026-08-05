"""Tier 1 -- board literacy.

Perception drills, no reasoning. Every question is answerable from the board
alone and every answer is computed by python-chess, so the tier is correct by
construction and needs no engine.

This is the tier that targets the measured failure: the Gemma 4 E2B baseline
placed the white king on g2 when it was on h2, invented bishops on empty
squares, and then reasoned confidently about the hallucinated position. No
amount of tactical training fixes a model that cannot read the board.

Two design constraints from the adversarial review are enforced here:

* **R6** -- claims come only from a closed set of machine-checkable predicates
  (occupancy, legality, attacks, checks). Nothing evaluative.
* **R7** -- every template family reserves a held-out phrasing pool that the
  ``train`` split never sees, so template memorization is measurable instead of
  hypothetical.

Phrasing variety lives in the *questions*. Answers stay canonical: Tier 1 is a
perception drill, and the natural-language variety belongs to Tier 3 traces.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import chess

from ..serialize import board_to_parts, move_to_parts
from ..vocab import EMPTY_TOKEN, SQUARE_TO_TOKEN, piece_token

N_HELDOUT_TEMPLATES = 2
"""Templates reserved per family for the held-out phrasing split."""

_PLACEHOLDER = re.compile(r"\{(\w+)\}")

PIECE_WORDS = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}
PIECE_PLURALS = {k: v + "s" for k, v in PIECE_WORDS.items()}
COLOR_WORDS = {chess.WHITE: "white", chess.BLACK: "black"}


@dataclass
class Example:
    task: str
    prompt_parts: list[str]
    answer_parts: list[str]
    meta: dict = field(default_factory=dict)

    def key(self) -> tuple:
        """Dedup key -- task plus the exact prompt."""
        return (self.task, tuple(self.prompt_parts))


def render(template: str, slots: dict[str, list[str]]) -> list[str]:
    """Expand ``{name}`` placeholders into chess-token runs, keeping prose whole.

    ``"what is on {sq} ?"`` with ``{"sq": ["<e4>"]}`` becomes
    ``["what is on ", "<e4>", " ?"]``.
    """
    parts: list[str] = []
    pos = 0
    for match in _PLACEHOLDER.finditer(template):
        prose = template[pos:match.start()]
        if prose:
            parts.append(prose)
        name = match.group(1)
        if name not in slots:
            raise KeyError(f"template {template!r} references unknown slot {name!r}")
        parts.extend(slots[name])
        pos = match.end()
    tail = template[pos:]
    if tail:
        parts.append(tail)
    return parts


def _templates(family: Sequence[str], split: str) -> list[str]:
    if len(family) <= N_HELDOUT_TEMPLATES:
        raise ValueError("template family too small to hold any out")
    if split == "train":
        return list(family[:-N_HELDOUT_TEMPLATES])
    if split == "heldout_phrasing":
        return list(family[-N_HELDOUT_TEMPLATES:])
    raise ValueError(f"unknown split {split!r}")


def _square_list(squares: Sequence[int]) -> list[str]:
    """Sorted square tokens with prose separators, or the literal 'none'."""
    if not squares:
        return ["none"]
    parts: list[str] = []
    for i, sq in enumerate(sorted(squares)):
        if i:
            parts.append(" , ")
        parts.append(SQUARE_TO_TOKEN[sq])
    return parts


# ---------------------------------------------------------------------------
# Template families. Last N_HELDOUT_TEMPLATES of each are never trained on.
# ---------------------------------------------------------------------------

T_PIECE_AT = [
    "what is on {sq} ?",
    "which piece stands on {sq} ?",
    "name the occupant of {sq} .",
    "is anything on {sq} ?",
    "contents of {sq} ?",
    "tell me what occupies {sq} .",
    "report the piece sitting on {sq} .",
]

T_FIND_PIECES = [
    "where are {color} 's {plural} ?",
    "list every {color} {piece} .",
    "find all {plural} belonging to {color} .",
    "on which squares are the {color} {plural} ?",
    "give the squares of {color} 's {plural} .",
    "locate the {color} {plural} .",
    "which squares hold {color} {plural} ?",
]

T_APPLY_MOVE = [
    "after the move {move} , what is on {sq} ?",
    "play {move} . what occupies {sq} then ?",
    "if {move} is played , what stands on {sq} ?",
    "make the move {move} and report the piece on {sq} .",
    "following {move} , name the occupant of {sq} .",
    "once {move} has been played , what is on {sq} ?",
    "apply {move} , then say what sits on {sq} .",
]

# Worded as *destinations*, not moves: a pawn promoting has four legal moves to
# one square, and listing that square four times would teach the model to pad
# its answers. The answer is the deduplicated destination set.
T_LEGAL_MOVES = [
    "list the legal destinations of the piece on {sq} .",
    "where can the piece on {sq} go ?",
    "give every square reachable from {sq} .",
    "which squares can the piece on {sq} reach ?",
    "enumerate the legal destinations from {sq} .",
    "to which squares may the piece standing on {sq} legally move ?",
    "show every landing square available to the piece on {sq} .",
]

T_IN_CHECK = [
    "is the side to move in check ?",
    "is the king of the side to move attacked ?",
    "does the player to move stand in check ?",
    "check status for the side to move ?",
    "is the moving side 's king under attack ?",
    "say whether the side to move is in check .",
    "report whether the king to move is being checked .",
]

T_ATTACKERS = [
    "which {color} pieces attack {sq} ?",
    "list {color} 's attackers of {sq} .",
    "what does {color} have bearing on {sq} ?",
    "give every {color} piece that hits {sq} .",
    "which of {color} 's pieces strike {sq} ?",
    "name the {color} units attacking {sq} .",
    "who among {color} 's pieces covers {sq} ?",
]

T_HANGING = [
    "list {color} 's hanging pieces .",
    "which {color} pieces are attacked and undefended ?",
    "find every undefended {color} piece under attack .",
    "what is {color} leaving en prise ?",
    "name {color} 's pieces that are attacked but not defended .",
    "which {color} units are hanging ?",
    "report the loose {color} pieces .",
]

T_PIECE_LIST = [
    "list every piece on the board .",
    "give the full piece placement .",
    "enumerate all pieces and their squares .",
    "read out the position piece by piece .",
    "what pieces are on the board , and where ?",
    "transcribe the whole position .",
    "state the location of every piece .",
]


# ---------------------------------------------------------------------------
# Task generators. Each returns None when the sampled position cannot pose a
# well-formed instance of that question.
# ---------------------------------------------------------------------------

def gen_piece_at(board: chess.Board, rng: random.Random, split: str) -> Example | None:
    sq = rng.choice(chess.SQUARES)
    piece = board.piece_at(sq)
    tmpl = rng.choice(_templates(T_PIECE_AT, split))
    prompt = board_to_parts(board) + render(tmpl, {"sq": [SQUARE_TO_TOKEN[sq]]})
    answer = [piece_token(piece)] if piece else [EMPTY_TOKEN]
    return Example("piece_at", prompt, answer,
                   {"square": chess.square_name(sq),
                    "piece": piece.symbol() if piece else None})


def gen_find_pieces(board: chess.Board, rng: random.Random, split: str) -> Example | None:
    color = rng.choice([chess.WHITE, chess.BLACK])
    ptype = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                        chess.ROOK, chess.QUEEN, chess.KING])
    squares = sorted(board.pieces(ptype, color))
    tmpl = rng.choice(_templates(T_FIND_PIECES, split))
    prompt = board_to_parts(board) + render(tmpl, {
        "color": [COLOR_WORDS[color]],
        "piece": [PIECE_WORDS[ptype]],
        "plural": [PIECE_PLURALS[ptype]],
    })
    return Example("find_pieces", prompt, _square_list(squares),
                   {"color": COLOR_WORDS[color], "piece_type": PIECE_WORDS[ptype],
                    "squares": [chess.square_name(s) for s in squares]})


def gen_apply_move(board: chess.Board, rng: random.Random, split: str) -> Example | None:
    legal = list(board.legal_moves)
    if not legal:
        return None
    move = rng.choice(legal)
    after = board.copy()
    after.push(move)
    # Ask about a square the move actually touched half the time, so the task is
    # not dominated by trivially-unchanged squares.
    if rng.random() < 0.5:
        sq = rng.choice([move.from_square, move.to_square])
    else:
        sq = rng.choice(chess.SQUARES)
    piece = after.piece_at(sq)
    tmpl = rng.choice(_templates(T_APPLY_MOVE, split))
    prompt = board_to_parts(board) + render(tmpl, {
        "move": move_to_parts(move), "sq": [SQUARE_TO_TOKEN[sq]],
    })
    answer = [piece_token(piece)] if piece else [EMPTY_TOKEN]
    return Example("apply_move", prompt, answer,
                   {"move": move.uci(), "square": chess.square_name(sq),
                    "piece_after": piece.symbol() if piece else None})


def gen_legal_moves_of(board: chess.Board, rng: random.Random, split: str) -> Example | None:
    occupied = [sq for sq in chess.SQUARES
                if (p := board.piece_at(sq)) and p.color == board.turn]
    if not occupied:
        return None
    sq = rng.choice(occupied)
    # set(): the four promotion choices all land on the same square
    dests = sorted({m.to_square for m in board.legal_moves if m.from_square == sq})
    tmpl = rng.choice(_templates(T_LEGAL_MOVES, split))
    prompt = board_to_parts(board) + render(tmpl, {"sq": [SQUARE_TO_TOKEN[sq]]})
    return Example("legal_moves_of", prompt, _square_list(dests),
                   {"from": chess.square_name(sq),
                    "destinations": [chess.square_name(d) for d in dests]})


def gen_in_check(board: chess.Board, rng: random.Random, split: str) -> Example | None:
    tmpl = rng.choice(_templates(T_IN_CHECK, split))
    prompt = board_to_parts(board) + render(tmpl, {})
    checkers = sorted(board.checkers())
    if checkers:
        answer = ["yes , checked from "] + _square_list(checkers)
    else:
        answer = ["no"]
    return Example("in_check", prompt, answer,
                   {"in_check": bool(checkers),
                    "checkers": [chess.square_name(s) for s in checkers]})


def gen_attackers_of(board: chess.Board, rng: random.Random, split: str) -> Example | None:
    color = rng.choice([chess.WHITE, chess.BLACK])
    # Bias toward occupied squares: attacks on empty squares are less useful.
    occupied = list(board.piece_map().keys())
    sq = rng.choice(occupied) if occupied and rng.random() < 0.7 else rng.choice(chess.SQUARES)
    attackers = sorted(board.attackers(color, sq))
    tmpl = rng.choice(_templates(T_ATTACKERS, split))
    prompt = board_to_parts(board) + render(tmpl, {
        "color": [COLOR_WORDS[color]], "sq": [SQUARE_TO_TOKEN[sq]],
    })
    return Example("attackers_of", prompt, _square_list(attackers),
                   {"color": COLOR_WORDS[color], "square": chess.square_name(sq),
                    "attackers": [chess.square_name(s) for s in attackers]})


def gen_hanging(board: chess.Board, rng: random.Random, split: str) -> Example | None:
    color = rng.choice([chess.WHITE, chess.BLACK])
    hanging = sorted(
        sq for sq, piece in board.piece_map().items()
        if piece.color == color
        and piece.piece_type != chess.KING
        and board.attackers(not color, sq)
        and not board.attackers(color, sq)
    )
    tmpl = rng.choice(_templates(T_HANGING, split))
    prompt = board_to_parts(board) + render(tmpl, {"color": [COLOR_WORDS[color]]})
    return Example("hanging", prompt, _square_list(hanging),
                   {"color": COLOR_WORDS[color],
                    "hanging": [chess.square_name(s) for s in hanging]})


def gen_piece_list(board: chess.Board, rng: random.Random, split: str) -> Example | None:
    tmpl = rng.choice(_templates(T_PIECE_LIST, split))
    prompt = board_to_parts(board) + render(tmpl, {})
    answer: list[str] = []
    for i, (sq, piece) in enumerate(sorted(board.piece_map().items())):
        if i:
            answer.append(" , ")
        answer.extend([piece_token(piece), " ", SQUARE_TO_TOKEN[sq]])
    if not answer:
        answer = ["none"]
    return Example("piece_list", prompt, answer,
                   {"n_pieces": len(board.piece_map())})


TASKS: dict[str, Callable[[chess.Board, random.Random, str], Example | None]] = {
    "piece_at": gen_piece_at,
    "find_pieces": gen_find_pieces,
    "apply_move": gen_apply_move,
    "legal_moves_of": gen_legal_moves_of,
    "in_check": gen_in_check,
    "attackers_of": gen_attackers_of,
    "hanging": gen_hanging,
    "piece_list": gen_piece_list,
}

DEFAULT_WEIGHTS = {
    # Pure lookups are learned fast and teach indexing; the computed predicates
    # are where the tier earns its tokens, so they carry more weight.
    "piece_at": 1.0,
    "find_pieces": 1.0,
    "piece_list": 1.0,
    "in_check": 1.0,
    "apply_move": 2.0,
    "legal_moves_of": 2.5,
    "attackers_of": 2.5,
    "hanging": 2.0,
}


# ---------------------------------------------------------------------------
# Position sources
# ---------------------------------------------------------------------------

def random_playout_positions(rng: random.Random, max_plies: int = 80) -> Iterator[chess.Board]:
    """Self-contained position source -- random legal games from the start.

    Needs no downloads, which keeps the go/no-go gate runnable anywhere. The
    distribution is not human-like; use ``puzzle_csv_positions`` for the real
    corpus.
    """
    while True:
        board = chess.Board()
        for _ in range(rng.randint(0, max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        yield board


def puzzle_csv_positions(path: str, rng: random.Random) -> Iterator[chess.Board]:
    """Positions from the Lichess puzzle CSV (CC0). Column 1 is the FEN."""
    import csv

    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header and header[1].strip().lower() != "fen":
            fh.seek(0)
            reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                yield chess.Board(row[1])
            except ValueError:
                continue


@dataclass
class PackedExample:
    """One board, many questions about it.

    The 72-token board plane is 81.6% of a single-question example (measured on
    a 50k shard), so re-emitting it per question spends most of the compute
    budget re-reading the same position. Amortizing one board over 8 questions
    lifts the answer share from 8.0% to 28.0% -- 3.5x more supervised signal for
    the same FLOPs -- and additionally forces the model to hold the board across
    a long span rather than answering off the most recent tokens.
    """
    board_parts: list[str]
    qa_pairs: list[tuple[list[str], list[str]]]
    tasks: list[str]
    metas: list[dict]


def generate_packed(positions: Iterator[chess.Board], n_boards: int, rng: random.Random,
                    split: str = "train", questions_per_board: tuple[int, int] = (6, 10),
                    weights: dict[str, float] | None = None,
                    seen: set | None = None) -> Iterator[PackedExample]:
    """Yield ``n_boards`` packed examples, each several questions on one board."""
    weights = weights or DEFAULT_WEIGHTS
    names = list(weights.keys())
    probs = [weights[k] for k in names]
    seen = seen if seen is not None else set()
    produced = 0
    attempts = 0
    budget = max(n_boards * 20, 1000)
    lo, hi = questions_per_board
    while produced < n_boards and attempts < budget:
        attempts += 1
        board = next(positions)
        board_parts = board_to_parts(board)
        want = rng.randint(lo, hi)
        pairs: list[tuple[list[str], list[str]]] = []
        tasks: list[str] = []
        metas: list[dict] = []
        for task in rng.choices(names, weights=probs, k=want * 3):
            if len(pairs) >= want:
                break
            example = TASKS[task](board, rng, split)
            if example is None:
                continue
            question = example.prompt_parts[len(board_parts):]
            key = (task, tuple(question))
            if key in {(t, tuple(q)) for t, (q, _) in zip(tasks, pairs)}:
                continue  # same question twice about the same board
            if example.key() in seen:
                continue
            seen.add(example.key())
            pairs.append((question, example.answer_parts))
            tasks.append(task)
            metas.append(example.meta)
        if not pairs:
            continue
        produced += 1
        yield PackedExample(board_parts, pairs, tasks, metas)
    if produced < n_boards:
        raise RuntimeError(
            f"only produced {produced}/{n_boards} packed examples in {attempts} attempts"
        )


def generate(positions: Iterator[chess.Board], n: int, rng: random.Random,
             split: str = "train", weights: dict[str, float] | None = None,
             seen: set | None = None) -> Iterator[Example]:
    """Yield ``n`` deduplicated Tier-1 examples drawn from ``positions``."""
    weights = weights or DEFAULT_WEIGHTS
    names = list(weights.keys())
    probs = [weights[k] for k in names]
    seen = seen if seen is not None else set()
    produced = 0
    attempts = 0
    budget = max(n * 20, 1000)
    while produced < n and attempts < budget:
        attempts += 1
        board = next(positions)
        task = rng.choices(names, weights=probs, k=1)[0]
        example = TASKS[task](board, rng, split)
        if example is None:
            continue
        key = example.key()
        if key in seen:
            continue
        seen.add(key)
        produced += 1
        yield example
    if produced < n:
        raise RuntimeError(
            f"only produced {produced}/{n} unique examples in {attempts} attempts; "
            "the position source is probably too small"
        )
