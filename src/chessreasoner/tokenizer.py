"""ChessReasoner tokenizer: atomic chess ids + BPE prose, in one id space.

Ids ``[0, CHESS_VOCAB_SIZE)`` are the fixed chess vocabulary; prose BPE ids are
offset above it. The two spaces cannot collide because every chess surface form
is angle-bracketed and the generators guarantee prose contains no ``<``.

Encoding input is a flat ``list[str]`` where each element is *either* an exact
chess token (``"<e4>"``) or a run of free prose (``"the knight on"``). That
keeps generator code readable while leaving tokenization unambiguous.

Every encoded example also carries per-token **segment ids** so the trainer can
weight the loss. This is load-bearing: 57% of the 64 board-plane tokens are the
empty symbol (measured over 3000 random-playout positions), so an unweighted LM
loss spends the majority of its gradient on near-zero-information tokens.
Masking the plane entirely is also wrong -- predicting it teaches a prior over
plausible piece configurations. Hence a weight, not a mask.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .vocab import (
    BOS,
    CHESS_TOKEN_TO_ID,
    CHESS_TOKENS,
    CHESS_VOCAB_SIZE,
    EOS,
    FEN_BEGIN,
    FEN_END,
    PAD,
    UNK,
)

SEG_BOARD = 0
SEG_PROMPT = 1
SEG_ANSWER = 2
SEG_PAD = 3

DEFAULT_SEGMENT_WEIGHTS = {SEG_BOARD: 0.1, SEG_PROMPT: 0.0, SEG_ANSWER: 1.0, SEG_PAD: 0.0}
"""Board planes in the prompt are cheap context; prompt prose is not supervised;
answers carry the signal. Tunable -- A/B testing these is one short run."""


class ChessTokenizer:
    """Hybrid tokenizer. Call :meth:`fit_prose` once, then encode."""

    def __init__(self, prose_tokenizer=None, prose_vocab_size: int = 0):
        self.prose = prose_tokenizer
        self.prose_vocab_size = prose_vocab_size

    # -- construction ------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return CHESS_VOCAB_SIZE + self.prose_vocab_size

    @classmethod
    def fit_prose(cls, texts: Iterable[str], vocab_size: int = 8192 - CHESS_VOCAB_SIZE,
                  min_frequency: int = 2) -> "ChessTokenizer":
        """Train the prose BPE on generated corpus text (chess tokens excluded)."""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

        tok = Tokenizer(models.BPE(unk_token=UNK))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[UNK],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(prose_tokenizer=tok, prose_vocab_size=tok.get_vocab_size())

    # -- encoding ----------------------------------------------------------

    def _encode_part(self, part: str) -> list[int]:
        chess_id = CHESS_TOKEN_TO_ID.get(part)
        if chess_id is not None:
            return [chess_id]
        if "<" in part or ">" in part:
            raise ValueError(
                f"prose may not contain angle brackets (got {part!r}); "
                "either it is a typo'd chess token or the generator leaked markup"
            )
        if self.prose is None:
            raise RuntimeError("prose tokenizer not fitted; call fit_prose() or load()")
        text = part.strip()
        if not text:
            return []
        return [i + CHESS_VOCAB_SIZE for i in self.prose.encode(text).ids]

    def encode(self, parts: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for part in parts:
            ids.extend(self._encode_part(part))
        return ids

    def encode_example(self, prompt_parts: Sequence[str], answer_parts: Sequence[str],
                       add_bos: bool = True, add_eos: bool = True) -> dict:
        """Encode a prompt/answer pair with per-token segment ids.

        Board spans **in the prompt** are ``SEG_BOARD``; everything else in the
        prompt is ``SEG_PROMPT``; the whole answer is ``SEG_ANSWER`` -- including
        board spans, since a board the model must *produce* (as in the
        apply-move task) is exactly what we want supervised at full weight.
        """
        ids: list[int] = []
        segs: list[int] = []

        if add_bos:
            ids.append(CHESS_TOKEN_TO_ID[BOS])
            segs.append(SEG_PROMPT)

        in_board = False
        for part in prompt_parts:
            if part == FEN_BEGIN:
                in_board = True
            piece_ids = self._encode_part(part)
            ids.extend(piece_ids)
            segs.extend([SEG_BOARD if in_board else SEG_PROMPT] * len(piece_ids))
            if part == FEN_END:
                in_board = False

        for part in answer_parts:
            piece_ids = self._encode_part(part)
            ids.extend(piece_ids)
            segs.extend([SEG_ANSWER] * len(piece_ids))

        if add_eos:
            ids.append(CHESS_TOKEN_TO_ID[EOS])
            segs.append(SEG_ANSWER)

        assert len(ids) == len(segs)
        return {"ids": ids, "segments": segs}

    def encode_packed(self, board_parts: Sequence[str],
                      qa_pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
                      add_bos: bool = True, add_eos: bool = True) -> dict:
        """Encode one board followed by several question/answer pairs.

        The board plane is emitted once and shared, so its cost is amortized
        across every answer in the example.
        """
        ids: list[int] = []
        segs: list[int] = []

        if add_bos:
            ids.append(CHESS_TOKEN_TO_ID[BOS])
            segs.append(SEG_PROMPT)

        board_ids = self.encode(board_parts)
        ids.extend(board_ids)
        segs.extend([SEG_BOARD] * len(board_ids))

        for question, answer in qa_pairs:
            q_ids = self.encode(question)
            ids.extend(q_ids)
            segs.extend([SEG_PROMPT] * len(q_ids))
            a_ids = self.encode(answer)
            ids.extend(a_ids)
            segs.extend([SEG_ANSWER] * len(a_ids))

        if add_eos:
            ids.append(CHESS_TOKEN_TO_ID[EOS])
            segs.append(SEG_ANSWER)

        assert len(ids) == len(segs)
        return {"ids": ids, "segments": segs}

    def loss_weights(self, segments: Sequence[int], weights: dict | None = None) -> list[float]:
        table = weights or DEFAULT_SEGMENT_WEIGHTS
        return [table[s] for s in segments]

    # -- decoding ----------------------------------------------------------

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        out: list[str] = []
        prose_run: list[int] = []

        def flush() -> None:
            if prose_run:
                out.append(self.prose.decode(prose_run))
                prose_run.clear()

        skip = {CHESS_TOKEN_TO_ID[t] for t in (PAD, BOS, EOS)} if skip_special else set()
        for i in ids:
            if i in skip:
                continue
            if i < CHESS_VOCAB_SIZE:
                flush()
                out.append(CHESS_TOKENS[i])
            else:
                prose_run.append(i - CHESS_VOCAB_SIZE)
        flush()
        return " ".join(out)

    # -- persistence -------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.prose is None:
            raise RuntimeError("nothing to save: prose tokenizer not fitted")
        self.prose.save(str(directory / "prose_bpe.json"))
        (directory / "chess_vocab.json").write_text(json.dumps({
            "chess_vocab_size": CHESS_VOCAB_SIZE,
            "prose_vocab_size": self.prose_vocab_size,
            "total_vocab_size": self.vocab_size,
            "chess_tokens": CHESS_TOKENS,
        }, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "ChessTokenizer":
        from tokenizers import Tokenizer

        directory = Path(directory)
        meta = json.loads((directory / "chess_vocab.json").read_text())
        if meta["chess_tokens"] != CHESS_TOKENS:
            raise ValueError(
                "saved chess vocabulary does not match the current vocab.py -- "
                "ids would shift and a trained checkpoint would be silently corrupted"
            )
        tok = Tokenizer.from_file(str(directory / "prose_bpe.json"))
        return cls(prose_tokenizer=tok, prose_vocab_size=meta["prose_vocab_size"])
