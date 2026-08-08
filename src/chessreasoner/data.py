"""Corpus shards -> packed training tensors.

Examples are encoded, concatenated and sliced into fixed ``max_seq_len``
windows. Every window carries a per-token loss weight (from the tokenizer's
segment ids) and, where a complete board span falls inside it, a 64-square
target for the board head.

Board targets are recovered from the token ids themselves rather than tracked
separately. The board span has a fixed layout, so relative to a ``</FEN>`` at
position ``p`` the 64 square-content tokens sit at ``p-70 .. p-7``:

    offset 0      <FEN>
    offset 1..64  square contents, a1 -> h8
    offset 65     side to move
    offset 66..69 castling rights
    offset 70     en-passant square
    offset 71     </FEN>

A span straddling a window boundary is skipped rather than half-read.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from .model import IGNORE_INDEX
from .tokenizer import DEFAULT_SEGMENT_WEIGHTS, SEG_PAD, ChessTokenizer
from .vocab import BOARD_CLASS_INDEX, CHESS_TOKEN_TO_ID, FEN_END, PAD

FEN_END_ID = CHESS_TOKEN_TO_ID[FEN_END]
BOARD_SQUARES_OFFSET = 70
"""Distance from ``</FEN>`` back to the first square-content token."""

# token id -> board-head class index (12 pieces + empty)
ID_TO_BOARD_CLASS = {CHESS_TOKEN_TO_ID[tok]: idx for tok, idx in BOARD_CLASS_INDEX.items()}
_BOARD_CLASS_LOOKUP = np.full(max(ID_TO_BOARD_CLASS) + 1, -1, dtype=np.int64)
for _id, _cls in ID_TO_BOARD_CLASS.items():
    _BOARD_CLASS_LOOKUP[_id] = _cls


def read_shard(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def encode_shard(path: str | Path, tokenizer: ChessTokenizer) -> list[dict]:
    """Encode one JSONL shard into per-example ``{ids, segments}`` records.

    Kept per-example rather than concatenated so :func:`pack_documents` can
    respect example boundaries.
    """
    out: list[dict] = []
    for record in read_shard(path):
        if record.get("packed"):
            out.append(tokenizer.encode_packed(
                record["board_parts"], [(q, a) for q, a in record["qa_pairs"]]))
        else:
            out.append(tokenizer.encode_example(
                record["prompt_parts"], record["answer_parts"]))
    return out


def pack(ids: np.ndarray, segments: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Naive contiguous slicing. **Do not use for training data** -- see
    :func:`pack_documents`.

    Kept only for throughput benchmarks and tests, where document boundaries do
    not matter. Slicing a concatenated stream cuts examples in half: measured on
    a real Tier-1 corpus, 15.9% of supervised answer tokens ended up in a window
    with no board span before them. The loss still demands those answers, so the
    model is trained to invent them -- which is exactly the hallucination this
    project exists to remove.
    """
    n = (len(ids) // seq_len) * seq_len
    if n == 0:
        raise ValueError(f"shard has {len(ids)} tokens, fewer than one {seq_len}-token window")
    return (ids[:n].reshape(-1, seq_len).copy(),
            segments[:n].reshape(-1, seq_len).copy())


def pack_documents(encoded: Sequence[dict], seq_len: int,
                   pad_id: int = CHESS_TOKEN_TO_ID[PAD]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Greedily fill windows with **whole** examples, padding the remainder.

    An example never straddles a window boundary, so every supervised answer
    token has its board in context. Padding is given ``SEG_PAD``, which maps to
    loss weight 0.

    Examples longer than ``seq_len`` cannot be placed and are dropped; the
    returned stats report how many, since a large count means the context is too
    short for the tier.
    """
    windows_ids: list[np.ndarray] = []
    windows_segs: list[np.ndarray] = []
    cur_ids: list[int] = []
    cur_segs: list[int] = []
    stats = {"examples": 0, "dropped_too_long": 0, "windows": 0, "pad_tokens": 0}

    def flush() -> None:
        if not cur_ids:
            return
        pad = seq_len - len(cur_ids)
        stats["pad_tokens"] += pad
        stats["windows"] += 1
        windows_ids.append(np.array(cur_ids + [pad_id] * pad, dtype=np.int32))
        windows_segs.append(np.array(cur_segs + [SEG_PAD] * pad, dtype=np.int8))
        cur_ids.clear()
        cur_segs.clear()

    for item in encoded:
        ids, segs = item["ids"], item["segments"]
        if len(ids) > seq_len:
            stats["dropped_too_long"] += 1
            continue
        if len(cur_ids) + len(ids) > seq_len:
            flush()
        cur_ids.extend(ids)
        cur_segs.extend(segs)
        stats["examples"] += 1
    flush()

    if not windows_ids:
        raise ValueError("no example fit in a single window")
    total = stats["windows"] * seq_len
    stats["utilization"] = 1.0 - stats["pad_tokens"] / total
    return np.stack(windows_ids), np.stack(windows_segs), stats


def board_targets_for(window_ids: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    """Positions of usable ``</FEN>`` markers and their 64-square class targets."""
    positions: list[int] = []
    targets: list[np.ndarray] = []
    for p in np.flatnonzero(window_ids == FEN_END_ID):
        start = p - BOARD_SQUARES_OFFSET
        if start < 0:
            continue  # span straddles the window start
        span = window_ids[start:start + 64]
        # Prose ids sit above the lookup table, so mask before indexing --
        # np.where would still evaluate the out-of-bounds gather.
        classes = np.full(64, -1, dtype=np.int64)
        in_range = span < len(_BOARD_CLASS_LOOKUP)
        classes[in_range] = _BOARD_CLASS_LOOKUP[span[in_range]]
        if (classes < 0).any():
            continue  # not a real board span -- a bare </FEN> from a truncated example
        positions.append(int(p))
        targets.append(classes)
    return positions, targets


class PackedDataset(torch.utils.data.Dataset):
    """Fixed-length windows over a packed corpus."""

    def __init__(self, ids: np.ndarray, segments: np.ndarray,
                 segment_weights: dict | None = None, with_board_targets: bool = True):
        self.ids = ids
        self.segments = segments
        self.with_board_targets = with_board_targets
        table = segment_weights or DEFAULT_SEGMENT_WEIGHTS
        self.weight_lookup = np.zeros(max(table) + 1, dtype=np.float32)
        for seg, weight in table.items():
            self.weight_lookup[seg] = weight

    @classmethod
    def from_shards(cls, paths: Sequence[str | Path], tokenizer: ChessTokenizer,
                    seq_len: int, verbose: bool = True, **kwargs) -> "PackedDataset":
        encoded: list[dict] = []
        for path in paths:
            encoded.extend(encode_shard(path, tokenizer))
        ids, segs, stats = pack_documents(encoded, seq_len)
        if verbose:
            print(f"packed {stats['examples']} examples into {stats['windows']} "
                  f"windows of {seq_len} ({stats['utilization']:.1%} utilization"
                  + (f", {stats['dropped_too_long']} too long to place)"
                     if stats["dropped_too_long"] else ")"))
        return cls(ids, segs, **kwargs)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        window = self.ids[i]
        item = {
            "input_ids": torch.from_numpy(window.astype(np.int64)),
            "loss_weights": torch.from_numpy(self.weight_lookup[self.segments[i]]),
        }
        if self.with_board_targets:
            positions, targets = board_targets_for(window)
            item["board_pos"] = torch.tensor(positions, dtype=torch.long)
            item["board_targets"] = (torch.from_numpy(np.stack(targets))
                                     if targets else torch.zeros((0, 64), dtype=torch.long))
        return item


def collate(batch: list[dict]) -> dict:
    """Stack windows and flatten per-example board positions into (N, 2) indices."""
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "loss_weights": torch.stack([b["loss_weights"] for b in batch]),
    }
    if "board_pos" in batch[0]:
        pos, targets = [], []
        for row, b in enumerate(batch):
            for p in b["board_pos"].tolist():
                pos.append([row, p])
            if len(b["board_targets"]):
                targets.append(b["board_targets"])
        if pos:
            out["board_pos"] = torch.tensor(pos, dtype=torch.long)
            out["board_targets"] = torch.cat(targets).long()
    return out


def aux_scale(step: int, total_steps: int, hold_fraction: float = 0.5) -> float:
    """Anneal the auxiliary-head weight to zero over the second half of training.

    Held at 1.0 for the first ``hold_fraction`` of the run, then linearly decayed
    so the model never learns to lean on heads it will not have at inference.
    """
    if total_steps <= 0:
        return 1.0
    progress = step / total_steps
    if progress <= hold_fraction:
        return 1.0
    return max(0.0, 1.0 - (progress - hold_fraction) / (1.0 - hold_fraction))
