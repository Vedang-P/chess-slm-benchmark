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
from .tokenizer import DEFAULT_SEGMENT_WEIGHTS, ChessTokenizer
from .vocab import BOARD_CLASS_INDEX, CHESS_TOKEN_TO_ID, FEN_END

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


def encode_shard(path: str | Path, tokenizer: ChessTokenizer) -> tuple[np.ndarray, np.ndarray]:
    """Encode one JSONL shard into flat (ids, segments) arrays."""
    ids: list[int] = []
    segs: list[int] = []
    for record in read_shard(path):
        if record.get("packed"):
            enc = tokenizer.encode_packed(
                record["board_parts"], [(q, a) for q, a in record["qa_pairs"]])
        else:
            enc = tokenizer.encode_example(record["prompt_parts"], record["answer_parts"])
        ids.extend(enc["ids"])
        segs.extend(enc["segments"])
    return np.asarray(ids, dtype=np.int32), np.asarray(segs, dtype=np.int8)


def pack(ids: np.ndarray, segments: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice the flat stream into ``(n_windows, seq_len)`` arrays, dropping the tail."""
    n = (len(ids) // seq_len) * seq_len
    if n == 0:
        raise ValueError(f"shard has {len(ids)} tokens, fewer than one {seq_len}-token window")
    return (ids[:n].reshape(-1, seq_len).copy(),
            segments[:n].reshape(-1, seq_len).copy())


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
                    seq_len: int, **kwargs) -> "PackedDataset":
        all_ids: list[np.ndarray] = []
        all_segs: list[np.ndarray] = []
        for path in paths:
            shard_ids, shard_segs = encode_shard(path, tokenizer)
            all_ids.append(shard_ids)
            all_segs.append(shard_segs)
        ids, segs = pack(np.concatenate(all_ids), np.concatenate(all_segs), seq_len)
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
