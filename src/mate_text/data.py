"""Data loading + collation for the MATE text transformer."""
from __future__ import annotations

import json
import torch
from torch.utils.data import Dataset, DataLoader

import chess

from src.mate_text.tokenizer import MateTokenizer

BOARD_TYPE = 0
TEXT_TYPE = 1
ANSWER_TYPE = 2


class MateTextDataset(Dataset):
    def __init__(self, path: str, tokenizer: MateTokenizer,
                 shuffle_seed: int = 42):
        self.tokenizer = tokenizer
        self.rows = [json.loads(l) for l in open(path)]
        self.rng = __import__("random").Random(shuffle_seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], int]:
        r = self.rows[idx]
        board = chess.Board(r["fen"])
        bid = self.tokenizer.board_ids(board)
        tid = self.tokenizer.text_ids(r["candidate_a"], r["candidate_b"])
        # NO answer token in the input — the model must decide from the
        # board + candidate text alone. The label is the target.
        tokens = bid + tid
        types = ([BOARD_TYPE] * len(bid)
                 + [TEXT_TYPE] * len(tid))
        label = 0 if r["truth"] == "A" else 1
        return tokens, types, label


def collate(batch, pad_id: int):
    maxlen = max(len(s[0]) for s in batch)
    tok_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    type_ids = torch.zeros((len(batch), maxlen), dtype=torch.long)
    labels = torch.tensor([s[2] for s in batch], dtype=torch.long)
    for i, (t, ty, _) in enumerate(batch):
        tok_ids[i, :len(t)] = torch.tensor(t)
        type_ids[i, :len(t)] = torch.tensor(ty)
    return tok_ids, type_ids, labels


def make_dataloader(path: str, tokenizer: MateTokenizer, batch_size: int,
                    shuffle: bool = True, num_workers: int = 2):
    ds = MateTextDataset(path, tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=lambda b: collate(b, tokenizer.special_pad()))
