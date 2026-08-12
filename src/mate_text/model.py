"""Hybrid MATE text transformer: board tokens + text, 2-way head.

Decoder-only transformer over the fused board+text token sequence, with
a 2-way classification head (MoveA vs MoveB) on the final hidden state.

Design (spec docs/superpowers/specs/2026-08-13-mate-text-transformer-design.md):
  - board: 64 piece tokens + metadata (side/castling/en-passant)
  - text:  MoveA:<uci> MoveB:<uci>
  - fused with type embeddings, decoder-only self-attention, causal mask
  - head: 2-way classification on the <cls>/final token

Scale: d_model 512, 8 layers, ~40M params (configurable).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MateTextConfig:
    vocab_size: int = 42
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq: int = 512
    n_types: int = 3  # board / text / answer
    n_classes: int = 2


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SelfAttention(nn.Module):
    def __init__(self, cfg: MateTextConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.out(y))


class MLP(nn.Module):
    def __init__(self, cfg: MateTextConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: MateTextConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = SelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MateTextTransformer(nn.Module):
    def __init__(self, cfg: MateTextConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.type_emb = nn.Embedding(cfg.n_types, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.ndim >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.normal_(p, 0.0, 0.02)

    def forward(self, tokens: torch.Tensor, types: torch.Tensor,
                answer_idx: torch.Tensor | None = None):
        B, T = tokens.shape
        assert T <= self.cfg.max_seq, f"seq {T} > max_seq {self.cfg.max_seq}"
        pos = torch.arange(T, device=tokens.device)
        x = (self.tok_emb(tokens)
             + self.type_emb(types)
             + self.pos_emb(pos))
        x = self.drop(x)
        # causal mask
        mask = torch.tril(torch.ones(T, T, device=tokens.device,
                                     dtype=torch.bool)).view(1, 1, T, T)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        # classify on the last position (the answer marker)
        last = x[:, -1, :]
        logits = self.head(last)  # [B, 2]
        if answer_idx is not None:
            loss = F.cross_entropy(logits, answer_idx)
            return logits, loss
        return logits, None

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
