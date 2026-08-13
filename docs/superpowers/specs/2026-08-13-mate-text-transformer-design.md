# MATE Text Transformer — Design

**Date:** 2026-08-13
**Status:** approved (design review 2026-08-13)
**Type:** side-arm experiment, not a pivot. The lucid/commentary track is the
primary research; this is a week-long build while Kaggle credits reset.

## Problem

Can a single, small, text-only transformer — trained jointly on all four MATE
subsets (strategy/tactic/noexplain/both) — beat (a) each per-subset fine-tune
of the same architecture, and (b) the frontier/SLM baselines on the MATE
2-candidate selection task? Bar: 85-93% on the noexplain testbed.

## Baselines (noexplain testset, 1000 positions)

| Model | Accuracy |
|---|---|
| random | 50% |
| gemma-4-E2B zero-shot | 61.1% (strategy arm; noexplain not yet measured) |
| MATE LLaMA-3-8B fine-tune (noexplain anchor) | 63.5% |
| deepseek-v4-flash thinking | 92.2% (922/1000) |
| MATE best combined fine-tune (both anchor) | 95.2% |

## Architecture (hybrid: board tokens + text head)

- **Board encoding:** FEN → 64 structured tokens (piece-on-square), plus
  side-to-move + castling/en-passant metadata tokens. Chessformer-style.
- **Text encoding:** candidate moves `MoveA:<uci>` / `MoveB:<uci>` as tokens,
  plus task framing.
- **Model:** compact decoder-only transformer, d_model 512, 8-12 layers,
  ~30-60M params. Concatenated board-token + text sequence, 2-way
  classification head (MoveA vs MoveB).
- **Vocab:** small (~200-500 tokens): pieces, squares, digits, MoveA/MoveB,
  specials. Learned embeddings.

## Dataset

- Source: `OutFlankShu/MATE_DATASET` (4 train zips + testset, ~185MB).
- Join all 4 subsets, dedup by FEN (deterministic priority), exclude MATE
  testset FENs, balance candidate order (50/50 truth=A/B), 95/5 train/val,
  seed 42.
- New `scripts/build_mate_text_data.py`.

## Training

- AdamW, cosine, label smoothing, bf16 (RTX 4050 sm_89), batch 32-64,
  seq ~512. Hours/epoch on the laptop; full train 1-2 days.
- wandb project `mate-text-transformer`.

## Evaluation

- Same `mate-selection-test-noexplain.json` (1k positions) as the campaign.
- Ablation: same arch trained per-subset vs the jointly-trained model.
- Report overall + per-subset accuracy.

## Week plan

1. Dataset builder + hygiene (day 1)
2. Tokenizer + architecture + training loop (days 1-2)
3. Train joint + 4 single-subset models (days 2-4)
4. Eval on shared testset + ablation table (days 4-5)
5. Buffer (days 5-7)
