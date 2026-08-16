# Decision Doc: Teaching gemma-4-E2B to reason from deepseek traces — cavegemma-style

Everything here is grounded in (a) the actual cavegemma repo
(JuliusBrussee/cavegemma, fetched 2026-08-16), (b) our measured numbers
(monitor/, wandb, HF), (c) the literature we've verified. No invented numbers.

---

## 1. What cavegemma ACTUALLY did (from the repo, not our old notes)

| Field | Real value (from README + repo) |
|---|---|
| Base | `google/gemma-4-31B-it` (31B, NOT 2B) |
| Method | QLoRA NF4 + double-quant + bf16 compute |
| LoRA | rank 16, α 32, dropout 0, **targets all linear** |
| Dataset | **1750 train + 193 eval** pairs (verbose → caveman rewrite) |
| Schedule | **3 epochs**, lr 2e-4 cosine, batch 2 × grad accum 8 (eff 16), `completion_only_loss=True` |
| Hardware | RunPod RTX PRO 6000 Blackwell 96GB, $1.89/hr |
| Wall time | ~50 min |
| Final loss | train 0.024, eval 0.72, eval acc 81.5% |
| Total spend | $4–5 |
| Result | 27% fewer tokens (weighted, 193 holdout), semantic 0.91–0.98 cosine, code fences byte-exact 96–100% |

**Pipeline** (from repo layout + reproduce section):
1. **Corpus**: 3000 rows pulled from 6 permissive HF sources → 1750 survived filtering (~58% yield)
2. **Synthesis**: `claude -p` / `codex exec` driven through the caveman SKILL.md ruleset, **two-step rewrite** (verbose → caveman), resumable by key-hash, `--workers 3`
3. **Filter**: fence-integrity + dedup + compression band (accepted ≤1.0× source length — **known filter bug**: gold is 0.54×, achieved 0.73×, so they undershoot gold; fix would be ≤0.70×)
4. **Split**: 90/10 with seed-pair pinning
5. **Train**: Unsloth + TRL SFT
6. **Eval**: compression ratio, article-density drop, code-fence exactness, semantic sim

Key fact: **1750 pairs was enough to imprint a STYLE on a 31B.** The eval measures *style fidelity* (compression + semantic preservation), NOT task accuracy.

## 2. What's the same and what's different for us

**Same:**
- Teacher synthesizes target-format text (deepseek writes lucid traces; claude/codex wrote caveman)
- Student imprints the format via SFT (QLoRA, all-linear, 3 epochs)
- Filter for integrity (we have a STRONGER one: Stockfish verification vs fence-integrity)
- Small data works: style imprint needs only ~1-2k pairs

**Different — and this is the critical honesty:**
- cavegemma is STYLE-ONLY: the task is "say the same thing in fewer tokens". There is no chess accuracy question. Their eval never asks "is the answer right."
- Ours must ALSO move MATE accuracy (61.1% → target). Imprinting a style does not by itself teach move selection.
- So the trace corpus has **two jobs**: (1) teach the lucid trace *format*, (2) carry *chess reasoning content* that improves the answer. cavegemma only had job 1.

## 3. Our measured physical facts (from this repo)

| Fact | Value | Source |
|---|---|---|
| base gemma accuracy, strategy 1000, thinking ON | 61.1% | monitor/gemma/state.json |
| base gemma with thinking ON, 1024 budget | **parse_rate 0.0 — never emits an answer** | src/models.py comment, observed in campaign |
| base gemma output budget used | 32768 max_new_tokens | monitor config |
| deepseek noexplain baseline | 92.2% (922/1000) | paper/main.tex (verified) |
| MATE 8B noexplain anchor | 63.5% | MATE paper |
| our 60k-label SFT at 0.107 epoch | loss 7.9→0.19 (format learned), live acc flat 0.56 | wandb run 39o5cvei |
| GPU throughput measured | ~370 steps/hr = ~6k rows/hr (batch2×accum8, 2B QLoRA) | wandb run 39o5cvei |

**The parse_rate 0.0 fact is decisive:** a base 2B with thinking enabled *cannot even emit a parseable answer*. There is no path to "gemma reasons" without training. **SFT is mandatory** — the only open question is the mix.

## 4. Do we need 3k deepseek positions? Evidence on trace count

| Evidence | N traces | Model | Domain | Result |
|---|---|---|---|---|
| cavegemma | 1750 | 31B | style transfer | works, 3 epochs |
| s1 (2501.19393) | **1000** | 32B base | math reasoning | o1-level with budget forcing |
| Reasoning Quality Emerges Early (2606.26797) | small curated sets | various | math | curation > volume |
| 2B-class learnability (2502.12143) | short+long mix | ≤3B | math | short traces learned BETTER |
| Self-training concise (2502.20122) | self-generated | various | math | conciseness from ~few k |
| DeepSeek R1-Distill | 800k | 1.5-70B | general | the BIG-data extreme (not our regime) |

**Honest synthesis:** For a 2B, the evidence points to *a few thousand well-curated traces* being the effective range — not 800k (R1-Distill is the anti-example for our compute), not 100 (too few to cover chess reasoning diversity). **3k raw is the right planning number, and here's the yield math:**

- cavegemma: 3000 pulled → 1750 kept = **58% yield** (filter loss)
- We will lose traces to Stockfish verification (final-choice mismatch, illegal steps, eval-instability). At deepseek's 92.2% final-choice accuracy and ~90-95% step-stability, expect **~80-85% yield** → 3000 raw → **~2.4-2.6k verified traces**
- 2.4k verified + cavegemma's 1750 precedent + s1's 1000 → **comfortably in the evidence-backed range**

So: **3k is not too many, not too few — it's the right call, and it's also the API budget ceiling you set. No reason to change it.** What we SHOULD do is spend the 3k smartly: difficulty-gated + phase-natural selection (we already have this machinery), NOT random sampling.

## 5. The training mix question — evidence-backed recommendation

We have three candidate data sources:
1. **Labels** (60k noexplain selection pairs — already pre-tokenized, already used)
2. **Verified deepseek lucid traces** (~2.4-2.6k after filter)
3. **Self-traces** (the model's own traces, sampled AFTER it learns the format, Stockfish-verified — on-policy)

**Decision: Train on labels + verified traces, in ONE SFT run.**

- Labels carry volume + the exact eval format (byte-identical prompt — already built)
- Traces carry the reasoning format + content (the cavegemma-style imprint)
- Evidence for mixing: 2502.12143 (mix long+short traces — 2B learns short better); LS-Mixture SFT (2505.03469) mixes long+short CoT in SFT; SuperCorrect mixes labels+traces
- Self-traces come LATER, after the model can reason (our verification-gap Phase 2)

**Mix ratio:** cavegemma did 3 epochs over 1750 pairs for style. We have 60k labels + 2.4k traces. Options:
- (a) 1 epoch over mixed 62.4k (traces at ~4% of data) — traces may get drowned
- (b) **repeated trace upsampling**: labels 1 epoch + traces repeated to ~15-20% of the epoch (trace-centric, cavegemma-style imprinting, labels for coverage). This matches the evidence: style imprint needs multiple exposures (3 epochs for cavegemma), and 2B learns short traces better.
- (c) 3 epochs over traces-only (pure cavegemma analog) + 1 epoch labels — the risk is forgetting the label format.

**Recommendation: (b)** — one SFT run, labels at 1 epoch + verified traces upsampled to ~15-20% of steps, LoRA r16/r32 all-linear, 3 effective passes over traces (cavegemma's 3 epochs, applied to the trace subset). This is directly testable in ONE 12h kernel, and we can A/B the trace fraction (10% vs 20%) if budget allows.

**On "do we need SFT on a subset":** evidence says NO subset needed — we have exactly the right scale. 60k labels is 6-10h of training; 2.4k traces add ~15-20% steps. One run.

## 6. What we will measure (the decision gates)

Before spending the full 3k API budget, run a **pilot on 300 traces**:
- Gate 1: after SFT with 300 verified traces, does gemma (a) emit parseable `MoveX:` answers with thinking ON, (b) produce lucid-style traces (measured: trace length distribution, token usage), (c) move accuracy on a 200-position probe?
- Gate 2: if Gate 1 passes, spend the remaining 2.7k, retrain, eval on the full 1000.
- Decision rule: if accuracy is flat at 61-63% after the full run, the finding is *"2B + traces imprints style but labels don't move selection — the contribution is the efficiency/verification story"* (still publishable). If it moves to 70-80%, the training story is real.

## 7. Cost & compute audit (real numbers)

| Item | Cost | Where |
|---|---|---|
| deepseek 3k traces | API budget (you set 3k) | gateway |
| Stockfish verification of 3k traces | ~1-2h CPU | local / Kaggle CPU kernel |
| SFT: 60k labels + 2.4k traces ×1 epoch | ~8-10h T4 | 1 Kaggle GPU kernel |
| Probe eval (200 pos, thinking ON) | ~1-2h GPU or CPU | same kernel tail |
| Full eval (1000 pos, thinking ON) | ~12h GPU | 1 Kaggle GPU kernel |
| **Total** | **~22h GPU + ~4h CPU** | within 30h/week |

## 8. The honest risk statement

1. **The biggest risk is that trace-SFT imprints style but does NOT move accuracy** (our 0.107-epoch labels-only run already showed format-learning with flat accuracy). If verified traces don't move selection either, we must pivot the paper to the efficiency/verification framing (which survives) — and we'll know this in ~3 days, not 3 weeks.
2. cavegemma's filter-bug lesson applies to us: don't accept traces that barely compress (we verify by Stockfish instead — a stronger filter, but the same lesson: quality gate matters more than volume).
3. We will NOT know the right trace count a priori; 3k is evidence-backed and budget-capped. The pilot-then-scale plan means we spend 10% of budget to de-risk before the rest.
