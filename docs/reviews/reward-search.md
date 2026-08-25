# Reward Design + Data Pool + Small-Model RL Stability — Evidence-Backed Review

**Target:** GRPO on small chess LMs (gemma-4-E2B / Qwen-0.5B–1.5B). **Date:** 2026-08-25. **Anchored:** 2607.02869 (reward granularity), 2508.05928 (S-GRPO), 2504.03380 (online filter), 2503.20783 (DrGRPO), 2608.03467 (CueGRPO), 2504.15777 (Tina), VERL handbook.

## 1. Reward Granularity — Outcome vs Process vs Hybrid (2607.02869)

**Setup:** Qwen2.5-0.5B VERL+vLLM GRPO on GSM8K (7473/1319), 45-Q slice, batch64 mini8 lr1e-6 KL0.001 5ep580steps temp0.6 G5 512/1024 1×A100. Rewards: R_proc=correct/total within1e-5 penalty>1.5×gold, R_out=1 if |a-a*|<1e-5, Hybrid R=λR_proc+(1-λ)R_out λ∈{0.9,0.5,0.1}.

**Results full 1319:**

| Regime | Acc% | Δ | Note |
|---|---|---|---|
| Base |33.13|—||
| Process-only |**63.73**|+30.60|best CoT-Pass0.64|
| Outcome-only |53.75|+20.62|~10pp gap|
| λ0.9/0.1 |61.10|+27.97|best hybrid validity0.60 ratio0.84|
| λ0.5/0.5 |57.40|+24.27||
| λ0.1/0.9 |**49.30**|+16.17|anomaly < outcome|

*Fig4 single-seed.* **Fidelity 45-Q:** validity 0.56→0.22proc0.40out0.60hybrid, deviation 3.00→6.49proc3.64out3.49hyb, step-ratio 0.79→0.64proc0.77out0.84hyb. Process maximizes accuracy but verbose contradictions (GPT-4o); outcome concise derivation errors; hybrid 0.9/0.1 balances. **λ0.1 underperforms outcome by4.45pp = conflicting optimization.** For chess rlvr-plan 1.0/0.3/0.1 likely too low; need λ≥0.5–0.9, A/B process-only vs 0.9hybrid, style gated (1-tokens/max only if correct).

## 2. Difficulty Gating & Curriculum — Near-Equal Chess + S-GRPO (2508.05928) + Filter (2504.03380)

**Theory 2504.03380:** p(x)=E[r], var p(1-p) max0.5. Improvement lower-bounded by Var[p], KL∝p(1-p). p≈0/1 zero signal. Balanced filter 0.25≤p≤0.75 → +10pp AIME +4pp avg, 60% time/data vs plain GRPO, parallel replacement to keep batch. Skewed fails.

**S-GRPO:** Think-Answer Mismatch false positive inflates advantage ~60% in 1/8 vs 4/8 (5.31vs3.32 N16, U-shaped). 10% noise slower, 20% GRPO collapses, S-GRPO stable.

**Formulas:**

```
ā=mean(r), σr²=ā(1-ā)+ε, A=(r-ā)/σr, a_pos=(N-k)/√k(N-k)
t=(ā-p)/(1-2p) clip0-1, σt²=t(1-t)+ε, Cov=(1-2p)t(1-t)
w*(N,k,p)=(1-2p)t(1-t)/(σr σt) ≤1-2p, A*=w*·A, L=min(ratio w*A, clip(ratio,1-ε,1+ε) w*A)
```

p=0 recovers GRPO, max k≈N/2, w*=0 when ā<p (p0.15 gates k≤1≥7 for N8). Empirics MATH8.5k 500steps: 7B 56.0+2.5, 1.5B 49.7+2.4, 3B+2.2 vs DrGRPO. p0.10 fastest dips80, p0.15 monotonic, optimal p inverse size (1.5B0.15 7B0.10). Entropy smooth vs jitter.

**Chess mapping:** |gap|≤60cp 50% Stockfish vs truth disagree = high noise p0.2-0.3 frontier p0.5. Use p0.15 for gemma-4-E2B (small) gating tails, balanced pool 5-50k stratified >150 60-150 ≤60, online 0.25-0.75 or w*. Hard-only from step0 →zero var; GRPO 20% noise collapses; DrGRPO lacks gating.

## 3. LoRA Stability — DrGRPO No-Norm + Rarity-Aware + LR/Clip (2503.20783, 2608.03467, 2504.15777)

**DrGRPO:** Length bias 1/|o| → short correct favored, long wrong penalized less → verbose wrong; difficulty bias ÷std → low-std overweighted. Fix Â=R-mean no std no length, loss masked_mean/MAX_TOKENS not mask.sum. 7B MATH L3-5 43.3% AIME 27h8×A100 β0. Chess length-agnostic must use DrGRPO (rlvr-plan clip1.0 no-norm correct).

**Adv formulas:**

```
GRPO: Â=(R-mean)/std +1/|o|
Dr:   Â=R-mean
S:    Â=w*(R-mean)/std
Cue:  Â^CR=Â·N|C|^{-α}/Σ|C|^{-α} if correct α0→GRPO 1→equal slope1→0.72 +6% vs +62% judge
```

Length norm → incorrect length ↑ monotonic, Dr token efficiency.

**LR cliff:** SFT 2e-4 r32 correct; RL 1e-5 r32 stable; VERL3B 3e-6 r64 stable; Tina1.5B 1e-6 optimal (5e-6 47.87 1e-6 48.47 5e-7 47.91) rank16 48.92>r32 48.47>r64 46.95. RL at 2e-4 collapses <1ep phase transition 30-60% best just before. Gemma-4B rec lr1e-5→5e-6→3e-6 cosine w10 r16-32 α32 save50 top3 early stop.

**Clip:** ε0.2 base 0.28 DAPO clip-higher if p→1, KL0/0.001, entropy0, grad1.0. Too tight slow, none ratio explode.

**CueGRPO 2608.03467:** 27 cues (\sqrt inom \pmod ... let x be Case1) → bag → suppress ρ → cosine cluster → ω above, slope confirmed.

## 4. Pool Sizes 335 vs 5k vs 50k

| Pool | Strat | Dyn G8~100/hr | Risk |
|---|---|---|---|
|335 hard≤60|hard|72-95× rep 12h 3-4k steps|overfit+starvation|
|5k bal|30/40/30|4.8-6.4×/ep|sweet Tina7k 50.60>93k49.26|
|50k|large|0.5×|under|
|1B Gigafish|distill|—|not T4 RL|

Tina quality>size, filter 60%. 335 narrow → plateau~40%. 50k needs G16-32. Staged 335 smoke→5k bal→50k if ≥65%.

## 5. Prescription

```python
R_out=1 if Move==stockfish else0
R_proc=verified/total # legality+|Δ|≤100 >1.5× penalty
R_style=(1-tokens/max) if R_out==1 else0
# ArmA 1.0/0.3/0.1 ArmB 0.1/0.9/0.1
A_raw=R-mean; t=clip((mean-p)/(1-2p),0,1); w*=(1-2p)t(1-t)/(σr σt); A=w*·A_raw
# α0.5-1.0 rarity, loss /MAX_TOKENS clip0.2
```

LoRA r16-32 lr1e-5 β0 save50, pool5k 0.25-0.75 replacement, log |gap| vs p.

## 6. Refs

-2607.02869 granularity, -2508.05928 S-GRPO, -2504.03380 filter, -2503.20783 DrGRPO, -2608.03467 Cue, -2504.15777 Tina, VERL handbook, rlvr-plan.
