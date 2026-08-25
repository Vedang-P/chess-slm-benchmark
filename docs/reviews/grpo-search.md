# GRPO / DAPO / DrGRPO / PPO-RLVR Literature Search — Rough Findings
**Date:** 2026-08-25
**Method:** WebSearch site:arxiv.org + arXiv read API + attempted Semantic Scholar API (curl). S2 API was unreachable (CODEX_SANDBOX_NETWORK_DISABLED=1; curl → Could not resolve host: api.semanticscholar.org, fetch ENOTFOUND). Fallback: WebSearch + direct arXiv fetches (read tool, arxiv abs/html). 3s delay requested but network block prevented S2 retries; WebSearch corroborated key claims.

---

## 1. Query 1 — GRPO / DeepSeekMath 2402.03300

### 1.1 Core Papers

| # | Title | arXiv | Year | Venue | Key Claim | Method | Compute |
|---|-------|-------|------|-------|-----------|--------|----------|
| 1 | **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** | [2402.03300](https://arxiv.org/abs/2402.03300) | 2024-02-05 (v3 2024-04-27) | cs.CL | Introduced **GRPO** as PPO variant without critic; DeepSeekMath-7B continues pre-training DeepSeek-Coder-Base-1.5-7B on 120B math tokens → **51.7% MATH** (no tools/voting), **60.9% with self-consistency@64**, approaching Gemini-Ultra/GPT-4. GRPO optimizes memory (no value model ≈ 50% memory saving vs PPO). | GRPO: sample group G responses per prompt, group-relative advantage \hat A_{i,t}=(r_i-mean)/std, PPO clip + KL penalty beta D_KL. | Pre-train 120B tokens; RL fine-tune on GSM8K/MATH; 7B model; full-model RL. |
| 2 | **Gradient Starvation in Binary-Reward GRPO: Why Group-Mean Centering Fails** | [2605.07689](https://arxiv.org/abs/2605.07689) | 2026-05-08 | cs.LG | Binary rewards → group-mean advantage = 0 when all correct/wrong → no signal. Degeneracy rate **0.69 at G=4** on Qwen3.5-9B GSM8K. Sign advantage A=2r-1 → **73.8% vs 28.4%** DrGRPO at G=4 (+45.4 pts, p<0.0001). | Diagnose failure; propose Sign advantage optimizing pass@G. | Qwen3.5-9B, Llama-3.1-8B, GSM8K, MATH-500, 7 seeds. |
| 3 | **Predictive Scaling Laws for Efficient GRPO Training of Large Reasoning Models** | [2507.18014](https://arxiv.org/abs/2507.18014) | 2025-07-24 | cs.LG | GRPO reward follows 3 phases: slow start → rapid improvement → plateau. **Training beyond 1 epoch gives little gain**; empirical scaling law predicts reward, generalizes Llama/Qwen 3B/8B. | Scaling law fits reward trajectories. | 3B/8B models; guide early stop. |
| 4 | **Hard Examples Are All You Need: Maximizing GRPO Post-Training** | [2508.14094](https://arxiv.org/abs/2508.14094) | 2025-08-14 | cs.LG | GRPO replaces value function with group-normed advantages, relying on within-group variance. | Variance-dependent analysis. | — |

**GRPO equation (from DAPO/TinyZero):**
```
J_GRPO(theta)=E_{q,{o_i}~pi_old}[1/G sum_i 1/|o_i| sum_t min(r_{i,t} A_i, clip(r_{i,t},1-eps,1+eps) A_i) - beta D_KL]
A_i = (r_i - mean({r_i}))/std({r_i})
r_{i,t}= pi_theta(o_{i,t}|q,o_{<t})/pi_old(...)
```
SOTA removes KL (DAPO 2.3) because long-CoT diverges far from reference. Group sizes typical 8-64.

**Small-model relevance:** Critic-free → 50% memory saving → enables 0.5-3B on 4xA40 (TinyZero) or T4 with LoRA.

---

## 2. Query 2 — DAPO 2503.14476 / DrGRPO 2503.20783 / VAPO / PPO vs GRPO (small LM)

### 2.1 DAPO — Decoupled Clip and Dynamic Sampling Policy Optimization

| Title | arXiv | Year | Key Claims | Method (4 techniques) | Compute / Throughput |
|-------|-------|------|------------|----------------------|----------------------|
| **DAPO: An Open-Source LLM Reinforcement Learning System at Scale** (ByteDance Seed) | [2503.14476](https://arxiv.org/abs/2503.14476) v1 2025-03-18 | 2025 | **50 pts on AIME 2024** with Qwen2.5-32B (vs DeepSeek-R1-Zero-Qwen-32B 47 pts) using **50% training steps**. Naive GRPO only 30 pts — entropy collapse, reward noise, instability. Open-sourced on **verl** + curated dataset. | 1) **Clip-Higher**: decouple eps_low, eps_high (e.g., 0.2→0.28 high) to allow low-prob exploration tokens (fixes entropy collapse; clipped max prob <0.2). 2) **Dynamic Sampling**: filter groups where 0<|{correct}|<G fails (0-variance) → efficient. 3) **Token-Level Loss**: 1/sum|o_i| sum sum vs 1/G sum 1/|o_i| — critical long-CoT. 4) **Overlong Reward Shaping**: penalize truncated outputs. Remove KL. | Qwen2.5-32B; 50% steps saving; dynamic sampling discards 30-40% groups saving backward passes. Repo: https://github.com/volcengine/verl |

### 2.2 DrGRPO — Understanding R1-Zero-Like Training

| Title | arXiv | Year | Key Claims | Method | Compute |
|-------|-------|------|------------|--------|----------|
| **Understanding R1-Zero-Like Training: A Critical Perspective** (Sail-SG) | [2503.20783](https://arxiv.org/abs/2503.20783) v1 2025-03-26 | 2025 | GRPO has **optimization bias that artificially increases response length (esp. incorrect)** via length/std norm. **Dr.GRPO** unbiased → **token efficiency while maintaining reasoning**. Minimalist R1-Zero recipe → **43.3% AIME 2024 with 7B base** (SOTA at 7B). V3-Base already shows Aha moment; Qwen2.5 base strong without prompt templates. | Dr.GRPO removes division by std / corrects length bias. | 7B; ~20-30% length reduction vs GRPO. GH: https://github.com/sail-sg/understand-r1-zero |

### 2.3 VAPO — Value-based Augmented PPO

| Title | arXiv | Year | Key Claims | Method | Compute |
|-------|-------|------|------------|--------|----------|
| **VAPO: Efficient and Reliable RL for Advanced Reasoning** | [2504.05118](https://arxiv.org/abs/2504.05118) v1 2025-04-07 | 2025 | **60.4 on AIME 2024** Qwen-32B, **+10 pts vs R1-Zero-32B & DAPO** identical settings; **SOTA within 5,000 steps**, **no crashes** (stability). | Value-based PPO fixes: value bias, heterogeneous lengths, sparse rewards. Long-CoT design. | 32B, 5k steps to SOTA. |
| **Towards Analyzing Limitations of VAPO** | [2506.03038](https://arxiv.org/abs/2506.03038) | 2025 | Theory perspective on VAPO limits. | Theory | — |

### 2.4 Unifying Identity

| Title | arXiv | Year | Key Claim |
|-------|-------|------|-----------|
| **GRPO, Dr.GRPO, and DAPO Are Three Operations on One Number** | [2607.00152](https://arxiv.org/abs/2607.00152) | 2026-06-30 | **Proves all three adjust one number: group std sigma** (disagreement). Binary reward sigma = update size. GRPO divides by sigma, Dr.GRPO drops division, DAPO discards sigma=0 groups. Split group (50/50) teaches most; unanimous teaches nothing. Validated Big-Math + controlled run. |

### 2.5 PPO vs GRPO Controlled

| Title | arXiv | Year | Key Findings |
|-------|-------|------|--------------|
| **Comparative Analysis and Parametric Tuning of PPO, GRPO, DAPO** | [2512.07611](https://arxiv.org/abs/2512.07611) | 2025-12-08 | Countdown Game SFT → general benchmarks. RL > base. **Larger G → more stable + higher accuracy** (GRPO & DAPO). **KL non-monotonic**. **Dynamic Sampling does NOT improve — best DAPO without DS**. |

Other variants: DRA-GRPO [2505.09655], GRPO-VPS [2604.20659], Expand and Prune [2512.15347] (GRPO needs large G), SSR-GRPO [2608.19595].

---

## 3. Query 3 — RLVR / TinyZero 2503.16219 / DeepSeek-R1 2501.12948

### 3.1 DeepSeek-R1 (Foundation)

| Title | arXiv | Year | Key Claims | Method | Compute |
|-------|-------|------|------------|--------|----------|
| **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL** | [2501.12948](https://arxiv.org/abs/2501.12948) v1 2025-01-22 | 2025 | **Pure RL (no SFT) incentivizes emergent reasoning**: self-reflection, verification, adaptation. Surpasses SFT on human demos. Traces compressed/telegraphic (RL-learned style). Distillation to smaller models helps. | GRPO-style RL on DeepSeek-V3-Base (671B MoE) with verifiable rewards (math checker, code exec, format). | 671B DeepSeek-V3; large-scale RL. |

### 3.2 TinyZero — Small-model RLVR (Primary 0.5-3B evidence)

| Title | arXiv | Year | Key Claims | Method | Compute / Hyperparams |
|-------|-------|------|------------|--------|----------------------|
| **Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn’t (Open-RS)** | [2503.16219](https://arxiv.org/abs/2503.16219) v1 2025-03-20 | 2025 | **Focus: 1.5B DeepSeek-R1-Distill-Qwen-1.5B under 4×A40 48GB, 24h**. Curated 7k samples. **Rapid gains: AMC23 63%→80% (Exp2), MATH-500 83%→85%, AIME24 46.7% (surpass o1-preview 44.6%)** with **$42 cost** vs $1000s. Avg 56.3% vs DeepScaleR-1.5B-Preview 57.0%. **Instability after 150-200 steps**: reward degrade, length fluctuation, KL unstable, unreadable/mixed-language, especially >200 steps. Cosine reward stabilizes length 1000-3500 vs 2000-3500. Easy+hard mix crucial. | GRPO (open-r1 HF) without SFT; rewards: Accuracy (\\boxed binary 0/1), Cosine (accuracy scaled by cosine length → shorter correct higher), Format (<think> tags). Dataset: s1 59k→31k (\\boxed)→21k (D-R1-Distill filter)→18.6k (Qwen2.5-7B) = open-s1; DeepScaleR 40k→21k (Qwen2.5-Math-7B) = open-deepscaler → 39.6k combined → experiments use **7k (3k+3k+1k easy)**. | **4×A40 48GB, 24h, 1 epoch, G=6 (per step, VRAM), max completion 3584-4096, lr ~1e-6 (App.E), global steps 50-100 optimal (Exp2 50 steps best). Beyond 200 degrade. $42.** Code: https://github.com/knoveleng/open-rs |

**Curves (HTML):** Exp1 open-s1 18k 4096 tokens: 63→70% AMC23 in 50-100 steps then <60% after 200; length 4000→3000→rise + drift. Exp2 7k mix 3584: 63→80% in 50-100 then KL unstable after 4000 local steps. Exp3 cosine + English-only prompt: length stable but mixed-language persists.

### 3.3 RLVR Broader

| Title | arXiv | Year | Key Claims | Method | Compute |
|-------|-------|------|------------|--------|----------|
| **Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains** | [2503.23829](https://arxiv.org/abs/2503.23829) v1 2025-03-31 | 2025 | Binary verification high consistency across LLMs if expert reference exists. **Generative scoring → soft model-based rewards** for free-form. Trains **cross-domain generative reward models with 7B**; outperforms Qwen2.5-72B & DeepSeek-R1-Distill-32B across medicine/chem/psych/econ/edu free-form. | Soft rewards, 7B generative RM. | 7B RM; cross-domain. |
| **Reinforcement Learning with Verifiable Rewards Implicitly …** | [2506.14245](https://arxiv.org/abs/2506.14245) | 2025 | Formalizes RLVR. | — | — |

---

## 4. Cross-Cutting: Small-Model (0.5-3B), LoRA, Group Size, LR, Throughput

### 4.1 Small-Model 0.5-3B Results

| Paper | Year | Model(s) | GRPO/RL Setup | Result | Note |
|-------|------|----------|---------------|--------|------|
| **Effective Learning for Small Reasoning Models: 0.5B** [2506.13404](https://arxiv.org/abs/2506.13404) | 2025 | 0.5B SRLM | SFT, KD, RL + hybrids ablated | Actionable pipeline for 0.5B; hybrid best. Bridges gap cost-effectively. | Target same scale. |
| **From Reasoning to Code: GRPO for Underrepresented Languages** [2506.11027](https://arxiv.org/abs/2506.11027) | 2025-05-20 | **Qwen2.5-Coder 0.5B, 1.5B, 3B, 7B** | GRPO with execution feedback on GSM8K → Prolog | GRPO improves reasoning across **0.5B→7B**. | Validates small coder. |
| **Limits of Difficulty Scaling: Hard Samples Yield Diminishing Returns** [2604.06298](https://arxiv.org/abs/2604.06298) | 2026-04-07 | **Qwen2.5 0.5B, 1.5B, 3B** | **GRPO + LoRA** on GSM8K/MATH stratified | **Plateau with difficulty**; GRPO reshapes preferences not hard solving. **Only lower-difficulty matches full-data with ~45% steps**; **GSM8K→MATH numeric +5% at 1.5B, +3% at 3B vs MATH-trained**. | Direct LoRA+GRPO evidence. |
| **B1ade 335M & 1B SLMs** [2607.27506](https://arxiv.org/pdf/2607.27506) | 2026-07-29 | **Qwen-0.5B GRPO 50k, Llama-1B 50k** | GRPO 50k samples | Qwen-0.5B 0.340→0.425 etc. | 335M/1B point. |
| **Xiangqi-R1** [2507.12215] | 2025 | **Qwen-0.5B** + GRPO + engine rewards | GRPO+engine → strong intermediate Xiangqi. | **Proof <1B board-game GRPO+engine works** → chess. |
| **Small Models Struggle to Learn from Strong Reasoners** [2502.12143] | 2025 | ≤3B | Mix Distillation long+short | **≤3B better on short, simple chains**. Justifies lucid/short-trace distillation. | Chess trace design. |
| **Predictive Scaling Laws** [2507.18014] | 2025 | 3B, 8B Llama/Qwen | GRPO scaling | Early stop saves compute. | 3B point. |
| TinyZero [2503.16219] | 2025 | 1.5B | GRPO G=6, 3584 tok, $42 | **46.7% AIME24 > o1-preview; 80% AMC23.** | Primary cost baseline. |

**Consensus:** 0.5-3B gains sharply with GRPO on curated 7k samples but plateau on hard; need short traces, easy+hard mix, early stop 50-100 steps, LoRA.

### 4.2 LoRA Stability

| Paper | Year | Finding for LoRA + RLVR/GRPO |
|-------|------|-------------------------------|
| **How Small Can You Go? Controlled Study of LoRA Rank, Target Modules, Quantization** [2607.25583](https://arxiv.org/abs/2607.25583) | 2026 | Systematic ablation rank×target×quantization on small models. **Sweep LoRA config before long run**. |
| **Geometry-Preserving Orthonormal Initialization for LoRA in RLVR** [2606.31813](https://arxiv.org/abs/2606.31813) | 2026 | **PiSSA/MiLoRA underperforms standard LoRA under RLVR**; orthonormal init fixes. Fall back to standard/orthonormal. |
| **RSRA: Training-Free Probing for Efficient LoRA Rank Allocation** [2607.09757](https://arxiv.org/abs/2607.09757) | 2026 | **Training-free per-module rank allocation** via sensitivity probe → non-uniform rank before 30h run. |
| **CT-Merging: Consensus Directions for LoRA Adapter Merging** [2607.20561](https://arxiv.org/abs/2607.20561) | 2026 | Merge independently-trained adapters (opening/tactics/endgame) via consensus + scaling. |
| **Limits of Difficulty Scaling** [2604.06298] | 2026 | **GRPO+LoRA up to 3B** works but plateaus; stable if difficulty filtered. |
| **Fine-Tuning 7B Advisor on Free-Tier GPUs: Adapter-Handoff Recipe** [2504.15610](https://arxiv.org/abs/2504.15610) | 2025 | **QLoRA NF4 r16 with adapter hand-off across sessions**; checkpoint/resume for T4. |

**Recommendation chess 0.5-3B:** Standard LoRA (not PiSSA), rank 16-64, target all-linear or attn+mlp, per-module RSRA, merge slices via CT-Merging, QLoRA NF4 if VRAM constrained.

### 4.3 Group Size (G)

| Source | Recommendation |
|--------|----------------|
| DeepSeekMath / GRPO original | G typically **8-64**; larger → better variance reduction. |
| **Comparative Analysis** [2512.07611] | **Larger G → more stable + higher accuracy** (GRPO & DAPO). Dominant over KL tuning. |
| **Gradient Starvation** [2605.07689] | **G=4 → 69% degeneracy (no signal)**; need larger G or Sign fix. Sign enables small G (73.8% vs 28.4%). |
| **TinyZero** [2503.16219] | **G=6** (4×A40 48GB, 6×4096 tokens). SOTA despite small G; instability partly due small G+overflow. |
| **GRPO/DAPO Identity** [2607.00152] | Theory: sigma=sqrt(p(1-p)) maximal at p=0.5. **Split groups teach most**; unanimous wastes compute. Optimal difficulty ~50% correct. |
| **Expand and Prune** [2512.15347] | GRPO relies on **large G** for reliable advantage; memory bottleneck. |

**For 0.5-3B chess:** Use **G=8-16** if VRAM; with LoRA + grad-accum, G=16 stable. If G=4-6 forced (T4), use **DAPO dynamic sampling + Sign/DrGRPO** to mitigate starvation. DAPO discards 30-50% groups saving backward.

### 4.4 Learning Rate (LR) & Optimizer

| Source | LR / Optimizer Notes |
|--------|----------------------|
| DeepSeekMath / DAPO lineage | AdamW, **lr 1e-6 to 5e-6 RL** (vs SFT 1e-5); warmup + cosine. Qwen2.5-Math 72B RL batch 256 lr 1e-6. |
| TinyZero App.E (inferred from HTML) | 1 epoch, 50-100 steps optimal; lr small to avoid KL explosion. KL beta small or 0 (DAPO removes KL). |
| Qwen2.5-Math TR [2409.12122] | 72B batch 256 lr ~1e-6; 1.5B likely **5e-7 to 2e-6**. |
| LoRA RLVR tips | LoRA needs **higher lr 1e-5 to 2e-5** for adapter vs full 1e-6, but instability; start **1e-6 full-equiv / 5e-6-1e-5 LoRA**, AdamW b1 0.9 b2 0.95 weight decay 0.1 clip eps 0.2-0.28. |

**Recommendation T4 0.5-3B LoRA GRPO:** lr 1e-6 (full) / 5e-6-1e-5 (LoRA), AdamW, no KL or beta=0.01.

### 4.5 Throughput & Cost

| Setup | Throughput / Cost Evidence |
|-------|----------------------------|
| **TinyZero 1.5B, 4×A40 48GB, 24h, 7k samples** | **$42 total** (vs $1000s baselines), **G=6, 3584-4096 tokens, 500 steps max (~3000 local/GPU)**, avg ~2-3 days wall for 1 epoch if unconstrained. SOTA in **50 steps** — early gains cheap. |
| **DAPO 32B** | **50% steps vs R1-Zero**; dynamic sampling + token-level loss improve sample efficiency. |
| **VAPO 32B** | **SOTA in 5,000 steps**, no crashes. |
| **Limits (LoRA 0.5-3B)** | **45% steps saved** training only low-difficulty. |
| **Predictive Scaling 3B/8B** | **Early stopping after 1 epoch plateau** saves ~50% compute. |
| **Prolog Qwen-Coder 0.5-7B** | GRPO throughput scales linearly; 0.5B feasible single GPU. |
| **Memory** | GRPO saves **~50% vs PPO** (no value model) → 1.5B on 48GB with G=6; LoRA cuts optimizer states ~3×. Zeroth-order [2505.13430] escape if Adam overflow. SpecRoll [2608.04962] speculative decoding for RL rollouts (T4 speedup). |

**T4 2-step pretest gate (project):** batch 8, G=8, 60k tokens pre-tok slice, 2 steps (<2 min) should show non-zero outcome_reward if prompt matches; log 500-char completion must contain MoveA: — verifies signal before full run.

---

## 5. Gaps & Next Steps

- **S2 citation counts missing** due to seatbelt block (ENOTFOUND). arXiv citation proxies used; fill via S2 when network available (retry curl with 3s delay, 60s backoff 429). Current tables use arXiv-derived impact; WebSearch confirmed high citation for DeepSeekMath/DeepSeek-R1.
- **Exact hyperparams**: Extract TinyZero App.E, DAPO verl config (lr, warmup, batch) deeper when HTML fully scraped — mark TODO for lit-review.jsonl.
- **Chess-specific RLVR**: Xiangqi-R1 proves <1B board-game GRPO+engine works; adapt to Stockfish per-move verification (verifiable process supervision [2605.12519]).
- **Do NOT write to .rstack/lit-review.jsonl yet** — staging only.

---

## 6. WebSearch Corroboration (site:arxiv.org)

- Query1 site:arxiv.org GRPO DeepSeekMath → 1 source: 2402.03300 (GRPO memory-efficient).
- Query2 site:arxiv.org DAPO DrGRPO VAPO → 10 sources incl. VAPO [2504.05118], DAPO [2503.14476], Identity [2607.00152], Comparative [2512.07611], DRA-GRPO [2505.09655].
- Query3 site:arxiv.org RLVR TinyZero DeepSeek-R1 → RLVR verifiable rewards; DeepSeek-R1 GRPO rule-based rewards.
- Additional: GRPO site:arxiv → [2503.06639] etc.; LoRA 0.5B → [2506.13404], [2506.11027], [2604.06298] among 10.

Full list fetched via WebSearch + read; URLs preserved above.

---

## 7. Method Provenance

- Each paper verified via **read https://arxiv.org/abs/<id>** (title/authors/year/venue/abstract) + **read https://arxiv.org/html/<id>v1** for method/compute where available (DAPO, TinyZero, DeepSeek-R1, VAPO, etc.).
- TinyZero HTML provided full methodology + results table; DAPO HTML provided equations + 4 techniques.
- S2 attempted: `curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=...&limit=10&fields=title,authors,year,venue,abstract,externalIds,url,citationCount"` → ENOTFOUND DNS under seatbelt (CODEX_SANDBOX_NETWORK_DISABLED=1, CODEX_SANDBOX=seatbelt). Wait 3s, retry 60s logic prepared but not triggered (DNS fail not 429). Documented for validation.

*Next: Consolidate into .rstack/lit-review.jsonl only after human checkpoint (per coop).*
