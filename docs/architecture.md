# ChessReasoner-120M — Architecture and Training Plan

A chess-native language model trained from scratch to **play chess well and
explain its play in natural language**, small enough to run on a phone, with no
engine and no search at inference.

Figure: `paper/figures/chessreasoner_arch.tex` → `chessreasoner_arch.pdf`.

---

## 1. Motivation

Two results bound the problem.

- **Karvonen, *Chess-GPT's Internal World Model*** — a 25M-parameter GPT trained
  only on Lichess PGN reaches ~1500 Elo, and its internal board state is
  linearly probeable at ~99% accuracy. Small transformers learn a real world
  model of the board.
- **Ruoss et al., *Grandmaster-Level Chess Without Search* (DeepMind, 2024)** —
  270M parameters trained on ~10M games annotated with Stockfish action-values
  reach 2895 blitz Elo with no search at inference.

So *playing* chess at small scale is solved. What is unsolved is a small model
that plays strongly **and narrates why in natural language**. Both results above
are policy networks: they emit a move and nothing else. Frontier reasoning
models narrate but are enormous and slow — our measured DeepSeek V4 Flash
reference spends **19,345 output tokens per position** for 91.0% on MATE
selection.

The gap this model targets: strong play, verbalized, at ~120 MB.

### 1.1 The design principle

> **Verbalize the search. Internalize the evaluation.**

Explicit text can express a search tree but not a deep one — each ply of
verbalized line costs 30–50 tokens, so five candidates at four ply is already
~800 tokens, and ten ply is not expressible at any sane length. A strong static
evaluation is thousands of tuned terms and is not verbalizable at all.

Our own measurement supports the split. On the MATE held-out set
(n=1000, `data/positions/mate-selection-test.json`):

| Method | Accuracy |
|---|---|
| Chance | 50.0% |
| Longer-explanation heuristic (no board) | 64.3% |
| Gemma 4 E2B, 4-bit, ~1150 output tokens | ~62% |
| Hand-written 2-ply forcing heuristic | 79.1% |
| **Stockfish depth 1 (quiescence only)** | **92.0%** |
| Stockfish depth 4 | 97.5% |
| Stockfish depth 8 / 14 | 100% |
| DeepSeek V4 Flash, thinking, ~19,345 tokens | 91.0% |

Depth-1 Stockfish — quiescence resolution and a static evaluation, no real
search — already beats the frontier model. The verbalized part only has to reach
a few *forcing* plies. Everything else must live in the weights.

### 1.2 The scaffolding constraint

Expanding the FEN into a piece list, or pre-applying candidate moves, hands the
model the exact sub-skill under test. It is off the table for input.

The alternative is to scaffold the **training objective** instead: auxiliary
heads that force an accurate internal board during training and are deleted
before inference. Input stays raw. Output stays pure natural language. The
evaluation is uncontaminated.

---

## 2. Model

### 2.1 Backbone

| | |
|---|---|
| Layers | 18 |
| $d_\text{model}$ | 768 |
| Attention | GQA, 12 query heads / 4 KV heads, $d_h = 64$ |
| FFN | SwiGLU, hidden 2048 |
| Normalization | RMSNorm, pre-norm, fp32 accumulation |
| Positional | dual-scheme RoPE (§4) |
| Biases | none |
| Embeddings | tied input/output |
| Vocabulary | 8192 |
| Context | 1024 |

Exact parameter count:

```
per layer   attention   W_q 768×768   =   589,824
                        W_k 768×256   =   196,608
                        W_v 768×256   =   196,608
                        W_o 768×768   =   589,824
            SwiGLU      3 × 768×2048  = 4,718,592
            RMSNorm     2 × 768        =     1,536
                                       -------------
                                          6,292,992
× 18 layers                             113,273,856
embeddings (tied)       8192×768       =   6,291,456
final norm                                       768
                                       -------------
backbone total                          119,566,080   ≈ 119.6 M
```

**Aspect ratio.** 18 layers at width 768 is deeper than the usual 120M
configuration (GPT-2 small and Pythia-160M are both 12 × 768). Depth is chosen
deliberately: serializing a search procedure is sequential composition, and
depth buys composition steps that width does not.

**Deployment.** 119.6M parameters ≈ 120 MB at int8, ≈ 240 MB at fp16. KV cache
at full 1024 context: $18 \times 2 \times 4 \times 64 \times 1024 \times 2$ bytes
= **18.9 MB**. GQA at 4 KV heads is chosen for exactly this — a 12-KV-head model
would need 56.6 MB of cache, which starts to matter on a phone.

### 2.2 Auxiliary heads (training only)

| Head | Shape | Fires at | Target |
|---|---|---|---|
| Board | $768 \to 64 \times 13$ | each `</LINE>` | occupancy at the leaf of the analyzed line |
| Value | $768 \to 128$ | `</FEN>` | HL-Gauss WDL distribution |
| Policy | $768 \to 1968$ | `</FEN>` | Stockfish best move |

Total 2.25M parameters, **discarded before inference**.

$$\mathcal{L} = \mathcal{L}_\text{LM} + \lambda_b \mathcal{L}_\text{board} + \lambda_v \mathcal{L}_\text{value} + \lambda_p \mathcal{L}_\text{policy}$$

with $\lambda_{\{b,v,p\}}$ annealed to 0 across the second half of training, so
the model never learns to lean on heads it will not have.

**Why the board head fires at `</LINE>`, not at `</FEN>`.** At `</FEN>` the board
is verbatim in context and the head is a copy — it teaches nothing. At the end of
an analyzed variation the model must have *simulated* the moves; the board there
is not copyable from anywhere. That placement is what makes it a world-model
objective rather than a parsing objective. A low-weight `</FEN>` term is retained
purely as an early-training parsing signal.

**Why value is a distribution, not a scalar.** Following Ruoss et al., regressing
a scalar centipawn value is badly conditioned; a 128-bin categorical target over
win probability with HL-Gauss smoothing (Gaussian centered on the true WDL,
σ ≈ 0.75 bin widths) trains far more stably.

**Why a policy head at all.** It is a candidate-generation prior. Without it the
model spends its verbalized budget analyzing moves no strong player would
consider. With it, the LM's candidate proposals are pulled toward the engine's
move distribution even though the head itself is thrown away.

---

## 3. Tokenizer

Vocabulary 8192, allocated:

| Range | Count | Contents |
|---|---|---|
| 0–15 | 16 | `<pad> <bos> <eos> <unk> <FEN> </FEN> <THINK> </THINK> <LINE> </LINE> <CAND> <MOVE> <EVAL>` … |
| 16–79 | 64 | square tokens `a1` … `h8` |
| 80–92 | 13 | piece tokens `P N B R Q K p n b r q k` and empty |
| 93–108 | 16 | promotion, check/mate, capture, castling, side-to-move, en-passant flags |
| 109–128 | 20 | digits and small integers |
| 129–8191 | ~8060 | BPE wordpieces learned on the corpus prose |

The chess tokens are **forced atomic before BPE ever sees the text**, so BPE
cannot split `f3` into `f` + `3` or merge `f3g5` into a single opaque unit.

Three consequences worth stating explicitly:

1. **Square embeddings are shared across roles.** `f3` is the same vector whether
   it names a board slot, a piece location in prose, or a move endpoint. The
   model's notion of the square is one object, not three.
2. **Moves are unambiguous by construction.** A move is exactly
   `(from, to[, promo])` = 2–3 tokens. The failure we observed in the Gemma
   baseline — reading `f5f6` as the single square `f5` and then reasoning about
   the wrong move for 400 tokens — is not expressible in this vocabulary.
3. **The board plane is emitted in full, not run-length encoded.** All 64 squares
   are written out, so the total board span is 70 tokens (`<FEN>` + 64 + `</FEN>`
   + side-to-move + castling + en-passant) against roughly 80 for a raw FEN under
   Gemma's tokenizer.

   The win here is **not compression** — it is *positional regularity*. Square
   $i$ is always at a computable offset from `<FEN>`, so an attention head can
   implement "scan the a1–h8 diagonal" as a fixed stride. Run-length FEN
   (`2r2r2/2qbbppk/...`) makes every square's position depend on decoding the
   digits before it.

---

## 4. Dual-scheme rotary encoding

Split $d_h = 64$ into three rotary bands:

```
 ┌──────────────────────────┬───────────────┬───────────────┐
 │  sequence phase 32 dims  │  rank 16 dims │  file 16 dims │
 └──────────────────────────┴───────────────┴───────────────┘
```

- **Board tokens** rotate all three bands. A token at square $s$ carries phase
  $(\text{pos}, \text{rank}(s), \text{file}(s))$.
- **Prose tokens** zero the rank/file bands, reducing to standard 1-D RoPE.

File neighbours are then at a fixed rotary offset of $+1$ in the file band, rank
neighbours $+1$ in the rank band, diagonal neighbours $+1$ in both. Board
geometry becomes a learnable stride pattern instead of something attention has to
recover from the token sequence.

This is the piece of the design that owning the model buys and that no amount of
fine-tuning a general LM can replicate.

**Ablation required:** dual-scheme vs. plain 1-D RoPE, same everything else. If
this does not measurably improve board-probe accuracy, it should be cut from the
paper.

---

## 5. Data

Everything is generated. All sources CC0.

### 5.1 Raw material

| Source | Scale | Provides |
|---|---|---|
| Lichess games DB (monthly PGN) | billions of games | positions, human move distributions, game continuity |
| **Lichess evaluation DB** | ~500M positions, pre-annotated | cp/mate, depth, full PV — **no Stockfish run required** |
| Lichess puzzle DB | 4M+ | tactical positions with named motif themes |
| Stockfish 17 | — | gap-fill only |

The evaluation DB is the unlock. It removes annotation as the bottleneck — the
expensive part of this pipeline was already done by Lichess, and the repo already
downloads it (`scripts/build_bestmove_evals.py`).

### 5.2 Corpus tiers

| Tier | Content | Examples | Tok/ex | Tokens |
|---|---|---|---|---|
| 1 | Board literacy | 6.0M | 95 | 570M |
| 2 | Tactical primitives | 5.0M | 155 | 775M |
| 3 | Full reasoning traces | 3.0M | 320 | 960M |
| 4 | Game continuity | 1.5M | 280 | 420M |
| 5 | General English (FineWeb-Edu) | — | — | 480M |
| | | | **total** | **≈ 3.2B** |

**Tier 1 — board literacy.** Pure perception, no reasoning. All generated from
python-chess, all trivially correct:

- FEN → full piece list
- "What is on e4?" / "Where are Black's rooks?"
- FEN + move → resulting board
- FEN → legal moves of the piece on ⟨sq⟩
- FEN → is the king in check, and from what
- FEN → every attacker and defender of ⟨sq⟩
- FEN → all currently hanging pieces

This tier is where the 62% problem actually gets fixed. It is free and unlimited.

**Tier 2 — tactical primitives.**

- Static exchange on a square, verbalized ply by ply
- "Is this a blunder?" with the concrete refutation
- Motif naming grounded in Lichess puzzle themes (fork, pin, skewer, deflection,
  discovered attack) — the themes are human-labelled, so the vocabulary is
  correctly attached
- Mate-in-1 and mate-in-2 with the line

**Tier 3 — full reasoning traces.** The core. Generated from evaluation-DB
multipv. Format in §6.

**Tier 4 — game continuity.** PGN prefix + current position → move with
reasoning. Teaches plans across moves rather than isolated tactics. This is what
makes the 100-game match work; a position-only model plays each move as if the
game started there.

**Tier 5 — general English at ~15%.** Buys robustness to phrasings the templates
never produced. Without it the model is fluent in chess-English and mute
elsewhere, which makes any human interaction brittle.

### 5.3 Three generation rules that decide whether this works

1. **Randomize everything structural.** Number of candidates discussed (2–5),
   their order, which static features get mentioned, whether material or threats
   comes first, and the surface phrasing (sample from ~200 hand-written templates
   per claim type). *If the best move is always discussed last, the corpus has a
   positional shortcut* — precisely the pathology already measured in MATE, where
   a bag-of-words classifier with no board access scores 63.7%.

2. **Every claim is generated from python-chess ground truth**, so the corpus is
   **100% factually correct by construction**. This is what distillation from a
   frontier model cannot offer: our DeepSeek traces contain fabricated engine
   evaluations ("Stockfish evaluation at depth 28 … +0.2 to +0.4" on a position
   never analyzed).

3. **Include corrective traces.** A fraction must explore a candidate, discover
   the refutation, and reject it. A corpus of only-correct-first-guess traces
   teaches confident assertion, not search.

### 5.4 Anti-leakage

Split by **game ID and by position**, not by row. Puzzle positions derived from
the same game must not straddle the split. Hold out MATE's held-out set entirely,
plus the existing `mate1-lichess`, `mate2-lichess`, `bestmove-8x8` sets.

---

## 6. Trace format

What the model emits at inference — raw FEN in, this out:

```
<THINK>
Material is level. My bishop on c4 bears on f7, defended only by the king.
The knight on f3 can join with Ng5, attacking f7 twice.

<CAND> f3 g5 </CAND>
  <LINE> f3g5 d7d5 e4d5 c6a5 </LINE>
  Black blocks the diagonal and I trade into a slightly better structure.
  <LINE> f3g5 g8h6 c4f7 h6f7 g5f7 </LINE>
  f7 falls: a pawn and the exchange.

<CAND> c4f7 </CAND>
  <LINE> c4f7 e8f7 f3g5 f7g8 </LINE>
  Two pieces for rook and pawn, and the initiative is gone. Premature.

Ng5 keeps the threat and forces a concession.
</THINK>
<MOVE> f3 g5 </MOVE>
```

Properties that matter:

- Forcing lines only, 2–4 ply. This is text-mode quiescence, and §1.1 shows
  quiescence is worth 92%.
- The board head fires at each `</LINE>`, so the model is scored on whether it
  actually tracked the position to the leaf.
- Every claim is checkable against python-chess — which is what makes the
  factuality metric in §9 possible.
- The choice is justified by a concrete variation, never by "space" or
  "mobility". The Gemma baseline's failure mode is exactly the reverse.

---

## 7. Training

### 7.1 Curriculum

| Stage | Tokens | Content | Active heads |
|---|---|---|---|
| 1 | 570M | Tier 1 — board literacy | board (high λ), value |
| 2 | 775M | Tier 2 — tactical primitives | board, value, policy |
| 3 | 960M | Tier 3 — full traces | all, annealing |
| 4 | 420M | Tier 4 — game continuity | annealed → 0 |
| — | 480M | Tier 5 English, **mixed throughout** at 15% | LM only |

Stage 5 (post-training, separate budget): **rejection-sampling SFT**. Sample $k$
traces per position, keep only those where the final move is correct **and** every
stated board fact verifies against python-chess. That double filter is the
difference between reinforcing reasoning and reinforcing lucky guesses with
hallucinated boards. Iterate 2–3 rounds. Optionally follow with ORPO on
(verified, refuted) trace pairs — no reference model, no reward model, no rollout
machinery.

### 7.2 Hyperparameters

| | |
|---|---|
| Optimizer | AdamW, $\beta=(0.9,0.95)$, weight decay 0.1 |
| LR | 3e-4 peak, cosine → 3e-5, 2000-step warmup |
| Grad clip | 1.0 |
| Precision | **fp16 + dynamic loss scaling** (T4 is Turing — no bf16) |
| Micro-batch | 16 × 1024 tokens |
| Grad accumulation | 16 → 262,144 tokens/step |
| Total steps | ≈ 12,200 |
| Attention kernel | PyTorch SDPA memory-efficient backend |

**T4 gotchas, in advance.** FlashAttention-2 requires Ampere (sm\_80+); the T4 is
sm\_75, so it will not build — use SDPA's memory-efficient backend, which does
support Turing. There is no bf16, so RMSNorm and the loss reduction must
accumulate in fp32 or training will diverge around step ~2k.

### 7.3 Compute

$$6ND = 6 \times 1.196\times10^{8} \times 3.2\times10^{9} \approx 2.3\times10^{18}\ \text{FLOPs}$$

T4 fp16 peak is 65 TFLOP/s; realistic MFU for a model this size is 25–30%,
so ~16–20 TFLOP/s achieved.

$$2.3\times10^{18} / 1.8\times10^{13} \approx 1.28\times10^{5}\ \text{s} \approx 35\ \text{h}$$

Add auxiliary heads, dataloading and checkpointing: **≈ 40 T4-hours** for the full
pretraining run. Memory: fp16 weights 0.24 GB + fp32 master and Adam moments
1.44 GB + activations ~2.3 GB at micro-batch 16 ≈ **4.3 GB**, comfortable inside
16 GB. Micro-batch 32 also fits if throughput favours it.

**Kaggle logistics.** The weekly GPU quota is 30 h and sessions cap at 9–12 h, so
40 h is ~5 sessions across two calendar weeks — or half that wall-clock on a
2×T4 kernel with DDP. The repo already has the checkpoint/resume and crash-safe
patterns this needs (`--resume`, schema-versioned cell skipping).

**Data generation** is CPU-bound and embarrassingly parallel; the evaluation DB
removes the Stockfish requirement for Tier 3, and Tiers 1–2 are pure
python-chess. Reuse the existing multi-worker Kaggle pattern.

---

## 8. Inference

FEN in, prose out. No engine, no search, no tool calls, no scaffolded input.
Auxiliary heads are not loaded. Greedy or low-temperature sampling; the `<MOVE>`
span is the answer and is parsed with the existing strict extractor in
`src/benchmarks/games/tasks.py`.

Expected budget: ~300–500 output tokens per decision, against DeepSeek's measured
19,345.

---

## 9. Evaluation

Trained on none of these.

| Metric | Why |
|---|---|
| **Puzzle Elo** on held-out Lichess puzzles across the rating spread | a number on a known scale — the most legible generality metric available |
| **Playing Elo** vs Stockfish levels 1–8 | tests generality no MATE-shaped eval can |
| 100-game match vs DeepSeek V4 Flash | the project's stated objective |
| `mate1-lichess`, `mate2-lichess`, `bestmove-8x8`, MATE held-out | existing batteries, transfer check |
| **Board-probe accuracy** | linear probe on frozen residuals for 64-square occupancy; evidences the world model and should climb across curriculum stages |
| **Trace factuality rate** | fraction of checkable claims in a generated trace that verify against python-chess |
| Tokens per decision | the efficiency axis |

**Trace factuality is the metric to lead with.** Nobody reports it, it is fully
automatable, and it measures faithful reasoning rather than leaderboard position.
If a 120M model produces 95%-factual traces where DeepSeek V4 Flash produces 70%
at 17× the tokens, that is a stronger workshop claim than beating it on accuracy —
and it is a claim about *reasoning*, which is what this project is actually about.

**Shortcut controls, mandatory.** Every MATE-family number must be reported
alongside always-A, longest-explanation, and text-only-TF-IDF baselines, plus an
explanation-stripped arm. A bag-of-words classifier scores 63.7% on that set; any
result not compared against it is not defensible.

---

## 10. Ablations for the paper

| # | Ablation | Question |
|---|---|---|
| A1 | − board head | Does world-model supervision matter, or does it emerge anyway? |
| A2 | − value head | Is internalized evaluation load-bearing? |
| A3 | − policy head | Does the candidate prior matter? |
| A4 | plain 1-D RoPE | Is dual-scheme encoding worth its complexity? |
| A5 | subword tokenizer (no atomic squares) | Isolates the tokenizer contribution |
| A6 | − Tier 1 (skip board literacy) | Is the curriculum's first stage necessary? |
| A7 | 60M / 120M / 240M | Scaling — how small can this go? |
| A8 | − Tier 5 English | Cost of dropping general language |

A1 and A4 are the paper's core claims. A7 is the "on-device" story.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| **Templated prose → template completion, not reasoning** | ~200 templates per claim type, randomized structure, 15% natural English; measured by held-out-phrasing eval and trace factuality |
| Model can only speak chess | Accepted and stated; Tier 5 limits the damage. If it matters more than expected, the fallback is Option C (fine-tune a 270M pretrained base) |
| Verbalized search caps at ~4 forcing plies | By design — §1.1 shows quiescence is worth 92%. Deeper positional judgment must come through the value head's influence on the LM |
| **The corpus teaches a shortcut we did not anticipate** | Run the same shortcut probes on our own corpus that we ran on MATE: train a no-board TF-IDF classifier on generated traces; if it exceeds ~55%, the generator leaks |
| fp16 divergence on T4 | fp32 accumulation in norms and loss; loss-scale monitoring from step 0 |
| 40 h across 9 h Kaggle sessions | Existing checkpoint/resume infrastructure; 2×T4 DDP if wall-clock matters |

**The honest uncertainty:** whether 120M parameters trained on 3.2B tokens of
synthetic chess text will *reason* or merely pattern-match is not known in
advance. The ablations and the factuality metric are designed to detect the
difference rather than paper over it. Given Karvonen's 25M model reaching 1500
Elo, a defensible expectation is roughly 1800–2200 puzzle Elo — a guess, and
flagged as one.

---

## 12. Build order

| # | Deliverable | Depends on |
|---|---|---|
| 1 | Tokenizer: atomic chess tokens + BPE over prose | — |
| 2 | Tier-1/2 generators (pure python-chess) | 1 |
| 3 | Board-literacy sanity run — 20M-param model, 200M tokens | 1, 2 |
| 4 | Tier-3 generator from the Lichess evaluation DB | 1 |
| 5 | Model code: backbone, dual RoPE, three heads | 1 |
| 6 | T4 throughput calibration — measure MFU, confirm the 40 h estimate | 5 |
| 7 | Full corpus generation (3.2B tokens) | 2, 4 |
| 8 | Pretraining run | 5, 6, 7 |
| 9 | Rejection-sampling SFT rounds | 8 |
| 10 | Evaluation harness: puzzle Elo, playing Elo, factuality | — (parallel) |
| 11 | Ablations | 8 |

Step 3 is the go/no-go gate. If a 20M model on 200M tokens of Tier-1 data cannot
learn to read a board and answer "what is on e4", nothing downstream will work,
and that costs about four T4-hours to find out.

---

## 13. Adversarial review (2026-08-06)

Findings against §1–12 before any code was written. Three of the four checked
claims came back against the design; the fixes are folded into §14.

### R1 — Dual-scheme RoPE double-counts the raster's benefit. **Demoted.**

§4 claims the dual-scheme encoding makes board geometry a fixed stride pattern.
But that is *already* what the fixed a1→h8 raster does, and it was §3's own
argument. Measured on `chess.SQUARE_NAMES` order:

```
file neighbour  e4->f4   offset  +1
rank neighbour  e4->e5   offset  +8
diagonal        e4->f5   offset  +9
diagonal        e4->d5   offset  +7
```

These are fixed 1-D offsets and standard RoPE encodes them well. The dual scheme
fixes exactly one thing the raster does not: **file wrap-around**. `h1->a2` is
also `+1` but is not a file step, which corrupts ~10.9% of file-neighbour pairs.

Real but small. Costs half the rotary dimensions (32 of 64) taken from sequence
phase, and the rank/file bands need their own rotary base (θ tuned for a range of
8, not 10000 — otherwise 8 frequency pairs over 8 positions are nearly constant
and the dims are wasted). **Not built in v1.** Demoted from a core claim to
ablation A4, expected small. If the paper leads with this it is overclaiming.

### R2 — 57% of board-plane tokens are the empty symbol. **Loss weighting required.**

Measured over 3000 random-playout positions: mean 36.2 empty squares of 64.

Under an unweighted LM loss the model gets very low loss on a majority of tokens
that carry almost no information, diluting the gradient. But fully masking the
board plane is also wrong — predicting it teaches a prior over plausible piece
configurations, which is most of what Chess-GPT learns from PGN.

**Fix:** emit per-token segment ids (`board` / `prompt` / `answer`) from the
tokenizer and let the trainer weight them. Default: board 0.1, prompt 0.0,
answer 1.0. This is a tunable, not a constant, and A-B testing it is cheap.

Consequence for §5.2: the corpus is ~3.2B tokens of *compute* but only ~2.1B of
*supervised signal*. FLOP estimates in §7.3 are unaffected; the token budget
claim needs the qualifier.

### R3 — The policy head knows the answer before the model reasons.

The policy head fires at `</FEN>`, before any reasoning tokens exist. Training it
to predict Stockfish's move there teaches the model the answer *prior to*
thinking, which is the textbook setup for post-hoc rationalization: the trace
becomes decoration over an answer already decided.

This is defensible as a System-1 / System-2 split — intuition proposes,
verbalized search verifies — and it is why the head is deleted at inference. But
it must be *measured*, not assumed. **New required experiment:** compare the
policy head's own top-1 accuracy against the full model's post-reasoning
accuracy. If they match, the reasoning is decorative and the paper must say so.

### R4 — The 1968-move vocabulary claim. **Verified, but compute it anyway.**

Recomputed from scratch (all sliding + knight from-to pairs, plus the four
promotions on each pawn-advance to the last rank): **1968**. The asserted number
was right. It is still hardcoded nowhere — `vocab.py` derives it, so a future
change to the encoding cannot silently desynchronize the head.

### R5 — Position-level leakage is worse than §5.4 says.

Lichess puzzles are *derived from* Lichess games. Splitting by game ID does not
prevent an evaluation puzzle's position appearing in Tier-4 training data drawn
from the same game corpus. Dedup must be **by position hash across the entire
corpus against every evaluation set**, not by game ID.

### R6 — "100% factually correct by construction" needs a scope.

python-chess can verify occupancy, legality, attacks, captures, checks and
material. It cannot verify "gains space", "the initiative is gone", or "improves
the worst piece". **Fix:** the generator emits claims only from a closed set of
machine-checkable predicates. The guarantee then holds exactly, at the cost of
drier prose. Any evaluative language must be introduced later and explicitly
excluded from the factuality metric.

### R7 — Template memorization is untestable as specified.

§11 lists it as a risk with no instrument. **Fix:** every template family
reserves a held-out pool that never appears in training. Reporting accuracy on
held-out phrasings alongside seen phrasings makes memorization visible instead of
hypothetical.

### R8 — Compute estimate is optimistic.

25–30% MFU on a T4 for a 120M model is high; small models are
memory-bandwidth-bound and the T4 has 320 GB/s. 15–20% is likelier, which moves
35 h to **50–60 h**. §7.3's range should be stated as 40–60 h pending the step-6
calibration, and micro-batch 16 × 1024 activations may need gradient
checkpointing — also a measurement, not an assumption.

---

## 14. v1 decisions (what actually gets built)

| | |
|---|---|
| Positional encoding | **standard 1-D RoPE**; dual-scheme deferred to ablation A4 |
| Board span | fixed **72 tokens**: `<FEN>` + 64 squares + stm + 4 castling + ep + `</FEN>` |
| Loss weighting | per-token segment ids; board 0.1 / prompt 0.0 / answer 1.0, tunable |
| Move vocabulary | derived, not hardcoded (currently 1968) |
| Dedup | position-hash across the whole corpus against all eval sets |
| Generated claims | closed set of machine-checkable predicates only |
| Templates | held-out phrasing pool per family, never trained on |
| Compute | 40–60 T4-hours, pending calibration |

### R9 — The board plane crowds out the supervised signal. **Fixed by packing.**

Measured on a real 50k-example Tier-1 shard built from Lichess puzzle positions:
the 72-token board plane is **81.6%** of every example and the answer is only
**8.0%**. Re-emitting the board per question spends the overwhelming majority of
the compute budget re-reading the same position.

**Fix:** pack 6–10 questions onto one board (`generate_packed` /
`encode_packed`, `--pack MIN MAX`). Measured on a 20k-board shard:

| | board share | answer share | tokens/example |
|---|---|---|---|
| unpacked | 81.6% | 8.0% | 88 |
| packed 6–10 | 38.3% | **26.1%** | 188 |

**3.3× more supervised signal for the same FLOPs**, and a useful side effect:
the model must now hold the board across a long span instead of answering off
the most recent tokens. Tier 3 traces should be packed the same way.

---

## 15. Implementation notes (as built)

`src/chessreasoner/model.py`, `data.py`, `scripts/calibrate_throughput.py`.

Backbone parameter count is **exactly 119,566,080**, matching the hand
computation in §2.1; auxiliary heads are 2,251,632 and detach cleanly via
`strip_aux_heads()`. The gate model (`gate_20m`) is 18,882,432.

Deviations and additions against §2:

* **Board-head placement, revised again.** §2.2 puts the board head at each
  `</LINE>`. Tier-1 data has no reasoning traces, so during the gate run the
  only available anchor is `</FEN>`, which is the low-weight parsing signal §2.2
  already describes. `data.board_targets_for` recovers the 64-square target from
  the token ids at fixed offsets (`</FEN>` at `p` implies squares at `p-70 ..
  p-7`) rather than tracking it alongside, so a span straddling a packing
  boundary is skipped instead of half-read.
* **All auxiliary targets are optional.** Value and policy need engine labels
  that Tier 1 does not carry, so the forward pass skips any head whose targets
  are absent rather than erroring.
* **`aux_scale`** implements the §2.2 anneal: held at 1.0 for the first half of
  training, then linear to 0.
* **Gradient checkpointing** added as a config flag (review item R8) and tested
  to be numerically transparent in both loss and gradients.

Verified by `scripts/test_model.py` (40 checks), beyond the parameter counts:

* attention is causal -- perturbing the last token leaves all earlier hidden
  states bit-identical
* KV-cached decoding matches a full forward pass
* zero loss-weight tokens contribute exactly zero
* board targets, reconstructed through `parts_to_board`, describe the true
  position for every span in a real packed corpus
* untrained losses sit at `ln(vocab_size)` and `ln(13)` as they must
* the model can drive a fixed batch to loss < 0.1, so a failed gate run cannot
  be blamed on the optimizer path

Two bugs the suite caught: an out-of-bounds gather in `board_targets_for`
(`np.where` still evaluates the out-of-range branch, so prose ids above the
lookup table would have raised) and an off-by-one in the board-span slice.

**§7.3's 40-60 h estimate is still unmeasured.** `calibrate_throughput.py`
sweeps micro-batch sizes and reports achieved TFLOP/s, MFU and peak memory
against the T4's 65 TFLOP/s fp16 peak. It must be run on a T4 before the corpus
is generated; it refuses to present an off-GPU projection as the answer.
