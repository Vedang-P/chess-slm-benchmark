"""GRPO RL of a MATE-selection model with Stockfish rewards (rlvr-plan.md).

r = 1.0 * outcome + 0.3 * process + 0.1 * style

- outcome: model's final MoveA/MoveB == oracle best at the position
- process: every UCI move the model mentions is legal (python-chess) and
  eval-stable (|Δeval| <= 100cp vs the position before the move, d12)
- style: brevity bonus (1 - tokens/max) gated on a correct outcome

Oracle: --oracle truth = mock (answer from the pool's expert label, no
engine) for the local CPU smoke; --oracle stockfish = real Stockfish at
--depth (the GPU run). Results are memoized per position.

Milestone 1 (local): validate the whole loop on CPU with a tiny base
model, one batch of rollouts + one optimizer step:
    python3 scripts/build_rlvr_pool.py \
        --train data/positions/noexplain-slice-smoke/train.jsonl \
        --out results/rlvr-pool/smoke.jsonl
    python3 scripts/train_mate_grpo.py \
        --base HuggingFaceTB/SmolLM2-135M-Instruct \
        --train results/rlvr-pool/smoke.jsonl \
        --out results/rlvr-smoke --cpu --smoke --max-steps 1

GPU run (Aug 22+, the gemma4 loader path proven by train_mate_lora.py):
    python3 scripts/train_mate_grpo.py \
        --base google/gemma-4-E2B-it --train results/rlvr-pool/train.jsonl \
        --out results/rlvr-adapter --rank 32 --lr 1e-5 \
        --oracle stockfish --max-steps 3500

The model-agnostic loader is a deliberate split: gemma-4-* takes the
multimodal 4-bit path (AutoModelForImageTextToText + processor), every
other base id takes the plain CausalLM path — so the smoke can run on a
small model through the same trainer/reward/processor code.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

# trl 0.17.0 + transformers >= 5.14 tuple-guard incompatibility: 5.14's
# _is_package_available always returns (bool, version), so trl's
# `_llm_blender_available = (False, None)` is TRUTHY and the guarded
# `import llm_blender` (missing) crashes the whole grpo_trainer import.
# This must run before ANY trl trainer import. Values are normalized to
# the bool the tuple's first element encodes — same semantics 5.13 had.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")  # re-enabled iff --wandb-project
# OOM fix (measured 2026-08-19, pretest v6: 15.13GB of 15.89GB used at the
# first backward; the SFT trainer that runs the same model on the same P100
# sets this + gradient checkpointing and fits batch 2x8 at 2048 tokens).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import trl.import_utils as _trl_iu  # noqa: E402

for _n, _v in list(vars(_trl_iu).items()):
    if (_n.startswith("_") and _n.endswith("_available")
            and isinstance(_v, tuple)):
        setattr(_trl_iu, _n, _v[0])

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---- answer parsing (provenance: scripts/run_mate_eval.py, same regexes
# and last-mention-wins rule so RL rewards score exactly like eval) ----
ANSWER_RE = re.compile(
    r"\bMove\s*([AB])\b\s*[:.\-]?\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)?", re.I)
UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?")

ANSWER_SPEC_FORCED = (
    "Answer with exactly one of: MoveA:<move> or MoveB:<move>. "
    "Only output the line, nothing else.\n"
    "You MUST output an answer. If you cannot determine which move is better "
    "with confidence, output your best guess from the two candidates anyway. "
    "An answer is required; refusing to answer is not acceptable."
)

MOCK_POSITION = ("r1bq1rk1/pp3ppp/1b5n/2p1p1N1/2Bn4/3Q4/PP3PPP/RNB2RK1 w - - 0 12")

# ---- HF checkpoint persistence (AGENTS.md requirement). The pattern is
# train_mate_lora.py's proven one: periodic upload of the latest
# checkpoint dir (adapter + optimizer + scheduler + trainer_state) so a
# killed Kaggle session resumes, final adapter upload at the end, and
# failure status so a crash needs no log download. ----


def _hf_api():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("HF_WRITE_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        raise RuntimeError("no HF_WRITE_TOKEN for HF checkpoint uploads")
    return HfApi(token=token)


class HfCheckpointCallback:
    """Upload the latest trainer checkpoint to HF every --hf-upload-every
    seconds (adapter + optimizer + scheduler + trainer_state, so a killed
    session resumes with --resume-from-hf). Uploads the final checkpoint
    at train end."""

    def __init__(self, api, repo_id: str, remote_dir: str,
                 interval_s: float, checkpoint_root: str):
        self.api = api
        self.repo_id = repo_id
        self.remote_dir = remote_dir.strip("/")
        self.interval_s = interval_s
        self.checkpoint_root = Path(checkpoint_root)
        self._last = time.time()
        self._uploaded = set()

    def _latest_checkpoint(self) -> Path | None:
        cps = sorted(self.checkpoint_root.glob("checkpoint-*"),
                     key=lambda p: int(p.name.split("-")[1]))
        return cps[-1] if cps else None

    def _upload_dir(self, cp: Path):
        rel = cp.name
        files = [f for f in cp.rglob("*") if f.is_file()]
        for f in files:
            rpath = f"{self.remote_dir}/{rel}/{f.relative_to(cp)}"
            self.api.upload_file(path_or_fileobj=str(f), path_in_repo=rpath,
                                 repo_id=self.repo_id, repo_type="dataset")
        self._uploaded.add(rel)
        print(f"[hf-cp] uploaded {rel} ({len(files)} files) -> "
              f"{self.repo_id}/{self.remote_dir}/", flush=True)

    def maybe_upload(self, force: bool = False):
        cp = self._latest_checkpoint()
        if cp is None:
            return
        if cp.name in self._uploaded:
            return
        if force or (time.time() - self._last) >= self.interval_s:
            try:
                self._upload_dir(cp)
                self._last = time.time()
            except Exception as e:
                print(f"[hf-cp] upload failed (will retry): {e}", flush=True)

    def final(self):
        cp = self._latest_checkpoint()
        if cp is not None:
            self._upload_dir(cp)


def download_hf_checkpoint(api, repo_id: str, remote_dir: str,
                           local_root: str) -> Path | None:
    """Fetch the latest checkpoint-* dir from HF into local_root; return
    its path (for Trainer resume_from_checkpoint) or None."""
    from huggingface_hub import hf_hub_download

    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"[resume] cannot list {repo_id}: {e}", flush=True)
        return None
    prefix = f"{remote_dir.strip('/')}/checkpoint-"
    # path layout: <remote_dir>/checkpoint-<N>/<file> — the checkpoint
    # dir name is component [1] (bug found 2026-08-19: [2] selected the
    # file name, so resume pointed at a nonexistent file path).
    cps = sorted({f.split("/")[1] for f in files
                  if f.startswith(prefix) and len(f.split("/")) > 2})
    if not cps:
        print(f"[resume] no checkpoints under {remote_dir} in {repo_id}",
              flush=True)
        return None
    latest = cps[-1]
    print(f"[resume] downloading {remote_dir}/{latest}", flush=True)
    for f in files:
        if not f.startswith(f"{remote_dir}/{latest}/"):
            continue
        hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset",
                        local_dir=str(local_root), token=api.token)
    return local_root / remote_dir / latest


def parse_choice(text: str, candidate_a: str, candidate_b: str):
    """(label, move) or (None, None) — last MoveX mention wins, bare uci
    falls back to the candidate it equals. Canonical: run_mate_eval."""
    if not text:
        return None, None
    matches = list(ANSWER_RE.finditer(text))
    if matches:
        m = matches[-1]
        label = m.group(1).upper()
        move = m.group(2) or (candidate_a if label == "A" else candidate_b)
        return label, move
    for token in reversed(UCI_RE.findall(text)):
        if token == candidate_a:
            return "A", candidate_a
        if token == candidate_b:
            return "B", candidate_b
    return None, None


def build_prompt(fen: str, candidate_a: str, candidate_b: str) -> str:
    """Byte-identical to run_mate_eval's forced-answer model input
    (instruction + input + ANSWER_SPEC_FORCED)."""
    instruction = ("You are an expert chess player. You are given a chess "
                   "board with FEN format. Your goal is to choose a better "
                   "move given two candidate moves.")
    board_text = (f'The FEN of the given chess board is "{fen}". Which move '
                  f"is better? MoveA:{candidate_a} MoveB:{candidate_b} ")
    return instruction + "\n" + board_text + "\n" + ANSWER_SPEC_FORCED


class _ThinkingProcessor:
    """AutoProcessor wrapper that renders gemma4's thought channel when
    --thinking is set (trl 0.17 can't pass enable_thinking itself)."""

    def __init__(self, processor, enable_thinking: bool):
        self._processor = processor
        self._enable_thinking = enable_thinking

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def apply_chat_template(self, conversation, **kwargs):
        kwargs.setdefault("enable_thinking", self._enable_thinking)
        return self._processor.apply_chat_template(conversation, **kwargs)


class Oracle:
    """Position scoring backend shared by the reward functions.

    truth: mock — best move is the pool's expert choice, no engine. Used
        by the CPU smoke so the whole loop validates without Stockfish.
    stockfish: real engine at --depth; best move via search, process
        reward's eval-stability via per-position analysis. Memoized per
        position (fen, candidate_a, candidate_b) and per fen for evals —
        the pool is finite and positions repeat across steps.
    """

    def __init__(self, mode: str, depth: int = 12, stockfish: str = ""):
        self.mode = mode
        self.depth = depth
        self.stockfish = stockfish
        self.engine = None
        self._best = {}
        self._evals = {}
        if mode == "stockfish":
            import shutil
            import chess.engine
            # Kaggle apt installs to /usr/games/stockfish; the homebrew
            # default is the local path. Resolve in order: explicit arg ->
            # PATH -> the two known install locations.
            if stockfish and not Path(stockfish).exists():
                print(f"[oracle] {stockfish} not found; resolving", flush=True)
                stockfish = ""
            if not stockfish:
                stockfish = shutil.which("stockfish") or ""
            if not stockfish:
                for cand in ("/usr/games/stockfish",
                             "/opt/homebrew/bin/stockfish"):
                    if Path(cand).exists():
                        stockfish = cand
                        break
            if not stockfish:
                raise RuntimeError("stockfish binary not found — install it "
                                   "(Kaggle: !apt-get install -y stockfish)")
            print(f"[oracle] stockfish: {stockfish}", flush=True)
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish)

    def close(self):
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def best_label(self, fen: str, a: str, b: str, truth_label: str | None,
                   ) -> str | None:
        """Oracle's answer label for the position, or None if undecided.

        Stockfish mode: pairwise eval comparison of the two candidates, not
        the global PV. The global PV can be a third move C (neither A nor B),
        so PV-vs-candidate mislabels every such position as None -> zero
        reward. Pairwise eval fixes it; the PV is only a fallback when evals
        fail. Tie -> truth_label tie-break to keep expert signal.
        """
        key = (fen, a, b)
        if key in self._best:
            return self._best[key]
        if self.mode == "truth":
            label = truth_label  # mock: the pool's expert answer
        else:
            # Pairwise Stockfish: which candidate gives the better position
            # for the mover. eval_cp returns opponent POV, so smaller opponent
            # score is better for mover.
            try:
                import chess
                board_a = chess.Board(fen)
                board_b = chess.Board(fen)
                try:
                    ma = chess.Move.from_uci(a)
                except Exception:
                    ma = None
                try:
                    mb = chess.Move.from_uci(b)
                except Exception:
                    mb = None
                if ma is None or ma not in board_a.legal_moves or mb is None or mb not in board_b.legal_moves:
                    best = self._best_move(fen)
                    label = ("A" if best == a else "B" if best == b else None)
                else:
                    board_a.push(ma)
                    board_b.push(mb)
                    ea = self.eval_cp(board_a.fen())
                    eb = self.eval_cp(board_b.fen())
                    if ea is None or eb is None:
                        best = self._best_move(fen)
                        label = ("A" if best == a else "B" if best == b else None)
                    else:
                        if ea < eb:
                            label = "A"
                        elif eb < ea:
                            label = "B"
                        else:
                            label = truth_label
            except Exception:
                label = truth_label
        self._best[key] = label
        return label

    def best_move(self, fen: str) -> str | None:
        if self.mode == "truth":
            return None
        return self._best_move(fen)

    def _best_move(self, fen: str) -> str | None:
        """Stockfish's best move at --depth (memoized by fen)."""
        import chess
        board = chess.Board(fen)
        try:
            info = self.engine.analyse(board, chess.engine.Limit(
                depth=self.depth))
        except Exception:
            return None
        pv = info.get("pv")
        return pv[0].uci() if pv else None

    def eval_cp(self, fen: str) -> float | None:
        """cp score of `fen` from the side-to-move's pov (memoized)."""
        if self.mode == "truth":
            return 0.0  # mock: no evals — stability trivially holds
        if fen in self._evals:
            return self._evals[fen]
        import chess
        board = chess.Board(fen)
        try:
            info = self.engine.analyse(board, chess.engine.Limit(
                depth=self.depth))
        except Exception:
            return None
        score = info.get("score")
        if score is None:
            return None
        cp = score.white().score(mate_score=100000)
        val = -cp if board.turn else cp
        self._evals[fen] = val
        return val


def _completion_text(completion) -> str:
    """trl 0.17 passes each completion as an OpenAI-style message list
    ([{role, content}]) or, on some paths, the raw dict/str — normalize
    all three (list shape caught by the 2026-08-18 CPU smoke)."""
    if isinstance(completion, list):
        return "".join(
            (m.get("content") or "") for m in completion if isinstance(m, dict))
    if isinstance(completion, dict):
        return completion.get("content") or ""
    return completion or ""


def _prompt_key(prompt) -> str:
    """Hashable prompt identity: trl passes prompts as message lists too,
    and dict keys must be hashable. Group members share one prompt text,
    so text is the correct outcome-memo key."""
    return _completion_text(prompt)


def _verify_trace(completion: str, fen: str, oracle: Oracle) -> float:
    """Process reward: fraction of the trace's UCI claims that hold.

    Walks the model's UCI moves in order from the position; a move counts
    as verified when it is legal AND eval-stable (|Δeval| <= 100cp vs the
    position before it). The first illegal/unstable move fails the rest of
    the claims (they are unverifiable from an invalid line). An empty
    trace scores 0 — saying nothing about the position is never rewarded.
    """
    import chess
    moves = UCI_RE.findall(completion)
    if not moves:
        return 0.0
    board = chess.Board(fen)
    before = board.fen()
    stable = 0
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        board.push(move)
        after = board.fen()
        ea = oracle.eval_cp(before)
        eb = oracle.eval_cp(after)
        before = after
        if ea is not None and eb is not None and abs(eb - ea) <= 100.0:
            stable += 1
        else:
            break
    return stable / len(moves)


def make_rewards(oracle: Oracle, tokenizer, max_completion_length: int,
                 max_steps: int):
    """The three reward functions as a trl-0.17 callable list. Each gets
    (prompts, completions, **kwargs) where kwargs carries the pool's extra
    columns (fen/candidate_a/candidate_b/truth_label) — verified against
    grpo_trainer._generate_and_score_completions: reward_kwargs = every
    input column except prompt/completion."""
    outcomes = {}  # (prompt_text, completion_index) -> 1/0.
    # Keyed per-completion, NOT per prompt: all group members share one
    # prompt, so a prompt-only key would let the LAST completion's outcome
    # gate the style reward for every member (bug found in the 2026-08-19
    # audit). trl calls the reward funcs with aligned, same-order lists
    # within a step, so the index is stable across calls.

    def outcome_reward(prompts, completions, **kwargs):
        fens = kwargs.get("fen") or [None] * len(completions)
        cas = kwargs.get("candidate_a") or [None] * len(completions)
        cbs = kwargs.get("candidate_b") or [None] * len(completions)
        truths = kwargs.get("truth_label") or [None] * len(completions)
        rewards = []
        for i, completion in enumerate(completions):
            text = _completion_text(completion)
            if fens[i] is None:
                rewards.append(0.0)
                continue
            label, move = parse_choice(text, cas[i], cbs[i])
            best = oracle.best_label(fens[i], cas[i], cbs[i], truths[i])
            if i == 0:
                try:
                    import chess as _ch
                    _ba = _ch.Board(fens[i]); _ba.push(_ch.Move.from_uci(cas[i])); _ea = oracle.eval_cp(_ba.fen())
                    _bb = _ch.Board(fens[i]); _bb.push(_ch.Move.from_uci(cbs[i])); _eb = oracle.eval_cp(_bb.fen())
                    print(f"[sample] text={text[:300]!r} label={label} move={move} best={best} a={cas[i]} b={cbs[i]} truth={truths[i]} ea={_ea} eb={_eb}", flush=True)
                except Exception as _e:
                    print(f"[sample] text={text[:200]!r} label={label} best={best} err={_e}", flush=True)
            ok = (best is not None and label == best
                  or (move and move == oracle.best_move(fens[i])))
            outcomes[(_prompt_key(prompts[i]), i)] = 1.0 if ok else 0.0
            rewards.append(outcomes[(_prompt_key(prompts[i]), i)])
        return rewards

    def process_reward(prompts, completions, **kwargs):
        fens = kwargs.get("fen") or [None] * len(completions)
        rewards = []
        for i, completion in enumerate(completions):
            text = _completion_text(completion)
            if fens[i] is None:
                rewards.append(0.0)
                continue
            rewards.append(_verify_trace(text, fens[i], oracle))
        return rewards

    def style_reward(prompts, completions, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            ok = outcomes.get((_prompt_key(prompts[i]), i), 0.0)
            if not ok:
                rewards.append(0.0)  # never reward short-and-wrong
                continue
            tokens = len(tokenizer.encode(_completion_text(completion)))
            rewards.append(1.0 - tokens / max_completion_length)
        return rewards

    return [outcome_reward, process_reward, style_reward]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E2B-it")
    ap.add_argument("--from-adapter", default="",
                    help="path to the stage-1 SFT adapter dir. Loaded on the "
                         "base and FROZEN, then a NEW trainable RL LoRA is "
                         "wrapped on top (peft nesting) — the SFT weights are "
                         "the skill, RL adjusts on top. Without this, RL "
                         "trains from the raw base.")
    ap.add_argument("--train", required=True,
                    help="pool jsonl from build_rlvr_pool.py")
    ap.add_argument("--out", default="results/rlvr-adapter")
    ap.add_argument("--oracle", choices=["truth", "stockfish"],
                    default="truth")
    ap.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--group", type=int, default=8,
                    help="GRPO group size (num_generations per step)")
    ap.add_argument("--optim", type=str, default="adamw_torch",
                    help="optimizer (adamw_torch for CPU smoke, "
                         "adamw_bnb_8bit to save ~0.5GB on P100)")
    ap.add_argument("--max-steps", type=int, default=3500)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--max-completion-length", type=int, default=2048,
                    help="rollout completion budget. 2048 = the eval "
                         "protocol's budget (unbounded thinking directive; "
                         "the SFT'd model averages ~127 tokens, so 2048 is "
                         "10x headroom, never binds). OOM fix = gradient "
                         "checkpointing + expandable segments, not a small "
                         "budget.")
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="rollout sampling temperature. The SFT'd model "
                         "degenerates under trl's default temp 1.0 sampling "
                         "(measured 2026-08-21: all 8 rollouts ran to the "
                         "2048 cap with no EOS -> zero rewards -> no "
                         "signal); ~0.7 matches the eval protocol's greedy "
                         "behavior while still sampling.")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="rollout nucleus sampling (0.9 with temperature "
                         "0.7 preserves the SFT policy's structure)")
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="cap pool rows (0 = all)")
    ap.add_argument("--thinking", action="store_true",
                    help="gemma4 only: enable the <|channel>thought block "
                         "during rollouts (the RLVR design's thinking-ON "
                         "variant)")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="disable gradient checkpointing: on sm_60 bnb's "
                         "4-bit fallback dequantizes to fp16 per layer and "
                         "checkpointing's backward recompute materializes "
                         "the WHOLE model in fp16 at once (~10GB, measured "
                         "2026-08-21: loss-phase peak 14.96GB at any group "
                         "size). Without it, per-layer dequant frees after "
                         "each layer's backward; activations at chunk "
                         "batch=1 are ~1GB.")
    ap.add_argument("--no-quant", action="store_true",
                    help="gemma4 GPU path: load the base in fp16 instead of "
                         "4-bit. REQUIRED with --from-adapter: peft 0.14's "
                         "merge_and_unload on a 4-bit bnb base crashes "
                         "(Params4bit._is_hf_initialized, measured "
                         "2026-08-19); fp16 merges cleanly, E2B fp16 is "
                         "~4GB (fits the P100 16GB), and fp16 generation "
                         "is typically faster than 4-bit dequant.")
    ap.add_argument("--cpu", action="store_true",
                    help="fp32 CPU load + use_cpu trainer (local smoke; "
                         "gemma-4 is too large for this — use a small "
                         "--base)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny defaults + no HF upload")
    ap.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark",
                    help="HF dataset repo for checkpoint upload/resume")
    ap.add_argument("--hf-tag", default="rlvr",
                    help="folder under --hf-repo for this run")
    ap.add_argument("--hf-upload-every", type=float, default=1800,
                    help="seconds between HF checkpoint uploads "
                         "(Kaggle T4 sessions die at ~12h)")
    ap.add_argument("--resume-from-hf", action="store_true",
                    help="download the latest checkpoint from --hf-repo "
                         "under --hf-tag and resume training from it")
    ap.add_argument("--save-steps", type=int, default=50,
                    help="local checkpoint cadence (uploaded to HF every "
                         "--hf-upload-every seconds). Default 50 matches the "
                         "P100 throughput reality (~200 steps per 12h kernel "
                         "at ~5.6 tok/s generation; the old 500 default "
                         "would upload nothing — same bug class as the SFT "
                         "run's save_steps=10000).")
    ap.add_argument("--wandb-project", default="",
                    help="wandb project name (empty = no wandb; requires "
                         "WANDB_API_KEY)")
    ap.add_argument("--progress-every", type=float, default=60,
                    help="seconds between HF progress.json heartbeats")
    ap.add_argument("--step-timeout-min", type=float, default=0,
                    help="SIGINT a step stuck longer than N minutes "
                         "(graceful checkpoint + upload, then exit); "
                         "0 disables")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback

    # P100 memory fix (measured pretest v5/v6, 2026-08-20): trl 0.17's
    # _compute_loss runs the training forward over the WHOLE group at once
    # (grpo_trainer.py:1197 passes no batch_size), materializing
    # group x completion_len x vocab fp16 logits — 4-8 x 2048 x 256k with
    # rollouts at the 2048 cap, on top of the model's fp16-equivalent
    # footprint (bnb has no sm_60 4-bit kernels, so the P100 dequantizes
    # to fp16). The ref/old logprob paths (grpo_trainer.py:1005/1014)
    # already chunk at per_device_train_batch_size=1; force the same for
    # the training forward: identical math (logps are concatenated), but
    # only one completion's logits are alive at a time during backward.
    import trl.trainer.grpo_trainer as _grpo_mod
    _orig_logps = _grpo_mod.GRPOTrainer._get_per_token_logps

    def _chunked_logps(self, model, input_ids, attention_mask,
                       logits_to_keep, batch_size=1):
        return _orig_logps(self, model, input_ids, attention_mask,
                           logits_to_keep, batch_size=batch_size or 1)

    _grpo_mod.GRPOTrainer._get_per_token_logps = _chunked_logps
    print("[memfix] trl GRPO training forward chunked to batch_size=1",
          flush=True)

    # GRPO in fp16 (Chess-R1/TinyZero run the loss in bf16 — fp16 is MORE
    # precise). accelerate's device_map dispatch wraps model calls with
    # convert_to_fp32, materializing a 2GB fp32 copy of the completion
    # logits at 256k vocab on long rollouts — the final OOM push on 16GB
    # GPUs (measured 2026-08-21 T4 pretest: peak 13.32/14.56GB at
    # tensor.float(); P100 variants identical). Patch the top-level
    # convert_to_fp32 used by Operations.__call__ to identity.
    try:
        import accelerate.utils.operations as _ops
        _ops.convert_to_fp32 = lambda t, *a, **k: t
        print("[memfix] accelerate fp32 logits conversion disabled",
              flush=True)
    except Exception as e:
        print(f"[memfix] fp32 patch failed: {e}", flush=True)

    # ---- memory diagnostics (v9): locate the ~14GB peak ----
    import torch as _torch
    _torch.cuda.reset_peak_memory_stats()

    def _mem(tag):
        print(f"[mem] {tag}: allocated={_torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={_torch.cuda.memory_reserved()/1e9:.2f}GB "
              f"peak={_torch.cuda.max_memory_allocated()/1e9:.2f}GB",
              flush=True)

    _orig_gsc = _grpo_mod.GRPOTrainer._generate_and_score_completions

    def _gsc(self, *a, **k):
        r = _orig_gsc(self, *a, **k)
        _mem("after generate_and_score (gen+ref+rewards)")
        return r

    _grpo_mod.GRPOTrainer._generate_and_score_completions = _gsc

    _orig_cl = _grpo_mod.GRPOTrainer._compute_loss

    def _cl(self, model, inputs):
        _mem("at _compute_loss start")
        return _orig_cl(self, model, inputs)

    _grpo_mod.GRPOTrainer._compute_loss = _cl

    is_gemma4 = "gemma-4" in args.base
    if args.smoke:
        args.max_steps = min(args.max_steps, 1)
    print(f"base={args.base} gemma4={is_gemma4} cpu={args.cpu} "
          f"oracle={args.oracle} steps={args.max_steps} group={args.group} "
          f"lr={args.lr}", flush=True)

    # ---- model: gemma4 multimodal 4-bit path (train_mate_lora-proven)
    # vs plain CausalLM path (CPU smoke on small models) ----
    if is_gemma4:
        processor = AutoProcessor.from_pretrained(args.base)
        tokenizer = processor.tokenizer
        if args.cpu:
            model = AutoModelForImageTextToText.from_pretrained(
                args.base, device_map="cpu", dtype=torch.float32,
                low_cpu_mem_usage=True)
        else:
            cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
            compute_dtype = torch.bfloat16 if cap >= (7, 5) else torch.float16
            if args.no_quant:
                model = AutoModelForImageTextToText.from_pretrained(
                    args.base, device_map={"": 0}, dtype=compute_dtype)
                print("base loaded fp16 (no 4-bit) for clean SFT merge",
                      flush=True)
            else:
                # same transformers 5.13.1 shim as train_mate_lora: bind
                # torch into quantization_config (is_torch_available() False
                # on the P100 stack -> BitsAndBytesConfig NameError)
                import transformers.utils.quantization_config as _qc
                if not hasattr(_qc, "torch"):
                    _qc.torch = torch
                # torch 2.4.1 lacks nn.Module.set_submodule (added 2.5+); bnb
                # 0.46.1's replace_with_bnb_linear needs it during 4-bit load
                # (same shim as train_mate_lora/src.models; missing here caused
                # the 2026-08-19 pretest failure).
                if not hasattr(torch.nn.Module, "set_submodule"):
                    def _set_submodule(self, target, module):
                        atoms = target.split(".")
                        parent = self
                        for atom in atoms[:-1]:
                            parent = getattr(parent, atom)
                        setattr(parent, atoms[-1], module)
                    torch.nn.Module.set_submodule = _set_submodule
                quant = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype)
                model = AutoModelForImageTextToText.from_pretrained(
                    args.base, quantization_config=quant, device_map={"": 0},
                    dtype=compute_dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        processor = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None:
            # base (non-instruct) checkpoints carry no chat template and
            # trl's prompt rendering requires one (caught by the
            # 2026-08-18 CPU smoke on SmolLM2-135M). Minimal fallback so
            # any small base validates the loop; the gemma4 path always
            # has the real processor template.
            tokenizer.chat_template = ("{% for m in messages %}{{'<|user|>\\n'"
                                       "+ m['content'] + '\\n<|assistant|>\\n'}}"
                                       "{% endfor %}")
        if args.cpu:
            model = AutoModelForCausalLM.from_pretrained(
                args.base, device_map="cpu", dtype=torch.float32,
                low_cpu_mem_usage=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.base, device_map={"": 0}, torch_dtype=torch.bfloat16)

    # ---- stage-1 SFT adapter, MERGED into the base ----
    # The SFT weights are baked into the base (peft merge_and_unload: on
    # the 4-bit gemma4 path this dequantizes to fp16 — ~4GB for E2B, fits
    # the P100, and fp16 generation is typically faster than 4-bit
    # dequant, to be measured in the pretest). The NEW RL LoRA then
    # targets the PLAIN base structure, so the saved RL adapter loads onto
    # the raw base with run_mate_eval's single-adapter path (the nested
    # PeftModel-on-PeftModel approach produced adapter keys that silently
    # failed to load onto the raw base — missing-keys warning, measured
    # 2026-08-19 in the local smoke).
    if args.from_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.from_adapter)
        model = model.merge_and_unload()
        print(f"SFT adapter MERGED into base: {args.from_adapter}",
              flush=True)

    # LoRA on the FULL model (all-linear matches Linear4bit by type;
    # proven by train_mate_lora — same wrap, same save/load symmetry)
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
        bias="none", target_modules="all-linear", task_type="CAUSAL_LM")
    try:
        model = get_peft_model(model, lora)
    except ValueError as e:
        # all-linear can't dispatch on the gemma4 wrap (Gemma4ClippableLinear
        # is not nn.Linear by type) — the proven fallback from the SFT
        # trainer: enumerate every linear by name (the stage-1 adapter used
        # this exact path: 529 modules).
        import torch.nn as nn
        lin_names = [n for n, mod in model.named_modules()
                     if isinstance(mod, nn.Linear)
                     or type(mod).__name__ in ("Linear4bit", "Linear8bitLt")]
        print(f"all-linear failed ({e}); explicit fallback with "
              f"{len(lin_names)} linear modules", flush=True)
        lora2 = LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
            bias="none",
            target_modules=lin_names[:512] or ["linear"],
            task_type="CAUSAL_LM")
        model = get_peft_model(model, lora2)
    # trl 0.17's GRPOTrainer.__init__ unconditionally does
    # model.warnings_issued["estimate_tokens"] = True; transformers 5.14
    # removed the PreTrainedModel.warnings_issued class attribute that
    # existed in 4.x, so a plain PeftModel raises AttributeError at
    # trainer construction (caught by the 2026-08-18 CPU smoke). Same
    # class of shim as the trl tuple-guard patch above.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    try:
        if not args.no_grad_checkpoint:
            model.gradient_checkpointing_enable()
            print("gradient checkpointing enabled", flush=True)
        else:
            print("gradient checkpointing DISABLED (sm_60 bnb dequant "
                  "retention fix)", flush=True)
    except Exception as e:
        print(f"grad checkpointing unavailable: {e}", flush=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable / 1e6:.1f}M "
          f"({trainable / sum(p.numel() for p in model.parameters()) * 100:.2f}%)",
          flush=True)

    if args.thinking and is_gemma4:
        # wrap the TOKENIZER (the processing_class), not the processor:
        # trl 0.17 reads processing_class.pad_token, which Gemma4Processor
        # lacks (AttributeError, measured 2026-08-19). The tokenizer has
        # pad_token + the chat template; rollouts are text-only.
        tokenizer = _ThinkingProcessor(tokenizer, enable_thinking=True)
        print("thinking channel ENABLED for rollouts", flush=True)

    # ---- pool -> trl dataset: prompt (messages) + oracle columns ----
    print(f"loading pool: {args.train}", flush=True)
    ds = load_dataset("json", data_files=str(args.train))["train"]
    if args.max_train_rows > 0:
        ds = ds.select(range(min(args.max_train_rows, len(ds))))

    def to_trl(row):
        prompt = build_prompt(row["fen"], row["candidate_a"],
                              row["candidate_b"])
        return {"prompt": [{"role": "user", "content": prompt}],
                "fen": row["fen"],
                "candidate_a": row["candidate_a"],
                "candidate_b": row["candidate_b"],
                "truth_label": row["truth_label"]}

    ds = ds.map(to_trl, remove_columns=ds.column_names)
    print(f"pool rows: {len(ds)}", flush=True)
    print("sample prompt:", ds[0]["prompt"][0]["content"][:180], "...", flush=True)

    oracle = Oracle(args.oracle, depth=args.depth, stockfish=args.stockfish)
    reward_funcs = make_rewards(oracle, tokenizer,
                                args.max_completion_length, args.max_steps)

    wandb_run = None
    if args.wandb_project:
        os.environ.pop("WANDB_DISABLED", None)
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.hf_tag,
                               config={"base": args.base,
                                       "from_adapter": args.from_adapter,
                                       "lr": args.lr, "beta": args.beta,
                                       "group": args.group,
                                       "max_steps": args.max_steps,
                                       "max_completion_length":
                                           args.max_completion_length,
                                       "pool": args.train,
                                       "oracle": args.oracle,
                                       "depth": args.depth})
        print(f"wandb: project={args.wandb_project} run={args.hf_tag}",
              flush=True)

    cfg = GRPOConfig(
        output_dir=str(Path(args.out)),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        # trl 0.17 requires effective batch (batch x accum x processes)
        # to be evenly divisible by num_generations, else it refuses the
        # trainer (caught by the 2026-08-18 CPU smoke). accum = group gives
        # effective = group, so group 8/4/2/1 all validate.
        gradient_accumulation_steps=args.group,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # scale warmup to the real step budget (P100: ~200 steps per 12h
        # kernel); the old fixed 50 was 25% of a 200-step run.
        warmup_steps=max(5, args.max_steps // 20),
        weight_decay=0.01,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.group,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        reward_weights=[1.0, 0.3, 0.1],
        use_cpu=args.cpu,
        bf16=False if args.cpu else (torch.cuda.get_device_capability(0)[0] >= 7 if torch.cuda.is_available() else False),
        fp16=False if args.cpu else (torch.cuda.get_device_capability(0)[0] < 7 if torch.cuda.is_available() else False),
        disable_dropout=True,
        optim=args.optim,
        gradient_checkpointing=not args.no_grad_checkpoint,
        disable_tqdm=args.smoke,
        logging_steps=1,
        log_completions=args.smoke,
        num_completions_to_print=1,
        save_strategy="steps" if not args.smoke else "no",
        save_steps=args.save_steps,
        save_total_limit=4,
        seed=42,
        report_to=["wandb"] if args.wandb_project else [],
    )
    print(f"GRPO: lr={cfg.learning_rate} beta={cfg.beta} "
          f"group={cfg.num_generations} max_completion={cfg.max_completion_length} "
          f"use_cpu={cfg.use_cpu}", flush=True)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=None,
    )

    # gemma-4-E2B terminates turns with <turn|> (id 106), not <eos> (id 1);
    # config.json declares eos_token_id=[1,106]. trl 0.17 builds the rollout
    # generation config with ONLY processing_class.eos_token_id (=1), so
    # rollouts never stop at <turn|> and run to the 2048 cap (measured:
    # 16/16 rollouts at 2048, all rewards 0, at temp 1.0 AND 0.7 — the eval
    # works because it uses the model's default generation config with both
    # ids, terminating ~127 tokens). Restore the full eos set.
    try:
        eos_ids = list(model.config.eos_token_id) if isinstance(
            model.config.eos_token_id, (list, tuple)) else [model.config.eos_token_id]
        trainer.generation_config.eos_token_id = eos_ids
        print(f"[eosfix] rollout eos_token_id = {eos_ids}", flush=True)
    except Exception as e:
        print(f"[eosfix] failed: {e}", flush=True)

    # HF checkpoint safety net + resume (AGENTS.md: killed kernels must
    # not strand progress; the latest checkpoint survives to HF). The
    # smoke path needs no HF token: it uploads nothing and resumes nothing.
    api = None
    hf_cb = None
    resume_from = None
    if not args.smoke:
        api = _hf_api()
        hf_cb = HfCheckpointCallback(api, args.hf_repo, args.hf_tag,
                                     args.hf_upload_every, args.out)
        if args.resume_from_hf:
            resume_from = download_hf_checkpoint(api, args.hf_repo, args.hf_tag,
                                                 str(Path(args.out).parent))
            if resume_from is not None:
                print(f"[resume] resuming from {resume_from}", flush=True)
            else:
                print("[resume] no remote checkpoint found; starting fresh",
                      flush=True)

    class _HfUploadCallback(TrainerCallback):
        """Trigger the HF safety-net upload on the trainer's cadence."""

        def on_step_end(self, args_, state, control, **kwargs):
            hf_cb.maybe_upload()

        def on_train_end(self, args_, state, control, **kwargs):
            hf_cb.maybe_upload(force=True)

    class _RewardLogCallback(TrainerCallback):
        """Per-step reward evidence: print the last logged metrics so a
        run visibly shows all three reward funcs returning real values
        (outcome/process/style), not silent zeros."""

        def on_step_end(self, args_, state, control, **kwargs):
            if not state.log_history:
                return
            entry = state.log_history[-1]
            hit = {k: round(v, 4) for k, v in entry.items()
                   if isinstance(v, (int, float)) and
                   ("reward" in k or k in ("kl", "completion_length"))}
            if hit:
                print(f"step {state.global_step}: {hit}", flush=True)
                if wandb_run is not None:
                    # Don't pass step: trl's global_step counts optimizer steps (1,2) but
                    # Trainer has already logged at higher micro-step counts (e.g. 57),
                    # so explicit step would be < current and wandb drops it. Let wandb auto-increment.
                    wandb_run.log(hit)

    class _ProgressHeartbeat(TrainerCallback):
        """Push a compact progress.json to HF on a timer so a kernel that
        shows only RUNNING is still observable (the Kaggle log API is
        unreliable; HF always is). A daemon thread fires EVERY interval
        during training (mid-step included — a single degenerate rollout
        can take ~50 min at the 2048 budget, and step-end-only would leave
        the run invisible that whole time). Fields: step, total, phase,
        elapsed, ETA, last reward metrics. Uploaded to <hf-tag>/progress.json
        (overwrites)."""

        def __init__(self, api, remote_dir, interval_s, total_steps,
                     step_timeout_min=0):
            self.api = api
            self.remote_dir = remote_dir.strip("/")
            self.interval_s = interval_s
            self.total_steps = total_steps
            self.state = None
            self.t0 = time.time()
            self._stop = threading.Event()
            self.step_timeout_min = step_timeout_min
            self._step_started = None
            self._watchdog_stop = threading.Event()

        def _payload(self):
            entry = (self.state.log_history[-1] if self.state
                     and self.state.log_history else {})
            hit = {k: v for k, v in entry.items()
                   if isinstance(v, (int, float)) and
                   ("reward" in k or k in ("kl", "completion_length"))}
            step = self.state.global_step if self.state else 0
            elapsed = time.time() - self.t0
            per = elapsed / max(step, 1)
            mem = None
            try:
                import torch as _t
                if _t.cuda.is_available():
                    mem = round(_t.cuda.memory_allocated() / 1e9, 2)
            except Exception:
                pass
            payload = {
                "step": step,
                "total_steps": self.total_steps,
                "phase": "training" if self.state and self.state.global_step
                         else "setup",
                "elapsed_s": round(elapsed),
                "eta_s": round(per * (self.total_steps - step))
                if step else None,
                "metrics": hit,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if mem is not None:
                payload["mem_gb"] = mem
            return payload

        def _push(self):
            try:
                self.api.upload_file(
                    path_or_fileobj=json.dumps(self._payload()).encode(),
                    path_in_repo=f"{self.remote_dir}/progress.json",
                    repo_id="vedangfake/chess-slm-benchmark",
                    repo_type="dataset",
                    commit_message=f"rlvr progress step "
                                   f"{self._payload()['step']}")
                print(f"[progress] step {self._payload()['step']}/"
                      f"{self.total_steps} -> HF progress.json", flush=True)
            except Exception as e:
                print(f"[progress] upload failed (will retry): {e}",
                      flush=True)

        def _loop(self):
            while not self._stop.wait(self.interval_s):
                self._push()

        def _watchdog(self):
            """Interrupt a step that exceeds the cap so a hang or a
            degenerate long rollout stops the run with a graceful
            checkpoint instead of burning GPU silently. SIGINT -> the
            trainer saves state + adapter and uploads them to HF, then
            exits nonzero, which fails the notebook cell."""
            if not self.step_timeout_min:
                return
            cap = self.step_timeout_min * 60
            while not self._watchdog_stop.wait(20):
                if self._step_started is None:
                    continue
                stuck = time.time() - self._step_started
                if stuck > cap:
                    print(f"[watchdog] step stuck {stuck / 60:.0f} min > "
                          f"cap {self.step_timeout_min} min; SIGINT for a "
                          f"graceful checkpoint", flush=True)
                    os.kill(os.getpid(), signal.SIGINT)
                    return

        def on_train_begin(self, args_, state, control, **kwargs):
            self.state = state
            self.t0 = time.time()
            threading.Thread(target=self._loop, daemon=True).start()
            threading.Thread(target=self._watchdog, daemon=True).start()
            self._push()

        def on_step_begin(self, args_, state, control, **kwargs):
            self._step_started = time.time()

        def on_step_end(self, args_, state, control, **kwargs):
            self._step_started = None
            self._push()

        def on_train_end(self, args_, state, control, **kwargs):
            self._step_started = None
            self._watchdog_stop.set()
            self._push()

    if not args.smoke:
        trainer.add_callback(_HfUploadCallback())
        trainer.add_callback(_ProgressHeartbeat(api, args.hf_tag,
                                                args.progress_every,
                                                args.max_steps,
                                                args.step_timeout_min))
    trainer.add_callback(_RewardLogCallback())

    print("training...", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    print(f"done in {(time.time() - t0) / 60:.1f}min", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    if is_gemma4 and not args.thinking:
        try:
            processor.save_pretrained(str(out))
        except Exception as e:
            print(f"processor save skipped: {e}", flush=True)
    print(f"adapter saved: {out}", flush=True)
    if not args.smoke:
        try:
            hf_cb.final()
        except Exception as e:
            print(f"[hf-cp] final upload failed: {e}", flush=True)

    oracle.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        try:
            api = _hf_api()
            body = (f"{type(e).__name__}: {e}\n" + _tb.format_exc()[-4000:])
            api.upload_file(path_or_fileobj=body.encode(),
                            path_in_repo="rlvr/run-status.txt",
                            repo_id="vedangfake/chess-slm-benchmark",
                            repo_type="dataset",
                            commit_message="rlvr failure status")
            print("[status] failure written to HF run-status.txt", flush=True)
        except Exception as e2:
            print(f"[status] failed to write run-status.txt: {e2}", flush=True)
        # Memory forensics: what was holding the GPU when it died.
        try:
            import torch as _t
            if _t.cuda.is_available():
                print(f"[mem] CRASH peak={_t.cuda.max_memory_allocated()/1e9:.2f}GB "
                      f"allocated={_t.cuda.memory_allocated()/1e9:.2f}GB",
                      flush=True)
                _t.cuda.memory_summary(abbreviated=True)
        except Exception as _e3:
            print(f"[mem] summary failed: {_e3}", flush=True)
        # Hard exit: wandb's non-daemon uploader thread and the heartbeat
        # loop would otherwise keep this process alive after a crash,
        # burning GPU until the notebook's wall-clock cap fires (measured
        # 2026-08-20 v6: 45 min of dead time). os._exit skips atexit and
        # kills every thread immediately; run-status.txt is already on HF.
        os._exit(1)
