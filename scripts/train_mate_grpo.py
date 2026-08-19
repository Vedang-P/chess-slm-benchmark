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
import sys
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
os.environ.setdefault("WANDB_DISABLED", "true")

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
    cps = sorted({f.split("/")[2] for f in files
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
            import chess.engine
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish)

    def close(self):
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def best_label(self, fen: str, a: str, b: str, truth_label: str | None,
                   ) -> str | None:
        """Oracle's answer label for the position, or None if undecided."""
        key = (fen, a, b)
        if key in self._best:
            return self._best[key]
        if self.mode == "truth":
            label = truth_label  # mock: the pool's expert answer
        else:
            best = self._best_move(fen)
            label = ("A" if best == a else "B" if best == b else None)
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
    outcomes = {}  # prompt -> 1/0 memo, shared so style gates on outcome

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
            ok = (best is not None and label == best
                  or (move and move == oracle.best_move(fens[i])))
            outcomes[_prompt_key(prompts[i])] = 1.0 if ok else 0.0
            rewards.append(outcomes[_prompt_key(prompts[i])])
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
            ok = outcomes.get(_prompt_key(prompts[i]), 0.0)
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
    ap.add_argument("--max-steps", type=int, default=3500)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="cap pool rows (0 = all)")
    ap.add_argument("--thinking", action="store_true",
                    help="gemma4 only: enable the <|channel>thought block "
                         "during rollouts (the RLVR design's thinking-ON "
                         "variant)")
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
    ap.add_argument("--save-steps", type=int, default=500,
                    help="local checkpoint cadence (uploaded to HF every "
                         "--hf-upload-every seconds)")
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
            # same transformers 5.13.1 shim as train_mate_lora: bind
            # torch into quantization_config (is_torch_available() False
            # on the P100 stack -> BitsAndBytesConfig NameError)
            import transformers.utils.quantization_config as _qc
            if not hasattr(_qc, "torch"):
                _qc.torch = torch
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

    # LoRA on the FULL model (all-linear matches Linear4bit by type;
    # proven by train_mate_lora — same wrap, same save/load symmetry)
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
        bias="none", target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # trl 0.17's GRPOTrainer.__init__ unconditionally does
    # model.warnings_issued["estimate_tokens"] = True; transformers 5.14
    # removed the PreTrainedModel.warnings_issued class attribute that
    # existed in 4.x, so a plain PeftModel raises AttributeError at
    # trainer construction (caught by the 2026-08-18 CPU smoke). Same
    # class of shim as the trl tuple-guard patch above.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable / 1e6:.1f}M "
          f"({trainable / sum(p.numel() for p in model.parameters()) * 100:.2f}%)",
          flush=True)

    if args.thinking and is_gemma4:
        processor = _ThinkingProcessor(processor, enable_thinking=True)
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
        warmup_steps=50,
        weight_decay=0.01,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.group,
        beta=args.beta,
        reward_weights=[1.0, 0.3, 0.1],
        use_cpu=args.cpu,
        bf16=False if args.cpu else (torch.cuda.get_device_capability(0)[0] >= 7 if torch.cuda.is_available() else False),
        fp16=False if args.cpu else (torch.cuda.get_device_capability(0)[0] < 7 if torch.cuda.is_available() else False),
        disable_dropout=True,
        disable_tqdm=args.smoke,
        logging_steps=1,
        log_completions=args.smoke,
        num_completions_to_print=1,
        save_strategy="steps" if not args.smoke else "no",
        save_steps=args.save_steps,
        save_total_limit=4,
        seed=42,
        report_to=[],
    )
    print(f"GRPO: lr={cfg.learning_rate} beta={cfg.beta} "
          f"group={cfg.num_generations} max_completion={cfg.max_completion_length} "
          f"use_cpu={cfg.use_cpu}", flush=True)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=ds,
        processing_class=processor,
        peft_config=None,
    )

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

    if not args.smoke:
        trainer.add_callback(_HfUploadCallback())
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
        raise
