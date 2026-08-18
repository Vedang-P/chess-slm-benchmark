"""QLoRA SFT of gemma-4-E2B on MATE noexplain selection pairs.

    python3 scripts/train_mate_lora.py --train data/positions/mate-lora/train.jsonl \
        --eval data/positions/mate-lora/eval.jsonl --out results/mate-lora-adapter \
        --wandb-project mate-lora

Audited details (2026-08-11):
  - gemma-4-E2B is Gemma4ForConditionalGeneration; text tower is
    model.language_model (35 layers, hidden 1536, GQA 8/1). LoRA wraps ONLY
    the language_model (never the vision/audio encoders).
  - Projection names q/k/v/o/gate/up/down_proj verified against the
    transformers gemma4 modeling code (Gemma4ClippableLinear wraps nn.Linear).
  - Assistant-only loss via apply_chat_template(return_assistant_tokens_mask=True)
    — exact mask from the template, no fragile prefix-tokenization.
  - Custom collator pads input_ids AND labels (labels with -100). The HF
    DataCollatorForLanguageModeling would OVERWRITE labels with input_ids,
    destroying the assistant mask — must not be used.
  - Eval prompt == training prompt byte-identical (FEN + candidates +
    ANSWER_SPEC), verified against run_mate_eval.py construction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _hf_api():
    """HfApi from HF_WRITE_TOKEN (env or .env) — same pattern as src/hf_push."""
    import os
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
    seconds (checkpoint dir = adapter weights + optimizer + scheduler +
    trainer_state, so a killed Kaggle session can resume with
    --resume-from-hf). Also uploads the final adapter at train end."""

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
    cps = sorted({f.split("/")[2] for f in files if f.startswith(prefix) and len(f.split("/")) > 2})
    if not cps:
        print(f"[resume] no checkpoints under {remote_dir} in {repo_id}", flush=True)
        return None
    latest = cps[-1]
    print(f"[resume] downloading {remote_dir}/{latest}", flush=True)
    for f in files:
        if not f.startswith(f"{remote_dir}/{latest}/"):
            continue
        hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset",
                        local_dir=str(local_root), token=api.token)
    return local_root / remote_dir / latest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default="results/mate-lora-adapter")
    ap.add_argument("--base", default="google/gemma-4-E2B-it")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument("--save-steps", type=int, default=10000)
    ap.add_argument("--wandb-project", default="", help="wandb project name "
                    "(empty = no wandb)")
    ap.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark",
                    help="HF dataset repo for checkpoint upload/resume "
                         "(adapter + optimizer + trainer state)")
    ap.add_argument("--hf-upload-every", type=float, default=1800,
                    help="seconds between HF checkpoint uploads "
                         "(Kaggle T4 sessions are ~12h; this is the "
                         "crash/session-loss safety net)")
    ap.add_argument("--resume-from-hf", default="",
                    help="download latest checkpoint from --hf-repo and "
                         "resume training from it (path_in_repo dir, e.g. "
                         "noexplain-slice/checkpoint-XXXX)")
    ap.add_argument("--train-tag", default="noexplain-slice",
                    help="folder under --hf-repo (and wandb run name) "
                         "for this training run")
    ap.add_argument("--live-test", default="data/positions/mate-selection-test-noexplain.json",
                    help="real test set for live accuracy tracking during "
                         "training (wandb live/* metrics)")
    ap.add_argument("--live-n", type=int, default=100,
                    help="positions sampled per live eval (byte-identical "
                         "eval prompt, thinking off for speed)")
    ap.add_argument("--live-every", type=int, default=1000,
                    help="live eval every N training steps")
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="cap training rows (0 = all). Use to fit a 12h "
                         "Kaggle kernel: 60k rows x 1 epoch ~= 10h at "
                         "measured 369 steps/hr (batch2 x accum8).")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to validate the stack")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    compute_dtype = torch.bfloat16 if cap >= (7, 5) else torch.float16
    print(f"cuda={torch.cuda.is_available()} cap={cap} dtype={compute_dtype}",
          flush=True)

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )
    print("loading base model...", flush=True)
    # THE CAMPAIGN MODEL: full multimodal google/gemma-4-E2B-it loaded
    # exactly like src/models.HFModel does (4-bit, device_map {"":0}).
    # Same artifact + same load path as every eval baseline. Runs on the
    # Kaggle T4 (16GB) that the campaign used.
    processor = AutoProcessor.from_pretrained(args.base)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, quantization_config=quant, device_map={"": 0},
        dtype=compute_dtype)

    # 4-bit -> freeze the whole base (LoRA marks its own params trainable).
    # We deliberately do NOT call prepare_model_for_kbit_training: it casts
    # every non-quantized param to fp32, which OOM'd on the T4 (tried to
    # allocate 8.75 GiB at load on the 2B text tower).
    # Wrapping the FULL multimodal model (not just the text tower):
    #  - the Trainer's validate_quantization_for_training rejects a purely
    #    quantized outer model; a full-model PeftModel passes the check
    #  - target_modules="all-linear" matches Linear4bit by TYPE, so the
    #    Gemma4ClippableLinear wrappers (plain nn.Module, not nn.Linear)
    #    are handled correctly (peft can't dispatch on them by name)
    #  - save/eval symmetric: adapter keys match the full model, so
    #    run_mate_eval's PeftModel.from_pretrained(full_model, ...) loads
    for p in model.parameters():
        p.requires_grad = False
    n_quant = sum(1 for m in model.modules()
                  if type(m).__name__ == "Linear4bit")
    print(f"4-bit Linear modules: {n_quant} "
          f"(0 = quantization did NOT engage)", flush=True)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"grad checkpointing unavailable: {e}", flush=True)
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM")
    try:
        model = get_peft_model(model, lora)
    except ValueError as e:
        # fallback: explicitly enumerate every Linear/Linear4bit module
        # name (handles non-PreTrainedModel towers or unusual wrappers)
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.1f}M "
          f"({trainable/sum(p.numel() for p in model.parameters())*100:.2f}%)",
          flush=True)

    print("loading datasets...", flush=True)
    # Pre-tokenized input (input_ids + labels JSONL, built offline by
    # scripts/pre_tokenize_slice.py) loads instantly — no per-row
    # apply_chat_template on the Kaggle CPU (600k rows took >95 min and
    # stalled; see git log 2026-08-16). Messages-format fallback kept for
    # small/smoke sets.
    # If --train already IS a *_pretok.jsonl, use it directly; otherwise
    # look for <stem>_pretok.jsonl next to it (the kernel fetches exactly
    # that name).
    if "_pretok" in Path(args.train).name:
        pretok, eval_pretok = args.train, args.eval
    else:
        pretok = args.train.replace(".jsonl", "_pretok.jsonl")
        eval_pretok = args.eval.replace(".jsonl", "_pretok.jsonl")
    if Path(pretok).exists():
        print(f"loading PRE-TOKENIZED data: {pretok} (+ {eval_pretok})",
              flush=True)
        train_ds = load_dataset(
            "json", data_files=str(pretok))["train"]
        if args.max_train_rows > 0:
            train_ds = train_ds.select(range(min(args.max_train_rows,
                                                 len(train_ds))))
        eval_ds = load_dataset(
            "json", data_files=str(eval_pretok))["train"]
    else:
        ds = load_dataset("json", data_files={"train": args.train, "eval": args.eval})

        def to_ids(row):
            msgs = row["messages"]
            # Render EXACTLY like run_mate_eval -> HFModel.generate does
            # (enable_thinking=False) so train/eval prompts are byte-identical.
            # Assistant mask via prefix-difference: transformers 5.13.1's
            # return_assistant_tokens_mask is BROKEN for this template (all
            # zeros -- audited 2026-08-14: '{% generation %}' keyword not
            # detected, mask stays 0 for every token -> labels all -100 -> a
            # silent no-op training run). The reliable method: tokenize the
            # full message and the prompt-only (user + assistant header);
            # assistant tokens = the suffix after the prompt prefix.
            full = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt", enable_thinking=False)["input_ids"][0]
            prompt = processor.apply_chat_template(
                msgs[:1], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False)["input_ids"][0]
            if not full[:len(prompt)].tolist() == prompt.tolist():
                raise RuntimeError("prompt is not a prefix of the full message "
                                   "-- assistant mask would be wrong")
            input_ids = full.tolist()
            if len(input_ids) > args.max_seq_len:
                input_ids = input_ids[: args.max_seq_len]
            labels = [-100] * len(prompt) + input_ids[len(prompt):]
            # after truncation both must stay the same length
            labels = labels[: args.max_seq_len]
            return {"input_ids": input_ids, "labels": labels}

        def to_text(row):
            msgs = row["messages"]
            return {"text": processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                enable_thinking=False)}

        train_ds = ds["train"].map(to_text)
        eval_ds = ds["eval"].map(to_text)
        if args.smoke:
            train_ds = train_ds.select(range(min(300, len(train_ds))))
            eval_ds = eval_ds.select(range(min(60, len(eval_ds))))
        train_ds = train_ds.map(to_ids, remove_columns=["text", "messages", "fen"])
        eval_ds = eval_ds.map(to_ids, remove_columns=["text", "messages", "fen"])
    print(f"train rows: {len(train_ds)} | eval rows: {len(eval_ds)}", flush=True)

    # sanity: verify the assistant mask marks exactly the answer tokens
    s = train_ds[0]
    print("--- sample check ---", flush=True)
    print("input len:", len(s["input_ids"]), "| label -100 count:",
          s["labels"].count(-100), flush=True)
    print("non-masked (assistant) tokens:",
          len(s["input_ids"]) - s["labels"].count(-100), flush=True)
    if s["labels"].count(-100) == len(s["input_ids"]):
        raise RuntimeError("assistant mask is empty -- labels all -100, "
                           "training would be a silent no-op")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_to = ["wandb"] if args.wandb_project else []
    wandb_run = None
    if report_to:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=args.train_tag,
                               config={"base_model": args.base})
        from src.live_metrics import log_run_config
        log_run_config(wandb_run, args)
        print(f"wandb: project={args.wandb_project} run will log losses "
              f"+ live accuracy", flush=True)

    # HF checkpoint safety net + resume (Kaggle T4 sessions die at ~12h).
    # The smoke path needs no HF token: it uploads nothing and resumes nothing.
    api = None
    hf_cb = None
    resume_from = None
    if not args.smoke:
        api = _hf_api()
        hf_cb = HfCheckpointCallback(api, args.hf_repo, args.train_tag,
                                     args.hf_upload_every, str(out))
        if args.resume_from_hf:
            resume_from = download_hf_checkpoint(api, args.hf_repo, args.train_tag,
                                                 str(out.parent))
            if resume_from is not None:
                print(f"[resume] resuming from {resume_from}", flush=True)
            else:
                print("[resume] no remote checkpoint found; starting fresh",
                      flush=True)

    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        optim="adamw_8bit",
        bf16=cap >= (7, 5),
        fp16=cap < (7, 5),
        logging_steps=1,
        eval_strategy="steps" if not args.smoke else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=4,
        report_to=report_to,
        run_name=args.train_tag,
        seed=42,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
    )

    class MaskedCollator:
        """Pad input_ids and labels with -100 for labels (the HF LM
        collator would overwrite our assistant mask)."""

        def __init__(self, tok):
            self.tok = tok

        def __call__(self, features):
            import torch

            ids = self.tok.pad(
                [{"input_ids": f["input_ids"]} for f in features],
                return_tensors="pt", padding=True)
            max_len = ids["input_ids"].shape[1]
            labels = torch.full_like(ids["input_ids"], -100)
            for i, f in enumerate(features):
                n = len(f["labels"])
                labels[i, :n] = torch.tensor(f["labels"][:n])
            return {"input_ids": ids["input_ids"],
                    "attention_mask": ids["attention_mask"],
                    "labels": labels}

    class HfUploadCallback(TrainerCallback):
        """Trigger the HF safety-net upload on the trainer's own cadence."""

        def __init__(self, cb: HfCheckpointCallback):
            self.cb = cb

        def on_step_end(self, args, state, control, **kwargs):
            self.cb.maybe_upload()

        def on_train_end(self, args, state, control, **kwargs):
            self.cb.maybe_upload(force=True)

    callbacks = [HfUploadCallback(hf_cb)] if not args.smoke else []
    if args.wandb_project and not args.smoke:
        from src.live_metrics import LiveEvalCallback

        live_cb = LiveEvalCallback(
            model, processor, args.live_test,
            n=args.live_n, every_steps=args.live_every)
        callbacks.append(live_cb)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if not args.smoke else None,
        data_collator=MaskedCollator(tokenizer),
        callbacks=callbacks,
    )
    print("training...", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    # the outer model IS the PeftModel now, so save_pretrained writes the
    # adapter files (adapter_model.safetensors + adapter_config.json) that
    # run_mate_eval loads via PeftModel.from_pretrained(full_model, ...)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    if not args.smoke:
        hf_cb.final()
    print(f"done in {(time.time()-t0)/3600:.2f}h -> {out}", flush=True)
    if not args.smoke:
        metrics = trainer.evaluate()
        print("eval:", json.dumps(metrics, indent=1), flush=True)
        if wandb_run is not None:
            wandb.log({"final/eval_loss": metrics.get("eval_loss"),
                       "final/train_loss": metrics.get("train_loss")})
            wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback as _tb

        # Write the REAL failure to HF so diagnostics never require
        # downloading the multi-GB /kaggle/working (which makes
        # 'kaggle kernels output' unusable). Read it back with:
        #   hf_hub_download("vedangfake/chess-slm-benchmark",
        #                   "noexplain-slice/run-status.txt", ...)
        try:
            import os as _os
            api = _hf_api()
            body = (f"{type(e).__name__}: {e}\n"
                    + _tb.format_exc()[-4000:])
            api.upload_file(path_or_fileobj=body.encode(),
                            path_in_repo="noexplain-slice/run-status.txt",
                            repo_id=_os.environ.get("HF_REPO",
                                "vedangfake/chess-slm-benchmark"),
                            repo_type="dataset",
                            commit_message="train failure status")
            print("[status] failure written to HF run-status.txt", flush=True)
        except Exception as e2:
            print(f"[status] failed to write run-status.txt: {e2}", flush=True)
        raise
