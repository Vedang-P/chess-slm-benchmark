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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default="results/mate-lora-adapter")
    ap.add_argument("--base", default="google/gemma-4-E2B-it")
    ap.add_argument("--epochs", type=int, default=1)
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
    ap.add_argument("--game-mode", action="store_true",
                    help="expect the full-game commentary format "
                    "('user: FEN+history+turn' / 'assistant: "
                    "<reasoning>\\nMove: <SAN>') instead of MATE selection "
                    "pairs; rows come from build_commentary_data.py --emit")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to validate the stack")
    args = ap.parse_args()
    if args.game_mode:
        args.max_seq_len = max(args.max_seq_len, 4096)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    compute_dtype = torch.bfloat16 if cap >= (7, 5) else torch.float16
    print(f"cuda={torch.cuda.is_available()} cap={cap} dtype={compute_dtype}",
          flush=True)

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )
    print("loading base model...", flush=True)
    if args.game_mode:
        # text-only checkpoint (extract_text_tower.py): causal LM, much
        # smaller than the multimodal E2B, fits a 6GB card in 4-bit.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base)
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=quant, device_map="auto",
            low_cpu_mem_usage=True, dtype=compute_dtype)
        processor = None
    else:
        processor = AutoProcessor.from_pretrained(args.base)
        tokenizer = processor.tokenizer
        model = AutoModelForImageTextToText.from_pretrained(
            args.base, quantization_config=quant, device_map={"": 0},
            dtype=compute_dtype)

    # 4-bit -> prepare for k-bit training, then wrap the language model.
    model = prepare_model_for_kbit_training(model)
    if args.game_mode:
        lang_model = model
    else:
        lang_model = model.model.language_model
    print("language_model params:",
          sum(p.numel() for p in lang_model.parameters()) / 1e6, "M",
          flush=True)
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    lang_model = get_peft_model(lang_model, lora)
    if not args.game_mode:
        model.model.language_model = lang_model
    else:
        model = lang_model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.1f}M "
          f"({trainable/sum(p.numel() for p in model.parameters())*100:.2f}%)",
          flush=True)

    print("loading datasets...", flush=True)
    ds = load_dataset("json", data_files={"train": args.train, "eval": args.eval})

    def to_ids(row):
        msgs = row["messages"]
        # render EXACTLY like run_mate_eval -> HFModel.generate does:
        # processor.apply_chat_template(enable_thinking=False). Thinking is
        # OFF for the LoRA eval, so the trainer must use the same rendering
        # or train/eval prompts diverge (a silent accuracy cap).
        if processor is not None:
            out = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                return_assistant_tokens_mask=True, return_dict=True,
                enable_thinking=False)
            input_ids = out["input_ids"]
            mask = out["assistant_tokens_mask"]
        else:
            # text-only causal LM: compute the assistant span from the
            # '<|turn>model\n' marker (the last assistant turn).
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            marker = tokenizer("<|turn>model\n",
                               add_special_tokens=False)["input_ids"]
            start = None
            for i in range(len(ids) - len(marker) + 1):
                if ids[i:i + len(marker)] == marker:
                    start = i
            input_ids = ids
            mask = [False] * len(ids)
            if start is not None:
                mask[start:] = [True] * (len(ids) - start)
        if len(input_ids) > args.max_seq_len:
            input_ids = input_ids[: args.max_seq_len]
            mask = mask[: args.max_seq_len]
        labels = [-100 if not m else i
                  for i, m in zip(input_ids, mask)]
        return {"input_ids": input_ids, "labels": labels}

    def to_text(row):
        msgs = row["messages"]
        if processor is not None:
            return {"text": processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                enable_thinking=False)}
        return {"text": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)}

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
    n_assistant = sum(1 for m in s["assistant_tokens_mask"]) if "assistant_tokens_mask" in s else None
    print("--- sample check ---", flush=True)
    print("input len:", len(s["input_ids"]), "| label -100 count:",
          s["labels"].count(-100), flush=True)
    print("non-masked (assistant) tokens:",
          len(s["input_ids"]) - s["labels"].count(-100), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_to = ["wandb"] if args.wandb_project else []
    if report_to:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        import wandb

        print(f"wandb: project={args.wandb_project} run will log losses",
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
        logging_steps=50,
        eval_strategy="steps" if not args.smoke else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=report_to,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if not args.smoke else None,
        data_collator=MaskedCollator(tokenizer),
    )
    print("training...", flush=True)
    t0 = time.time()
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    if processor is not None:
        processor.save_pretrained(str(out))
    print(f"done in {(time.time()-t0)/3600:.2f}h -> {out}", flush=True)
    if not args.smoke:
        metrics = trainer.evaluate()
        print("eval:", json.dumps(metrics, indent=1), flush=True)


if __name__ == "__main__":
    main()
