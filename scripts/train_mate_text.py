"""Train the MATE text transformer (joint or per-subset).

    python3 scripts/train_mate_text.py \
        --data data/raw/mate-text/train.jsonl \
        --val data/raw/mate-text/val.jsonl \
        --out results/mate-text-model --d-model 512 --layers 8

`--subset` filters the training data to one MATE subset for the
single-task ablation arms (strategy|tactic|noexplain|both).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.mate_text.data import make_dataloader, MateTextDataset  # noqa: E402
from src.mate_text.model import MateTextConfig, MateTextTransformer  # noqa: E402
from src.mate_text.tokenizer import MateTokenizer  # noqa: E402


def build_ds(path, tokenizer, subset):
    rows = [json.loads(l) for l in open(path)]
    if subset:
        rows = [r for r in rows if r.get("subset") == subset]
    tmp = Path("/tmp") / f"mate-text-{subset or 'all'}-{hash(path)}.jsonl"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return MateTextDataset(str(tmp), tokenizer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="results/mate-text-model")
    ap.add_argument("--subset", default="",
                    help="filter training to one MATE subset for ablation")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="mate-text-transformer")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 = run full epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} cuda_cap="
          f"{torch.cuda.get_device_capability(0) if device=='cuda' else (0,0)}",
          flush=True)

    tokenizer = MateTokenizer()
    cfg = MateTextConfig(
        vocab_size=tokenizer.vocab_size, d_model=args.d_model,
        n_layers=args.layers, n_heads=args.heads, d_ff=args.d_ff,
    )
    model = MateTextTransformer(cfg).to(device)
    if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 7:
        model = model.bfloat16()
    print(f"params: {model.count_params()/1e6:.1f}M "
          f"(d_model {args.d_model}, {args.layers} layers)", flush=True)

    train_ds = build_ds(args.data, tokenizer, args.subset)
    val_ds = build_ds(args.val, tokenizer, args.subset)
    print(f"train rows: {len(train_ds)} | val rows: {len(val_ds)} "
          f"| subset: {args.subset or 'all'}", flush=True)

    from torch.utils.data import DataLoader

    def loader(ds):
        return DataLoader(ds, batch_size=args.batch, shuffle=True,
                          num_workers=2,
                          collate_fn=lambda b: _collate(b, tokenizer))

    train_loader = loader(train_ds)
    val_loader = loader(val_ds)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    if args.max_steps:
        total_steps = args.max_steps

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
            run = wandb.init(project=args.wandb_project,
                             config=vars(args), reinit=True)
        except Exception:
            use_wandb = False

    step = 0
    best_val = -1.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for tok_ids, type_ids, labels in train_loader:
            tok_ids, type_ids, labels = (tok_ids.to(device),
                                         type_ids.to(device),
                                         labels.to(device))
            logits, loss = model(tok_ids, type_ids, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} "
                      f"loss {loss.item():.4f} "
                      f"({(time.time()-t0)/60:.1f}min)",
                      flush=True)
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "step": step})

            if step % args.eval_every == 0 or args.max_steps == step:
                val_acc = evaluate(model, val_loader, device)
                print(f"  val acc: {val_acc:.4f}", flush=True)
                if use_wandb:
                    wandb.log({"val/acc": val_acc, "step": step})
                if val_acc > best_val:
                    best_val = val_acc
                    torch.save({"model": model.state_dict(),
                                "cfg": cfg.__dict__,
                                "vocab": tokenizer.vocab_size},
                               out / "best.pt")
                    print(f"  saved best ({best_val:.4f})", flush=True)

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    # final
    val_acc = evaluate(model, val_loader, device)
    print(f"final val acc: {val_acc:.4f} (best {best_val:.4f})", flush=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "vocab": tokenizer.vocab_size},
               out / "final.pt")
    print(f"done in {(time.time()-t0)/3600:.2f}h -> {out}", flush=True)


def _collate(batch, tokenizer):
    from src.mate_text.data import collate
    return collate(batch, tokenizer.special_pad())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for tok_ids, type_ids, labels in loader:
        tok_ids, type_ids, labels = (tok_ids.to(device),
                                     type_ids.to(device),
                                     labels.to(device))
        logits, _ = model(tok_ids, type_ids)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    model.train()
    return correct / total


if __name__ == "__main__":
    main()
