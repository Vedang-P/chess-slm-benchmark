"""Extract a text-only gemma-4-E2B checkpoint from the multimodal one.

The full model is 10.25GB (vision+audio+text). Chess needs only the
text tower (`model.language_model.*`, 4.65B params). This writes a
standalone `Gemma4ForCausalLM` checkpoint that QLoRA can load on a
6GB card (2.3GB in 4-bit).

    python scripts/extract_text_tower.py \
        --src data/models/gemma-4-E2B-it \
        --dst data/models/gemma-4-E2B-it-text
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source model dir")
    ap.add_argument("--dst", required=True, help="output text-only dir")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    src_weights = src / "model.safetensors"
    if not src_weights.exists():
        sys.exit(f"no {src_weights}")

    out = {}
    with safe_open(src_weights, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.language_model."):
                out[k[len("model.language_model."):]] = f.get_tensor(k)

    # remap to Gemma4ForCausalLM naming: the causal LM uses `model.` keys
    save_file(out, str(dst / "model.safetensors"))
    print(f"wrote {len(out)} text-tower tensors -> {dst}/model.safetensors")

    cfg = json.loads((src / "config.json").read_text())
    text_cfg = cfg.get("text_config", {}).copy()
    text_cfg.update({
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4_text",
        "torch_dtype": "bfloat16",
    })
    (dst / "config.json").write_text(json.dumps(text_cfg, indent=1))
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "generation_config.json"):
        p = src / name
        if p.exists():
            (dst / name).write_bytes(p.read_bytes())
    print(f"wrote config -> {dst}/config.json")


if __name__ == "__main__":
    main()
