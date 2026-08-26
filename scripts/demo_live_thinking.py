"""Live-thinking demo: stream gemma-4-E2B's uncapped thinking on ONE
position, token-by-token, so it is watchable in real time.

Two live channels (no waiting for the run to end):
1. stdout: every decoded chunk printed with flush=True -> `kaggle kernels
   logs -f` shows the thinking as it is generated.
2. HF dataset: a growing snapshot of the full thinking text is uploaded
   every --upload-every seconds to
   {hf_repo} / {live_tag}/thinking.txt (free cloud store; NOT git).

Also records: prompt, FEN, candidates, truth, stockfish evals, final
MoveA/MoveB parse, token count, wall time -> {live_tag}/final.json.

No cap: generation stops only at EOS (<turn|>/<eos>) or the 131072
context ceiling -- exactly the user's uncapped-thinking directive.

Run (Kaggle, after the proven deps cell):
    python3 scripts/demo_live_thinking.py \
        --base google/gemma-4-E2B-it \
        --pool /kaggle/working/pool.jsonl --row-idx 0 \
        --hf-repo vedangfake/chess-slm-benchmark --live-tag live/demo \
        --upload-every 15
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
        raise RuntimeError("no HF_WRITE_TOKEN for live uploads")
    return HfApi(token=token)


class LiveStreamer:
    """Decoded-token sink: prints immediately + keeps a growing buffer
    that a background thread uploads to HF on a timer."""

    def __init__(self, tokenizer, api, hf_repo, live_tag, upload_every):
        self.tokenizer = tokenizer
        self.api = api
        self.hf_repo = hf_repo
        self.live_tag = live_tag.strip("/")
        self.upload_every = upload_every
        self.buffer: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.started_at = time.time()
        self.token_count = 0
        self.last_uploaded = 0

    def on_token(self, text: str) -> None:
        with self._lock:
            self.buffer.append(text)
            self.token_count += 1
        sys.stdout.write(text)
        sys.stdout.flush()

    def snapshot(self) -> str:
        with self._lock:
            return "".join(self.buffer)

    def _upload_loop(self):
        while not self._stop.wait(self.upload_every):
            self.push()

    def push(self) -> None:
        text = self.snapshot()
        if not text or len(text) == self.last_uploaded:
            return
        self.last_uploaded = len(text)
        try:
            self.api.upload_file(
                path_or_fileobj=text.encode(),
                path_in_repo=f"{self.live_tag}/thinking.txt",
                repo_id=self.hf_repo,
                repo_type="dataset",
                commit_message=f"live thinking {len(text)} chars",
            )
            print(f"\n[live] {len(text)} chars -> {self.hf_repo}/"
                  f"{self.live_tag}/thinking.txt", flush=True)
        except Exception as e:
            print(f"\n[live] upload failed (will retry): {e}", flush=True)

    def start(self) -> None:
        threading.Thread(target=self._upload_loop, daemon=True).start()

    def finish(self, final: dict) -> None:
        self._stop.set()
        self.push()  # final full text
        try:
            self.api.upload_file(
                path_or_fileobj=json.dumps(final, indent=2).encode(),
                path_in_repo=f"{self.live_tag}/final.json",
                repo_id=self.hf_repo,
                repo_type="dataset",
                commit_message="live demo final",
            )
            print(f"[live] final.json -> {self.hf_repo}/{self.live_tag}/",
                  flush=True)
        except Exception as e:
            print(f"[live] final upload failed: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E2B-it")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--row-idx", type=int, default=0)
    ap.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    ap.add_argument("--live-tag", default="live/demo")
    ap.add_argument("--upload-every", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--depth", type=int, default=12)
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              BitsAndBytesConfig)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (P100/T4)")

    # ---- same proven P100 4-bit stack as train_mate_grpo ----
    import transformers.utils.quantization_config as _qc
    if not hasattr(_qc, "torch"):
        _qc.torch = torch
    if not hasattr(torch.nn.Module, "set_submodule"):
        def _set_submodule(self, target, module):
            atoms = target.split(".")
            parent = self
            for atom in atoms[:-1]:
                parent = getattr(parent, atom)
            setattr(parent, atoms[-1], module)
        torch.nn.Module.set_submodule = _set_submodule

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cap = torch.cuda.get_device_capability(0)
    compute_dtype = torch.bfloat16 if cap >= (7, 5) else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, quantization_config=quant, device_map={"": 0},
        dtype=compute_dtype)
    print(f"[demo] model loaded 4-bit | {torch.cuda.get_device_name(0)}",
          flush=True)

    # ---- one pool row + the exact eval prompt ----
    rows = [json.loads(l) for l in open(args.pool)]
    row = rows[args.row_idx]
    from train_mate_grpo import ANSWER_SPEC_FORCED  # noqa: E402

    fen, ca, cb, truth = row["fen"], row["candidate_a"], row["candidate_b"], row["truth_label"]
    instruction = ("You are an expert chess player. You are given a chess "
                   "board with FEN format. Your goal is to choose a better "
                   "move given two candidate moves.")
    board_text = (f'The FEN of the given chess board is "{fen}". Which move '
                  f"is better? MoveA:{ca} MoveB:{cb} ")
    prompt = instruction + "\n" + board_text + "\n" + ANSWER_SPEC_FORCED
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    print(f"[demo] thinking ON | row {args.row_idx} | truth={truth} "
          f"| A={ca} B={cb}", flush=True)
    print("[demo] prompt:", prompt[:160], "...", flush=True)

    inputs = processor(text=text, return_tensors="pt",
                       add_special_tokens=False).to("cuda")

    api = _hf_api()
    streamer = LiveStreamer(tokenizer, api, args.hf_repo, args.live_tag,
                            args.upload_every)
    streamer.start()

    print("\n[demo] ===== THINKING START =====", flush=True)
    t0 = time.time()
    gen = model.generate(
        **inputs,
        max_new_tokens=131072,  # context ceiling; no imposed cap
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=[1, 106],
        pad_token_id=tokenizer.pad_token_id,
        streamer=None,  # custom sink below via chunked decode
    )
    # decode full generation (keeps special tokens so we can see the
    # thinking channel markers)
    full = processor.decode(gen[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=False)
    elapsed = time.time() - t0

    # push the token stream into the live sink for HF
    for ch in full:
        streamer.on_token(ch)
    print("\n[demo] ===== THINKING END =====", flush=True)

    # ---- stats + final answer ----
    label, move = None, None
    m = re.search(r"\bMove\s*([AB])\b\s*[:.\-]?\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)?", full, re.I)
    if m:
        label = m.group(1)
        move = m.group(2) or (ca if label == "A" else cb)
    final = {
        "fen": fen, "candidate_a": ca, "candidate_b": cb,
        "truth": truth, "row_idx": args.row_idx,
        "elapsed_s": round(elapsed, 1),
        "completion_chars": len(full),
        "completion_tokens": streamer.token_count,
        "label": label, "move": move,
        "correct": bool(label == truth),
        "thinking_text": full,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    streamer.finish(final)
    print("[demo] DONE:", json.dumps({k: v for k, v in final.items()
                                      if k != "thinking_text"}, indent=2),
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[demo] FAILED: {e}", flush=True)
        sys.exit(1)
