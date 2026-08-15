"""Generate kaggle_stack_check.ipynb — ONE-shot diagnostic that resolves the
LoRA-wrap issue on the real gemma-4-E2B and validates a tiny training step.

Strategy: the notebook itself iterates. It installs deps, loads the model,
PRINTS the actual text-tower module tree (the missing fact so far), tries
multiple wrap strategies in order, runs a 3-step training smoke, and writes
a full report to HF run-status.txt. One launch, all answers — no repeated
kernel pushes per hypothesis.

    python notebooks/build_stack_check_kernel_notebook.py
    kaggle kernels push -p notebooks/push_stack_check
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
OWNER = os.environ.get("STACK_OWNER", "vedanggggg")
SLUG = "stack-check-gemma4"
WANDB_PROJECT = "chess-slm-benchmark"

CLONE_CELL = r'''
import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
if REPO.exists():
    shutil.rmtree(REPO)

def find_token():
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        v = os.environ.get(name)
        if v:
            return v
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        return None

url = "https://github.com/Vedang-P/chess-slm-benchmark.git"
url = url.replace("https://", f"https://x-access-token:{find_token()}@")
res = subprocess.run(["git", "clone", "--quiet", "-b", "main", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed: " + res.stderr[-300:])
os.chdir(REPO)
print("cwd:", Path.cwd())
'''.strip()

DEPS_CELL = r'''
import subprocess, sys, torch
# Same proven stack as the SFT kernel: torch 2.5.1 cu121 --no-deps + nvidia
# stack from the PyTorch index (cudnn 9.1.1.17 replaces the yanked
# 9.1.0.70), transformers 5.13.1, peft 0.14.0, bnb 0.46.1.
CU121 = "https://download.pytorch.org/whl/cu121"
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
                "--index-url", CU121, "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "nvidia-cudnn-cu12==9.1.1.17", "--index-url", CU121], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "nvidia-cublas-cu12", "nvidia-cuda-cupti-cu12",
                "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-runtime-cu12",
                "nvidia-cufft-cu12", "nvidia-curand-cu12",
                "nvidia-cusolver-cu12", "nvidia-cusparse-cu12",
                "nvidia-nccl-cu12", "nvidia-nvjitlink-cu12",
                "nvidia-nvtx-cu12", "triton", "--index-url", CU121], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "bitsandbytes==0.46.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "--quiet", "-y",
                "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "peft==0.14.0"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "transformers==5.13.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U", "wandb"], check=True)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''.strip()

DIAG_CELL = r'''
import os, sys, json, traceback
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

REPORT = []

def log(msg):
    print(msg, flush=True)
    REPORT.append(str(msg))

try:
    cap = torch.cuda.get_device_capability(0)
    log(f"GPU: {torch.cuda.get_device_name(0)} cap={cap}")
    compute_dtype = torch.bfloat16 if cap >= (7, 5) else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )
    log("loading model...")
    processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-4-E2B-it", quantization_config=quant,
        device_map={"": 0}, dtype=compute_dtype)
    log("model loaded")

    lang = model.model.language_model
    keys = [n for n, _ in lang.named_modules()]
    log(f"text tower modules: {len(keys)}")
    log("first 15 keys: " + json.dumps(keys[:15]))
    qkeys = [k for k in keys if "q_proj" in k]
    log("q_proj keys: " + json.dumps(qkeys[:5]))
    lkeys = [k for k in keys if k.endswith(".linear")]
    log(f"keys ending .linear: {len(lkeys)} -> " + json.dumps(lkeys[:4]))
    l4 = [k for k in keys if "Linear4bit" in str(type(dict(lang.named_modules()).get(k)))]
    log(f"Linear4bit count: {sum(1 for _, m in lang.named_modules() if type(m).__name__=='Linear4bit')}")

    # freeze + stub
    for p in lang.parameters():
        p.requires_grad = False
    if not hasattr(lang, "prepare_inputs_for_generation"):
        lang.prepare_inputs_for_generation = lambda *a, **k: None
    lang.config.use_cache = False
    try:
        lang.gradient_checkpointing_enable()
    except Exception as e:
        log(f"gcp: {e}")

    # try wrap strategies in order; stop at first success
    strategies = [
        ("exact-linear", ["linear"]),
        ("regex-any-linear", r".*\.linear"),
        ("wrapper-names", ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]),
        ("all-linear", ".*(?:q|k|v|o|gate|up|down)_proj\\.linear"),
    ]
    wrapped = None
    for name, target in strategies:
        try:
            lora = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, bias="none",
                              target_modules=target, task_type="CAUSAL_LM")
            lang = get_peft_model(lang, lora)
            model.model.language_model = lang
            t = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f"STRATEGY {name} (target={target}) WORKS; trainable={t/1e6:.1f}M")
            wrapped = name
            break
        except Exception as e:
            log(f"strategy {name} failed: {type(e).__name__}: {str(e)[:150]}")

    if wrapped is None:
        raise RuntimeError("NO wrap strategy worked")

    # tiny 3-step training smoke using the committed 300-row dataset
    from datasets import load_dataset
    ds = load_dataset("json", data_files={
        "train": "data/positions/noexplain-slice-smoke/train.jsonl",
        "eval": "data/positions/noexplain-slice-smoke/eval.jsonl"})

    def to_ids(row):
        msgs = row["messages"]
        full = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", enable_thinking=False)["input_ids"][0]
        prompt = processor.apply_chat_template(
            msgs[:1], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False)["input_ids"][0]
        assert full[:len(prompt)].tolist() == prompt.tolist()
        iids = full.tolist()[:2048]
        labels = [-100] * len(prompt) + iids[len(prompt):]
        return {"input_ids": iids, "labels": labels[:2048]}

    tr = ds["train"].select(range(64)).map(to_ids)
    ev = ds["eval"].select(range(16)).map(to_ids)
    s = tr[0]
    n_ass = len(s["input_ids"]) - s["labels"].count(-100)
    log(f"mask check: assistant tokens = {n_ass}")
    assert n_ass > 0

    class MaskedCollator:
        def __init__(self, tok): self.tok = tok
        def __call__(self, features):
            ids = self.tok.pad([{"input_ids": f["input_ids"]} for f in features],
                               return_tensors="pt", padding=True)
            labels = torch.full_like(ids["input_ids"], -100)
            for i, f in enumerate(features):
                labels[i, :len(f["labels"])] = torch.tensor(f["labels"][:len(f["labels"])])
            return {"input_ids": ids["input_ids"], "attention_mask": ids["attention_mask"],
                    "labels": labels}

    from transformers import Trainer, TrainingArguments
    args = TrainingArguments(
        output_dir="results/stack-check",
        num_train_epochs=0.1, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_steps=0, weight_decay=0.0,
        optim="adamw_8bit", bf16=cap >= (7, 5), fp16=cap < (7, 5),
        logging_steps=1, save_strategy="no", eval_strategy="no",
        report_to=[], seed=42, max_grad_norm=1.0, dataloader_pin_memory=False)
    trainer = Trainer(model=model, args=args, train_dataset=tr,
                      data_collator=MaskedCollator(processor.tokenizer))
    log("training 3 steps...")
    trainer.train()
    log("TRAINING SMOKE OK")

    # save adapter roundtrip check
    out = Path("results/stack-check-adapter")
    out.mkdir(parents=True, exist_ok=True)
    lang.save_pretrained(str(out))
    files = sorted(f.name for f in out.iterdir())
    log("adapter saved: " + json.dumps(files))
    model2 = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-4-E2B-it", quantization_config=quant,
        device_map={"": 0}, dtype=compute_dtype)
    lang2 = model2.model.language_model
    if not hasattr(lang2, "prepare_inputs_for_generation"):
        lang2.prepare_inputs_for_generation = lambda *a, **k: None
    lang2 = PeftModel.from_pretrained(lang2, str(out))
    model2.model.language_model = lang2
    log("ADAPTER ROUNDTRIP OK")
    log("RESULT: ALL CHECKS PASSED; strategy=" + wrapped)
except Exception as e:
    log(f"FAILED: {type(e).__name__}: {e}")
    log(traceback.format_exc()[-2000:])

# write report to HF (fast diagnostics channel)
try:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
    api.upload_file(path_or_fileobj=("\n".join(REPORT)).encode(),
                    path_in_repo="noexplain-slice/run-status.txt",
                    repo_id="vedangfake/chess-slm-benchmark",
                    repo_type="dataset",
                    commit_message="stack check report")
    print("[status] report written to HF run-status.txt", flush=True)
except Exception as e2:
    print("[status] upload failed:", e2, flush=True)
'''.strip()


def load_env() -> dict:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def inject_secrets(nb: dict, env: dict, names: list[str]) -> None:
    lines = ["import os\n"]
    for name in names:
        if name not in env or not env[name]:
            raise RuntimeError(f"missing secret {name} in .env")
        lines.append(f'os.environ[{name!r}] = {env[name]!r}\n')
    lines.append("print('secrets set:', "
                 + ", ".join(f"bool(os.environ.get({n!r}))" for n in names)
                 + ")\n")
    cell = {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": lines,
    }
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source") or [])
        if "secrets are injected at build time" in src:
            nb["cells"][i] = cell
            return
    raise RuntimeError("placeholder secrets cell not found")


def main() -> None:
    cells = [
        _md("# Stack check — gemma-4-E2B LoRA wrap + tiny training smoke\n\n"
            "One launch, all answers: real module tree, wrap strategies, "
            "3-step training, adapter roundtrip. Report -> HF run-status.txt."),
        _md("## 1. Secrets (injected at build time)"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies"),
        _code(DEPS_CELL),
        _md("## 4. Diagnose + wrap + train + roundtrip"),
        _code(DIAG_CELL),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env, ["GITHUB_TOKEN", "HF_WRITE_TOKEN", "WANDB_API_KEY"])

    push_dir = NB_DIR / "push_stack_check"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = "kaggle_stack_check.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}",
        "title": "Stack check gemma4",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }, indent=1))
    print(f"wrote {push_dir}/{code_file}")
    print(f"push with: kaggle kernels push -p notebooks/push_stack_check")


if __name__ == "__main__":
    main()
