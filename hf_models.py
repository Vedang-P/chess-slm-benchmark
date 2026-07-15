"""Unified model inference wrapper for the spatial-reasoning project.

One backend behind a .load()/.generate() interface so eval.py and the
training scripts don't need per-model branching:
  - HFModel: plain HF transformers + bitsandbytes 4-bit, for every model
    including Gemma 4 E2B/E4B -- see .load()'s comment for why Gemma 4 no
    longer routes through Unsloth (it did originally; every crash that
    caused turned out to be inside Unsloth's own import chain or patched
    forward pass, not anything about plain transformers' ability to load
    Gemma 4, and peft>=0.19.0 ships proper Gemma4ClippableLinear support --
    see requirements.txt -- closing the original reason Unsloth was needed
    at all). LoRA attachment (.attach_lora() for inference-time use,
    load_trainable_model() below for training) works uniformly across every
    model now, Gemma 4 included.
  - OllamaModel: local Ollama server, quantized models. Alternative backend
    for Gemma 4 E2B on very low VRAM machines that already have Ollama set up.

Both backends default to greedy decoding (temperature=0, do_sample=False) so
eval numbers are a fixed, repeatable point rather than one stochastic draw --
set temperature > 0 explicitly if a call site wants sampling.
"""

import re
import time
from typing import Optional


def configure_quiet_logging():
    """Suppress library log/warning noise so the only things printed are this
    project's own progress bars and diagnostic lines. Call this as the very
    first thing in a script's entry point -- before importing anything else.

    Ordering matters for one specific reason: alphamaze_reference/benchmark/
    utils.py (imported by eval.py for faithful MazeBench scoring) calls
    logging.basicConfig(level=logging.INFO) at import time. basicConfig() is
    a no-op if the root logger already has a handler, so calling it here
    FIRST (claiming the root logger at a quiet level) prevents that import
    from silently escalating every other library's logger to INFO globally
    -- which is what was actually causing most of the "HTTP Request: ...",
    "TensorFlow version ... available", "NumExpr defaulting to N threads"
    noise. None of that is this project's own output; it's other libraries'
    loggers picking up the verbosity AlphaMaze's utils.py's basicConfig call
    left behind, once something (anything) imports it.
    """
    import logging
    import os
    import warnings

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Not just quieting output: trl's base_trainer.py calls transformers'
    # is_wandb_available() at import time (module-level, unconditional --
    # runs whether or not a Trainer is ever instantiated, regardless of our
    # own report_to=[] settings), which does a real `import wandb` to check.
    # On Kaggle specifically, the preinstalled wandb's SDK and its own
    # bundled generated protobuf file were observed out of sync ("cannot
    # import name 'Imports' from wandb.proto.wandb_telemetry_pb2"), which
    # crashes that check and takes down the entire `import unsloth` chain
    # with it -- confirmed fix (Unsloth's own Kaggle guidance): disable wandb
    # before anything imports it, not just suppress its logging.
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    # PyTorch's own suggested fix (printed directly in the CUDA OOM error
    # text on this project's actual Kaggle runs) for a specific fragmentation
    # pattern: a later model load OOMs while the error message itself reports
    # several GB "reserved but unallocated" -- i.e. VRAM is sitting in the
    # allocator's pool, not actually in use, just not contiguous enough to
    # satisfy the new allocation. Must be set before CUDA initializes.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")
    for name in ("transformers", "datasets", "huggingface_hub", "urllib3", "filelock", "numexpr"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # Best-effort only: this is a cosmetic step, never let it crash the
    # actual run (e.g. an incomplete/mocked module, or a library version
    # that has moved these functions, should just be silently skipped).
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    try:
        import datasets
        datasets.logging.set_verbosity_error()
    except Exception:
        pass


MODEL_IDS = {
    # confirmed trainable on 6 GB with 4-bit LoRA (+GRPO) via bitsandbytes+peft
    # -- see check_finetune_feasibility.py
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # marginal on 6 GB (fine on Kaggle's 16 GB T4)
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    # Gemma 4: the unsloth/ org's re-upload, not google/gemma-4-*-it. Kept
    # even though this project no longer loads Gemma 4 through Unsloth's own
    # FastLanguageModel loader (see load_trainable_model()'s docstring) --
    # it's still just a standard HF checkpoint, loadable via plain
    # AutoModelForCausalLM like any other model_id here, and switching to it
    # was what surfaced (not caused) the per-layer-projection dtype gap that
    # _fix_gemma4_dtype_mismatches() now handles regardless of loader.
    "gemma4-e2b": "unsloth/gemma-4-E2B-it",
    "gemma4-e4b": "unsloth/gemma-4-E4B-it",
}

OLLAMA_MODEL_TAGS = {
    "deepseek-r1-distill-qwen-1.5b": "deepseek-r1:1.5b",
    "smollm2-1.7b": "smollm2:1.7b",
    "qwen2.5-3b": "qwen2.5:3b",
    "gemma4-e2b": "gemma4:e2b",
}

# End-of-thinking delimiters to check for, in the order a real generation
# would contain them. NOT independently confirmed against real Gemma 4
# output as of this writing -- Google's own docs and community reports
# disagree with each other on the exact tokens (some describe
# "<|channel>thought\n...<channel|>", others plain "<think>...</think>"), and
# at least one HuggingFace thread reports the chat template itself is
# inconsistent. VERIFY THIS against the raw `records[i]["raw"]` field of your
# first real Gemma 4 eval run before trusting downstream scores -- if the
# model's real output uses a different marker, extract_reported_answer will
# silently fall through to "no marker found" behavior instead of correctly
# isolating the answer, which undercounts (fails closed) rather than
# overclaims, but should still be corrected once you've seen real output.
THINKING_END_MARKERS = ["<channel|>", "</think>", "<end_of_thought>"]

GEMMA4_THINK_SYSTEM_PROMPT = "<|think|>"


def is_gemma4(model_key: str) -> bool:
    return "gemma4" in model_key.lower() or "gemma-4" in model_key.lower()


def _fix_gemma4_dtype_mismatches(model):
    """Work around a real, confirmed gap in Unsloth's current Gemma 4
    support (their own source labels this architecture's patches
    "temporary_patches" -- actively in flux, not a finished implementation):
    on GPUs forced into a float32 fallback (no bf16 hardware support, and
    Unsloth's own printed banner says float16 "won't work" for Gemma 4 --
    both true for a T4), that float32 cast doesn't reach every submodule.
    Confirmed directly: per_layer_model_projection, a Gemma-4-specific
    architectural component (not a standard attention/MLP projection, so
    outside any LoRA target_modules list), stays in the checkpoint's
    original dtype while the rest of the model is float32, crashing the
    first forward pass with "expected mat1 and mat2 to have the same dtype".

    Rather than patch around that one named layer, walk every NON-quantized
    parameter (skip anything bitsandbytes 4-bit-quantized, identified by the
    `quant_state` attribute those carry and plain parameters don't -- must
    not touch those, they need to stay in their packed 4-bit representation)
    and force it to match the model's real compute dtype, read off the
    embedding layer specifically because embeddings are never 4-bit
    quantized regardless of GPU/dtype policy, so it's a reliable source of
    "what dtype is this model actually supposed to be". This catches this
    layer and any sibling with the same problem, not just the one the
    traceback happened to name first.
    """
    ref_dtype = model.get_input_embeddings().weight.dtype
    fixed = []
    for name, param in model.named_parameters():
        if hasattr(param, "quant_state"):
            continue
        if param.dtype.is_floating_point and param.dtype != ref_dtype:
            param.data = param.data.to(ref_dtype)
            fixed.append(name)
    if fixed:
        print(f"  [dtype fixup] cast {len(fixed)} non-quantized param(s) to {ref_dtype} "
              f"to match the embedding layer (Gemma 4 per-layer-projection dtype gap): "
              f"{fixed[0]}{f' (+{len(fixed) - 1} more)' if len(fixed) > 1 else ''}")
    return model


class HFModel:
    """Loads a HF causal LM once (optionally 4-bit + LoRA), reused across all
    generations in a run. Primary backend for every model's baseline
    inference in this project, including Gemma 4 (LoRA/GRPO training on
    Gemma 4 uses load_trainable_model() instead, see below)."""

    def __init__(self, model_key: str, smoke_test: bool = False,
                 load_in_4bit: bool = True, system_prompt: Optional[str] = None,
                 adapter_path: Optional[str] = None):
        self.model_key = model_key
        self.smoke_test = smoke_test
        self.load_in_4bit = load_in_4bit
        # Gemma 4's thinking mode is opt-in via a "<|think|>" control token in
        # the system message -- it is NOT automatic. Enable it by default for
        # any Gemma 4 model so it actually gets "leverage to think fully"
        # rather than silently answering without reasoning; pass an explicit
        # system_prompt to override.
        if system_prompt is None and is_gemma4(model_key):
            system_prompt = GEMMA4_THINK_SYSTEM_PROMPT
        self.system_prompt = system_prompt
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.smoke_test:
            # No real model load in smoke-test mode -- keep preflight fast and CPU-only.
            return self

        model_id = MODEL_IDS.get(self.model_key, self.model_key)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # A LoRA adapter checkpoint (what train_sft.py/train_grpo.py save) only
        # contains adapter weights + config -- it must be loaded onto the base
        # model it was trained from, not passed to from_pretrained directly.
        tokenizer_source = self.adapter_path if self.adapter_path else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # device_map={"": 0}, not "auto": these models are all small enough to
        # fit on one GPU, and on a multi-GPU box (e.g. Kaggle's "T4 x2") "auto"
        # shards the model's layers across both devices by default -- then
        # anything that does a plain .cuda() (defaults to cuda:0) mismatches
        # against layers auto-placed on cuda:1. Pinning to one device avoids
        # that whole class of bug; there's no benefit to sharding a model
        # this size across GPUs anyway.
        #
        # Gemma 4 loads through this exact same plain path now, not Unsloth --
        # every crash hit trying to route it through Unsloth (ConstantLengthDataset
        # import, a broken wandb install, a dtype mismatch in a Gemma-4-specific
        # layer) turned out to be inside Unsloth's own import chain or its
        # patched/compiled forward pass, not anything about whether plain
        # transformers can load Gemma 4 -- it already was, successfully, every
        # single time, from inside Unsloth's own loader (which itself just
        # calls AutoModelForCausalLM.from_pretrained under the hood).
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map={"": 0},
                trust_remote_code=True, dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map={"": 0}, trust_remote_code=True, dtype=torch.bfloat16,
            )
        if is_gemma4(self.model_key):
            _fix_gemma4_dtype_mismatches(self.model)

        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()
        return self

    def attach_lora(self, r: int = 16, lora_alpha: int = 16):
        """Attach a LoRA adapter for fine-tuning. Call after load(), before
        training. Not usable for Gemma 4 -- see load_trainable_model()."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(
            r=r, lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.enable_input_require_grads()
        return self

    def generate(self, prompt: str, max_new_tokens: int = 4096, temperature: float = 0.0,
                 top_p: float = 1.0, repetition_penalty: float = 1.0) -> dict:
        """Returns dict(content, input_tokens, output_tokens, latency_ms, finished).
        `finished` is True iff generation ended on its own (EOS) rather than
        hitting max_new_tokens -- a truncated generation shouldn't be trusted
        as a complete answer, see extract_reported_answer()."""
        t0 = time.time()

        if self.smoke_test:
            # Deterministic stub response so the parsing/scoring pipeline is
            # exercised end-to-end without loading a multi-GB model.
            content = "(0,0) (0,1) (0,2)"
            return {
                "content": content,
                "input_tokens": len(prompt.split()),
                "output_tokens": len(content.split()),
                "latency_ms": (time.time() - t0) * 1000,
                "finished": True,
            }

        import torch

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        do_sample = temperature > 0.0
        gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                           pad_token_id=self.tokenizer.eos_token_id)
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["repetition_penalty"] = repetition_penalty

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        output_ids = outputs[0][input_len:]
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return {
            "content": content,
            "input_tokens": int(input_len),
            "output_tokens": int(output_ids.shape[-1]),
            "latency_ms": (time.time() - t0) * 1000,
            "finished": int(output_ids.shape[-1]) < max_new_tokens,
        }


class OllamaModel:
    """Same .load()/.generate() interface as HFModel, backed by a local
    Ollama server instead of HF transformers. Alternative backend for Gemma 4
    E2B on machines that already have Ollama set up with a quantized model.
    """

    def __init__(self, model_key: str, base_url: str = "http://localhost:11434",
                 smoke_test: bool = False):
        self.model_key = model_key
        self.smoke_test = smoke_test
        self.base_url = base_url
        self.env = None

    def load(self):
        if self.smoke_test:
            return self
        from src.ollama_env import OllamaEnv
        self.env = OllamaEnv(model=OLLAMA_MODEL_TAGS[self.model_key],
                              base_url=self.base_url, temperature=0.0)
        return self

    def generate(self, prompt: str, max_new_tokens: int = 4096, temperature: float = 0.0) -> dict:
        if self.smoke_test:
            content = "(0,0) (0,1) (0,2)"
            return {
                "content": content,
                "input_tokens": len(prompt.split()),
                "output_tokens": len(content.split()),
                "latency_ms": 0.0,
                "finished": True,
            }
        result = self.env.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens, temperature=temperature,
        )
        output_tokens = result.get("output_tokens", 0)
        return {
            "content": result.get("content", ""),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": output_tokens,
            "latency_ms": result.get("latency_ms", 0.0),
            "finished": output_tokens < max_new_tokens,
        }


def load_trainable_model(model_id: str, load_in_4bit: bool = True, max_seq_length: int = 2048,
                          lora_r: int = 16, lora_alpha: int = 16, adapter_path: str = None,
                          force_gemma4: bool = None):
    """Load a model + attach a LoRA adapter, ready for SFTTrainer/GRPOTrainer.
    Used by train_sft.py and train_grpo.py so both scripts share one place
    that knows which models need which backend.

    Every model, Gemma 4 included, goes through the same plain
    bitsandbytes+peft path -- Gemma 4 used to route through Unsloth instead
    (its FastLanguageModel loader patches around Gemma4ClippableLinear,
    which older peft didn't recognize as an attachable target module type).
    That's now the wrong call for two independent reasons: (1) every crash
    hit going through Unsloth (a broken ConstantLengthDataset import, a
    broken wandb install, a dtype mismatch inside Unsloth's own
    patched/compiled forward pass) was inside Unsloth's own code, not
    anything about whether plain transformers can load and train Gemma 4 --
    it already was loading successfully every single time, from inside
    Unsloth's own loader, which itself just calls plain
    AutoModelForCausalLM.from_pretrained under the hood; (2) peft>=0.19.0
    (see requirements.txt) ships its own built-in Gemma4ClippableLinear
    support, scoped to the language-model layers via a regex when
    target_modules is omitted, which is exactly what's needed here and
    closes the original reason Unsloth was necessary at all.

    adapter_path: if given, `model_id` is still the BASE model (e.g. the
    shorthand/HF ID GRPO should keep resolving VRAM/backend warnings against),
    and an EXISTING LoRA adapter is loaded on top of it from this local
    directory, continuing training from there (e.g. GRPO continuing from an
    SFT warm-start) -- rather than attaching a fresh, randomly-initialized
    adapter, which is what happens if this is left unset. Without this,
    "continue training" would silently discard whatever the prior stage
    actually learned and start over from the untrained base model, which is
    exactly the bug this parameter exists to close (see train_grpo.py's
    --model_path).

    force_gemma4: explicitly say whether this is a Gemma 4 model, overriding
    the is_gemma4(model_id) string-match. Needed because a local checkpoint
    directory (e.g. "./results/sft_run_1") won't necessarily contain
    "gemma4"/"gemma-4" in its name the way a model shorthand/HF ID does --
    callers should pass is_gemma4(<the original --model shorthand>) here
    whenever `model_id` might be a local path rather than that shorthand.

    Returns (model, tokenizer, backend_name) -- backend_name is always
    "bnb_peft" now; kept as a return value since callers/logging already
    expect it, in case a model ever needs a genuinely different backend again.
    """
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
    use_gemma4 = is_gemma4(model_id) if force_gemma4 is None else force_gemma4

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    tokenizer_source = adapter_path if adapter_path else model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map={"": 0}, not "auto" -- see the comment in HFModel.load() above.
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map={"": 0},
            trust_remote_code=True, dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": 0}, trust_remote_code=True, dtype=torch.bfloat16,
        )
    if use_gemma4:
        _fix_gemma4_dtype_mismatches(model)
    model = prepare_model_for_kbit_training(model)
    if adapter_path:
        # is_trainable=True: peft's documented pattern for resuming training
        # on a saved adapter -- without it, PeftModel.from_pretrained loads
        # the adapter frozen (inference-only), and GRPO would have nothing
        # trainable to update, silently or via an obscure error.
        print(f"  [load_trainable_model] loading saved adapter from {adapter_path} "
              f"(is_trainable=True) on top of base {model_id}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    elif use_gemma4:
        # No target_modules: peft>=0.19.0 ships its own Gemma4ClippableLinear-
        # aware defaults, scoped to the language-model layers via a regex,
        # which is exactly what's needed for text-only spatial reasoning --
        # our own target_modules string list (q_proj/k_proj/... below) names
        # standard attention/MLP projections that don't match how Gemma 4
        # wraps them, so passing it here would just fail to attach anything.
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                                  lora_dropout=0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
    else:
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
                                  lora_dropout=0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    return model, tokenizer, "bnb_peft"


def extract_coords_from_text(text: str) -> list:
    """Language-agnostic coordinate extraction: numbers don't need
    translation, so this works regardless of instruction language."""
    patterns = [r"\((\d+),\s*(\d+)\)", r"\[(\d+),\s*(\d+)\]"]
    coords = []
    for pat in patterns:
        for x, y in re.findall(pat, text):
            coords.append((int(x), int(y)))
    return coords


def extract_reported_answer(text: str, finished: bool = True, require_marker: bool = False) -> Optional[str]:
    """Return the model's own clean final-answer text -- never mine
    coordinates out of a thinking trace the model didn't itself present as
    its answer. Policy, in order:

      1. If any marker in THINKING_END_MARKERS is present, thinking is done;
         work only with the text after the LAST such marker from here on.
      2. Within that (or the whole text, if no thinking marker appeared): if
         a "FINAL ANSWER:" marker is present, return what follows it.
      3. Otherwise, if `require_marker` is True (this prompt format explicitly
         asked for a "FINAL ANSWER:" line and didn't get one), return None --
         that is a failure to follow the requested format, not something to
         recover by scanning raw text.
      4. Otherwise (no marker was ever requested for this format -- e.g. the
         token-maze format just asks for move tokens), return the remaining
         text as-is, UNLESS no thinking marker was found at all AND the
         generation was truncated (`finished=False`) -- in that case we can't
         tell whether the model was still mid-thought, so there is nothing
         reliable to extract.
    """
    remaining = text
    end = -1
    for marker in THINKING_END_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            end = max(end, idx + len(marker))
    if end != -1:
        remaining = text[end:]
    elif not finished:
        return None

    m = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", remaining, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    if require_marker:
        return None
    return remaining.strip() or None


def parse_path_response(content: str, start: tuple, goal: tuple) -> Optional[list]:
    """Extract a start->goal coordinate sequence from free-form model output,
    independent of the instruction/response language.

    This is the ONE canonical path parser for the NL/coordinate format --
    every eval and training script should import this rather than
    reimplementing extraction. Reasoning models routinely restate the start
    (and sometimes goal) coordinate in prose before giving the actual path
    (e.g. "start is (1,3), ... the path is [(1,3), (1,4), ...]"), so this
    takes the LAST occurrence of `start` in the text and pairs it with the
    FIRST `goal` occurrence after it -- i.e. the shortest, cleanest start->goal
    span nearest the end of the response, skipping over earlier preamble
    mentions. An earlier version of this function took the first `start` and
    the longest possible span to `goal`, which could scoop up a spurious
    preamble mention of `start` as an extra leading coordinate -- turning a
    correct answer into a path with a zero-length first "step" that then
    fails the unit-step check downstream. Never validates a fragment that
    doesn't actually connect the real endpoints.

    Callers evaluating a model's answer (as opposed to training-time reward,
    where seeing some rambling is expected) should run this on the output of
    extract_reported_answer(), not the raw generation, so a model that never
    reported a clean answer at all doesn't get a path mined out of its
    thinking trace.
    """
    coords = extract_coords_from_text(content)
    if len(coords) < 2:
        return None

    for i in range(len(coords) - 1, -1, -1):
        if coords[i] != start:
            continue
        for j in range(i + 1, len(coords)):
            if coords[j] == goal:
                return coords[i:j + 1]
    return None
