"""Unified model inference wrapper for the spatial-reasoning project.

Two backends behind one .load()/.generate() interface so eval.py and the
training scripts don't need per-model branching:
  - HFModel: plain HF transformers + bitsandbytes 4-bit. Used for every
    model's baseline INFERENCE (DeepSeek-R1-Distill-Qwen-1.5B, SmolLM2-1.7B,
    Qwen2.5-*, AlphaMaze-v0.2-1.5B, and Gemma 4 E2B/E4B). LoRA attachment via
    .attach_lora() works for all of these except Gemma 4 -- see
    load_trainable_model() below for why Gemma 4 needs a different path for
    training specifically (not for plain inference/baselining).
  - OllamaModel: local Ollama server, quantized models. Alternative backend
    for Gemma 4 E2B on very low VRAM machines that already have Ollama set up.

Both backends default to greedy decoding (temperature=0, do_sample=False) so
eval numbers are a fixed, repeatable point rather than one stochastic draw --
set temperature > 0 explicitly if a call site wants sampling.
"""

import re
import time
from typing import Optional


MODEL_IDS = {
    # confirmed trainable on 6 GB with 4-bit LoRA (+GRPO) via bitsandbytes+peft
    # -- see check_finetune_feasibility.py
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # marginal on 6 GB (fine on Kaggle's 16 GB T4)
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    # Gemma 4: fine for baseline INFERENCE via plain HFModel below (needs
    # transformers>=5.13 to recognize the architecture at all, no Unsloth
    # required just to run it). LoRA/GRPO training on these two specifically
    # goes through load_trainable_model()'s Unsloth path instead, since plain
    # peft can't attach LoRA to Gemma 4's Gemma4ClippableLinear layers.
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
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

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # A LoRA adapter checkpoint (what train_sft.py/train_grpo.py save) only
        # contains adapter weights + config -- it must be loaded onto the base
        # model it was trained from, not passed to from_pretrained directly.
        model_id = MODEL_IDS.get(self.model_key, self.model_key)
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
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map={"": 0},
                trust_remote_code=True, torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.bfloat16,
            )

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
                          lora_r: int = 16, lora_alpha: int = 16):
    """Load a model + attach a LoRA adapter, ready for SFTTrainer/GRPOTrainer.
    Used by train_sft.py and train_grpo.py so both scripts share one place
    that knows which models need which backend.

    Routes Gemma 4 through Unsloth (its FastLanguageModel loader patches
    around Gemma4ClippableLinear, which plain peft's LoraConfig doesn't
    recognize as an attachable target module type -- confirmed by direct
    test on this project's hardware, not a guess). Everything else goes
    through the plain bitsandbytes+peft path used throughout this project.

    Returns (model, tokenizer, backend_name) -- backend_name is "unsloth" or
    "bnb_peft", useful for logging/debugging which path was taken.
    """
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

    if is_gemma4(model_id):
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=max_seq_length,
            dtype=None, load_in_4bit=load_in_4bit,
        )
        # Explicit finetune_*_layers/modules flags, not left to Unsloth's
        # defaults -- Unsloth intersects target_modules with these filters
        # ("adapters attach only where both select"), and Gemma 4's Unsloth
        # loader is vision-model-aware even for text-only "it" checkpoints.
        # We're doing text-only spatial reasoning, so vision layers are off.
        model = FastLanguageModel.get_peft_model(
            model, r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=42,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer, "unsloth"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    model = prepare_model_for_kbit_training(model)
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
