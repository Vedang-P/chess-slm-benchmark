"""Gemma 4 E2B model wrapper for on-device inference.

Supports:
- Function calling (native tool use)
- Thinking mode (chain-of-thought reasoning)
- Quantized inference (via bitsandbytes / llama.cpp)
- Structured JSON output with grammar constraints
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Optional


class Gemma4Env:
    """Gemma 4 E2B inference environment for on-device pathfinding."""

    MODEL_ID = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
    ORIGINAL_MODEL_ID = "google/gemma-4-E2B-it"

    def __init__(
        self,
        device: str = "auto",
        load_in_4bit: bool = True,
        enable_thinking: bool = True,
        model_id: str = None,
    ):
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.enable_thinking = enable_thinking
        self.model_id = model_id or self.MODEL_ID
        self.model = None
        self.processor = None

    def load(self):
        """Load Gemma 4 E2B model. Uses pre-quantized 4-bit by default."""
        import transformers
        hf_version = tuple(int(x) for x in transformers.__version__.split(".")[:3])
        dtype_kwarg = "dtype" if hf_version >= (5, 13, 0) else "torch_dtype"

        load_kwargs = {
            "device_map": self.device,
            "attn_implementation": "sdpa",
            dtype_kwarg: torch.bfloat16,
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self

    def generate(
        self,
        messages: list,
        tools: Optional[list] = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> dict:
        """Generate with optional function calling."""
        text = self.processor.apply_chat_template(
            messages,
            tools=tools or [],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        return self.processor.parse_response(response)

    def measure_vram(self) -> dict:
        """Measure current VRAM usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return {"allocated_gb": allocated, "reserved_gb": reserved, "total_gb": total, "free_gb": total - reserved}
        return {"error": "CUDA not available"}


# Test
if __name__ == "__main__":
    env = Gemma4Env(load_in_4bit=True, enable_thinking=True)
    env.load()
    print(env.measure_vram())
