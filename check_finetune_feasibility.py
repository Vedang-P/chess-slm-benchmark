"""Feasibility check: can Gemma 4 E2B be loaded and LoRA-attached for
fine-tuning on available VRAM? Run this before building the full GRPO
pipeline -- if this fails, we need to know now, not after building
everything on top of it.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"\nLoading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading model in bf16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto"
)
print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\nAttempting LoRA attachment via peft...")
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
print(f"VRAM after LoRA attach: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\nAttempting a forward + backward pass...")
prompt = "Navigate from (0,0) to (2,2) on a 5x5 grid. Move up/down/left/right."
inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)
labels = inputs["input_ids"].clone()
outputs = peft_model(**inputs, labels=labels)
outputs.loss.backward()
print(f"Loss: {outputs.loss.item():.4f}")
print(f"VRAM after backward pass: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("\nSUCCESS -- LoRA fine-tuning is feasible on this hardware.")
