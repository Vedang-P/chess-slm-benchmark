"""Quick test: load Gemma 4 E2B 4-bit and run inference."""
import torch, time, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Free: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
print(f"Used before load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

MODEL_PATH = "/home/vedang/Desktop/Research/neuro-symbolic-pathfinding/data/models/gemma-4-E2B-4bit"

from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Tokenizer loaded in {time.time()-t0:.1f}s")

from accelerate import infer_auto_device_map
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

t0 = time.time()
print("Loading model (4-bit, cpu offload)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    offload_folder="/tmp/offload",
    trust_remote_code=True,
)
print(f"Model loaded in {time.time()-t0:.1f}s")

mem = torch.cuda.memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM used after load: {mem:.2f} GB / {total:.2f} GB")

prompt = "What is 2+2? Answer briefly."
t0 = time.time()
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=20)
response = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Inference in {time.time()-t0:.1f}s")
print(f"Response: {response}")

del model, inputs, out
torch.cuda.empty_cache()
print(f"VRAM after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print("SUCCESS!")
