import sys, os
sys.path.insert(0, "/home/vedang/Desktop/Research/neuro-symbolic-pathfinding/venv/lib/python3.12/site-packages")

from huggingface_hub import snapshot_download, login
import logging
logging.basicConfig(level=logging.INFO)

MODEL_DIR = "/home/vedang/Desktop/Research/neuro-symbolic-pathfinding/data/models/gemma-4-E2B-4bit"
TOKEN = "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"

login(token=TOKEN)

# Download with progress
result = snapshot_download(
    repo_id="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
    force_download=False,
    token=TOKEN
)
print(f"Download complete! Files at: {result}")

# List files
for f in os.listdir(MODEL_DIR):
    fp = os.path.join(MODEL_DIR, f)
    size = os.path.getsize(fp)
    print(f"  {f}: {size/1e9:.2f} GB")
