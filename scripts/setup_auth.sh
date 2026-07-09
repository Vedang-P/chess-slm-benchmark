#!/bin/bash
set -euo pipefail

# Authenticate with HuggingFace and W&B
source /home/vedang/Desktop/Research/neuro-symbolic-pathfinding/venv/bin/activate

echo "=== HuggingFace Auth ==="
huggingface-cli login --token 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 --add-to-git-credential

echo "=== W&B Auth ==="
wandb login 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

echo "=== Verify HF ==="
python -c "from huggingface_hub import whoami; print(f'Logged in as: {whoami()[\"name\"]}')"

echo "=== Verify W&B ==="
python -c "import wandb; wandb.init(project='neuro-symbolic-pathfinding', mode='disabled'); print('W&B OK')"

echo "=== Verify CUDA ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "=== All auth complete ==="
