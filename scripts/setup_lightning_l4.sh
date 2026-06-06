#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade uv
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -e ".[wandb,cloud]"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Switch the Lightning Studio to an NVIDIA L4.")
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory / 2**30
print("GPU:", name)
print(f"VRAM: {memory:.1f} GiB")
if "L4" not in name.upper():
    raise SystemExit(f"Expected NVIDIA L4, detected {name}")
if memory < 22:
    raise SystemExit(f"Expected approximately 24 GiB VRAM, detected {memory:.1f} GiB")
PY

echo "Environment ready. Run: source .venv/bin/activate"
