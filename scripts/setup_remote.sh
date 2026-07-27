#!/bin/bash
# setup_remote.sh — Run on AutoDL server to set up PI0.5 training environment
# Usage: ssh -p 29167 root@connect.westd.seetacloud.com 'bash -s' < setup_remote.sh

set -e

echo "=== Installing system dependencies ==="
apt-get update -qq && apt-get install -y -qq ffmpeg libsm6 libxext6 git tmux

echo "=== Installing LeRobot with PI0.5 support ==="
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git" -q

echo "=== Hugging Face Login ==="
# IMPORTANT: Run this manually with your token first:
#   huggingface-cli login
# Then visit https://huggingface.co/google/paligemma-3b-pt-224 to accept the license
echo ">> Reminder: run 'huggingface-cli login' with your HF token"
echo ">> Reminder: accept PaliGemma license at https://huggingface.co/google/paligemma-3b-pt-224"

echo "=== Verifying installation ==="
python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
import torch
print('LeRobot + PI05 import OK')
print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')
"

echo "=== Done! Ready to run train.sh ==="
