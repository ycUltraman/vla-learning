#!/bin/bash
# download_checkpoint.sh — Download fine-tuned checkpoint from AutoDL to local
# Usage: bash scripts/download_checkpoint.sh

SERVER="root@connect.westd.seetacloud.com"
PORT="29167"
CHECKPOINT_DIR="outputs/pi05_sim_teleop/checkpoints/10000"
LOCAL_DIR="./checkpoints/pi05_sim_teleop"

echo "=== Downloading checkpoint from ${SERVER} ==="
mkdir -p "${LOCAL_DIR}"

scp -P ${PORT} -r "${SERVER}:~/${CHECKPOINT_DIR}/." "${LOCAL_DIR}/"

echo "=== Verifying downloaded checkpoint ==="
python -c "
from pathlib import Path
import os
p = Path('${LOCAL_DIR}')
files = list(p.rglob('*'))
print(f'Downloaded {len(files)} files to {p.resolve()}')
safetensors = list(p.rglob('*.safetensors'))
bins = list(p.rglob('*.bin'))
if safetensors: print(f'safetensors: {[f.name for f in safetensors]}')
if bins: print(f'bin files: {[f.name for f in bins]}')
configs = list(p.rglob('config*.json'))
if configs: print(f'config: {[f.name for f in configs]}')
"

echo "=== Done! Checkpoint at ${LOCAL_DIR} ==="
