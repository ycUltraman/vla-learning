#!/bin/bash
# train.sh — Launch PI0.5 LoRA fine-tuning on the remote server
# Run on AutoDL server inside tmux:
#   tmux new -s pi05_train
#   bash train.sh

set -e

echo "=== Verifying dataset access ==="
python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('zeno-labs/sim-teleop')
print(f'Dataset: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames')
print(f'State shape: {ds[0][\"observation.state\"].shape}')
print(f'Action shape: {ds[0][\"action\"].shape}')
print(f'Front image: {ds[0][\"observation.images.front\"].shape}')
print(f'Wrist image: {ds[0][\"observation.images.wrist\"].shape}')
"

echo "=== Starting PI0.5 LoRA fine-tuning ==="
lerobot-train \
    --dataset.repo_id=zeno-labs/sim-teleop \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.use_lora=true \
    --policy.lora_rank=16 \
    --output_dir=./outputs/pi05_sim_teleop \
    --job_name=pi05_lora_v1 \
    --steps=10000 \
    --batch_size=2 \
    --save_freq=2000 \
    --log_freq=100 \
    --wandb.enable=false

echo "=== Training complete! Checkpoint saved to ./outputs/pi05_sim_teleop/ ==="
