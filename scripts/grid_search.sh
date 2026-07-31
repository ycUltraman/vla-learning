#!/bin/bash
# Grid search: BC loss × res_scale
# Usage: bash grid_search.sh

CHECKPOINT="/root/autodl-tmp/output_my_data/checkpoints/020000/pretrained_model"
EPISODES=100
LOG_DIR="/root/autodl-tmp/grid_logs"
VENV="/root/autodl-tmp/venv_lerobot/bin/activate"

mkdir -p "$LOG_DIR"

for BC in 2 5; do
  for RS in 0.01 0.02; do
    NAME="bc${BC}_rs${RS}"
    LOG="$LOG_DIR/${NAME}.log"
    echo "=============================================="
    echo "Starting: BC_loss=$BC  res_scale=$RS"
    echo "Log: $LOG"
    echo "=============================================="

    source "$VENV"
    export HF_HOME=/root/autodl-tmp/.hf_cache HF_HUB_OFFLINE=1
    MUJOCO_GL=egl python -u /root/autodl-tmp/train_ppo_pi05.py \
      --checkpoint "$CHECKPOINT" \
      --episodes "$EPISODES" \
      --bc_loss "$BC" \
      --res_scale "$RS" \
      --save "/root/autodl-tmp/grid_logs/${NAME}_checkpoint.pt" \
      2>&1 | tee "$LOG"

    echo "Finished: $NAME"
    echo ""
  done
done

echo "All grid searches done."
echo "Logs in: $LOG_DIR"
echo ""
echo "Summary:"
for BC in 2 5; do
  for RS in 0.01 0.02; do
    NAME="bc${BC}_rs${RS}"
    LOG="$LOG_DIR/${NAME}.log"
    SUCCESS=$(grep -c "success: True" "$LOG" 2>/dev/null || echo 0)
    echo "  $NAME: $SUCCESS/$EPISODES success"
  done
done
