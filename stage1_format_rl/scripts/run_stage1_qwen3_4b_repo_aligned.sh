#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${ALLOW_STAGE1_TRAINING:-0}" != "1" ]]; then
    echo "Stage 1 training is disabled. Set ALLOW_STAGE1_TRAINING=1 only after explicit user approval."
    exit 2
fi

WORKSPACE="/root/autodl-tmp/rods-workspace"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
CONFIG_DIR="$STAGE_ROOT/configs"
LOG_FILE="$STAGE_ROOT/logs/stage1_qwen3_4b_k16_repo_aligned.log"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rods
export PYTHONPATH="$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="/root/autodl-tmp/triton_cache_stage1"
export NCCL_IB_TIMEOUT=22
export NCCL_TIMEOUT=9999999999
mkdir -p "$(dirname "$LOG_FILE")" "$TRITON_CACHE_DIR"
cd "$AWORLD/EnvTuning"

python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name=stage1_qwen3_4b_k16_repo_aligned \
    2>&1 | tee "$LOG_FILE"

