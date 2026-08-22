#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ACTOR="${1:?usage: merge_stage1_gate_checkpoint.sh SOURCE_ACTOR TARGET_DIR}"
TARGET_DIR="${2:?usage: merge_stage1_gate_checkpoint.sh SOURCE_ACTOR TARGET_DIR}"
test -d "$SOURCE_ACTOR"

if [[ -s "$TARGET_DIR/model.safetensors.index.json" && -s "$TARGET_DIR/config.json" ]]; then
    echo "Merged model already exists: $TARGET_DIR"
    exit 0
fi
if [[ -e "$TARGET_DIR" ]]; then
    echo "Refusing to overwrite partial merge target: $TARGET_DIR"
    exit 2
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rods
export OMP_NUM_THREADS=8
export PYTHONPATH="/root/autodl-tmp/rods-workspace/code/AWorld-RL-stage1-worktree/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$SOURCE_ACTOR" \
    --target_dir "$TARGET_DIR" \
    --use_cpu_initialization

test -s "$TARGET_DIR/config.json"
test -s "$TARGET_DIR/model.safetensors.index.json"
find "$TARGET_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' -size +0c | grep -q .
