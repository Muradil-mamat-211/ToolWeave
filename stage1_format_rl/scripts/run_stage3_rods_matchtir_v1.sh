#!/usr/bin/env bash
# Training Branch launcher.  Verification is the default; formal training needs
# an explicit environment guard and was not invoked during implementation.
set -Eeuo pipefail

WORKSPACE="/root/autodl-tmp/rods-workspace"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
CONFIG_NAME="stage3_rods_matchtir_v1_training_branch"
CONFIG_PATH="$STAGE_ROOT/configs/$CONFIG_NAME.yaml"
MODEL_PATH="$STAGE_ROOT/artifacts/stage2_eval/merged/global_step_25"
TRAIN_PATH="$STAGE_ROOT/data/bfcl_stage3_train_all_400_shuffled_seed42.parquet"

verify() {
    local failed=0
    for path in "$CONFIG_PATH" "$MODEL_PATH/model.safetensors.index.json" "$TRAIN_PATH"; do
        if [[ -e "$path" ]]; then
            echo "[ok] $path"
        else
            echo "[missing] $path"
            failed=1
        fi
    done
    (( failed == 0 )) || return 1
    PYTHONPATH="$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl" \
      /root/miniconda3/envs/rods/bin/python -m pytest -q \
      "$STAGE_ROOT/tests/test_rods_matchtir_v1_matching.py" \
      "$STAGE_ROOT/tests/test_rods_matchtir_v1_advantage.py" \
      "$STAGE_ROOT/tests/test_rods_matchtir_v1_integration.py" \
      "$STAGE_ROOT/tests/test_rods_stage3_lifecycle.py" \
      "$STAGE_ROOT/tests/test_stage3_rods_matchtir_config.py"
}

run_full() {
    if [[ "${ALLOW_RODS_MATCHTIR_STAGE3_TRAINING:-0}" != "1" ]]; then
        echo "Stage 3 formal training is disabled. Set ALLOW_RODS_MATCHTIR_STAGE3_TRAINING=1 explicitly."
        return 2
    fi
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate rods
    export PYTHONPATH="$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    mkdir -p "$STAGE_ROOT/artifacts/stage3_queues"
    cd "$AWORLD/EnvTuning"
    exec python -m verl.trainer.main_ppo \
      --config-path="$STAGE_ROOT/configs" \
      --config-name="$CONFIG_NAME"
}

case "${1:---verify}" in
    --verify) verify ;;
    --full) run_full ;;
    *) echo "usage: $0 [--verify|--full]"; exit 2 ;;
esac
