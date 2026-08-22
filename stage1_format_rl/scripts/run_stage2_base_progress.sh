#!/usr/bin/env bash
# Stage 2 launcher: stage2_qwen3_4b_k16_base_progress_batch20_plain_env
#   bash run_stage2_base_progress.sh --verify    # is_augmented=false assertion (no GPU)
#   bash run_stage2_base_progress.sh --smoke     # 1-group smoke, no checkpoint
#   bash run_stage2_base_progress.sh --full      # formal Stage 2 (background tmux)
set -Eeuo pipefail

WORKSPACE="/root/autodl-tmp/rods-workspace"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
CONFIG_DIR="$STAGE_ROOT/configs"
CONFIG_NAME="stage2_qwen3_4b_k16_base_progress_batch20_plain_env"
OUTPUT_ROOT="$WORKSPACE/outputs/stage2_qwen3_4b_k16_base_progress_batch20_plain_env"
CKPT_DIR="$OUTPUT_ROOT/checkpoints"
LOG_DIR="$STAGE_ROOT/logs"
SMOKE_DIR="$STAGE_ROOT/artifacts/gpu_smoke/stage2_smoke"
TMP_ROOT="/tmp/r1g2"
MODEL_PATH="$STAGE_ROOT/artifacts/gate_vs_base/merged/global_step_25"
TRAIN_FILE="$STAGE_ROOT/data/bfcl_stage1_train_base_100_shuffled_seed42.parquet"
VAL_BASE100="$STAGE_ROOT/data/checkpoint_gate_eval/val_base_100.parquet"

setup_env() {
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate rods
    export PYTHONPATH="$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES=0,1
    export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 NUMEXPR_MAX_THREADS=48
    export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=48
    export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN NCCL_TIMEOUT=3600
    export PYTHONUNBUFFERED=1
    unset PYTORCH_CUDA_ALLOC_CONF
    rm -rf "$TMP_ROOT"; mkdir -p "$TMP_ROOT/ray" "$TMP_ROOT/triton"
    export TRITON_CACHE_DIR="$TMP_ROOT/triton"
    export RAY_TMPDIR="$TMP_ROOT/ray"
    export TMPDIR="$TMP_ROOT"
}

require_ok() {
    local label="$1" path="$2"
    [[ -e "$path" ]] && echo "  [ok] $label: $path" || { echo "  [FAIL] $label missing: $path"; return 1; }
}

verify_plain_env() {
    echo "== Stage 2 plain-environment assertion =="
    /root/miniconda3/envs/rods/bin/python "$STAGE_ROOT/scripts/verify_stage2_plain_env.py" || exit 1
    local fail=0
    require_ok "start model"  "$MODEL_PATH"   || fail=1
    require_ok "train data"   "$TRAIN_FILE"   || fail=1
    require_ok "val base100"  "$VAL_BASE100"  || fail=1
    require_ok "config"       "$CONFIG_DIR/$CONFIG_NAME.yaml" || fail=1
    require_ok "reward bfcl"  "$AWORLD/EnvTuning/env_tuning/bfcl_reward.py" || fail=1
    if (( fail )); then echo "== VERIFY FAILED =="; exit 1; fi
    echo "== VERIFY OK =="
}

run_smoke() {
    setup_env
    verify_plain_env
    echo "== Stage 2 smoke: 1 prompt group (K=16), no checkpoint =="
    rm -rf "$SMOKE_DIR"; mkdir -p "$SMOKE_DIR"
    ( cd "$AWORLD/EnvTuning" && \
      python -m verl.trainer.main_ppo \
        --config-path="$CONFIG_DIR" --config-name="$CONFIG_NAME" \
        data.train_batch_size=1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=1 \
        trainer.total_training_steps=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.resume_mode=disable \
        trainer.experiment_name=stage2_smoke \
        trainer.default_local_dir="$SMOKE_DIR/checkpoints" \
        trainer.rollout_data_dir=null \
        trainer.validation_data_dir="$SMOKE_DIR/rollouts" \
    ) 2>&1 | tee "$LOG_DIR/stage2_smoke.log"
    local status=${PIPESTATUS[0]}
    echo "== Stage 2 smoke exit=$status =="
    echo "  verify: 0/1 codes present; score==progress mean; -3/-2/-1 excluded;"
    echo "          group variance & advantage nonzero; backward/KL/grad_norm OK; no ckpt."
    [[ -d "$SMOKE_DIR/checkpoints" ]] && { echo "  [FAIL] smoke wrote checkpoints!"; return 1; }
    return "$status"
}

run_full() {
    setup_env
    verify_plain_env
    echo "== Stage 2 full (base progress, plain env, max 25 updates) =="
    mkdir -p "$CKPT_DIR" "$LOG_DIR"
    ( cd "$AWORLD/EnvTuning" && \
      python -m verl.trainer.main_ppo \
        --config-path="$CONFIG_DIR" --config-name="$CONFIG_NAME" ) \
      2>&1 | tee "$LOG_DIR/stage2_train.log"
    exit "${PIPESTATUS[0]}"
}

mode="${1:---verify}"
case "$mode" in
    --verify) verify_plain_env ;;
    --smoke)  run_smoke ;;
    --full)   run_full ;;
    *) echo "unknown: $mode"; exit 2 ;;
esac
