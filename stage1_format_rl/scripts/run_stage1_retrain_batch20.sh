#!/usr/bin/env bash
# RODS Stage 1 retrain (batch 20, adaptive KL) launcher.
#
#   bash run_stage1_retrain_batch20.sh --preflight   # 环境/路径/磁盘检查，不碰 GPU
#   bash run_stage1_retrain_batch20.sh --smoke       # 2 步 GPU smoke（显存+稳定性）
#   bash run_stage1_retrain_batch20.sh --tmux        # 在 tmux 会话里启动全量训练（可 Ctrl-C 后同命令续跑）
#   bash run_stage1_retrain_batch20.sh --full        # 前台全量训练（等价的默认动作）
#
# Resume: Ctrl-C 后重跑同一命令（--full 或 --tmux），veRL resume_mode=auto 会
# 从 default_local_dir 里最新的、经完整性校验的 checkpoint 无缝续跑。
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
CONFIG_DIR="$STAGE_ROOT/configs"
CONFIG_NAME="stage1_qwen3_4b_k16_retrain_batch20"
CKPT_DIR="$TOOLWEAVE_OUTPUTS_ROOT/stage1_qwen3_4b_k16_retrain_batch20/checkpoints"
SMOKE_DIR="$TOOLWEAVE_ARTIFACTS_ROOT/gpu_smoke/retrain_smoke"
BASE_MODEL="$TOOLWEAVE_MODELS_ROOT/Qwen3-4B"
TRAIN_FILE="$TOOLWEAVE_DATA_ROOT/bfcl_stage1_train_base_100_shuffled_seed42.parquet"
VAL_FILE="$TOOLWEAVE_DATA_ROOT/val_100_stratified_seed42.parquet"
TMP_ROOT="$TOOLWEAVE_SHORT_TEMP_ROOT/stage1-retrain"
SESSION="rods-stage1-retrain"

require_ok() {
    local label="$1" path="$2"
    if [[ -e "$path" ]]; then echo "  [ok] $label: $path"; else echo "  [FAIL] $label missing: $path"; return 1; fi
}

setup_env() {
    toolweave_activate_conda
    export PYTHONPATH="$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
    toolweave_apply_topology learner
    export TOKENIZERS_PARALLELISM=true
    export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN NCCL_TIMEOUT=3600
    export PYTHONUNBUFFERED=1
    # MUST remain unset: expandable_segments breaks TorchMemorySaver, which the
    # hybrid engine needs to swap weights between SGLang and FSDP on one GPU.
    unset PYTORCH_CUDA_ALLOC_CONF
    toolweave_safe_rm_rf "$TMP_ROOT" 2>/dev/null
    mkdir -p "$TMP_ROOT/ray" "$TMP_ROOT/triton"
    export TRITON_CACHE_DIR="$TMP_ROOT/triton"
    export RAY_TMPDIR="$TMP_ROOT/ray"
    export TMPDIR="$TMP_ROOT"
}

checkpoint_complete() {
    local dir="$1"
    local world_size="${TOOLWEAVE_LEARNER_WORLD_SIZE:?topology plan was not applied}"
    [[ -s "$dir/data.pt" ]] || return 1
    for ((rank=0; rank<world_size; rank++)); do
        [[ -s "$dir/actor/model_world_size_${world_size}_rank_${rank}.pt" ]] || return 1
        [[ -s "$dir/actor/optim_world_size_${world_size}_rank_${rank}.pt" ]] || return 1
        [[ -s "$dir/actor/extra_state_world_size_${world_size}_rank_${rank}.pt" ]] || return 1
    done
    return 0
}

report_resume() {
    local latest="$CKPT_DIR/latest_checkpointed_iteration.txt"
    if [[ -f "$latest" ]]; then
        local step
        step="$(tr -dc '0-9' < "$latest")"
        echo "Resume target: global_step_$step ($latest)"
        if checkpoint_complete "$CKPT_DIR/global_step_$step"; then
            echo "  checkpoint complete -> 同命令重跑将从 step $step 无缝续训。"
        else
            echo "  WARNING: checkpoint global_step_$step INCOMPLETE（可能被中断写坏）。"
            echo "  → 若这是全新训练，请先清空 $CKPT_DIR 再启动；"
            echo "  → 若确定要续，请先手工核验后再运行 --full。"
            return 1
        fi
    else
        echo "No previous checkpoint -> 全新训练，从 Qwen3-4B (step 0) 开始。"
    fi
    return 0
}

preflight() {
    echo "== Preflight (no GPU) =="
    toolweave_activate_conda
    toolweave_apply_topology learner
    local fail=0
    require_ok "base model"   "$BASE_MODEL"            || fail=1
    require_ok "config"       "$CONFIG_DIR/$CONFIG_NAME.yaml" || fail=1
    require_ok "train data"   "$TRAIN_FILE"            || fail=1
    require_ok "val 100"      "$VAL_FILE"              || fail=1
    require_ok "reward"       "$AWORLD/EnvTuning/env_tuning/format_reward.py" || fail=1
    require_ok "interaction"  "$AWORLD/EnvTuning/env_tuning/config/multi_turn_fc_interaction_config.yaml" || fail=1
    local avail
    avail="$(df -P "$TOOLWEAVE_ASSET_ROOT" | tail -1 | awk '{print $4}')"
    echo "  disk free: ${avail} KB (建议 ≥ 60GB)"
    if (( avail < 60*1024*1024 )); then echo "  [FAIL] disk low"; fail=1; fi
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null \
        | sed 's/^/  gpu /' || { echo "  [FAIL] nvidia-smi unavailable"; fail=1; }
    if [[ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]]; then
        report_resume || fail=1
    else
        echo "  resume: none (fresh)"
    fi
    if (( fail )); then echo "== PREFLIGHT FAILED =="; exit 1; fi
    echo "== PREFLIGHT OK =="
}

run_smoke() {
    setup_env
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        echo "Full training tmux '$SESSION' is alive; abort smoke to avoid GPU contention."; exit 3
    fi
    echo "== Smoke: 2 steps, batch 4, no save, no val, fresh dir =="
    mkdir -p "$SMOKE_DIR/rollouts"
    ( cd "$AWORLD/EnvTuning" && \
      python -m verl.trainer.main_ppo \
        --config-path="$CONFIG_DIR" \
        --config-name="$CONFIG_NAME" \
        data.train_batch_size=4 \
        actor_rollout_ref.actor.ppo_mini_batch_size=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        trainer.total_training_steps=2 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.resume_mode=disable \
        trainer.experiment_name=stage1_qwen3_4b_k16_retrain_smoke \
        trainer.default_local_dir="$SMOKE_DIR/checkpoints" \
        trainer.rollout_data_dir=null \
        trainer.validation_data_dir="$SMOKE_DIR/rollouts" \
    ) 2>&1 | tee "$TOOLWEAVE_LOGS_ROOT/retrain_smoke.log"
    local status=${PIPESTATUS[0]}
    echo "== Smoke exit=$status =="
    echo "  检查: 1) 无 OOM/Traceback  2) critic/score 有限  3) response_length/clip_ratio 低"
    return "$status"
}

run_full() {
    setup_env
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        echo "Refusing: tmux session '$SESSION' already running (防止重复启动)."
        exit 3
    fi
    echo "== Full training (batch 20, adaptive KL) =="
    report_resume
    mkdir -p "$CKPT_DIR"
    ( cd "$AWORLD/EnvTuning" && \
      python -m verl.trainer.main_ppo \
        --config-path="$CONFIG_DIR" \
        --config-name="$CONFIG_NAME" \
    ) 2>&1 | tee "$TOOLWEAVE_LOGS_ROOT/retrain_train.log"
    exit "${PIPESTATUS[0]}"
}

mode="${1:---full}"
case "$mode" in
    --preflight) preflight ;;
    --smoke)     run_smoke ;;
    --tmux)
        if tmux has-session -t "=$SESSION" 2>/dev/null; then
            echo "Already running in tmux '$SESSION'."; exit 3
        fi
        tmux new-session -d -s "$SESSION" "bash '$0' --full"
        echo "Started full training in tmux session '$SESSION'. Attach: tmux attach -t '$SESSION'"
        ;;
    --full)      run_full ;;
    *) echo "unknown mode: $mode"; exit 2 ;;
esac
