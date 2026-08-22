#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${ALLOW_STAGE1_GPU_SMOKE:-0}" != "1" ]]; then
    echo "GPU smoke is disabled. Set ALLOW_STAGE1_GPU_SMOKE=1 only for an approved smoke phase."
    exit 2
fi

if [[ "${ALLOW_STAGE1_TRAINING:-0}" == "1" ]]; then
    echo "Refusing to run: formal training guard must remain disabled during smoke."
    exit 3
fi

PHASE="${1:?usage: run_gpu_smoke_phase.sh smoke1|smoke2|smoke3|smoke4}"
case "$PHASE" in
    smoke1|smoke2|smoke3|smoke4) ;;
    *) echo "Unknown phase: $PHASE"; exit 4 ;;
esac

SMOKE2_CASE="${2:-}"
if [[ "$PHASE" == "smoke2" && -n "$SMOKE2_CASE" ]]; then
    case "$SMOKE2_CASE" in
        0|1|2|3) ;;
        *) echo "Smoke 2 case must be one of 0, 1, 2, or 3"; exit 4 ;;
    esac
fi
RUN_ID="$PHASE"
if [[ "$PHASE" == "smoke2" && -n "$SMOKE2_CASE" ]]; then
    RUN_ID="smoke2_case_${SMOKE2_CASE}"
fi
if [[ -n "${SMOKE_RUN_SUFFIX:-}" ]]; then
    if [[ ! "$SMOKE_RUN_SUFFIX" =~ ^[a-zA-Z0-9_]+$ ]]; then
        echo "SMOKE_RUN_SUFFIX may contain only letters, numbers, and underscores"
        exit 4
    fi
    RUN_ID="${RUN_ID}_${SMOKE_RUN_SUFFIX}"
fi

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
CONFIG_DIR="$STAGE_ROOT/configs"
DATA_DIR="$TOOLWEAVE_DATA_ROOT/smoke"
LOG_DIR="$TOOLWEAVE_LOGS_ROOT/gpu_smoke"
ARTIFACT_DIR="$TOOLWEAVE_ARTIFACTS_ROOT/gpu_smoke"
PHASE_ARTIFACT_DIR="$ARTIFACT_DIR/$RUN_ID"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"
GPU_CSV="$LOG_DIR/${RUN_ID}_gpu.csv"
CPU_CSV="$LOG_DIR/${RUN_ID}_cpu.csv"
# Ray's plasma-store UNIX socket must fit within the 107-byte AF_UNIX limit.
TMP_ROOT="$TOOLWEAVE_SHORT_TEMP_ROOT/stage1-smoke/$RUN_ID"

mkdir -p "$LOG_DIR" "$PHASE_ARTIFACT_DIR" "$TMP_ROOT"
toolweave_safe_rm_rf "$TMP_ROOT/ray" "$TMP_ROOT/triton"
mkdir -p "$TMP_ROOT/ray" "$TMP_ROOT/triton"

toolweave_activate_conda

export PYTHONPATH="$STAGE_ROOT/smoke_instrumentation:$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
toolweave_apply_topology learner
export TOKENIZERS_PARALLELISM=true
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
# SGLang's TorchMemorySaver rejects expandable_segments. Keep the default
# allocator so model-weight offload and resume remain available.
unset PYTORCH_CUDA_ALLOC_CONF
export TRITON_CACHE_DIR="$TMP_ROOT/triton"
export RAY_TMPDIR="$TMP_ROOT/ray"
export TMPDIR="$TMP_ROOT"

scan_forbidden() {
    find "$STAGE_ROOT" -type f \( \
        -name 'optimizer.pt' -o \
        -name 'model.safetensors' -o \
        -name 'pytorch_model.bin' -o \
        -iname '*trainer_state*' \
    \) -print
    find "$STAGE_ROOT" -type d \( \
        -name 'global_step_*' -o \
        -name 'checkpoint-*' -o \
        -name actor -o \
        -name final_model \
    \) -print
}

BEFORE_FORBIDDEN="$(scan_forbidden)"
if [[ -n "$BEFORE_FORBIDDEN" ]]; then
    echo "Forbidden checkpoint artifacts already exist under Stage 1 smoke root:"
    echo "$BEFORE_FORBIDDEN"
    exit 5
fi

monitor_gpu() {
    echo 'timestamp,index,memory.used_mib,memory.total_mib,utilization.gpu_pct,utilization.memory_pct,power.draw_w,temperature_c' > "$GPU_CSV"
    while true; do
        local now
        now="$(date -Ins)"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
            --format=csv,noheader,nounits | sed "s/^/$now,/" >> "$GPU_CSV"
        sleep 1
    done
}

monitor_cpu() {
    echo 'timestamp,load1,load5,load15,mem_available_kib' > "$CPU_CSV"
    while true; do
        local now load1 load5 load15 mem_available
        now="$(date -Ins)"
        read -r load1 load5 load15 _ < /proc/loadavg
        mem_available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
        echo "$now,$load1,$load5,$load15,$mem_available" >> "$CPU_CSV"
        sleep 1
    done
}

monitor_gpu &
GPU_MONITOR_PID=$!
monitor_cpu &
CPU_MONITOR_PID=$!

cleanup() {
    kill "$GPU_MONITOR_PID" "$CPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" "$CPU_MONITOR_PID" 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

COMMON_OVERRIDES=(
    "trainer.save_freq=-1"
    "trainer.resume_mode=disable"
    "trainer.default_hdfs_dir=null"
    "trainer.default_local_dir=$PHASE_ARTIFACT_DIR/no_checkpoints"
    "trainer.experiment_name=qwen3_4b_${RUN_ID}_no_save"
    "actor_rollout_ref.actor.use_kl_loss=true"
    "actor_rollout_ref.actor.kl_loss_coef=0.01"
    "actor_rollout_ref.actor.entropy_coeff=0.001"
    "actor_rollout_ref.actor.clip_ratio_low=0.2"
    "actor_rollout_ref.actor.clip_ratio_high=0.28"
    "actor_rollout_ref.actor.clip_ratio_c=10.0"
    "algorithm.use_kl_in_reward=false"
    "algorithm.kl_ctrl.kl_coef=0.0"
    "trainer.n_gpus_per_node=$TOOLWEAVE_LEARNER_GPUS_PER_NODE"
    "trainer.nnodes=$TOOLWEAVE_NNODES"
)
if [[ -n "${SMOKE_MAX_RESPONSE_LENGTH:-}" ]]; then
    COMMON_OVERRIDES+=("data.max_response_length=$SMOKE_MAX_RESPONSE_LENGTH")
fi

PHASE_OVERRIDES=()
case "$PHASE" in
    smoke1)
        PHASE_OVERRIDES=(
            "data.val_files=$DATA_DIR/smoke1_functional.parquet"
            "data.val_batch_size=1"
            "actor_rollout_ref.rollout.n=2"
            "actor_rollout_ref.rollout.val_kwargs.n=2"
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.75"
            "trainer.val_before_train=true"
            "trainer.val_only=true"
            "trainer.validation_data_dir=$PHASE_ARTIFACT_DIR/rollouts"
            "trainer.rollout_data_dir=null"
        )
        ;;
    smoke2)
        SMOKE2_VAL_FILE="$DATA_DIR/smoke2_extremes_4.parquet"
        if [[ -n "$SMOKE2_CASE" ]]; then
            SMOKE2_VAL_FILE="$DATA_DIR/smoke2_extreme_${SMOKE2_CASE}.parquet"
        fi
        PHASE_OVERRIDES=(
            "data.val_files=$SMOKE2_VAL_FILE"
            "data.val_batch_size=1"
            "actor_rollout_ref.rollout.n=16"
            "actor_rollout_ref.rollout.val_kwargs.n=16"
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.75"
            "trainer.val_before_train=true"
            "trainer.val_only=true"
            "trainer.validation_data_dir=$PHASE_ARTIFACT_DIR/rollouts"
            "trainer.rollout_data_dir=null"
        )
        ;;
    smoke3)
        PHASE_OVERRIDES=(
            "data.val_files=$DATA_DIR/smoke3_stress_4.parquet"
            "data.val_batch_size=4"
            "actor_rollout_ref.rollout.n=16"
            "actor_rollout_ref.rollout.val_kwargs.n=16"
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.75"
            "trainer.val_before_train=true"
            "trainer.val_only=true"
            "trainer.validation_data_dir=$PHASE_ARTIFACT_DIR/rollouts"
            "trainer.rollout_data_dir=null"
        )
        ;;
    smoke4)
        export SMOKE_MAX_OPTIMIZER_STEPS_PER_PROCESS=1
        export SMOKE_OPTIMIZER_AUDIT_DIR="$PHASE_ARTIFACT_DIR/optimizer_step_audit"
        mkdir -p "$SMOKE_OPTIMIZER_AUDIT_DIR"
        PHASE_OVERRIDES=(
            "data.train_files=$DATA_DIR/smoke4_train_4.parquet"
            "data.train_batch_size=4"
            "actor_rollout_ref.rollout.n=16"
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.75"
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768"
            "trainer.val_before_train=false"
            "trainer.val_only=false"
            "trainer.total_epochs=1"
            "trainer.total_training_steps=1"
            "trainer.test_freq=-1"
            "trainer.validation_data_dir=null"
            "trainer.rollout_data_dir=$PHASE_ARTIFACT_DIR/rollouts"
        )
        ;;
esac

printf '%q ' python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name=stage1_gpu_smoke_no_save \
    "${COMMON_OVERRIDES[@]}" "${PHASE_OVERRIDES[@]}" > "$PHASE_ARTIFACT_DIR/launch_command.txt"
printf '\n' >> "$PHASE_ARTIFACT_DIR/launch_command.txt"

cd "$AWORLD/EnvTuning"
set +e
python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name=stage1_gpu_smoke_no_save \
    "${COMMON_OVERRIDES[@]}" \
    "${PHASE_OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

cleanup
trap - EXIT INT TERM

AFTER_FORBIDDEN="$(scan_forbidden)"
if [[ -n "$AFTER_FORBIDDEN" ]]; then
    echo "Forbidden checkpoint artifacts appeared during $PHASE:"
    echo "$AFTER_FORBIDDEN"
    exit 90
fi

echo "$STATUS" > "$PHASE_ARTIFACT_DIR/exit_code.txt"
exit "$STATUS"
