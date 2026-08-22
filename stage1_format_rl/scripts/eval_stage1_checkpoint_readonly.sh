#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${ALLOW_STAGE1_CHECKPOINT_EVAL:-0}" != "1" ]]; then
    echo "Checkpoint evaluation is disabled. Set ALLOW_STAGE1_CHECKPOINT_EVAL=1 explicitly."
    exit 2
fi
if [[ "${ALLOW_STAGE1_TRAINING:-0}" == "1" ]]; then
    echo "Refusing evaluation while the formal-training guard is enabled."
    exit 3
fi

MODEL_PATH="${1:?usage: eval_stage1_checkpoint_readonly.sh MODEL_PATH LABEL}"
LABEL="${2:?usage: eval_stage1_checkpoint_readonly.sh MODEL_PATH LABEL}"
if [[ ! "$LABEL" =~ ^[a-zA-Z0-9_]+$ ]]; then
    echo "LABEL may contain only letters, numbers, and underscores"
    exit 4
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model directory does not exist: $MODEL_PATH"
    exit 5
fi

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
CONFIG_DIR="$STAGE_ROOT/configs"
VAL_FILE="${STAGE1_CHECKPOINT_EVAL_VAL_FILE:-$TOOLWEAVE_DATA_ROOT/checkpoint_eval/heldout_base_20.parquet}"
EVAL_DIR="$TOOLWEAVE_ARTIFACTS_ROOT/checkpoint_eval/$LABEL"
LOG_DIR="$TOOLWEAVE_LOGS_ROOT/checkpoint_eval"
LOG_FILE="$LOG_DIR/$LABEL.log"
TMP_ROOT="$TOOLWEAVE_SHORT_TEMP_ROOT/stage1-checkpoint-eval/$LABEL"

test -f "$VAL_FILE"
mkdir -p "$EVAL_DIR/rollouts" "$LOG_DIR" "$TMP_ROOT/ray" "$TMP_ROOT/triton"
toolweave_safe_rm_rf "$TMP_ROOT/ray" "$TMP_ROOT/triton"
mkdir -p "$TMP_ROOT/ray" "$TMP_ROOT/triton"

toolweave_activate_conda
toolweave_apply_topology learner

export PYTHONPATH="$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=true
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
unset PYTORCH_CUDA_ALLOC_CONF
export TRITON_CACHE_DIR="$TMP_ROOT/triton"
export RAY_TMPDIR="$TMP_ROOT/ray"
export TMPDIR="$TMP_ROOT"

cleanup() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

COMMAND=(
    python -m verl.trainer.main_ppo
    --config-path="$CONFIG_DIR"
    --config-name=stage1_qwen3_4b_k16_formal_5epoch
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "data.val_files=$VAL_FILE"
    "data.val_batch_size=4"
    "actor_rollout_ref.rollout.n=1"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=false"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.70"
    "trainer.val_before_train=true"
    "trainer.val_only=true"
    "trainer.save_freq=-1"
    "trainer.resume_mode=disable"
    "trainer.default_hdfs_dir=null"
    "trainer.default_local_dir=$EVAL_DIR/no_checkpoints"
    "trainer.validation_data_dir=$EVAL_DIR/rollouts"
    "trainer.rollout_data_dir=null"
    "trainer.logger=[console]"
    "trainer.experiment_name=checkpoint_eval_$LABEL"
)

printf '%q ' "${COMMAND[@]}" > "$EVAL_DIR/launch_command.txt"
printf '\n' >> "$EVAL_DIR/launch_command.txt"
printf '%s\n' "$MODEL_PATH" > "$EVAL_DIR/model_path.txt"

cd "$AWORLD/EnvTuning"
set +e
"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

cleanup
trap - EXIT INT TERM

find "$EVAL_DIR" -mindepth 1 -type f \( \
    -name 'model.safetensors' -o \
    -name 'pytorch_model.bin' -o \
    -name 'optimizer.pt' -o \
    -iname '*trainer_state*' \
\) -print > "$EVAL_DIR/forbidden_artifacts.txt"
find "$EVAL_DIR" -mindepth 1 -type d \( \
    -name 'global_step_*' -o \
    -name 'checkpoint-*' -o \
    -name 'final_model' \
\) -print >> "$EVAL_DIR/forbidden_artifacts.txt"
if [[ -s "$EVAL_DIR/forbidden_artifacts.txt" ]]; then
    echo "Read-only evaluation unexpectedly wrote checkpoint artifacts:"
    cat "$EVAL_DIR/forbidden_artifacts.txt"
    exit 90
fi

printf '%s\n' "$STATUS" > "$EVAL_DIR/exit_code.txt"
exit "$STATUS"
