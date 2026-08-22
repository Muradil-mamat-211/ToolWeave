#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${ALLOW_STAGE1_CHECKPOINT_EVAL:-0}" != "1" ]]; then
    echo "Gate evaluation is disabled. Set ALLOW_STAGE1_CHECKPOINT_EVAL=1 explicitly."
    exit 2
fi
if [[ "${ALLOW_STAGE1_TRAINING:-0}" == "1" ]]; then
    echo "Refusing evaluation while the formal-training guard is enabled."
    exit 3
fi

MODEL_PATH="${1:?usage: eval_stage1_gate_dual_gpu.sh MODEL_PATH LABEL VAL_FILE}"
LABEL="${2:?usage: eval_stage1_gate_dual_gpu.sh MODEL_PATH LABEL VAL_FILE}"
VAL_FILE="${3:?usage: eval_stage1_gate_dual_gpu.sh MODEL_PATH LABEL VAL_FILE}"

[[ "$LABEL" =~ ^[a-zA-Z0-9_]+$ ]] || { echo "invalid LABEL"; exit 4; }
test -d "$MODEL_PATH"
test -f "$VAL_FILE"

WORKSPACE="/root/autodl-tmp/rods-workspace"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
EVAL_ROOT="$STAGE_ROOT/artifacts/checkpoint_gate_eval"
EVAL_DIR="$EVAL_ROOT/runs/$LABEL"
LOG_DIR="$STAGE_ROOT/logs/checkpoint_gate_eval"
LOG_FILE="$LOG_DIR/$LABEL.log"
TMP_ROOT="/tmp/r1g"

if [[ -e "$EVAL_DIR/SUCCESS" ]]; then
    echo "Evaluation already completed: $LABEL"
    exit 0
fi
if [[ -e "$EVAL_DIR" ]]; then
    echo "Refusing to overwrite existing incomplete evaluation: $EVAL_DIR"
    exit 5
fi
mkdir -p "$EVAL_DIR/rollouts" "$LOG_DIR"
rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT/ray" "$TMP_ROOT/triton"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rods

export PYTHONPATH="$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export NUMEXPR_MAX_THREADS=48
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=48
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
    --config-path="$STAGE_ROOT/configs"
    --config-name=stage1_qwen3_4b_k16_recovery_from_step25
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.model.enable_gradient_checkpointing=false"
    "actor_rollout_ref.actor.use_kl_loss=false"
    "actor_rollout_ref.actor.entropy_coeff=0.0"
    "actor_rollout_ref.rollout.n=1"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.70"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=false"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0"
    "data.val_files=$VAL_FILE"
    "data.val_batch_size=16"
    "trainer.n_gpus_per_node=2"
    "trainer.nnodes=1"
    "trainer.val_before_train=true"
    "trainer.val_only=true"
    "trainer.save_freq=-1"
    "trainer.resume_mode=disable"
    "trainer.default_hdfs_dir=null"
    "trainer.default_local_dir=$EVAL_DIR/no_checkpoints"
    "trainer.validation_data_dir=$EVAL_DIR/rollouts"
    "trainer.rollout_data_dir=null"
    "trainer.logger=[console]"
    "trainer.experiment_name=gate_eval_$LABEL"
    "ray_init.num_cpus=48"
)

printf '%q ' "${COMMAND[@]}" > "$EVAL_DIR/launch_command.txt"
printf '\n' >> "$EVAL_DIR/launch_command.txt"
printf '%s\n' "$MODEL_PATH" > "$EVAL_DIR/model_path.txt"
printf '%s\n' "$VAL_FILE" > "$EVAL_DIR/validation_file.txt"

cd "$AWORLD/EnvTuning"
set +e
"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$EVAL_DIR/exit_code.txt"
if [[ "$STATUS" -ne 0 ]]; then
    exit "$STATUS"
fi

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
    echo "Read-only gate evaluation wrote forbidden checkpoint artifacts"
    cat "$EVAL_DIR/forbidden_artifacts.txt"
    exit 90
fi

touch "$EVAL_DIR/SUCCESS"
