#!/usr/bin/env bash
# Stage-2 gate eval (read-only): val_only rollout of one HF model on one val file,
# using the EXACT Stage-2 training config (stage2_qwen3_4b_k16_base_progress_batch20_plain_env):
#   - Stage-2 fixed-denominator progress reward wrapper (rods_stage2_progress_reward.py)
#   - standard non-augmented env (is_augmented=false, func_source_code_wo_aug) via
#     multi_turn_fc_interaction_config_stage2.yaml
#   - deterministic decoding: val_kwargs n=1, do_sample=false, temperature=0
#   - max_response_length=10000 (from Stage-2 data config)
# Produces rollouts/<shard>.jsonl; writes SUCCESS marker when done. Read-only w.r.t.
# checkpoints (save_freq=-1, val_only=true, resume disabled).
set -Eeuo pipefail

if [[ "${ALLOW_STAGE2_GATE_EVAL:-0}" != "1" ]]; then
    echo "Stage-2 gate evaluation is disabled. Set ALLOW_STAGE2_GATE_EVAL=1 explicitly."
    exit 2
fi
if [[ "${ALLOW_STAGE1_TRAINING:-0}" == "1" ]]; then
    echo "Refusing evaluation while the formal-training guard is enabled."
    exit 3
fi

MODEL_PATH="${1:?usage: eval_stage2_gate_dual_gpu.sh MODEL_PATH LABEL VAL_FILE}"
LABEL="${2:?usage: eval_stage2_gate_dual_gpu.sh MODEL_PATH LABEL VAL_FILE}"
VAL_FILE="${3:?usage: eval_stage2_gate_dual_gpu.sh MODEL_PATH LABEL VAL_FILE}"

[[ "$LABEL" =~ ^[a-zA-Z0-9_]+$ ]] || { echo "invalid LABEL"; exit 4; }
test -d "$MODEL_PATH"
test -f "$VAL_FILE"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
EVAL_ROOT="$TOOLWEAVE_ARTIFACTS_ROOT/stage2_eval"
EVAL_DIR="$EVAL_ROOT/runs/$LABEL"
LOG_DIR="$TOOLWEAVE_LOGS_ROOT/stage2_eval"
LOG_FILE="$LOG_DIR/$LABEL.log"
TMP_ROOT="$TOOLWEAVE_SHORT_TEMP_ROOT/stage2-gate-dual/$LABEL"

if [[ -e "$EVAL_DIR/SUCCESS" ]]; then
    echo "Evaluation already completed: $LABEL"
    exit 0
fi
if [[ -e "$EVAL_DIR" ]]; then
    echo "Refusing to overwrite existing incomplete evaluation: $EVAL_DIR"
    exit 5
fi
mkdir -p "$EVAL_DIR/rollouts" "$LOG_DIR"
toolweave_safe_rm_rf "$TMP_ROOT"
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

# Stage-2 config with val-only overrides; ONLY model.path / val_files / dirs differ
# between models. Everything else is byte-identical across the four eval runs.
COMMAND=(
    python -m verl.trainer.main_ppo
    --config-path="$STAGE_ROOT/configs"
    --config-name=stage2_qwen3_4b_k16_base_progress_batch20_plain_env
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.model.enable_gradient_checkpointing=false"
    "actor_rollout_ref.actor.use_kl_loss=false"
    "actor_rollout_ref.actor.entropy_coeff=0.0"
    "actor_rollout_ref.rollout.n=1"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$TOOLWEAVE_ROLLOUT_TP"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.70"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=false"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0"
    "data.val_files=$VAL_FILE"
    "data.val_batch_size=16"
    "trainer.n_gpus_per_node=$TOOLWEAVE_LEARNER_GPUS_PER_NODE"
    "trainer.nnodes=$TOOLWEAVE_NNODES"
    "trainer.val_before_train=true"
    "trainer.val_only=true"
    "trainer.save_freq=-1"
    "trainer.resume_mode=disable"
    "trainer.default_hdfs_dir=null"
    "trainer.default_local_dir=$EVAL_DIR/no_checkpoints"
    "trainer.validation_data_dir=$EVAL_DIR/rollouts"
    "trainer.rollout_data_dir=null"
    "trainer.logger=[console]"
    "trainer.experiment_name=stage2_gate_eval_$LABEL"
    "ray_init.num_cpus=$TOOLWEAVE_RAY_NUM_CPUS"
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
