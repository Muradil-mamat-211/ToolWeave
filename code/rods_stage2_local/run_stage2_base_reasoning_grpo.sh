#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/rods-workspace}"
if [[ ! -d "$WORKSPACE" ]]; then
  WORKSPACE="/root/rods-workspace"
fi
LOCAL="$WORKSPACE/code/rods_stage2_local"
VERL="$WORKSPACE/code/verl"
AWORLD="$WORKSPACE/code/AWorld-RL/EnvTuning"
OUTPUT="${OUTPUT:-$WORKSPACE/outputs/stage2_base_reasoning_qwen3_1p7b}"
MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    STEPS=5
    K=4
    BATCH_SIZE=2
    MINI_BATCH_SIZE=2
    RESPONSE_LENGTH=512
    SAVE_FREQ=5
    EPOCHS=1
    EXPERIMENT="qwen3_1p7b_stage2_smoke"
    CHECKPOINT_DIR="$OUTPUT/smoke_checkpoints_reference_v2"
    ROLLOUT_DIR="$OUTPUT/logs/smoke_rollouts_reference_v2"
    METRICS_OUT="$OUTPUT/stage2_smoke_metrics_reference_v2.jsonl"
    ;;
  formal)
    STEPS="${TRAIN_STEPS:-100}"
    K=16
    BATCH_SIZE=2
    MINI_BATCH_SIZE=2
    RESPONSE_LENGTH=1024
    SAVE_FREQ=100
    EPOCHS=2
    EXPERIMENT="qwen3_1p7b_stage2_base_reasoning"
    CHECKPOINT_DIR="$OUTPUT/checkpoints"
    ROLLOUT_DIR="$OUTPUT/logs/rollouts"
    METRICS_OUT="$OUTPUT/stage2_metrics.jsonl"
    ;;
  *)
    echo "Usage: $0 {smoke|formal}" >&2
    exit 2
    ;;
esac

mkdir -p "$CHECKPOINT_DIR" "$ROLLOUT_DIR" "$OUTPUT/logs" "$OUTPUT/final_model"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$LOCAL:$VERL:$AWORLD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VLLM_DO_NOT_TRACK=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1

cp "$LOCAL/stage2_train_config.yaml" "$OUTPUT/train_config.yaml"
conda run --no-capture-output -n rods python - <<'PY' | tee "$OUTPUT/logs/flash_attention_check.txt"
import flash_attn
import torch
print("flash_attn", flash_attn.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flash_attention_2_requested=true")
PY
{
  echo "mode=$MODE"
  echo "entrypoint=python -m verl.trainer.main_ppo"
  echo "model_path=$WORKSPACE/outputs/stage1_format_qwen3_1p7b/final_model"
  echo "steps=$STEPS"
  echo "rollout_n=$K"
  echo "max_prompt_length=2048"
  echo "max_response_length=$RESPONSE_LENGTH"
  echo "learning_rate=5e-7"
  echo "kl_loss_coef=0.002"
  echo "attention_implementation=flash_attention_2"
  echo "tensor_model_parallel_size=2"
  echo "gpu_memory_utilization=0.92"
} > "$OUTPUT/logs/${MODE}_run_config.txt"

cd "$WORKSPACE"
conda run --no-capture-output -n rods python -m verl.trainer.main_ppo \
  --config-path="$LOCAL" \
  --config-name=stage2_train_config \
  data.train_batch_size="$BATCH_SIZE" \
  data.val_batch_size=2 \
  data.max_response_length="$RESPONSE_LENGTH" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.rollout.n="$K" \
  actor_rollout_ref.actor.rollout_n="$K" \
  actor_rollout_ref.rollout.response_length="$RESPONSE_LENGTH" \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs="$EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=0 \
  trainer.val_before_train=false \
  trainer.experiment_name="$EXPERIMENT" \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DIR" \
  trainer.validation_data_dir="$OUTPUT/logs/validation" \
  2>&1 | tee "$OUTPUT/logs/${MODE}_train.log"

conda run --no-capture-output -n rods python "$WORKSPACE/scripts/summarize_stage2_rollouts.py" \
  --rollouts "$ROLLOUT_DIR" \
  --out "$METRICS_OUT" \
  --expected_steps "$STEPS"

latest="$CHECKPOINT_DIR/global_step_$STEPS"
test -d "$latest/actor" || { echo "Expected checkpoint missing: $latest/actor" >&2; exit 1; }
printf '%s\n' "$latest" > "$OUTPUT/logs/${MODE}_latest_checkpoint.txt"

if [[ "$MODE" == "formal" ]]; then
  test -z "$(find "$OUTPUT/final_model" -mindepth 1 -maxdepth 1 -print -quit)" || {
    echo "Final model directory is not empty; refusing to overwrite it." >&2
    exit 1
  }
  conda run --no-capture-output -n rods python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$latest/actor" \
    --target_dir "$OUTPUT/final_model" \
    2>&1 | tee "$OUTPUT/logs/merge_final_model.log"
  rm -rf "$CHECKPOINT_DIR"
fi
