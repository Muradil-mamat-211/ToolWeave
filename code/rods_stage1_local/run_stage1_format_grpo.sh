#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/rods-workspace}"
LOCAL="$WORKSPACE/code/rods_stage1_local"
VERL="$WORKSPACE/code/verl"
OUTPUT="$WORKSPACE/outputs/stage1_format_qwen3_1p7b"
MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    STEPS=5
    K=2
    BATCH_SIZE=4
    MINI_BATCH_SIZE=4
    EXPERIMENT="qwen3_1p7b_stage1_smoke"
    SAVE_FREQ=5
    EPOCHS=1
    CHECKPOINT_DIR="$OUTPUT/checkpoints/format_prompt_smoke"
    ;;
  formal)
    STEPS="${TRAIN_STEPS:-100}"
    K=8
    BATCH_SIZE=2
    # veRL expands this by K before FSDP sharding: 2 prompts * K=8 -> 16 rollouts.
    # A prompt-level mini-batch of 2 therefore gives an 8-row local FSDP mini-batch.
    MINI_BATCH_SIZE=2
    EXPERIMENT="qwen3_1p7b_stage1_format"
    SAVE_FREQ=20
    EPOCHS=2
    CHECKPOINT_DIR="$OUTPUT/checkpoints"
    ;;
  *)
    echo "Usage: $0 {smoke|formal}" >&2
    exit 2
    ;;
esac

mkdir -p "$CHECKPOINT_DIR" "$OUTPUT/final_model" "$OUTPUT/logs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$VERL:$WORKSPACE/code/AWorld-RL/EnvTuning:$LOCAL:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1

cp "$LOCAL/stage1_train_config.yaml" "$OUTPUT/train_config.yaml"
{
  echo "mode=$MODE"
  echo "steps=$STEPS"
  echo "rollout_n=$K"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "gpu_memory_utilization=0.90"
  echo "tensor_model_parallel_size=2"
  echo "trainer_n_gpus_per_node=2"
  echo "checkpoint_dir=$CHECKPOINT_DIR"
} > "$OUTPUT/logs/run_config.txt"

cd "$WORKSPACE"
conda run --no-capture-output -n rods python -m verl.trainer.main_ppo \
  --config-path="$LOCAL" \
  --config-name=stage1_train_config \
  data.train_batch_size="$BATCH_SIZE" \
  data.val_batch_size=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
  actor_rollout_ref.rollout.n="$K" \
  actor_rollout_ref.actor.rollout_n="$K" \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs="$EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=0 \
  trainer.val_before_train=false \
  trainer.experiment_name="$EXPERIMENT" \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.rollout_data_dir="$OUTPUT/logs/rollouts" \
  trainer.validation_data_dir="$OUTPUT/logs/validation" \
  2>&1 | tee "$OUTPUT/logs/${MODE}_train.log"

latest="$(find "$OUTPUT/checkpoints" -maxdepth 2 -type d -name 'global_step_*' | sort -V | tail -1 || true)"
if [[ -n "$latest" ]]; then
  printf '%s\n' "$latest" > "$OUTPUT/logs/latest_checkpoint.txt"
  echo "latest_checkpoint=$latest"
else
  echo "No global_step checkpoint was found" >&2
fi
