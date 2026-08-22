#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
WORKSPACE=/root/autodl-tmp/rods-workspace

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rods

AWORLD="$WORKSPACE/code/AWorld-RL"
ENV_ROOT="$AWORLD/EnvTuning"
DATA="$WORKSPACE/data/stage2_official/bfcl_v3_multiturn_base_official.parquet"
VAL="$WORKSPACE/data/stage2_official/bfcl_val_official.parquet"
MODEL="$WORKSPACE/models/Qwen3-1.7B"

if [[ "$MODE" == "smoke" ]]; then
  OUT="$WORKSPACE/outputs/stage1_official_protocol_qwen3_1p7b_smoke"
  K=4
  BATCH=1
  MINI=4
  MICRO=1
  STEPS=5
  SAVE_FREQ=5
  TEST_FREQ=0
  LOGGER="[console]"
  EXPERIMENT="stage1_official_protocol_qwen3_1p7b_smoke"
else
  OUT="$WORKSPACE/outputs/stage1_official_protocol_qwen3_1p7b"
  K=8
  BATCH=2
  MINI=8
  MICRO=1
  STEPS=100
  SAVE_FREQ=25
  TEST_FREQ=25
  LOGGER="[console,tensorboard]"
  EXPERIMENT="stage1_official_protocol_qwen3_1p7b"
fi

mkdir -p "$OUT" "$OUT/logs" "$OUT/training_rollouts" "$OUT/checkpoints"
export CUDA_VISIBLE_DEVICES="0,1"
export NVIDIA_VISIBLE_DEVICES="all"
export PYTHONPATH="$ENV_ROOT:$ENV_ROOT/verl:$WORKSPACE/code/verl:${PYTHONPATH:-}"
export NCCL_IB_TIMEOUT=22
export NCCL_TIMEOUT=9999999999
export TRITON_CACHE_DIR="/tmp/triton_cache_$(id -u)"
export OMP_NUM_THREADS=1

cd "$ENV_ROOT"

CMD=(python -m verl.trainer.main_ppo
  --config-path="$ENV_ROOT/env_tuning/config"
  --config-name=multi_turn_fc_grpo_stage1
  algorithm.adv_estimator=grpo
  data.train_batch_size="$BATCH"
  data.val_batch_size=2
  data.filter_overlong_prompts=False
  data.truncation=error
  data.return_raw_chat=True
  data.max_prompt_length=2048
  data.max_response_length=1024
  data.train_files="$DATA"
  data.val_files="$VAL"
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_fused_kernels=True
  actor_rollout_ref.actor.entropy_checkpointing=True
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80
  actor_rollout_ref.rollout.n="$K"
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  actor_rollout_ref.rollout.max_model_len=4096
  actor_rollout_ref.rollout.prompt_length=2048
  actor_rollout_ref.rollout.response_length=1024
  actor_rollout_ref.rollout.temperature=0.7
  actor_rollout_ref.rollout.top_p=0.9
  actor_rollout_ref.rollout.do_sample=True
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_user_turns=100
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=100
  actor_rollout_ref.rollout.multi_turn.interaction_config_path=env_tuning/config/multi_turn_fc_interaction_config.yaml
  actor_rollout_ref.rollout.multi_turn.format=qwen
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
  trainer.logger="$LOGGER"
  trainer.project_name=bfcl_multi_turn_grpo
  trainer.experiment_name="$EXPERIMENT"
  trainer.rollout_data_dir="$OUT/training_rollouts"
  trainer.default_local_dir="$OUT/checkpoints"
  trainer.n_gpus_per_node=2
  trainer.nnodes=1
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.val_before_train=False
  trainer.total_training_steps="$STEPS"
  trainer.total_epochs=1
  trainer.resume_mode=disable)

printf '%q ' "${CMD[@]}" > "$OUT/launch_command.sh"
printf '\n' >> "$OUT/launch_command.sh"
cp "$AWORLD/EnvTuning/env_tuning/config/multi_turn_fc_grpo_stage1.yaml" "$OUT/official_stage1_config.yaml"
printf '%s\n' "MODE=$MODE K=$K STEPS=$STEPS" | tee "$OUT/run_metadata.txt"
"${CMD[@]}" 2>&1 | tee "$OUT/train.log"
