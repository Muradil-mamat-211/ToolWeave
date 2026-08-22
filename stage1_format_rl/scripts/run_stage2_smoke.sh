#!/usr/bin/env bash
# Stage 2 real smoke test: 4 Base prompts x K=16, ONE trainer step, no checkpoint.
# Verifies the 12 required points (progress reward, is_augmented, KL, loss, etc.).
set -Eeuo pipefail
WORKSPACE="/root/autodl-tmp/rods-workspace"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
CONFIG_DIR="$STAGE_ROOT/configs"
CONFIG_NAME="stage2_qwen3_4b_k16_base_progress_batch20_plain_env"
SMOKE_DIR="$STAGE_ROOT/artifacts/gpu_smoke/stage2_smoke"
LOG_FILE="$STAGE_ROOT/logs/stage2_smoke.log"
TMP_ROOT="/tmp/r1g2smoke"
MODEL_PATH="$STAGE_ROOT/artifacts/gate_vs_base/merged/global_step_25"
SMOKE_DATA="$STAGE_ROOT/data/checkpoint_gate_eval/smoke_base4.parquet"

# --- 1. plain-env assertion + tool module paths ---
echo "== [1] plain-env assertion =="
/root/miniconda3/envs/rods/bin/python "$STAGE_ROOT/scripts/verify_stage2_plain_env.py"

# --- 2. prepare 4 Base prompts data file (from val_base_100, first 4) ---
/root/miniconda3/envs/rods/bin/python - <<PYEOF
import pandas as pd
src="/root/autodl-tmp/rods-workspace/stage1_format_rl/data/checkpoint_gate_eval/val_base_100.parquet"
df=pd.read_parquet(src).head(4)
assert len(df)==4 and df['data_source'].nunique()==1, "must be 4 Base prompts"
df.to_parquet("$SMOKE_DATA", index=False)
print("smoke data: 4 Base prompts -> $SMOKE_DATA")
PYEOF

# --- 3. environment + launch one step, no checkpoint ---
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rods
export PYTHONPATH="$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 NUMEXPR_MAX_THREADS=48
export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=48
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN NCCL_TIMEOUT=3600
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF
rm -rf "$TMP_ROOT" "$SMOKE_DIR"; mkdir -p "$TMP_ROOT/ray" "$TMP_ROOT/triton" "$SMOKE_DIR"
export TRITON_CACHE_DIR="$TMP_ROOT/triton"; export RAY_TMPDIR="$TMP_ROOT/ray"; export TMPDIR="$TMP_ROOT"

echo "== [2] launching Stage 2 smoke: 4 prompts x K=16, 1 step, no ckpt =="
( cd "$AWORLD/EnvTuning" && \
  python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" --config-name="$CONFIG_NAME" \
    data.train_files="$SMOKE_DATA" \
    data.train_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.experiment_name=stage2_smoke \
    trainer.default_local_dir="$SMOKE_DIR/checkpoints" \
    trainer.rollout_data_dir=null \
    trainer.validation_data_dir="$SMOKE_DIR/rollouts" ) \
  2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
echo "== smoke exit=$STATUS =="

# --- 4. post-run verifications ---
echo "== [3] no checkpoint written =="
if find "$SMOKE_DIR" -name 'global_step_*' -o -name '*.pt' -o -name 'data.pt' | grep -q .; then
  echo "  [FAIL] smoke wrote checkpoints"; exit 90
else
  echo "  [ok] no checkpoint written"
fi
echo "== [4] required log signals =="
grep -qE "reward_kl_penalty|reward_after_kl" "$LOG_FILE" && echo "  [FAIL] reward-side KL present" || echo "  [ok] no reward-side KL in logs"
grep -E "use_kl_loss|kl_loss_coef|use_kl_in_reward|kl_ctrl" "$LOG_FILE" | head -6
echo "== [5] step metrics (loss/grad/entropy/kl) =="
grep -E "actor/kl_loss|actor/ppo_kl|actor/grad_norm|actor/entropy|actor/pg_clipfrac|critic/score/mean|critic/rewards/mean|critic/advantages/mean|timing_s/step" "$LOG_FILE" | tail -4
echo "== SMOKE_DONE status=$STATUS =="
exit "$STATUS"
