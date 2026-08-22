#!/usr/bin/env bash
# Stage 2 real smoke test: 4 Base prompts x K=16, ONE trainer step, no checkpoint.
# Verifies the 12 required points (progress reward, is_augmented, KL, loss, etc.).
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
CONFIG_DIR="$STAGE_ROOT/configs"
CONFIG_NAME="stage2_qwen3_4b_k16_base_progress_batch20_plain_env"
SMOKE_DIR="$TOOLWEAVE_ARTIFACTS_ROOT/gpu_smoke/stage2_smoke"
LOG_FILE="$TOOLWEAVE_LOGS_ROOT/stage2_smoke.log"
TMP_ROOT="$TOOLWEAVE_SHORT_TEMP_ROOT/stage2-smoke"
MODEL_PATH="$TOOLWEAVE_ARTIFACTS_ROOT/gate_vs_base/merged/global_step_25"
SMOKE_DATA="$TOOLWEAVE_DATA_ROOT/checkpoint_gate_eval/smoke_base4.parquet"

# --- 1. plain-env assertion + tool module paths ---
echo "== [1] plain-env assertion =="
"$TOOLWEAVE_PYTHON" "$STAGE_ROOT/scripts/verify_stage2_plain_env.py"

# --- 2. prepare 4 Base prompts data file (from val_base_100, first 4) ---
"$TOOLWEAVE_PYTHON" - <<PYEOF
import pandas as pd
src="$TOOLWEAVE_DATA_ROOT/checkpoint_gate_eval/val_base_100.parquet"
df=pd.read_parquet(src).head(4)
assert len(df)==4 and df['data_source'].nunique()==1, "must be 4 Base prompts"
df.to_parquet("$SMOKE_DATA", index=False)
print("smoke data: 4 Base prompts -> $SMOKE_DATA")
PYEOF

# --- 3. environment + launch one step, no checkpoint ---
toolweave_activate_conda
export PYTHONPATH="$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
toolweave_apply_topology learner
export TOKENIZERS_PARALLELISM=true
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN NCCL_TIMEOUT=3600
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF
toolweave_safe_rm_rf "$TMP_ROOT" "$SMOKE_DIR"
mkdir -p "$TMP_ROOT/ray" "$TMP_ROOT/triton" "$SMOKE_DIR"
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
