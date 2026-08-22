#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${ALLOW_STAGE1_RECOVERY_TRAINING:-0}" != "1" ]]; then
    echo "Recovery training is disabled. Set ALLOW_STAGE1_RECOVERY_TRAINING=1 only after explicit approval."
    exit 2
fi

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
CONFIG_DIR="$STAGE_ROOT/configs"
CONFIG_NAME="stage1_qwen3_4b_k16_recovery_from_step25"
MODEL="$TOOLWEAVE_ARTIFACTS_ROOT/checkpoint_eval/merged/global_step_25"
TRAIN_DATA="$TOOLWEAVE_DATA_ROOT/bfcl_stage1_train_base_100_shuffled_seed42.parquet"
OUTPUT_ROOT="$TOOLWEAVE_OUTPUTS_ROOT/stage1_format_qwen3_4b_recovery_from_step25"
CHECKPOINT_ROOT="$OUTPUT_ROOT/checkpoints"
FINAL_MODEL="$OUTPUT_ROOT/final_model"
STAGING_MODEL="$OUTPUT_ROOT/.final_model_staging"
LOG_DIR="$TOOLWEAVE_LOGS_ROOT/recovery_from_step25"
LOG_FILE="$LOG_DIR/training.log"
GPU_CSV="$LOG_DIR/gpu.csv"
CPU_CSV="$LOG_DIR/cpu.csv"
STATUS_FILE="$LOG_DIR/status.txt"
TMP_ROOT="$TOOLWEAVE_SHORT_TEMP_ROOT/stage1-recovery"

for path in "$AWORLD/EnvTuning" "$AWORLD/EnvTuning/verl" "$MODEL" "$TRAIN_DATA"; do
    [[ -e "$path" ]] || { echo "Required path missing: $path"; exit 3; }
done
if [[ -e "$FINAL_MODEL" || -e "$STAGING_MODEL" ]]; then
    echo "Refusing to overwrite an existing recovery final model: $OUTPUT_ROOT"
    exit 4
fi

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT" "$LOG_DIR" "$TMP_ROOT/ray" "$TMP_ROOT/triton"
touch "$LOG_FILE" "$GPU_CSV" "$CPU_CSV"
printf '%q ' python -m verl.trainer.main_ppo --config-path="$CONFIG_DIR" --config-name="$CONFIG_NAME" > "$OUTPUT_ROOT/launch_command.sh"
printf '\n' >> "$OUTPUT_ROOT/launch_command.sh"

toolweave_activate_conda

export PYTHONPATH="$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
toolweave_apply_topology learner
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

monitor_gpu() {
    [[ -s "$GPU_CSV" ]] || echo 'timestamp,index,memory.used_mib,memory.total_mib,utilization.gpu_pct,utilization.memory_pct,power.draw_w,temperature_c' > "$GPU_CSV"
    while true; do
        now="$(date -Ins)"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits | sed "s/^/$now,/" >> "$GPU_CSV"
        sleep 5
    done
}

monitor_cpu() {
    [[ -s "$CPU_CSV" ]] || echo 'timestamp,load1,load5,load15,mem_available_kib' > "$CPU_CSV"
    while true; do
        now="$(date -Ins)"
        read -r load1 load5 load15 _ < /proc/loadavg
        mem_available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
        echo "$now,$load1,$load5,$load15,$mem_available" >> "$CPU_CSV"
        sleep 5
    done
}

monitor_gpu & GPU_MONITOR_PID=$!
monitor_cpu & CPU_MONITOR_PID=$!
cleanup() {
    kill "$GPU_MONITOR_PID" "$CPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" "$CPU_MONITOR_PID" 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "RUNNING $(date -Is)" > "$STATUS_FILE"
cd "$AWORLD/EnvTuning"
set +e
python -m verl.trainer.main_ppo --config-path="$CONFIG_DIR" --config-name="$CONFIG_NAME" 2>&1 | tee -a "$LOG_FILE"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

cleanup
trap - EXIT INT TERM
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    echo "FAILED exit=$TRAIN_STATUS $(date -Is)" > "$STATUS_FILE"
    exit "$TRAIN_STATUS"
fi

LATEST_STEP="$(tr -dc '0-9' < "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")"
[[ "$LATEST_STEP" == "100" ]] || { echo "Expected recovery step 100, found $LATEST_STEP"; exit 20; }
LATEST_DIR="$CHECKPOINT_ROOT/global_step_$LATEST_STEP"
ACTOR_DIR="$LATEST_DIR/actor"

toolweave_safe_rm_rf "$STAGING_MODEL"
python -m verl.model_merger merge --backend fsdp --local_dir "$ACTOR_DIR" --target_dir "$STAGING_MODEL" --use_cpu_initialization

FINAL_PATH="$STAGING_MODEL" SOURCE_CHECKPOINT="$LATEST_DIR" python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["FINAL_PATH"])
payload = {
    "stage": "stage1_format_rl",
    "recovery_reason": "original run collapsed after epoch 1; held-out selection chose global_step_25",
    "original_base_model": str(Path(os.environ["TOOLWEAVE_MODELS_ROOT"]) / "Qwen3-4B"),
    "continuation_model": str(Path(os.environ["TOOLWEAVE_ARTIFACTS_ROOT"]) / "checkpoint_eval/merged/global_step_25"),
    "source_checkpoint": os.environ["SOURCE_CHECKPOINT"],
    "logical_global_step": 125,
    "outer_epochs_total": 5,
    "outer_epochs_before_recovery": 1,
    "outer_epochs_in_recovery": 4,
    "algorithm": "GRPO",
    "rollout_n": 16,
    "learning_rate": 1e-7,
    "kl_loss_coef": 0.01,
    "reward_side_kl": False,
    "reward": "EnvTuning/env_tuning/format_reward.py::compute_score",
    "created_at": datetime.now(timezone.utc).isoformat(),
}
(path / "training_provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

FINAL_PATH="$STAGING_MODEL" python - <<'PY'
import json
import os
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

path = Path(os.environ["FINAL_PATH"])
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
index_path = path / "model.safetensors.index.json"
if index_path.exists():
    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing = sorted({name for name in index["weight_map"].values() if not (path / name).is_file()})
    if missing:
        raise RuntimeError(f"missing final shards: {missing}")
elif not (path / "model.safetensors").is_file():
    raise RuntimeError("no final safetensors weights")
print("FINAL_MODEL_RESOURCE_VERIFIED", config.model_type, type(tokenizer).__name__)
PY

mv "$STAGING_MODEL" "$FINAL_MODEL"
echo "COMPLETED final_model=$FINAL_MODEL logical_step=125 $(date -Is)" > "$STATUS_FILE"
echo "$FINAL_MODEL"
