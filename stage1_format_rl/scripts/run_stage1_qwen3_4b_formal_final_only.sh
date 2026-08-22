#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${ALLOW_STAGE1_TRAINING:-0}" != "1" ]]; then
    echo "Stage 1 training is disabled. Set ALLOW_STAGE1_TRAINING=1 only after explicit user approval."
    exit 2
fi

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
AWORLD="$WORKSPACE/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
CONFIG_DIR="$STAGE_ROOT/configs"
CONFIG_NAME="stage1_qwen3_4b_k16_formal_final_only"
MODEL="$TOOLWEAVE_MODELS_ROOT/Qwen3-4B"
TRAIN_DATA="$TOOLWEAVE_DATA_ROOT/bfcl_stage1_train_base_100.parquet"
OUTPUT_ROOT="$TOOLWEAVE_OUTPUTS_ROOT/stage1_format_qwen3_4b"
TEMP_EXPORT="$OUTPUT_ROOT/.final_export_tmp"
FINAL_MODEL="$OUTPUT_ROOT/final_model"
LOG_DIR="$TOOLWEAVE_LOGS_ROOT/formal"
LOG_FILE="$LOG_DIR/stage1_qwen3_4b_k16_formal.log"
GPU_CSV="$LOG_DIR/stage1_qwen3_4b_k16_gpu.csv"
CPU_CSV="$LOG_DIR/stage1_qwen3_4b_k16_cpu.csv"
STATUS_FILE="$LOG_DIR/stage1_qwen3_4b_k16_status.txt"
TMP_ROOT="$TOOLWEAVE_SHORT_TEMP_ROOT/stage1-formal"

for path in "$AWORLD/EnvTuning" "$AWORLD/EnvTuning/verl" "$MODEL" "$TRAIN_DATA"; do
    [[ -e "$path" ]] || { echo "Required path missing: $path"; exit 3; }
done

if [[ -e "$FINAL_MODEL" || -e "$TEMP_EXPORT" ]]; then
    echo "Refusing to overwrite an existing final or temporary export: $OUTPUT_ROOT"
    exit 4
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$TMP_ROOT/ray" "$TMP_ROOT/triton"
: > "$STATUS_FILE"

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
export TENSORBOARD_DIR="$LOG_DIR/tensorboard"

monitor_gpu() {
    echo 'timestamp,index,memory.used_mib,memory.total_mib,utilization.gpu_pct,utilization.memory_pct,power.draw_w,temperature_c' > "$GPU_CSV"
    while true; do
        local now
        now="$(date -Ins)"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
            --format=csv,noheader,nounits | sed "s/^/$now,/" >> "$GPU_CSV"
        sleep 5
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
        sleep 5
    done
}

monitor_gpu &
GPU_MONITOR_PID=$!
monitor_cpu &
CPU_MONITOR_PID=$!

cleanup_runtime() {
    kill "$GPU_MONITOR_PID" "$CPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" "$CPU_MONITOR_PID" 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_runtime EXIT INT TERM

echo "RUNNING $(date -Is)" > "$STATUS_FILE"
cd "$AWORLD/EnvTuning"
set +e
python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    2>&1 | tee "$LOG_FILE"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

cleanup_runtime
trap - EXIT INT TERM

if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    echo "FAILED exit=$TRAIN_STATUS $(date -Is)" > "$STATUS_FILE"
    toolweave_safe_rm_rf "$TEMP_EXPORT"
    exit "$TRAIN_STATUS"
fi

mapfile -t STEP_DIRS < <(find "$TEMP_EXPORT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | sort)
if [[ "${#STEP_DIRS[@]}" -ne 1 ]]; then
    echo "Expected exactly one final-step export, found ${#STEP_DIRS[@]}" >&2
    exit 20
fi

STEP_DIR="${STEP_DIRS[0]}"
ACTOR_DIR="$STEP_DIR/actor"
HF_DIR="$ACTOR_DIR/huggingface"

if find "$TEMP_EXPORT" -type f \( \
    -name 'optim_world_size_*.pt' -o \
    -name 'optimizer.pt' -o \
    -name 'extra_state_world_size_*.pt' -o \
    -name 'model_world_size_*.pt' \
\) -print -quit | grep -q .; then
    echo "Unexpected sharded checkpoint or optimizer state was saved" >&2
    exit 21
fi

[[ -f "$HF_DIR/config.json" ]] || { echo "Final config missing"; exit 22; }
if [[ ! -f "$HF_DIR/model.safetensors" && ! -f "$HF_DIR/model.safetensors.index.json" ]]; then
    echo "Final safetensors weights missing" >&2
    exit 23
fi

STAGING_MODEL="$OUTPUT_ROOT/.final_model_staging"
toolweave_safe_rm_rf "$STAGING_MODEL"
mv "$HF_DIR" "$STAGING_MODEL"
GLOBAL_STEP="${STEP_DIR##*global_step_}"

GLOBAL_STEP="$GLOBAL_STEP" FINAL_PATH="$STAGING_MODEL" "$TOOLWEAVE_PYTHON" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["FINAL_PATH"])
payload = {
    "stage": "stage1_format_rl",
    "base_model": str(Path(os.environ["TOOLWEAVE_MODELS_ROOT"]) / "Qwen3-4B"),
    "train_data": str(Path(os.environ["TOOLWEAVE_DATA_ROOT"]) / "bfcl_stage1_train_base_100.parquet"),
    "algorithm": "GRPO",
    "rollout_n": 16,
    "prompt_batch": 4,
    "epochs": 1,
    "global_step": int(os.environ["GLOBAL_STEP"]),
    "kl_loss_coef": 0.01,
    "reward_side_kl": False,
    "reward": "EnvTuning/env_tuning/format_reward.py::compute_score",
    "intermediate_checkpoints": 0,
    "created_at": datetime.now(timezone.utc).isoformat(),
}
(path / "training_provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

mv "$STAGING_MODEL" "$FINAL_MODEL"
toolweave_safe_rm_rf "$TEMP_EXPORT"

FINAL_MODEL="$FINAL_MODEL" "$TOOLWEAVE_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

path = Path(os.environ["FINAL_MODEL"])
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
index_path = path / "model.safetensors.index.json"
if index_path.exists():
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing final-model shards: {missing}")
elif not (path / "model.safetensors").is_file():
    raise RuntimeError("No final safetensors weights found")
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
num_params = sum(parameter.numel() for parameter in model.parameters())
if not 3_000_000_000 <= num_params <= 5_000_000_000:
    raise RuntimeError(f"Unexpected parameter count: {num_params}")
print("FINAL_MODEL_VERIFIED", cfg.model_type, type(tok).__name__, num_params)
PY

if find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name final_model -print -quit | grep -q .; then
    echo "Unexpected non-final output directory remains" >&2
    exit 24
fi

echo "COMPLETED final_model=$FINAL_MODEL $(date -Is)" > "$STATUS_FILE"
echo "$FINAL_MODEL"
