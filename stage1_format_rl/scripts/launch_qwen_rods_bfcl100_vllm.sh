#!/usr/bin/env bash
# Launch Qwen3-4B-RODS for the read-only BFCL100 evaluation on Blackwell.
# The rods-synth stack is reused because it has native sm_120 support and was
# already exercised by the completed Gemma generation run.
set -Eeuo pipefail

if [[ "${RODS_ALLOW_QWEN_EVAL_AFTER_GENERATOR:-0}" != "1" ]]; then
  echo "Refusing to start: Generator terminal reconciliation has not been acknowledged." >&2
  exit 2
fi

WORKSPACE=/root/autodl-tmp/rods-workspace
MODEL="$WORKSPACE/models/Qwen3-4B-RODS"
PORT="${RODS_EVAL_PORT:-31000}"
ENV_ROOT=/root/autodl-tmp/conda/envs/rods-synth
MAX_MODEL_LEN="${RODS_EVAL_MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${RODS_EVAL_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${RODS_EVAL_MAX_NUM_SEQS:-32}"
MAX_BATCHED_TOKENS="${RODS_EVAL_MAX_BATCHED_TOKENS:-65536}"

if pgrep -af 'gemma-4-31B|vllm.entrypoints.openai.api_server.*gemma' >/dev/null; then
  echo "Refusing to start Qwen eval while the Gemma generator is present." >&2
  pgrep -af 'gemma-4-31B|vllm.entrypoints.openai.api_server.*gemma' >&2 || true
  exit 3
fi

used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if [[ ! "$used_mib" =~ ^[0-9]+$ ]] || (( used_mib > 2048 )); then
  echo "Refusing to start: GPU0 is not released (used=${used_mib:-unknown} MiB)." >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# The completed Gemma run established this Blackwell-safe transport setting.
# Eval decoding is greedy, so this changes only the sampler implementation.
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="$ENV_ROOT/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

exec "$ENV_ROOT/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --api-key EMPTY \
  --served-model-name Qwen3-4B-RODS \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --enable-prefix-caching \
  --language-model-only \
  --skip-mm-profiling \
  --generation-config vllm
