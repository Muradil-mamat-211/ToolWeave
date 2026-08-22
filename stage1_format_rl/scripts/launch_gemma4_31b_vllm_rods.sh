#!/usr/bin/env bash
set -euo pipefail

# This launcher is an implemented deployment interface, not a tested runtime.
# It was not executed during the no-GPU implementation/audit.
if [[ "${RODS_ALLOW_VLLM_SERVER:-0}" != "1" ]]; then
  echo "Refusing to launch vLLM. Set RODS_ALLOW_VLLM_SERVER=1 explicitly." >&2
  exit 2
fi

RODS_GEMMA_MODEL="${RODS_GEMMA_MODEL:-/root/autodl-tmp/rods-workspace/models/gemma-4-31B-it-manual}"
RODS_VLLM_HOST="${RODS_VLLM_HOST:-127.0.0.1}"
RODS_VLLM_PORT="${RODS_VLLM_PORT:-8000}"
RODS_TENSOR_PARALLEL_SIZE="${RODS_TENSOR_PARALLEL_SIZE:-2}"
RODS_SERVED_MODEL_NAME="${RODS_SERVED_MODEL_NAME:-${RODS_GEMMA_MODEL}}"

RODS_VLLM_ARGS=(
  serve "${RODS_GEMMA_MODEL}"
  --host "${RODS_VLLM_HOST}"
  --port "${RODS_VLLM_PORT}"
  --tensor-parallel-size "${RODS_TENSOR_PARALLEL_SIZE}"
  --served-model-name "${RODS_SERVED_MODEL_NAME}"
)
if [[ -n "${RODS_MAX_MODEL_LEN:-}" ]]; then
  RODS_VLLM_ARGS+=(--max-model-len "${RODS_MAX_MODEL_LEN}")
fi
if [[ -n "${RODS_GPU_MEMORY_UTILIZATION:-}" ]]; then
  RODS_VLLM_ARGS+=(--gpu-memory-utilization "${RODS_GPU_MEMORY_UTILIZATION}")
fi

exec vllm "${RODS_VLLM_ARGS[@]}"
