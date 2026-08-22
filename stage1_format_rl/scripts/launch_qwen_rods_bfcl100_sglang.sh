#!/usr/bin/env bash
# Launch the released Qwen3-4B-RODS model for the read-only BFCL100 eval.
# This script deliberately refuses to run while the Gemma generator owns GPU0.
set -Eeuo pipefail

if [[ "${RODS_ALLOW_QWEN_EVAL_AFTER_GENERATOR:-0}" != "1" ]]; then
  echo "Refusing to start: set RODS_ALLOW_QWEN_EVAL_AFTER_GENERATOR=1 only after Generator terminal reconciliation." >&2
  exit 2
fi

WORKSPACE=/root/autodl-tmp/rods-workspace
MODEL="$WORKSPACE/models/Qwen3-4B-RODS"
PORT="${RODS_EVAL_PORT:-31000}"

if pgrep -af 'gemma-4-31B|vllm.entrypoints.openai.api_server' >/dev/null; then
  echo "Refusing to start Qwen eval while Gemma/vLLM is still present." >&2
  pgrep -af 'gemma-4-31B|vllm.entrypoints.openai.api_server' >&2 || true
  exit 3
fi

used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if [[ ! "$used_mib" =~ ^[0-9]+$ ]] || (( used_mib > 2048 )); then
  echo "Refusing to start: GPU0 is not released (used=${used_mib:-unknown} MiB)." >&2
  exit 4
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rods
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name Qwen3-4B-RODS \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tp-size 1 \
  --dtype bfloat16 \
  --context-length 32768 \
  --mem-fraction-static 0.90 \
  --max-running-requests 24 \
  --cuda-graph-max-bs 32 \
  --attention-backend triton \
  --sampling-backend flashinfer \
  --enable-tokenizer-batch-encode
