#!/usr/bin/env bash
# Launch Qwen3-4B-RODS for the read-only BFCL100 evaluation on Blackwell.
# The rods-synth stack is reused because it has native sm_120 support and was
# already exercised by the completed Gemma generation run.
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
export TOOLWEAVE_PROFILE="${TOOLWEAVE_SINGLE_GPU_PROFILE:-$STAGE_ROOT/configs/layers/profiles/single_gpu_eval.yaml}"

case "${1:-}" in
  --dry-run) mode=() ;;
  --execute|"") mode=(--execute) ;;
  *) echo "usage: $0 [--dry-run|--execute]" >&2; exit 2 ;;
esac

exec "$TOOLWEAVE_PYTHON" -m stage1_format_rl.infrastructure.cli \
  --profile "$TOOLWEAVE_PROFILE" evaluation-server --backend vllm "${mode[@]}"
