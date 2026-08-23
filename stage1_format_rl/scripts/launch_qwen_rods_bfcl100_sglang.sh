#!/usr/bin/env bash
# Launch the released Qwen3-4B-RODS model for the read-only BFCL100 eval.
# Runtime and physical GPU ownership come from the selected layered profile.
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
export TOOLWEAVE_PROFILE="${TOOLWEAVE_SINGLE_GPU_PROFILE:-$STAGE_ROOT/configs/layers/profiles/single_gpu_eval.yaml}"
toolweave_activate_conda

case "${1:-}" in
  --dry-run) mode=() ;;
  --execute|"") mode=(--execute) ;;
  *) echo "usage: $0 [--dry-run|--execute]" >&2; exit 2 ;;
esac

exec "$TOOLWEAVE_PYTHON" -m stage1_format_rl.infrastructure.cli \
  --profile "$TOOLWEAVE_PROFILE" evaluation-server --backend sglang "${mode[@]}"
