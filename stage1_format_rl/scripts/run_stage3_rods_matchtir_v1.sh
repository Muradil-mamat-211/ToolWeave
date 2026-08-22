#!/usr/bin/env bash
# Thin Stage 3 launcher: load machine environment, resolve, preflight, launch.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
if [[ -f "$SOURCE_ROOT/environment/env.local.sh" ]]; then
    # shellcheck source=/dev/null
    source "$SOURCE_ROOT/environment/env.local.sh"
else
    # shellcheck source=/dev/null
    source "$SOURCE_ROOT/environment/env.template.sh"
fi

PROFILE="${TOOLWEAVE_STAGE3_PROFILE:-$SOURCE_ROOT/stage1_format_rl/configs/layers/profiles/stage3_reference.yaml}"
CLI=("$TOOLWEAVE_PYTHON" -m stage1_format_rl.infrastructure.cli --profile "$PROFILE")

case "${1:---verify}" in
    --verify)
        "${CLI[@]}" preflight --check-assets
        PYTHONPATH="$TOOLWEAVE_SOURCE_ROOT/code/AWorld-RL-stage1-worktree/EnvTuning:$TOOLWEAVE_SOURCE_ROOT/code/AWorld-RL-stage1-worktree/EnvTuning/verl" \
          "$TOOLWEAVE_PYTHON" -m pytest -q \
          "$TOOLWEAVE_SOURCE_ROOT/stage1_format_rl/tests/test_rods_matchtir_v1_matching.py" \
          "$TOOLWEAVE_SOURCE_ROOT/stage1_format_rl/tests/test_rods_matchtir_v1_advantage.py" \
          "$TOOLWEAVE_SOURCE_ROOT/stage1_format_rl/tests/test_rods_matchtir_v1_integration.py" \
          "$TOOLWEAVE_SOURCE_ROOT/stage1_format_rl/tests/test_rods_stage3_lifecycle.py" \
          "$TOOLWEAVE_SOURCE_ROOT/stage1_format_rl/tests/test_stage3_rods_matchtir_config.py"
        ;;
    --dry-run) "${CLI[@]}" preflight ;;
    --full)
        if [[ "${ALLOW_RODS_MATCHTIR_STAGE3_TRAINING:-0}" != "1" ]]; then
            echo "Stage 3 formal training is disabled. Set ALLOW_RODS_MATCHTIR_STAGE3_TRAINING=1 only after explicit approval."
            exit 2
        fi
        "${CLI[@]}" launch --execute --observe-hardware
        ;;
    *) echo "usage: $0 [--verify|--dry-run|--full]" >&2; exit 2 ;;
esac
