#!/usr/bin/env bash
# Thin Generator server launcher; GPU role and vLLM settings come from profile.
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

PROFILE="${TOOLWEAVE_STAGE3_ONLINE_PROFILE:-$SOURCE_ROOT/stage1_format_rl/configs/layers/profiles/stage3_online_2gpu.yaml}"
ARGS=("$TOOLWEAVE_PYTHON" -m stage1_format_rl.infrastructure.cli --profile "$PROFILE" generator-server)
if [[ "${1:---dry-run}" == "--execute" ]]; then
    ARGS+=(--execute)
elif [[ "${1:---dry-run}" != "--dry-run" ]]; then
    echo "usage: $0 [--dry-run|--execute]" >&2
    exit 2
fi
exec "${ARGS[@]}"
