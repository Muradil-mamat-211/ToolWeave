#!/usr/bin/env bash
# Resolve and launch the Generator daemon without embedding paths or GPU IDs.
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
exec "$TOOLWEAVE_PYTHON" -m stage1_format_rl.infrastructure.cli \
    --profile "$PROFILE" generator-daemon "$@"
