#!/usr/bin/env bash
# Shared machine-local bootstrap for historical qualification scripts.

_toolweave_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
_toolweave_source_root="$(cd "$_toolweave_script_dir/../.." && pwd -P)"
if [[ -f "$_toolweave_source_root/environment/env.local.sh" ]]; then
    # shellcheck source=/dev/null
    source "$_toolweave_source_root/environment/env.local.sh"
else
    # shellcheck source=/dev/null
    source "$_toolweave_source_root/environment/env.template.sh"
fi
WORKSPACE="$TOOLWEAVE_SOURCE_ROOT"
AWORLD="$TOOLWEAVE_SOURCE_ROOT/code/AWorld-RL-stage1-worktree"
STAGE_ROOT="$TOOLWEAVE_SOURCE_ROOT/stage1_format_rl"
export PYTHONPATH="$TOOLWEAVE_SOURCE_ROOT:$AWORLD/EnvTuning:$AWORLD/EnvTuning/verl${PYTHONPATH:+:$PYTHONPATH}"
export TOOLWEAVE_PROFILE="${TOOLWEAVE_PROFILE:-$STAGE_ROOT/configs/layers/profiles/stage3_reference.yaml}"

toolweave_activate_conda() {
    if [[ -n "${TOOLWEAVE_CONDA_ENV:-}" ]]; then
        if ! command -v conda >/dev/null 2>&1; then
            echo "TOOLWEAVE_CONDA_ENV is set but conda is unavailable" >&2
            return 2
        fi
        eval "$(conda shell.bash hook)"
        conda activate "$TOOLWEAVE_CONDA_ENV"
        export TOOLWEAVE_PYTHON="$(command -v python)"
    fi
}

toolweave_apply_topology() {
    local role="${1:?toolweave_apply_topology requires a runtime role}"
    local exports
    exports="$("$TOOLWEAVE_PYTHON" -m stage1_format_rl.infrastructure.cli \
        --profile "$TOOLWEAVE_PROFILE" shell-env --role "$role")" || return
    eval "$exports"
}

toolweave_safe_rm_rf() {
    local target resolved root resolved_root resolved_home allowed
    resolved_home=""
    if [[ -n "${HOME:-}" ]]; then
        resolved_home="$(realpath -m -- "$HOME")"
    fi
    for target in "$@"; do
        [[ -n "$target" ]] || { echo "refusing to remove an empty path" >&2; return 2; }
        resolved="$(realpath -m -- "$target")"
        allowed=0
        for root in \
            "$TOOLWEAVE_OUTPUTS_ROOT" \
            "$TOOLWEAVE_ARTIFACTS_ROOT" \
            "$TOOLWEAVE_LOGS_ROOT" \
            "$TOOLWEAVE_CACHE_ROOT" \
            "$TOOLWEAVE_TEMP_ROOT" \
            "$TOOLWEAVE_SHORT_TEMP_ROOT"; do
            resolved_root="$(realpath -m -- "$root")"
            if [[ "$resolved_root" == "/" \
                || ( -n "$resolved_home" && "$resolved_root" == "$resolved_home" ) \
                || "$resolved_root" == "$(realpath -m -- "$TOOLWEAVE_SOURCE_ROOT")" \
                || "$resolved_root" == "$(realpath -m -- "$TOOLWEAVE_ASSET_ROOT")" ]]; then
                continue
            fi
            if [[ "$resolved" == "$resolved_root"/* ]]; then
                allowed=1
                break
            fi
        done
        if (( ! allowed )); then
            echo "refusing to remove path outside a configured runtime root: $resolved" >&2
            return 2
        fi
        rm -rf -- "$resolved"
    done
}

unset _toolweave_script_dir _toolweave_source_root
