#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../stage1_format_rl/scripts" && pwd -P)/_machine.sh"
LOG="$TOOLWEAVE_LOGS_ROOT/modelscope_qwen3_4b_download.log"
TEMP="$TOOLWEAVE_CACHE_ROOT/downloads/modelscope-Qwen3-4B"
mkdir -p "$TOOLWEAVE_LOGS_ROOT" "$(dirname "$TEMP")" "$TOOLWEAVE_MODELS_ROOT/Qwen3-4B"
exec > >(tee -a "$LOG") 2>&1

echo "=== ModelScope Qwen3-4B download started: $(date -Is) ==="
if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi
if [[ -n "${http_proxy:-}" ]]; then export HTTP_PROXY="$http_proxy"; fi
if [[ -n "${https_proxy:-}" ]]; then export HTTPS_PROXY="$https_proxy"; fi
if [[ -n "${HTTP_PROXY:-}" ]]; then export http_proxy="$HTTP_PROXY"; fi
if [[ -n "${HTTPS_PROXY:-}" ]]; then export https_proxy="$HTTPS_PROXY"; fi
echo "Proxy variables in download child:"
env | grep -Ei '^(http|https|all)_proxy=' || true

toolweave_activate_conda
mkdir -p "$TEMP"

modelscope download Qwen/Qwen3-4B \
  --local-dir "$TEMP" \
  --max-workers 8

echo "=== ModelScope download finished: $(date -Is) ==="
du -sh "$TEMP"
find "$TEMP" -type f \( -name '*.incomplete' -o -name '*.lock' -o -name '*.tmp' \) -print || true
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY || true
