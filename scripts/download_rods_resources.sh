#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../stage1_format_rl/scripts" && pwd -P)/_machine.sh"
LOG_DIR="$TOOLWEAVE_LOGS_ROOT"
DOWNLOAD_DIR="$TOOLWEAVE_CACHE_ROOT/downloads"
mkdir -p "$LOG_DIR" "$DOWNLOAD_DIR" "$WORKSPACE/code" "$TOOLWEAVE_SHARED_DATA_ROOT" "$TOOLWEAVE_MODELS_ROOT" "$WORKSPACE/scripts" "$TOOLWEAVE_REPORTS_ROOT"

exec > >(tee -a "$LOG_DIR/resource_download.log") 2>&1

echo "=== Download started: $(date -Is) ==="
echo "WORKSPACE=$WORKSPACE"

# Academic acceleration is intentionally scoped to this detached child only.
if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
    echo "Loaded /etc/network_turbo inside download subprocess"
else
    echo "WARNING: /etc/network_turbo not found"
fi
# /etc/network_turbo may only set lowercase names.  Synchronize all names
# inside this child so inherited proxy values cannot override the accelerator.
if [[ -n "${http_proxy:-}" ]]; then export HTTP_PROXY="$http_proxy"; fi
if [[ -n "${https_proxy:-}" ]]; then export HTTPS_PROXY="$https_proxy"; fi
if [[ -n "${all_proxy:-}" ]]; then export ALL_PROXY="$all_proxy"; fi
if [[ -n "${HTTP_PROXY:-}" ]]; then export http_proxy="$HTTP_PROXY"; fi
if [[ -n "${HTTPS_PROXY:-}" ]]; then export https_proxy="$HTTPS_PROXY"; fi
if [[ -n "${ALL_PROXY:-}" ]]; then export all_proxy="$ALL_PROXY"; fi
echo "Proxy variables inside download subprocess:"
env | grep -Ei '^(http|https|all)_proxy=' || true

echo "=== Repository refresh (fast-forward only) ==="
if [[ -d "$WORKSPACE/code/AWorld-RL/.git" ]]; then
    git -C "$WORKSPACE/code/AWorld-RL" pull --ff-only || echo "WARNING: AWorld-RL fast-forward update failed; retaining existing checkout"
    git -C "$WORKSPACE/code/AWorld-RL" submodule update --init --recursive || echo "WARNING: AWorld-RL submodule update failed"
else
    git clone --recurse-submodules https://github.com/inclusionAI/AWorld-RL.git "$WORKSPACE/code/AWorld-RL"
fi
if [[ -d "$WORKSPACE/code/gorilla/.git" ]]; then
    git -C "$WORKSPACE/code/gorilla" pull --ff-only || echo "WARNING: gorilla fast-forward update failed; retaining existing checkout"
    git -C "$WORKSPACE/code/gorilla" submodule update --init --recursive || echo "WARNING: gorilla submodule update failed"
else
    git clone --recurse-submodules https://github.com/ShishirPatil/gorilla.git "$WORKSPACE/code/gorilla"
fi

echo "=== Hugging Face client ==="
toolweave_activate_conda
python -m pip install -U huggingface_hub hf_transfer

HF_CMD=""
if command -v hf >/dev/null 2>&1; then
    HF_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD="huggingface-cli"
else
    echo "ERROR: neither hf nor huggingface-cli is available" >&2
    exit 1
fi

BFCL_DIR="$TOOLWEAVE_SHARED_DATA_ROOT/Berkeley-Function-Calling-Leaderboard"
MODEL_DIR="$TOOLWEAVE_MODELS_ROOT/Qwen3-4B"
mkdir -p "$BFCL_DIR" "$MODEL_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
# The current unauthenticated route returns 401 for Xet CAS.  Force the
# standard Hub file endpoint so the academic proxy can resume safetensors.
export HF_HUB_DISABLE_XET=1

echo "=== BFCL full snapshot download/resume ==="
if [[ "$HF_CMD" == "hf" ]]; then
    hf download gorilla-llm/Berkeley-Function-Calling-Leaderboard --repo-type dataset --local-dir "$BFCL_DIR" || {
        echo "WARNING: BFCL download with academic proxy failed; unsetting child proxy and retrying with HF mirror"
        unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
        export HF_ENDPOINT=https://hf-mirror.com
        hf download gorilla-llm/Berkeley-Function-Calling-Leaderboard --repo-type dataset --local-dir "$BFCL_DIR"
    }
else
    huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard --repo-type dataset --local-dir "$BFCL_DIR" || {
        echo "WARNING: BFCL download with academic proxy failed; unsetting child proxy and retrying with HF mirror"
        unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY ALL_PROXY all_proxy
        export HF_ENDPOINT=https://hf-mirror.com
        huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard --repo-type dataset --local-dir "$BFCL_DIR"
    }
fi

echo "=== Qwen3-4B full Transformers snapshot download/resume ==="
unset HF_ENDPOINT || true
export HF_HUB_ENABLE_HF_TRANSFER=1
if [[ "$HF_CMD" == "hf" ]]; then
    hf download Qwen/Qwen3-4B --local-dir "$MODEL_DIR" || {
        echo "WARNING: Qwen3-4B download with academic proxy failed; unsetting child proxy and retrying with HF mirror"
        unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
        export HF_ENDPOINT=https://hf-mirror.com
        hf download Qwen/Qwen3-4B --local-dir "$MODEL_DIR"
    }
else
    huggingface-cli download Qwen/Qwen3-4B --local-dir "$MODEL_DIR" || {
        echo "WARNING: Qwen3-4B download with academic proxy failed; unsetting child proxy and retrying with HF mirror"
        unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
        export HF_ENDPOINT=https://hf-mirror.com
        huggingface-cli download Qwen/Qwen3-4B --local-dir "$MODEL_DIR"
    }
fi

echo "=== Download tree summary ==="
du -sh "$BFCL_DIR" "$MODEL_DIR" || true
find "$MODEL_DIR" -type f \( -name '*.incomplete' -o -name '*.lock' -o -name '*.tmp' \) -print || true
echo "=== Download finished: $(date -Is) ==="

# Explicitly clear only this child process's temporary proxy variables.
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY HF_ENDPOINT || true
