#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_DIR="/root/autodl-tmp/rods-workspace/models/gemma-4-31B-it"
LOG_FILE="/root/autodl-tmp/rods-workspace/logs/gemma4_31b_download.log"

# Preserve the proxy inherited by the detached tmux child.  If the AutoDL
# academic proxy is slow, restore this child-only proxy before HF transfer.
ORIG_http_proxy="${http_proxy-}"
ORIG_https_proxy="${https_proxy-}"
ORIG_HTTP_PROXY="${HTTP_PROXY-}"
ORIG_HTTPS_PROXY="${HTTPS_PROXY-}"
ORIG_ALL_PROXY="${ALL_PROXY-}"
ORIG_all_proxy="${all_proxy-}"

mkdir -p "$MODEL_DIR" "$(dirname "$LOG_FILE")"

exec > >(tee -a "$LOG_FILE") 2>&1

if [[ -f /etc/network_turbo ]]; then
    # Academic acceleration is scoped to this detached download child.
    source /etc/network_turbo
    # Do not allow inherited Codex proxy variables to override the academic
    # proxy inside this download child.
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
fi

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate rods
fi

python -m pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
# Use the regular HTTP downloader with one worker to avoid the previous
# downloader subprocess being killed; the model source and files are unchanged.
export HF_HUB_DISABLE_XET=1

# Remove only incomplete artifacts from earlier interrupted attempts.  The
# metadata and tokenizer files remain untouched.
rm -f "$MODEL_DIR"/model-*.safetensors
find "$MODEL_DIR/.cache/huggingface/download" -type f -name '*.incomplete' -delete 2>/dev/null || true

# Fixed byte ranges make retries idempotent: a failed request can never
# truncate an already completed portion of an official safetensors shard.
CHUNK_SIZE=67108864
PARALLEL=4
PART_ROOT="$MODEL_DIR/.download_parts"
mkdir -p "$PART_ROOT"

download_range() {
    local name="$1" total="$2" index="$3"
    local start=$((index * CHUNK_SIZE))
    local end=$((start + CHUNK_SIZE - 1))
    if (( end >= total )); then end=$((total - 1)); fi
    local expected=$((end - start + 1))
    local part_dir="$PART_ROOT/$name"
    local part_file="$part_dir/$(printf '%06d.part' "$index")"
    local url="https://huggingface.co/google/gemma-4-31B-it/resolve/main/$name?download=true"
    mkdir -p "$part_dir"

    while :; do
        if [[ -f "$part_file" ]] && [[ "$(stat -c '%s' "$part_file")" == "$expected" ]]; then
            return 0
        fi
        rm -f "$part_file"
        if curl -fsSL --retry 50 --retry-all-errors --retry-delay 3 \
            --connect-timeout 30 --max-time 180 \
            -H "Range: bytes=$start-$end" -H 'Accept-Encoding: identity' \
            -o "$part_file" "$url" >> "$LOG_FILE" 2>&1; then
            if [[ "$(stat -c '%s' "$part_file" 2>/dev/null || echo 0)" == "$expected" ]]; then
                printf 'RANGE_OK %s %s-%s\n' "$name" "$start" "$end" >> "$LOG_FILE"
                return 0
            fi
        fi
        rm -f "$part_file"
        sleep 3
    done
}

download_shard() {
    local name="$1" total="$2"
    local count=$(( (total + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    local running=0
    local pids=()
    for index in $(seq 0 $((count - 1))); do
        download_range "$name" "$total" "$index" &
        pids+=("$!")
        running=$((running + 1))
        if (( running >= PARALLEL )); then
            wait "${pids[@]}"
            pids=()
            running=0
        fi
    done
    if (( running > 0 )); then wait "${pids[@]}"; fi

    local output="$MODEL_DIR/$name"
    local tmp="$output.assembling"
    rm -f "$tmp"
    : > "$tmp"
    for index in $(seq 0 $((count - 1))); do
        cat "$PART_ROOT/$name/$(printf '%06d.part' "$index")" >> "$tmp"
        rm -f "$PART_ROOT/$name/$(printf '%06d.part' "$index")"
    done
    mv "$tmp" "$output"
    rmdir "$PART_ROOT/$name"
}

download_shard model-00001-of-00002.safetensors 49784788364 &
PID_1=$!
download_shard model-00002-of-00002.safetensors 12761389388 &
PID_2=$!
wait "$PID_1" "$PID_2"
rmdir "$PART_ROOT"

python - <<'PY'
from pathlib import Path
import json

path = Path("/root/autodl-tmp/rods-workspace/models/gemma-4-31B-it")

assert (path / "config.json").exists()
assert (path / "tokenizer_config.json").exists()

index_file = path / "model.safetensors.index.json"
single_file = path / "model.safetensors"

if index_file.exists():
    index = json.loads(index_file.read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (path / name).exists()]
    assert not missing, f"Missing shards: {missing}"
    expected = {
        "model-00001-of-00002.safetensors": 49784788364,
        "model-00002-of-00002.safetensors": 12761389388,
    }
    for name, size in expected.items():
        actual = (path / name).stat().st_size
        assert actual == size, f"Unexpected size for {name}: {actual} != {size}"
elif not single_file.exists():
    raise RuntimeError("No complete safetensors weights found")

incomplete = list(path.rglob("*.incomplete"))
assert not incomplete, f"Incomplete files: {incomplete}"

print("DOWNLOAD_VERIFIED")
PY

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY || true
