#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
toolweave_apply_topology learner
OUTPUT_ROOT="$TOOLWEAVE_OUTPUTS_ROOT/stage1_format_qwen3_4b"
SOURCE_ROOT="$OUTPUT_ROOT/checkpoints"
ARCHIVE_ROOT="$OUTPUT_ROOT/weight_checkpoints"
LOG_FILE="$TOOLWEAVE_LOGS_ROOT/formal_5epoch/stage1_weight_checkpoint_archive.log"
STATUS_FILE="$TOOLWEAVE_LOGS_ROOT/formal_5epoch/stage1_weight_checkpoint_archive_status.txt"
EXPECTED_STEPS=(25 50 75 100 125)

mkdir -p "$ARCHIVE_ROOT" "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

archive_step() {
    local step="$1"
    local source="$SOURCE_ROOT/global_step_$step/actor"
    local target="$ARCHIVE_ROOT/global_step_$step"
    local staging="$ARCHIVE_ROOT/.global_step_${step}.staging"
    local -a shards

    [[ ! -e "$target" ]] || return 0
    [[ -d "$source/huggingface" && -f "$source/fsdp_config.json" ]] || return 1
    mapfile -t shards < <(find "$source" -maxdepth 1 -type f \
        -name "model_world_size_${TOOLWEAVE_LEARNER_WORLD_SIZE}_rank_*.pt" | sort)
    [[ "${#shards[@]}" -eq "$TOOLWEAVE_LEARNER_WORLD_SIZE" ]] || return 1
    local shard
    for shard in "${shards[@]}"; do [[ -s "$shard" ]] || return 1; done

    toolweave_safe_rm_rf "$staging"
    mkdir -p "$staging/actor"
    cp -al "${shards[@]}" "$staging/actor/"
    cp -al "$source/huggingface" "$staging/actor/"
    cp -al "$source/fsdp_config.json" "$staging/actor/"

    STEP="$step" TARGET="$staging" SOURCE="$source" WORLD_SIZE="$TOOLWEAVE_LEARNER_WORLD_SIZE" "$TOOLWEAVE_PYTHON" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

step = int(os.environ["STEP"])
target = Path(os.environ["TARGET"])
actor = target / "actor"
world_size = int(os.environ["WORLD_SIZE"])
shards = sorted(path.name for path in actor.glob(f"model_world_size_{world_size}_rank_*.pt"))
forbidden = sorted(
    path.name
    for path in actor.rglob("*")
    if path.is_file()
    and (
        path.name.startswith("optim_")
        or path.name.startswith("extra_state_")
        or path.name == "data.pt"
    )
)
if len(shards) != world_size or forbidden:
    raise RuntimeError(f"Invalid model-only archive: shards={shards}, forbidden={forbidden}")

manifest = {
    "stage": "stage1_format_rl",
    "global_step": step,
    "epoch": step // 25,
    "base_model": str(Path(os.environ["TOOLWEAVE_MODELS_ROOT"]) / "Qwen3-4B"),
    "source_actor_checkpoint": os.environ["SOURCE"],
    "checkpoint_type": "model_weights_only_fsdp_sharded",
    "world_size": world_size,
    "model_shards": shards,
    "optimizer_state_saved": False,
    "scheduler_state_saved": False,
    "rng_state_saved": False,
    "dataloader_state_saved": False,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "merge_command": (
        "python -m verl.model_merger merge --backend fsdp "
        f"--local_dir={target.parent / ('global_step_' + str(step)) / 'actor'} "
        "--target_dir=<TARGET_HF_DIR> --use_cpu_initialization"
    ),
}
(target / "checkpoint_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

    mv "$staging" "$target"
    echo "ARCHIVED step=$step size=$(du -sh "$target" | awk '{print $1}') at=$(date -Is)"
}

echo "RUNNING expected_steps=${EXPECTED_STEPS[*]} started=$(date -Is)" > "$STATUS_FILE"
echo "=== Weight checkpoint archive watcher started: $(date -Is) ==="

while true; do
    archived=0
    for step in "${EXPECTED_STEPS[@]}"; do
        target="$ARCHIVE_ROOT/global_step_$step"
        if [[ -d "$target" ]]; then
            archived=$((archived + 1))
            continue
        fi

        tracker="$SOURCE_ROOT/latest_checkpointed_iteration.txt"
        if [[ -s "$tracker" ]] && [[ "$(tr -dc '0-9' < "$tracker")" == "$step" ]]; then
            archive_step "$step" || true
        fi
        [[ -d "$target" ]] && archived=$((archived + 1))
    done

    echo "RUNNING archived=$archived/5 checked=$(date -Is)" > "$STATUS_FILE"
    if [[ "$archived" -eq 5 ]]; then
        echo "COMPLETED archived=5/5 finished=$(date -Is)" > "$STATUS_FILE"
        echo "=== All model-only checkpoints archived: $(date -Is) ==="
        exit 0
    fi
    sleep 5
done
