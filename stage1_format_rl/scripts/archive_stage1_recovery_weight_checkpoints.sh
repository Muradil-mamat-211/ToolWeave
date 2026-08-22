#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"
toolweave_apply_topology learner
OLD_ARCHIVE="$TOOLWEAVE_OUTPUTS_ROOT/stage1_format_qwen3_4b/weight_checkpoints"
OUTPUT_ROOT="$TOOLWEAVE_OUTPUTS_ROOT/stage1_format_qwen3_4b_recovery_from_step25"
SOURCE_ROOT="$OUTPUT_ROOT/checkpoints"
ARCHIVE_ROOT="$OUTPUT_ROOT/weight_checkpoints_repaired"
LOG_DIR="$TOOLWEAVE_LOGS_ROOT/recovery_from_step25"
LOG_FILE="$LOG_DIR/weight_checkpoint_archive.log"
STATUS_FILE="$LOG_DIR/weight_checkpoint_archive_status.txt"

mkdir -p "$ARCHIVE_ROOT" "$LOG_DIR"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

seed_epoch_one() {
    local source="$OLD_ARCHIVE/global_step_25"
    local target="$ARCHIVE_ROOT/global_step_25"
    local staging="$ARCHIVE_ROOT/.global_step_25.staging"
    [[ -d "$target" ]] && return 0
    [[ -d "$source/actor" ]] || { echo "Missing epoch-1 checkpoint: $source"; exit 10; }
    toolweave_safe_rm_rf "$staging"
    cp -al "$source" "$staging"
    mv "$staging" "$target"
    echo "SEEDED logical_step=25 source=$source at=$(date -Is)"
}

archive_step() {
    local run_step="$1"
    local logical_step="$2"
    local source="$SOURCE_ROOT/global_step_$run_step/actor"
    local target="$ARCHIVE_ROOT/global_step_$logical_step"
    local staging="$ARCHIVE_ROOT/.global_step_${logical_step}.staging"
    local -a shards

    [[ -d "$target" ]] && return 0
    [[ -d "$source/huggingface" && -f "$source/fsdp_config.json" ]] || return 1
    mapfile -t shards < <(find "$source" -maxdepth 1 -type f -name "model_world_size_${TOOLWEAVE_LEARNER_WORLD_SIZE}_rank_*.pt" | sort)
    [[ "${#shards[@]}" -eq "$TOOLWEAVE_LEARNER_WORLD_SIZE" ]] || return 1
    local shard
    for shard in "${shards[@]}"; do [[ -s "$shard" ]] || return 1; done

    toolweave_safe_rm_rf "$staging"
    mkdir -p "$staging/actor"
    cp -al "${shards[@]}" "$staging/actor/"
    cp -al "$source/huggingface" "$staging/actor/"
    cp -al "$source/fsdp_config.json" "$staging/actor/"

    RUN_STEP="$run_step" LOGICAL_STEP="$logical_step" TARGET="$staging" SOURCE="$source" WORLD_SIZE="$TOOLWEAVE_LEARNER_WORLD_SIZE" "$TOOLWEAVE_PYTHON" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

run_step = int(os.environ["RUN_STEP"])
logical_step = int(os.environ["LOGICAL_STEP"])
target = Path(os.environ["TARGET"])
actor = target / "actor"
world_size = int(os.environ["WORLD_SIZE"])
shards = sorted(path.name for path in actor.glob(f"model_world_size_{world_size}_rank_*.pt"))
forbidden = [
    str(path)
    for path in actor.rglob("*")
    if path.is_file()
    and (path.name.startswith("optim_") or path.name.startswith("extra_state_") or path.name == "data.pt")
]
if len(shards) != world_size or forbidden:
    raise RuntimeError(f"invalid archive: shards={shards}, forbidden={forbidden}")
payload = {
    "stage": "stage1_format_rl_recovery",
    "logical_global_step": logical_step,
    "logical_epoch": logical_step // 25,
    "recovery_run_step": run_step,
    "base_checkpoint": str(Path(os.environ["TOOLWEAVE_OUTPUTS_ROOT"]) / "stage1_format_qwen3_4b/weight_checkpoints/global_step_25"),
    "model_start_path": str(Path(os.environ["TOOLWEAVE_ARTIFACTS_ROOT"]) / "checkpoint_eval/merged/global_step_25"),
    "source_actor_checkpoint": os.environ["SOURCE"],
    "checkpoint_type": "model_weights_only_fsdp_sharded",
    "world_size": world_size,
    "model_shards": shards,
    "optimizer_state_saved": False,
    "created_at": datetime.now(timezone.utc).isoformat(),
}
(target / "checkpoint_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

    mv "$staging" "$target"
    echo "ARCHIVED run_step=$run_step logical_step=$logical_step size=$(du -sh "$target" | awk '{print $1}') at=$(date -Is)"
}

seed_epoch_one
echo "RUNNING archived=1/5 started=$(date -Is)" > "$STATUS_FILE"

while true; do
    archived=1
    for mapping in 25:50 50:75 75:100 100:125; do
        run_step="${mapping%%:*}"
        logical_step="${mapping##*:}"
        target="$ARCHIVE_ROOT/global_step_$logical_step"
        if [[ -d "$target" ]]; then
            archived=$((archived + 1))
            continue
        fi
        tracker="$SOURCE_ROOT/latest_checkpointed_iteration.txt"
        if [[ -s "$tracker" ]] && [[ "$(tr -dc '0-9' < "$tracker")" == "$run_step" ]]; then
            archive_step "$run_step" "$logical_step" || true
        fi
        [[ -d "$target" ]] && archived=$((archived + 1))
    done
    echo "RUNNING archived=$archived/5 checked=$(date -Is)" > "$STATUS_FILE"
    if [[ "$archived" -eq 5 ]]; then
        echo "COMPLETED archived=5/5 finished=$(date -Is)" > "$STATUS_FILE"
        exit 0
    fi
    sleep 5
done
