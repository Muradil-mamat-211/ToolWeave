#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${ALLOW_STAGE1_CHECKPOINT_EVAL:-0}" != "1" ]]; then
    echo "Three-checkpoint gate evaluation is disabled."
    exit 2
fi
if tmux has-session -t '=rods-stage1-recovery' 2>/dev/null; then
    echo "Refusing to start evaluation while recovery training tmux is running."
    exit 3
fi

WORKSPACE="/root/autodl-tmp/rods-workspace"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
OUTPUT_ROOT="$WORKSPACE/outputs/stage1_format_qwen3_4b_recovery_from_step25"
FULL_STEP50="$OUTPUT_ROOT/checkpoints/global_step_50"
ARCHIVE_ROOT="$OUTPUT_ROOT/weight_checkpoints_repaired"
EVAL_ROOT="$STAGE_ROOT/artifacts/checkpoint_gate_eval"
MERGED_ROOT="$EVAL_ROOT/merged"
DATA_ROOT="$STAGE_ROOT/data/checkpoint_gate_eval"
LOG_ROOT="$STAGE_ROOT/logs/checkpoint_gate_eval"

OLD_MODEL="$STAGE_ROOT/artifacts/checkpoint_eval/merged/global_step_25"
RECOVERY25_MODEL="$MERGED_ROOT/recovery_run_step25_logical50"
RECOVERY50_MODEL="$MERGED_ROOT/recovery_run_step50_logical75"

test -d "$OLD_MODEL"
test -d "$FULL_STEP50/actor"
test -d "$ARCHIVE_ROOT/global_step_50/actor"
test -d "$ARCHIVE_ROOT/global_step_75/actor"
[[ "$(tr -dc '0-9' < "$OUTPUT_ROOT/checkpoints/latest_checkpointed_iteration.txt")" == "50" ]]
mkdir -p "$MERGED_ROOT" "$LOG_ROOT"

"$STAGE_ROOT/scripts/merge_stage1_gate_checkpoint.sh" \
    "$ARCHIVE_ROOT/global_step_50/actor" "$RECOVERY25_MODEL"
"$STAGE_ROOT/scripts/merge_stage1_gate_checkpoint.sh" \
    "$ARCHIVE_ROOT/global_step_75/actor" "$RECOVERY50_MODEL"

run_model_wave() {
    local model="$1"
    local label="$2"
    echo "START wave=$label at=$(date -Is)"
    ALLOW_STAGE1_CHECKPOINT_EVAL=1 \
        "$STAGE_ROOT/scripts/eval_stage1_gate_dual_gpu.sh" \
        "$model" "$label" "$DATA_ROOT/val_400_combined.parquet" \
        > "$LOG_ROOT/${label}.driver.log" 2>&1
    local status=$?
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate rods
    ray stop --force >/dev/null 2>&1 || true
    if [[ "$status" -ne 0 ]]; then
        echo "FAILED wave=$label status=$status"
        return 1
    fi
    echo "DONE wave=$label at=$(date -Is)"
}

run_model_wave "$OLD_MODEL" old_run_step25
run_model_wave "$RECOVERY25_MODEL" recovery_run_step25_logical50
run_model_wave "$RECOVERY50_MODEL" recovery_run_step50_logical75

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rods
python "$STAGE_ROOT/scripts/summarize_stage1_gate_eval.py" \
    --eval-root "$EVAL_ROOT" \
    --manifest-dir "$DATA_ROOT"

echo "GATE_EVAL_COMPLETED $(date -Is)"
