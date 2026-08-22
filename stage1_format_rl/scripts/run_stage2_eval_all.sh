#!/usr/bin/env bash
# Stage-2 final comparison evals: each model x eval_400 (val_400_combined.parquet),
# all with the identical Stage-2 config (same wrapper/env/decoding/lengths).
# Serial on 2 GPUs; each run is skipped if its SUCCESS marker exists.
set -Eeuo pipefail
WORKSPACE="/root/autodl-tmp/rods-workspace"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
EVAL_SCRIPT="$STAGE_ROOT/scripts/eval_stage2_gate_dual_gpu.sh"
EVAL_ROOT="$STAGE_ROOT/artifacts/stage2_eval"
LOG_DIR="$STAGE_ROOT/logs/stage2_eval"

BASE_MODEL="$WORKSPACE/models/Qwen3-4B"
STAGE1_MODEL="$STAGE_ROOT/artifacts/gate_vs_base/merged/global_step_25"
STAGE2_STEP25_MODEL="$STAGE_ROOT/artifacts/stage2_eval/merged/global_step_25"
# Stage-2 update-20 weights were deleted by the trainer (max_actor_ckpt_to_keep=1);
# set STAGE2_STEP20_MODEL here if a backup appears.
STAGE2_STEP20_MODEL="${STAGE2_STEP20_MODEL:-}"

VAL_400="$STAGE_ROOT/data/checkpoint_gate_eval/val_400_combined.parquet"

mkdir -p "$LOG_DIR"

run_one() {
    local model="$1" label="$2"
    if [[ -e "$EVAL_ROOT/runs/$label/SUCCESS" ]]; then
        echo "== SKIP (done) $label =="
        return 0
    fi
    echo "== EVAL $label start $(date -Is) =="
    ALLOW_STAGE2_GATE_EVAL=1 bash "$EVAL_SCRIPT" "$model" "$label" "$VAL_400" \
        > "$LOG_DIR/${label}.driver.log" 2>&1
    local status=$?
    source /root/miniconda3/etc/profile.d/conda.sh; conda activate rods
    ray stop --force >/dev/null 2>&1 || true
    if [[ "$status" -ne 0 ]]; then
        echo "== EVAL $label FAILED status=$status =="; return 1
    fi
    echo "== EVAL $label done $(date -Is) =="
}

run_one "$BASE_MODEL"      base_qwen3_4b_s2
run_one "$STAGE1_MODEL"    stage1_step25_s2
run_one "$STAGE2_STEP25_MODEL" stage2_step25_s2
if [[ -n "$STAGE2_STEP20_MODEL" ]]; then
    run_one "$STAGE2_STEP20_MODEL" stage2_step20_s2
fi

echo "ALL_STAGE2_EVALS_COMPLETE $(date -Is)"
