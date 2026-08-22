#!/usr/bin/env bash
# Stage-1 gate evals (efficient): eval1 base x Base-100 (may already be done) +
# base/step20/step25 x val_400, from which Base-100 is extracted via manifest join.
set -Eeuo pipefail
WORKSPACE="/root/autodl-tmp/rods-workspace"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
EVAL_SCRIPT="$STAGE_ROOT/scripts/eval_stage1_gate_dual_gpu.sh"
METRIC_SCRIPT="$STAGE_ROOT/scripts/compute_stage1_gate_metrics.py"
EVAL_ROOT="$STAGE_ROOT/artifacts/checkpoint_gate_eval"
METRIC_ROOT="$STAGE_ROOT/artifacts/gate_vs_base/metrics"
BASE_MODEL="$WORKSPACE/models/Qwen3-4B"
STEP20_MODEL="$STAGE_ROOT/artifacts/gate_vs_base/merged/global_step_20"
STEP25_MODEL="$STAGE_ROOT/artifacts/gate_vs_base/merged/global_step_25"
VAL_BASE100="$STAGE_ROOT/data/checkpoint_gate_eval/val_base_100.parquet"
VAL_400="$STAGE_ROOT/data/checkpoint_gate_eval/val_400_combined.parquet"
MAN_BASE100="$STAGE_ROOT/data/checkpoint_gate_eval/val_base_100.manifest.json"
MAN_400="$STAGE_ROOT/data/checkpoint_gate_eval/val_400_combined.manifest.json"

mkdir -p "$METRIC_ROOT"

run_one() {
    local model="$1" label="$2" val="$3" manifest="$4" dataset="$5"
    if [[ -e "$EVAL_ROOT/runs/$label/SUCCESS" ]]; then
        echo "== SKIP (done) $label =="
    else
        echo "== EVAL $label start $(date -Is) =="
        ALLOW_STAGE1_CHECKPOINT_EVAL=1 bash "$EVAL_SCRIPT" "$model" "$label" "$val" \
            > "$STAGE_ROOT/logs/checkpoint_gate_eval/${label}.driver.log" 2>&1
        local status=$?
        source /root/miniconda3/etc/profile.d/conda.sh; conda activate rods
        ray stop --force >/dev/null 2>&1 || true
        if [[ "$status" -ne 0 ]]; then
            echo "== EVAL $label FAILED status=$status =="; return 1
        fi
    fi
    echo "== METRIC $label =="
    /root/miniconda3/envs/rods/bin/python "$METRIC_SCRIPT" \
        --rollout-dir "$EVAL_ROOT/runs/$label/rollouts" \
        --manifest "$manifest" --model "$label" --dataset "$dataset" \
        --out-json "$METRIC_ROOT/${label}.json"
}

# eval1: base x Base-100 (may already be running/done)
run_one "$BASE_MODEL" base_qwen3_4b_valbase "$VAL_BASE100" "$MAN_BASE100" "base_100"
# val_400 for all three models (Base-100 extracted from these via manifest)
run_one "$BASE_MODEL"  base_qwen3_4b_val400 "$VAL_400" "$MAN_400" "base_400"
run_one "$STEP20_MODEL" step20_val400 "$VAL_400" "$MAN_400" "step20_400"
run_one "$STEP25_MODEL" step25_val400 "$VAL_400" "$MAN_400" "step25_400"

echo "ALL_GATE_EVALS_COMPLETE $(date -Is)"
