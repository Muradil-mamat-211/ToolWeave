#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="/root/autodl-tmp/rods-workspace"
STAGE_ROOT="$WORKSPACE/stage1_format_rl"
OUTPUT_ROOT="$WORKSPACE/outputs/stage1_format_qwen3_4b_recovery_from_step25"
CHECKPOINT="$OUTPUT_ROOT/checkpoints/global_step_50"
ARCHIVE="$OUTPUT_ROOT/weight_checkpoints_repaired/global_step_75"
LATEST="$OUTPUT_ROOT/checkpoints/latest_checkpointed_iteration.txt"
LOG_DIR="$STAGE_ROOT/logs/checkpoint_gate_eval"
LOG_FILE="$LOG_DIR/gate_boundary_pause.log"
STATUS_FILE="$LOG_DIR/gate_boundary_pause.status"

mkdir -p "$LOG_DIR"
exec >> "$LOG_FILE" 2>&1
echo "WATCHING $(date -Is)"
echo "WATCHING" > "$STATUS_FILE"

checkpoint_complete() {
    [[ -f "$LATEST" ]] || return 1
    [[ "$(tr -dc '0-9' < "$LATEST")" == "50" ]] || return 1
    [[ -s "$CHECKPOINT/data.pt" ]] || return 1
    for rank in 0 1; do
        [[ -s "$CHECKPOINT/actor/model_world_size_2_rank_${rank}.pt" ]] || return 1
        [[ -s "$CHECKPOINT/actor/optim_world_size_2_rank_${rank}.pt" ]] || return 1
        [[ -s "$CHECKPOINT/actor/extra_state_world_size_2_rank_${rank}.pt" ]] || return 1
    done
}

archive_complete() {
    [[ -s "$ARCHIVE/checkpoint_manifest.json" ]] || return 1
    for rank in 0 1; do
        [[ -s "$ARCHIVE/actor/model_world_size_2_rank_${rank}.pt" ]] || return 1
        local source_stat archive_stat
        source_stat="$(stat -c '%d:%i:%s' "$CHECKPOINT/actor/model_world_size_2_rank_${rank}.pt")"
        archive_stat="$(stat -c '%d:%i:%s' "$ARCHIVE/actor/model_world_size_2_rank_${rank}.pt")"
        [[ "$source_stat" == "$archive_stat" ]] || return 1
    done
}

while tmux has-session -t '=rods-stage1-recovery' 2>/dev/null; do
    if checkpoint_complete && archive_complete; then
        echo "VERIFIED checkpoint=$CHECKPOINT archive=$ARCHIVE at=$(date -Is)"
        echo "VERIFIED_PAUSING" > "$STATUS_FILE"
        tmux send-keys -t rods-stage1-recovery C-c
        for _ in $(seq 1 120); do
            if ! tmux has-session -t '=rods-stage1-recovery' 2>/dev/null; then
                echo "PAUSED $(date -Is)"
                echo "PAUSED_FOR_GATE_EVAL" > "$STATUS_FILE"
                exit 0
            fi
            sleep 5
        done
        echo "ERROR training tmux did not exit after SIGINT" | tee "$STATUS_FILE"
        exit 20
    fi
    sleep 10
done

echo "ERROR training tmux exited before gate checkpoint was verified" | tee "$STATUS_FILE"
exit 21
