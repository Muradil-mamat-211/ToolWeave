#!/usr/bin/env bash
set -u

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_machine.sh"

ROOT_PID="${1:-858226}"
LOG_DIR="$TOOLWEAVE_LOGS_ROOT/checkpoint_gate_eval"
LOG_FILE="$LOG_DIR/monitor_6min_tmux.log"

mkdir -p "$LOG_DIR"

while kill -0 "$ROOT_PID" 2>/dev/null; do
    ACTIVE_LOG="$(ls -1t "$LOG_DIR"/*.driver.log 2>/dev/null | head -n 1 || true)"
    {
        date -Is
        echo "active_log=$ACTIVE_LOG"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
        if [[ -n "$ACTIVE_LOG" ]]; then
            rg -o "len reward_extra_infos_dict\\['score'\\]: [0-9]+" "$ACTIVE_LOG" | tail -n 1 || true
            rg -n "Traceback|CUDA out of memory|OutOfMemory|NaN|Error executing task|DONE wave|START wave|GATE_EVAL_COMPLETED" "$ACTIVE_LOG" | tail -n 5 || true
            stat -c "log_size=%s log_mtime=%y" "$ACTIVE_LOG"
        fi
        echo
    } >> "$LOG_FILE" 2>&1
    sleep 360
done

echo "$(date -Is) EVAL_ROOT_PROCESS_EXITED" >> "$LOG_FILE"
