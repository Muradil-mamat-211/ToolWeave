#!/usr/bin/env python3
"""Print compact health metrics for the active Stage 1 recovery run."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import fmean

from machine_paths import project_roots

ROOTS = project_roots()
LOG_ROOT = ROOTS.short_temp_root / "stage1-recovery/ray/ray/session_latest/logs"
STATUS = ROOTS.logs_root / "recovery_from_step25/status.txt"
ARCHIVE_STATUS = (
    ROOTS.logs_root
    / "recovery_from_step25/weight_checkpoint_archive_status.txt"
)
STEP_RE = re.compile(r"^step:(\d+)\s+-\s+(.*)$")


def metric_file() -> Path:
    for path in sorted(LOG_ROOT.glob("worker*.out")):
        if "step:" in path.read_text(encoding="utf-8", errors="replace"):
            return path
    raise FileNotFoundError(f"no metric worker log found below {LOG_ROOT}")


def rows(path: Path) -> list[dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP_RE.match(line)
        if not match:
            continue
        step = int(match.group(1))
        row: dict[str, float] = {"step": float(step)}
        for field in match.group(2).split(" - "):
            if ":" not in field:
                continue
            key, raw_value = field.rsplit(":", 1)
            try:
                row[key] = float(raw_value)
            except ValueError:
                pass
        result[step] = row
    return [result[step] for step in sorted(result)]


def average(items: list[dict[str, float]], key: str) -> float | None:
    values = [item[key] for item in items if key in item and math.isfinite(item[key])]
    return fmean(values) if values else None


def main() -> None:
    path = metric_file()
    parsed = rows(path)
    latest = parsed[-1] if parsed else {}
    recent = parsed[-5:]
    zero_tail = 0
    for item in reversed(parsed):
        if item.get("critic/score/max") == 0.0:
            zero_tail += 1
        else:
            break
    payload = {
        "status": STATUS.read_text(encoding="utf-8").strip() if STATUS.exists() else "MISSING",
        "archive_status": (
            ARCHIVE_STATUS.read_text(encoding="utf-8").strip()
            if ARCHIVE_STATUS.exists()
            else "MISSING"
        ),
        "metric_log": str(path),
        "completed_updates": int(latest.get("step", 0)),
        "latest": {
            "score_mean": latest.get("critic/score/mean"),
            "score_max": latest.get("critic/score/max"),
            "kl_loss": latest.get("actor/kl_loss"),
            "pg_loss": latest.get("actor/pg_loss"),
            "entropy": latest.get("actor/entropy"),
            "grad_norm": latest.get("actor/grad_norm"),
            "response_mean": latest.get("response_length/mean"),
            "response_max": latest.get("response_length/max"),
            "response_clip_ratio": latest.get("response_length/clip_ratio"),
            "step_seconds": latest.get("timing_s/step"),
        },
        "last_5_mean": {
            "score": average(recent, "critic/score/mean"),
            "kl_loss": average(recent, "actor/kl_loss"),
            "response_length": average(recent, "response_length/mean"),
            "response_clip_ratio": average(recent, "response_length/clip_ratio"),
            "step_seconds": average(recent, "timing_s/step"),
        },
        "consecutive_zero_reward_updates": zero_tail,
        "health_alert": zero_tail >= 3,
    }
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
