#!/usr/bin/env python3
"""Low-overhead 30-second GPU/host/queue telemetry for Generator replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def gpu_sample() -> dict[str, float | str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(command, text=True, timeout=5).strip().split(", ")
        return {
            "name": raw[0],
            "memory_used_mib": float(raw[1]),
            "memory_total_mib": float(raw[2]),
            "utilization_percent": float(raw[3]),
            "power_w": float(raw[4]),
            "temperature_c": float(raw[5]),
            "sm_clock_mhz": float(raw[6]),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def host_sample() -> dict[str, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0])
    load1, load5, load15 = os.getloadavg()
    return {
        "memory_used_gib": (values["MemTotal"] - values["MemAvailable"]) / 1024**2,
        "memory_available_gib": values["MemAvailable"] / 1024**2,
        "load1": load1,
        "load5": load5,
        "load15": load15,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    output = root / "00_manifest/generator_telemetry.jsonl"
    stop = root / "00_manifest/STOP_TELEMETRY"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        while not stop.exists():
            terminal = root / "03_replay/runtime/terminal_results.jsonl"
            candidates = root / "05_validated/validated_candidates_all.jsonl"
            seeds = root / "03_replay/boundary_seeds_replayed.jsonl"
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gpu": gpu_sample(),
                "host": host_sample(),
                "queue": {
                    "dispatched": line_count(seeds),
                    "completed_terminal": line_count(terminal),
                    "validated_unique_content": line_count(candidates),
                    "pending": max(0, line_count(seeds) - line_count(terminal)),
                },
                "operational_limits": {"cpu_cores": 25, "ram_gb": 120},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
