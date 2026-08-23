#!/usr/bin/env python3
"""One-second GPU/host telemetry for the standalone BFCL100 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from stage1_format_rl.infrastructure.cli import selected_role_physical_gpu
from stage1_format_rl.infrastructure.resolver import resolve_profile


DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/layers/profiles/single_gpu_eval.yaml"
)


def gpu_snapshot(selected_physical_gpu: int) -> dict[str, float | str]:
    fields = (
        "name,memory.used,memory.total,utilization.gpu,power.draw,"
        "temperature.gpu,clocks.sm"
    )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={selected_physical_gpu}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().split(", ")
    return {
        "name": output[0],
        "memory_used_mib": float(output[1]),
        "memory_total_mib": float(output[2]),
        "utilization_percent": float(output[3]),
        "power_w": float(output[4]),
        "temperature_c": float(output[5]),
        "sm_clock_mhz": float(output[6]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(
            os.environ.get("TOOLWEAVE_SINGLE_GPU_PROFILE", DEFAULT_PROFILE)
        ),
    )
    args = parser.parse_args()
    selected_gpu = selected_role_physical_gpu(resolve_profile(args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        while not args.stop_file.exists():
            memory = psutil.virtual_memory()
            load = os.getloadavg()
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gpu": gpu_snapshot(selected_gpu),
                "host": {
                    "load1": load[0],
                    "load5": load[1],
                    "load15": load[2],
                    "memory_used_gib": (memory.total - memory.available) / 2**30,
                    "memory_available_gib": memory.available / 2**30,
                },
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            time.sleep(args.interval)
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
