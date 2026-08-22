#!/usr/bin/env python3
"""One-second GPU, cgroup-system, and queue telemetry for the Stage-3 smoke."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


STOP = False


def _stop(*_: object) -> None:
    global STOP
    STOP = True


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return records


def _read_cgroup_number(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return None if raw == "max" else int(raw)
    except (OSError, ValueError):
        return None


def _read_cgroup_cpu_usage_usec() -> int | None:
    try:
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text(encoding="utf-8").splitlines():
            key, value = line.split()
            if key == "usage_usec":
                return int(value)
    except (OSError, ValueError):
        return None
    return None


def _read_cgroup_cpu_quota_cores() -> float | None:
    try:
        quota_raw, period_raw = Path("/sys/fs/cgroup/cpu.max").read_text(
            encoding="utf-8"
        ).split()
        if quota_raw == "max":
            return None
        return int(quota_raw) / int(period_raw)
    except (OSError, ValueError, ZeroDivisionError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    telemetry = args.artifact_root / "telemetry"
    telemetry.mkdir(parents=True, exist_ok=True)
    gpu_path = telemetry / "gpu_1s.csv"
    system_path = telemetry / "system_1s.csv"
    queue_path = telemetry / "queue_1s.jsonl"
    fields = [
        "timestamp",
        "index",
        "uuid",
        "name",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_pct",
        "power_draw_w",
        "temperature_c",
        "sm_clock_mhz",
        "memory_clock_mhz",
    ]
    query = (
        "index,uuid,name,memory.used,memory.total,utilization.gpu,"
        "power.draw,temperature.gpu,clocks.sm,clocks.mem"
    )
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    system_fields = [
        "timestamp",
        "cgroup_cpu_utilization_pct",
        "cpu_quota_cores",
        "load_1m",
        "load_5m",
        "load_15m",
        "cgroup_memory_current_gib",
        "cgroup_memory_max_gib",
    ]
    cpu_quota_cores = _read_cgroup_cpu_quota_cores()
    previous_cpu_usage = _read_cgroup_cpu_usage_usec()
    previous_wall = time.monotonic()
    with gpu_path.open("a", newline="", encoding="utf-8") as gpu_handle, queue_path.open(
        "a", encoding="utf-8"
    ) as queue_handle, system_path.open("a", newline="", encoding="utf-8") as system_handle:
        writer = csv.DictWriter(gpu_handle, fieldnames=fields)
        if gpu_path.stat().st_size == 0:
            writer.writeheader()
        system_writer = csv.DictWriter(system_handle, fieldnames=system_fields)
        if system_path.stat().st_size == 0:
            system_writer.writeheader()
        while not STOP:
            timestamp = datetime.now(timezone.utc).isoformat()
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                    if len(row) == 10:
                        writer.writerow(dict(zip(fields, [timestamp, *row])))
            current_wall = time.monotonic()
            current_cpu_usage = _read_cgroup_cpu_usage_usec()
            cpu_utilization = None
            if (
                current_cpu_usage is not None
                and previous_cpu_usage is not None
                and cpu_quota_cores is not None
                and current_wall > previous_wall
            ):
                used_seconds = (current_cpu_usage - previous_cpu_usage) / 1_000_000
                capacity_seconds = (current_wall - previous_wall) * cpu_quota_cores
                cpu_utilization = 100.0 * used_seconds / capacity_seconds
            previous_cpu_usage = current_cpu_usage
            previous_wall = current_wall
            load_1m, load_5m, load_15m = os.getloadavg()
            memory_current = _read_cgroup_number(Path("/sys/fs/cgroup/memory.current"))
            memory_max = _read_cgroup_number(Path("/sys/fs/cgroup/memory.max"))
            system_writer.writerow(
                {
                    "timestamp": timestamp,
                    "cgroup_cpu_utilization_pct": (
                        "" if cpu_utilization is None else f"{cpu_utilization:.4f}"
                    ),
                    "cpu_quota_cores": "" if cpu_quota_cores is None else cpu_quota_cores,
                    "load_1m": load_1m,
                    "load_5m": load_5m,
                    "load_15m": load_15m,
                    "cgroup_memory_current_gib": (
                        "" if memory_current is None else f"{memory_current / 1024**3:.4f}"
                    ),
                    "cgroup_memory_max_gib": (
                        "" if memory_max is None else f"{memory_max / 1024**3:.4f}"
                    ),
                }
            )
            seed_records = _read_jsonl(args.artifact_root / "queues" / "boundary_seeds.jsonl")
            candidates = _read_jsonl(args.artifact_root / "queues" / "validated_candidates.jsonl")
            terminal = _read_jsonl(args.artifact_root / "generator" / "expanded" / "terminal_results.jsonl")
            succeeded_ids = {
                item.get("seed_id") for item in terminal if item.get("status") == "SUCCEEDED"
            }
            dropped_ids = {
                item.get("seed_id") for item in terminal if item.get("status") == "DROPPED"
            }
            terminal_ids = succeeded_ids | dropped_ids
            pending = [item for item in seed_records if item.get("sample_id") not in terminal_ids]
            queue_record = {
                "timestamp": timestamp,
                "seed_depth_total": len(seed_records),
                "candidate_depth_total": len(candidates),
                "terminal_depth_total": len(terminal),
                "pending_seed_count": len(pending),
                "succeeded_seed_count": len(succeeded_ids),
                "dropped_seed_count": len(dropped_ids),
                "oldest_pending_seed_age_seconds": (
                    max(time.time() - (args.artifact_root / "queues" / "boundary_seeds.jsonl").stat().st_mtime, 0.0)
                    if pending and (args.artifact_root / "queues" / "boundary_seeds.jsonl").exists()
                    else 0.0
                ),
            }
            queue_handle.write(json.dumps(queue_record, sort_keys=True) + "\n")
            gpu_handle.flush()
            system_handle.flush()
            queue_handle.flush()
            os.fsync(gpu_handle.fileno())
            os.fsync(system_handle.fileno())
            os.fsync(queue_handle.fileno())
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
