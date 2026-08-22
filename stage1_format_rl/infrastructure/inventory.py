"""Read-only host inventory used only by explicit preflight."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Any

from .models import ConfigError, HardwareConfig


_GPU_MEMORY_REPORTING_TOLERANCE_GIB = 0.5


def _effective_cpu_cores() -> int:
    """Return scheduler-visible CPU capacity, including cgroup quotas."""

    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count() or 0
    quota_capacity: int | None = None
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        quota, period = cpu_max.read_text(encoding="utf-8").split()[:2]
        if quota != "max" and int(period) > 0:
            quota_capacity = max(1, math.ceil(int(quota) / int(period)))
    return min(affinity, quota_capacity) if quota_capacity is not None else affinity


def _effective_ram_gib(host_memory_kib: int) -> float:
    """Return the smaller of host RAM and an active cgroup memory limit."""

    host_bytes = host_memory_kib * 1024
    limits = [host_bytes] if host_bytes else []
    for candidate in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if not candidate.is_file():
            continue
        raw = candidate.read_text(encoding="utf-8").strip()
        if raw and raw != "max":
            value = int(raw)
            # cgroup v1 commonly uses an enormous sentinel for "unlimited".
            if value < (1 << 60):
                limits.append(value)
    effective = min(limits) if limits else 0
    return effective / (1024.0**3)


def discover_local_inventory() -> dict[str, Any]:
    memory_kib = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_kib = int(line.split()[1])
                break
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
        gpus = []
        for line in completed.stdout.splitlines():
            index, name, memory_mib = (part.strip() for part in line.split(",", 2))
            gpus.append(
                {
                    "id": int(index),
                    "model": name,
                    "memory_gib": int(memory_mib) / 1024.0,
                }
            )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        gpus = []
    return {
        "cpu_cores": _effective_cpu_cores(),
        "ram_gib": _effective_ram_gib(memory_kib),
        "gpus": gpus,
    }


def validate_local_inventory(hardware: HardwareConfig, observed: dict[str, Any]) -> None:
    """Generic capacity check; model identity belongs to qualification."""

    if len(hardware.nodes) != 1:
        raise ConfigError("local inventory validation currently supports one node")
    requested = hardware.nodes[0]
    observed_ids = {int(item["id"]) for item in observed.get("gpus", [])}
    missing = sorted(set(requested.gpu_ids) - observed_ids)
    if missing:
        raise ConfigError(f"requested GPU IDs are unavailable: {missing}")
    if int(observed.get("cpu_cores", 0)) < requested.cpu_cores:
        raise ConfigError(
            f"CPU resources insufficient: requested {requested.cpu_cores}, "
            f"observed {observed.get('cpu_cores', 0)}"
        )
    if float(observed.get("ram_gib", 0.0)) < requested.ram_gib:
        raise ConfigError(
            f"RAM resources insufficient: requested {requested.ram_gib:g} GiB, "
            f"observed {float(observed.get('ram_gib', 0.0)):g} GiB"
        )
    memory_by_id = {
        int(item["id"]): float(item.get("memory_gib", 0.0))
        for item in observed.get("gpus", [])
    }
    for gpu_id in requested.gpu_ids:
        # nvidia-smi reports usable binary GiB, while hardware manifests use
        # nominal capacity. Reference identity remains a separate check.
        if (
            memory_by_id[gpu_id] + _GPU_MEMORY_REPORTING_TOLERANCE_GIB
            < requested.gpu_memory_gib
        ):
            raise ConfigError(
                f"GPU {gpu_id} memory insufficient: requested "
                f"{requested.gpu_memory_gib:g} GiB, observed {memory_by_id[gpu_id]:g} GiB"
            )
