"""Strict reference reproduction checks, separate from portable validation."""

from __future__ import annotations

from typing import Any, Mapping

from .models import ConfigError, HardwareConfig, RuntimeConfig, TopologyPlan


def qualify_reference(
    raw: Mapping[str, Any] | None,
    hardware: HardwareConfig,
    runtime: RuntimeConfig,
    plan: TopologyPlan,
) -> None:
    if not raw:
        raise ConfigError("mode=reference requires a qualification profile")
    expected = raw.get("qualification", raw)
    if not isinstance(expected, Mapping):
        raise ConfigError("qualification must be a mapping")

    actual_scalar = {
        "nnodes": plan.nnodes,
        "total_gpus": hardware.total_gpus,
        "cpu_cores": hardware.total_cpu_cores,
        "ram_gib": hardware.total_ram_gib,
        "learner_world_size": plan.learner_world_size,
        "rollout_tp": plan.rollout_tp,
        "rollout_dp": plan.rollout_dp,
        "sequence_parallel_size": runtime.sequence_parallel_size,
    }
    for key, actual in actual_scalar.items():
        if key in expected and expected[key] != actual:
            raise ConfigError(
                f"reference qualification mismatch for {key}: "
                f"expected {expected[key]!r}, got {actual!r}"
            )
    if "gpu_model" in expected:
        models = {node.gpu_model for node in hardware.nodes}
        if models != {str(expected["gpu_model"])}:
            raise ConfigError(
                f"reference qualification mismatch for gpu_model: "
                f"expected {expected['gpu_model']!r}, got {sorted(models)!r}"
            )
    if "gpu_memory_gib" in expected:
        memories = {node.gpu_memory_gib for node in hardware.nodes if node.gpu_ids}
        if memories != {float(expected["gpu_memory_gib"])}:
            raise ConfigError(
                "reference qualification mismatch for gpu_memory_gib: "
                f"expected {expected['gpu_memory_gib']!r}, got {sorted(memories)!r}"
            )
    expected_roles = expected.get("role_assignments")
    if expected_roles is not None:
        normalized = {
            str(role): {str(node): tuple(int(item) for item in ids) for node, ids in nodes.items()}
            for role, nodes in expected_roles.items()
        }
        actual = {role: dict(nodes) for role, nodes in plan.role_assignments.items()}
        if normalized != actual:
            raise ConfigError(
                f"reference qualification mismatch for role_assignments: "
                f"expected {normalized!r}, got {actual!r}"
            )


def qualify_observed_reference(
    raw: Mapping[str, Any] | None, observed: Mapping[str, Any]
) -> None:
    """Compare an observed host to strict reference-machine identity.

    Generic resource validation intentionally accepts equivalent hardware
    models. This reference-only check verifies identity and nominal capacity
    when the caller explicitly requests host observation.
    """

    if not raw:
        raise ConfigError("mode=reference requires a qualification profile")
    expected = raw.get("qualification", raw)
    if not isinstance(expected, Mapping):
        raise ConfigError("qualification must be a mapping")
    gpus = observed.get("gpus", [])
    if not isinstance(gpus, list):
        raise ConfigError("observed hardware gpus must be a list")

    scalar_pairs = {
        "total_gpus": (len(gpus), int),
        "cpu_cores": (observed.get("cpu_cores", 0), int),
    }
    for key, (actual, cast) in scalar_pairs.items():
        if key in expected and cast(actual) != cast(expected[key]):
            raise ConfigError(
                f"observed reference qualification mismatch for {key}: "
                f"expected {expected[key]!r}, got {actual!r}"
            )

    if "gpu_model" in expected:
        models = {str(item.get("model", "")) for item in gpus}
        wanted = {str(expected["gpu_model"])}
        if models != wanted:
            raise ConfigError(
                "observed reference qualification mismatch for gpu_model: "
                f"expected {sorted(wanted)!r}, got {sorted(models)!r}"
            )

    if "gpu_memory_gib" in expected:
        tolerance = float(expected.get("gpu_memory_tolerance_gib", 0.5))
        wanted = float(expected["gpu_memory_gib"])
        mismatched = [
            float(item.get("memory_gib", 0.0))
            for item in gpus
            if abs(float(item.get("memory_gib", 0.0)) - wanted) > tolerance
        ]
        if mismatched:
            raise ConfigError(
                "observed reference qualification mismatch for gpu_memory_gib: "
                f"expected {wanted:g} +/- {tolerance:g}, got {mismatched!r}"
            )

    if "ram_gib" in expected:
        tolerance = float(expected.get("ram_tolerance_gib", 1.0))
        wanted = float(expected["ram_gib"])
        actual = float(observed.get("ram_gib", 0.0))
        if abs(actual - wanted) > tolerance:
            raise ConfigError(
                "observed reference qualification mismatch for ram_gib: "
                f"expected {wanted:g} +/- {tolerance:g}, got {actual:g}"
            )
