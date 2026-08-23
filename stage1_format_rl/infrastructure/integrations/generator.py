"""Adapter for the independent RODS Data-Generation process."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..loader import dotted_set, expand_templates
from ..models import AssetSpec, ConfigError, MachineConfig, RuntimeConfig, TopologyPlan


@dataclass(frozen=True)
class _OwnedGeneratorField:
    owner: str
    resolve: Callable[[RuntimeConfig, TopologyPlan], Any]


_AUTHORITATIVE_GENERATOR_FIELDS = {
    "deployment.cuda_visible_devices": _OwnedGeneratorField(
        "topology", lambda _runtime, plan: plan.cuda_visible_devices["generator"]
    ),
    "deployment.gpu_count": _OwnedGeneratorField(
        "topology", lambda _runtime, plan: plan.generator_world_size
    ),
    "deployment.tensor_parallel_size": _OwnedGeneratorField(
        "topology", lambda _runtime, plan: plan.generator_tp
    ),
    "deployment.data_parallel_size": _OwnedGeneratorField(
        "topology", lambda _runtime, plan: plan.generator_dp
    ),
    "deployment.dtype": _OwnedGeneratorField(
        "runtime", lambda runtime, _plan: runtime.generator_dtype
    ),
    "deployment.max_model_len": _OwnedGeneratorField(
        "runtime", lambda runtime, _plan: runtime.generator_max_model_len
    ),
    "deployment.gpu_memory_utilization": _OwnedGeneratorField(
        "runtime", lambda runtime, _plan: runtime.generator_gpu_memory_utilization
    ),
    "llm.backend": _OwnedGeneratorField(
        "runtime", lambda runtime, _plan: runtime.generator_backend
    ),
    "llm.endpoint": _OwnedGeneratorField(
        "runtime",
        lambda runtime, _plan: (
            f"http://{runtime.generator_host}:{runtime.generator_port}/v1"
        ),
    ),
    "llm.concurrency": _OwnedGeneratorField(
        "runtime", lambda runtime, _plan: runtime.generator_backend_concurrency
    ),
    "seed_worker_count": _OwnedGeneratorField(
        "runtime", lambda runtime, _plan: runtime.generator_seed_workers
    ),
}

GENERATOR_FIELD_OWNERS: Mapping[str, str] = MappingProxyType(
    {
        destination: field.owner
        for destination, field in _AUTHORITATIVE_GENERATOR_FIELDS.items()
    }
)


def _leaf_paths(mapping: Mapping[str, Any], prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            paths.extend(_leaf_paths(value, path))
        else:
            paths.append(path)
    return tuple(paths)


def _paths_overlap(left: str, right: str) -> bool:
    return (
        left == right
        or left.startswith(f"{right}.")
        or right.startswith(f"{left}.")
    )


def _owned_collision(path: str) -> str | None:
    return next(
        (
            destination
            for destination in GENERATOR_FIELD_OWNERS
            if _paths_overlap(path, destination)
        ),
        None,
    )


def build_generator_config(
    raw: Mapping[str, Any],
    assets: Mapping[str, AssetSpec],
    machine: MachineConfig,
    runtime: RuntimeConfig,
    plan: TopologyPlan,
) -> dict[str, Any]:
    """Resolve Generator paths and deployment values without algorithm changes."""

    section = raw.get("generator")
    if not isinstance(section, Mapping):
        raise ConfigError("generator experiment must contain a generator mapping")
    for path in _leaf_paths(section):
        destination = _owned_collision(path)
        if destination is not None:
            raise ConfigError(
                "generator experiment illegally owns runtime/topology field "
                f"{destination} (authoritative owner: "
                f"{GENERATOR_FIELD_OWNERS[destination]})"
            )
    config: dict[str, Any] = deepcopy(dict(section))
    bindings = raw.get("asset_bindings", {})
    if not isinstance(bindings, Mapping):
        raise ConfigError("generator.asset_bindings must be a mapping")
    outputs = raw.get("output_bindings", {})
    if not isinstance(outputs, Mapping):
        raise ConfigError("generator.output_bindings must be a mapping")
    section_paths = _leaf_paths(section)
    for label, destinations in (
        ("generator.asset_bindings", bindings),
        ("generator.output_bindings", outputs),
    ):
        for raw_destination in destinations:
            destination = str(raw_destination)
            collision = _owned_collision(destination)
            if collision is not None:
                raise ConfigError(
                    f"{label} destination {destination} collides with "
                    f"{GENERATOR_FIELD_OWNERS[collision]}-owned field {collision}"
                )
            if any(_paths_overlap(destination, path) for path in section_paths):
                raise ConfigError(
                    f"{label} destination {destination} collides with "
                    "generator experiment ownership"
                )
    for asset_destination in bindings:
        if any(
            _paths_overlap(str(asset_destination), str(output_destination))
            for output_destination in outputs
        ):
            raise ConfigError(
                f"generator asset/output binding ownership collision at "
                f"{asset_destination}"
            )
    for destination, asset_name in bindings.items():
        if asset_name not in assets:
            raise ConfigError(f"unknown logical asset {asset_name!r} for {destination}")
        dotted_set(config, str(destination), str(assets[str(asset_name)].path))
    for destination, value in expand_templates(outputs, machine.template_values()).items():
        dotted_set(config, str(destination), value)

    generator_role = runtime.roles.get("generator")
    if generator_role and generator_role.gpu_count:
        for destination, field in _AUTHORITATIVE_GENERATOR_FIELDS.items():
            dotted_set(config, destination, field.resolve(runtime, plan))
    return config
