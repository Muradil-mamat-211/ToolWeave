"""Adapter for the independent RODS Data-Generation process."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..loader import dotted_set, expand_templates
from ..models import AssetSpec, ConfigError, MachineConfig, RuntimeConfig, TopologyPlan


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
    config: dict[str, Any] = deepcopy(dict(section))
    bindings = raw.get("asset_bindings", {})
    if not isinstance(bindings, Mapping):
        raise ConfigError("generator.asset_bindings must be a mapping")
    for destination, asset_name in bindings.items():
        if asset_name not in assets:
            raise ConfigError(f"unknown logical asset {asset_name!r} for {destination}")
        dotted_set(config, str(destination), str(assets[str(asset_name)].path))
    outputs = raw.get("output_bindings", {})
    if not isinstance(outputs, Mapping):
        raise ConfigError("generator.output_bindings must be a mapping")
    for destination, value in expand_templates(outputs, machine.template_values()).items():
        dotted_set(config, str(destination), value)

    generator_role = runtime.roles.get("generator")
    if generator_role and generator_role.gpu_count:
        # These are deployment diagnostics consumed by the generic launcher;
        # GeneratorConfig itself retains the published algorithm parameters.
        config.setdefault("deployment", {})
        config["deployment"].update(
            {
                "cuda_visible_devices": plan.cuda_visible_devices["generator"],
                "gpu_count": generator_role.gpu_count,
                "tensor_parallel_size": plan.generator_tp,
                "data_parallel_size": plan.generator_dp,
                "dtype": runtime.generator_dtype,
                "max_model_len": runtime.generator_max_model_len,
                "gpu_memory_utilization": runtime.generator_gpu_memory_utilization,
            }
        )
        dotted_set(config, "llm.backend", runtime.generator_backend)
        dotted_set(
            config,
            "llm.endpoint",
            f"http://{runtime.generator_host}:{runtime.generator_port}/v1",
        )
        dotted_set(config, "llm.concurrency", runtime.generator_backend_concurrency)
        dotted_set(config, "seed_worker_count", runtime.generator_seed_workers)
    return config
