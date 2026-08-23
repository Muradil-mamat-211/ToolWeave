"""Resolve layered ToolWeave profiles into framework-ready configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .assets import load_asset_specs
from .integrations.generator import build_generator_config
from .integrations.verl import build_verl_config
from .loader import (
    find_project_root,
    load_machine_config,
    load_yaml,
    resolve_relative_path,
)
from .models import (
    AssetSpec,
    ConfigError,
    HardwareConfig,
    MachineConfig,
    RuntimeConfig,
    TopologyPlan,
)
from .qualification import qualify_reference
from .topology import build_topology_plan


_LAYER_SCHEMAS = {
    "hardware": "toolweave.hardware.v1",
    "runtime": "toolweave.runtime.v1",
    "assets": "toolweave.assets.v1",
    "experiment": "toolweave.experiment.v1",
    "generator": "toolweave.generator-experiment.v1",
    "qualification": "toolweave.qualification.v1",
}


@dataclass(frozen=True)
class ResolvedProjectConfig:
    profile_path: Path
    mode: str
    machine: MachineConfig
    assets: dict[str, AssetSpec]
    hardware: HardwareConfig
    runtime: RuntimeConfig
    topology: TopologyPlan
    experiment: dict[str, Any]
    effective_verl: dict[str, Any]
    effective_generator: dict[str, Any] | None = None
    qualification: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "profile": str(self.profile_path),
            "mode": self.mode,
            "machine": {
                key.removeprefix("machine."): value
                for key, value in self.machine.template_values().items()
            },
            "assets": {
                name: {
                    "kind": asset.kind,
                    "path": str(asset.path),
                    "sha256": asset.sha256,
                    "row_count": asset.row_count,
                }
                for name, asset in self.assets.items()
            },
            "topology": self.topology.to_dict(),
            "effective_verl": self.effective_verl,
            "effective_generator": self.effective_generator,
            "qualification": self.qualification,
        }


def _layer(profile_path: Path, profile: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = profile.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"profile.{key} must name a YAML file")
    return load_yaml(
        resolve_relative_path(profile_path, value),
        expected_schema=_LAYER_SCHEMAS[key],
    )


def resolve_profile(
    profile_path: str | Path,
    *,
    mode: str | None = None,
    machine_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ResolvedProjectConfig:
    profile_path = Path(profile_path).expanduser().resolve()
    project_root = find_project_root(profile_path)
    profile = load_yaml(profile_path, expected_schema="toolweave.profile.v1")
    selected_mode = str(mode or profile.get("mode", "portable"))
    if selected_mode not in {"portable", "reference"}:
        raise ConfigError("mode must be 'portable' or 'reference'")
    machine_value = machine_path or profile.get("machine")
    if not machine_value:
        raise ConfigError("profile.machine is required")
    resolved_machine_path = (
        Path(machine_value).expanduser().resolve()
        if machine_path
        else resolve_relative_path(profile_path, str(machine_value))
    )
    machine = load_machine_config(
        resolved_machine_path, project_root=project_root, environ=environ
    )
    hardware = HardwareConfig.from_mapping(_layer(profile_path, profile, "hardware"))
    runtime = RuntimeConfig.from_mapping(_layer(profile_path, profile, "runtime"))
    topology = build_topology_plan(hardware, runtime)
    assets = load_asset_specs(_layer(profile_path, profile, "assets"), machine)
    experiment = _layer(profile_path, profile, "experiment")
    effective = build_verl_config(experiment, assets, machine, runtime, topology)
    generator_raw = (
        _layer(profile_path, profile, "generator") if profile.get("generator") else None
    )
    effective_generator = (
        build_generator_config(generator_raw, assets, machine, runtime, topology)
        if generator_raw is not None
        else None
    )
    qualification_raw = (
        _layer(profile_path, profile, "qualification")
        if profile.get("qualification")
        else None
    )
    if selected_mode == "reference":
        qualify_reference(qualification_raw, hardware, runtime, topology)
    return ResolvedProjectConfig(
        profile_path=profile_path,
        mode=selected_mode,
        machine=machine,
        assets=assets,
        hardware=hardware,
        runtime=runtime,
        topology=topology,
        experiment=experiment,
        effective_verl=effective,
        effective_generator=effective_generator,
        qualification=qualification_raw,
    )
