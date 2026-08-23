"""Strict YAML loading and machine-local template expansion."""

from __future__ import annotations

import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import ConfigError, MachineConfig


_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.-]*)\}")
_SCHEMA_VERSION = re.compile(r"^toolweave\.[a-z][a-z0-9-]*\.v[1-9][0-9]*$")

SUPPORTED_LAYER_SCHEMAS = frozenset(
    {
        "toolweave.machine.v1",
        "toolweave.profile.v1",
        "toolweave.hardware.v1",
        "toolweave.runtime.v1",
        "toolweave.assets.v1",
        "toolweave.experiment.v1",
        "toolweave.generator-experiment.v1",
        "toolweave.qualification.v1",
    }
)


def load_yaml(
    path: str | Path, *, expected_schema: str | None = None
) -> dict[str, Any]:
    """Load YAML, optionally enforcing one exact ToolWeave layer contract.

    Generic callers deliberately remain schema-agnostic so historical Hydra
    configs, upstream YAML, and reproduction snapshots are not pulled into the
    layered configuration contract.
    """

    path = Path(path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file does not exist: {path}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    if expected_schema is not None:
        if expected_schema not in SUPPORTED_LAYER_SCHEMAS:
            raise ConfigError(f"unsupported expected ToolWeave schema: {expected_schema}")
        observed = value.get("schema_version")
        observed_label = "<missing>" if observed is None else repr(observed)
        if (
            not isinstance(observed, str)
            or _SCHEMA_VERSION.fullmatch(observed) is None
            or observed != expected_schema
        ):
            raise ConfigError(
                f"schema_version mismatch for {path}: expected schema "
                f"{expected_schema!r}; observed schema {observed_label}"
            )
    return value


def find_project_root(start: str | Path) -> Path:
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "stage1_format_rl").is_dir() and (candidate / "code").is_dir():
            return candidate
    raise ConfigError(f"cannot discover ToolWeave project root from {start}")


def expand_templates(value: Any, variables: Mapping[str, Any]) -> Any:
    """Recursively expand exact `${name}` placeholders, failing on unknowns."""

    if isinstance(value, dict):
        return {key: expand_templates(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_templates(item, variables) for item in value]
    if not isinstance(value, str):
        return value

    unresolved: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            unresolved.add(key)
            return match.group(0)
        return str(variables[key])

    result = value
    for _ in range(10):
        updated = _PLACEHOLDER.sub(replace, result)
        if updated == result:
            break
        result = updated
    remaining = set(_PLACEHOLDER.findall(result)) | unresolved
    if remaining:
        raise ConfigError(f"unresolved configuration placeholders: {sorted(remaining)}")
    return result


def machine_environment_defaults(
    project_root: Path, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    source_root = Path(env.get("TOOLWEAVE_SOURCE_ROOT", env.get("TOOLWEAVE_ROOT", project_root))).expanduser().resolve()
    asset_root = Path(env.get("TOOLWEAVE_ASSET_ROOT", source_root)).expanduser().resolve()
    defaults = {
        "TOOLWEAVE_ROOT": str(source_root),
        "TOOLWEAVE_SOURCE_ROOT": str(source_root),
        "TOOLWEAVE_ASSET_ROOT": str(asset_root),
        "TOOLWEAVE_MODELS_ROOT": str(asset_root / "models"),
        "TOOLWEAVE_DATA_ROOT": str(asset_root / "stage1_format_rl" / "data"),
        "TOOLWEAVE_SHARED_DATA_ROOT": str(asset_root / "data"),
        "TOOLWEAVE_ARTIFACTS_ROOT": str(asset_root / "stage1_format_rl" / "artifacts"),
        "TOOLWEAVE_OUTPUTS_ROOT": str(asset_root / "outputs"),
        "TOOLWEAVE_LOGS_ROOT": str(asset_root / "stage1_format_rl" / "logs"),
        "TOOLWEAVE_REPORTS_ROOT": str(asset_root / "reports"),
        "TOOLWEAVE_EVALS_ROOT": str(asset_root / "evals"),
        "TOOLWEAVE_CACHE_ROOT": str(asset_root / ".cache"),
        "TOOLWEAVE_TEMP_ROOT": str(asset_root / ".runtime"),
        "TOOLWEAVE_SHORT_TEMP_ROOT": str(
            Path(env.get("TMPDIR", "/tmp")) / "toolweave"
        ),
        "TOOLWEAVE_PYTHON": sys.executable,
        "TOOLWEAVE_SYNTH_PYTHON": sys.executable,
        "TOOLWEAVE_CONDA_ENV": "",
    }
    defaults.update({key: value for key, value in env.items() if key.startswith("TOOLWEAVE_")})
    return defaults


def load_machine_config(
    path: str | Path,
    *,
    project_root: Path,
    environ: Mapping[str, str] | None = None,
) -> MachineConfig:
    variables = machine_environment_defaults(project_root, environ)
    raw = expand_templates(
        load_yaml(path, expected_schema="toolweave.machine.v1"), variables
    )
    return MachineConfig.from_mapping(raw)


def resolve_relative_path(owner: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (owner.parent / path).resolve()


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic recursive merge used only for configuration composition."""

    result = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def dotted_set(target: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"cannot assign {key}: {part} is not a mapping")
        current = child
    current[parts[-1]] = deepcopy(value)


def dotted_get(target: Mapping[str, Any], key: str) -> Any:
    current: Any = target
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"missing configuration field: {key}")
        current = current[part]
    return current
