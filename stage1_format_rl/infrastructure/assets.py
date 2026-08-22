"""Logical asset manifests and integrity validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .loader import expand_templates
from .models import AssetSpec, ConfigError, MachineConfig


def load_asset_specs(raw: Mapping[str, Any], machine: MachineConfig) -> dict[str, AssetSpec]:
    root = raw.get("assets", raw)
    if not isinstance(root, Mapping) or not root:
        raise ConfigError("asset manifest must define a nonempty assets mapping")
    expanded = expand_templates(root, machine.template_values())
    result: dict[str, AssetSpec] = {}
    for name, value in expanded.items():
        if not isinstance(value, Mapping):
            raise ConfigError(f"asset {name} must be a mapping")
        path_text = str(value.get("path", "")).strip()
        if not path_text:
            raise ConfigError(f"asset {name}.path is required")
        required_files_raw = value.get("required_files", {})
        if not isinstance(required_files_raw, Mapping):
            raise ConfigError(f"asset {name}.required_files must be a mapping")
        row_count_raw = value.get("row_count")
        result[str(name)] = AssetSpec(
            name=str(name),
            kind=str(value.get("kind", "file")),
            path=Path(path_text).expanduser().resolve(),
            required=bool(value.get("required", True)),
            sha256=(str(value["sha256"]) if value.get("sha256") else None),
            row_count=(int(row_count_raw) if row_count_raw is not None else None),
            required_files={
                str(relative): (str(digest) if digest else None)
                for relative, digest in required_files_raw.items()
            },
        )
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assets(assets: Mapping[str, AssetSpec], *, check_rows: bool = True) -> None:
    """Fail closed on missing assets, checksum drift, or declared row-count drift."""

    for asset in assets.values():
        if not asset.path.exists():
            if asset.required:
                raise ConfigError(f"required asset does not exist: {asset.name}={asset.path}")
            continue
        if asset.sha256:
            if not asset.path.is_file():
                raise ConfigError(f"asset {asset.name} has a file checksum but is not a file")
            actual = sha256_file(asset.path)
            if actual != asset.sha256:
                raise ConfigError(
                    f"asset checksum mismatch for {asset.name}: expected {asset.sha256}, got {actual}"
                )
        for relative, expected in asset.required_files.items():
            target = asset.path / relative
            if not target.is_file():
                raise ConfigError(f"asset {asset.name} is missing required file {relative}")
            if expected:
                actual = sha256_file(target)
                if actual != expected:
                    raise ConfigError(
                        f"asset checksum mismatch for {asset.name}/{relative}: "
                        f"expected {expected}, got {actual}"
                    )
        if asset.row_count is not None and check_rows:
            if asset.path.suffix != ".parquet":
                raise ConfigError(f"row_count is currently supported only for parquet: {asset.name}")
            import pyarrow.parquet as pq

            # Metadata-only: validates large datasets without materializing rows.
            actual_rows = pq.ParquetFile(asset.path).metadata.num_rows
            if actual_rows != asset.row_count:
                raise ConfigError(
                    f"asset row count mismatch for {asset.name}: "
                    f"expected {asset.row_count}, got {actual_rows}"
                )
