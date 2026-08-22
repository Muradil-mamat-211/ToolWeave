"""Compatibility path resolver for historical one-off scripts.

New launchers use ``stage1_format_rl.infrastructure``. Historical utilities
consume the same machine-local environment names through this small adapter;
source, input assets, outputs, logs, caches, and temporary files can therefore
live under independent roots without embedding host paths in Python.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectRoots:
    source_root: Path
    asset_root: Path
    models_root: Path
    shared_data_root: Path
    stage_data_root: Path
    artifacts_root: Path
    outputs_root: Path
    logs_root: Path
    reports_root: Path
    evals_root: Path
    cache_root: Path
    temp_root: Path
    short_temp_root: Path

def project_roots() -> ProjectRoots:
    discovered = Path(__file__).resolve().parents[2]
    source = Path(
        os.environ.get(
            "TOOLWEAVE_SOURCE_ROOT", os.environ.get("TOOLWEAVE_ROOT", discovered)
        )
    ).expanduser().resolve()
    asset = Path(os.environ.get("TOOLWEAVE_ASSET_ROOT", source)).expanduser().resolve()
    def configured(name: str, fallback: Path) -> Path:
        return Path(os.environ.get(name, fallback)).expanduser().resolve()

    return ProjectRoots(
        source_root=source,
        asset_root=asset,
        models_root=configured("TOOLWEAVE_MODELS_ROOT", asset / "models"),
        shared_data_root=configured("TOOLWEAVE_SHARED_DATA_ROOT", asset / "data"),
        stage_data_root=configured(
            "TOOLWEAVE_DATA_ROOT", asset / "stage1_format_rl/data"
        ),
        artifacts_root=configured(
            "TOOLWEAVE_ARTIFACTS_ROOT", asset / "stage1_format_rl/artifacts"
        ),
        outputs_root=configured("TOOLWEAVE_OUTPUTS_ROOT", asset / "outputs"),
        logs_root=configured(
            "TOOLWEAVE_LOGS_ROOT", asset / "stage1_format_rl/logs"
        ),
        reports_root=configured("TOOLWEAVE_REPORTS_ROOT", asset / "reports"),
        evals_root=configured("TOOLWEAVE_EVALS_ROOT", asset / "evals"),
        cache_root=configured("TOOLWEAVE_CACHE_ROOT", asset / ".cache"),
        temp_root=configured("TOOLWEAVE_TEMP_ROOT", asset / ".runtime"),
        short_temp_root=configured(
            "TOOLWEAVE_SHORT_TEMP_ROOT", Path(os.environ.get("TMPDIR", "/tmp")) / "toolweave"
        ),
    )
