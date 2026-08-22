"""Machine-local paths for repository utilities; no server defaults live here."""

from __future__ import annotations

import os
from pathlib import Path


SOURCE_ROOT = Path(
    os.environ.get("TOOLWEAVE_SOURCE_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
ASSET_ROOT = Path(os.environ.get("TOOLWEAVE_ASSET_ROOT", SOURCE_ROOT)).expanduser().resolve()
MODELS_ROOT = Path(os.environ.get("TOOLWEAVE_MODELS_ROOT", ASSET_ROOT / "models")).expanduser().resolve()
SHARED_DATA_ROOT = Path(
    os.environ.get("TOOLWEAVE_SHARED_DATA_ROOT", ASSET_ROOT / "data")
).expanduser().resolve()
STAGE_DATA_ROOT = Path(
    os.environ.get("TOOLWEAVE_DATA_ROOT", ASSET_ROOT / "stage1_format_rl/data")
).expanduser().resolve()
ARTIFACTS_ROOT = Path(
    os.environ.get("TOOLWEAVE_ARTIFACTS_ROOT", ASSET_ROOT / "stage1_format_rl/artifacts")
).expanduser().resolve()
OUTPUTS_ROOT = Path(os.environ.get("TOOLWEAVE_OUTPUTS_ROOT", ASSET_ROOT / "outputs")).expanduser().resolve()
LOGS_ROOT = Path(
    os.environ.get("TOOLWEAVE_LOGS_ROOT", ASSET_ROOT / "stage1_format_rl/logs")
).expanduser().resolve()
REPORTS_ROOT = Path(os.environ.get("TOOLWEAVE_REPORTS_ROOT", ASSET_ROOT / "reports")).expanduser().resolve()
EVALS_ROOT = Path(os.environ.get("TOOLWEAVE_EVALS_ROOT", ASSET_ROOT / "evals")).expanduser().resolve()
