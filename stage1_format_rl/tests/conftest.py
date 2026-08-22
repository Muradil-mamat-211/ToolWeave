from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get("TOOLWEAVE_ASSET_ROOT", SOURCE_ROOT)).expanduser().resolve()
STAGE_ROOT = SOURCE_ROOT / "stage1_format_rl"
ASSET_STAGE_ROOT = ASSET_ROOT / "stage1_format_rl"
AWORLD = SOURCE_ROOT / "code" / "AWorld-RL-stage1-worktree"
ENVTUNING = AWORLD / "EnvTuning"
ADAPTED_VERL = ENVTUNING / "verl"

for path in (STAGE_ROOT / "scripts", ENVTUNING, ADAPTED_VERL):
    sys.path.insert(0, str(path))


def normalize(value):
    if isinstance(value, np.ndarray):
        return [normalize(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


@pytest.fixture(scope="session")
def workspace() -> Path:
    """Read-only root for heavyweight model/data assets used by contract tests."""

    return ASSET_ROOT


@pytest.fixture(scope="session")
def stage_root() -> Path:
    return STAGE_ROOT


@pytest.fixture(scope="session")
def asset_stage_root() -> Path:
    return ASSET_STAGE_ROOT


@pytest.fixture(scope="session")
def envtuning_root() -> Path:
    return ENVTUNING


@pytest.fixture(scope="session")
def train_rows():
    path = ASSET_STAGE_ROOT / "data" / "bfcl_stage1_train_base_100.parquet"
    return [normalize(row) for row in pd.read_parquet(path).to_dict(orient="records")]


@pytest.fixture(scope="session")
def validation_rows():
    path = ASSET_STAGE_ROOT / "data" / "bfcl_val_400.parquet"
    return [normalize(row) for row in pd.read_parquet(path).to_dict(orient="records")]
