from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


WORKSPACE = Path("/root/autodl-tmp/rods-workspace")
STAGE_ROOT = WORKSPACE / "stage1_format_rl"
AWORLD = WORKSPACE / "code" / "AWorld-RL-stage1-worktree"
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
    return WORKSPACE


@pytest.fixture(scope="session")
def stage_root() -> Path:
    return STAGE_ROOT


@pytest.fixture(scope="session")
def envtuning_root() -> Path:
    return ENVTUNING


@pytest.fixture(scope="session")
def train_rows():
    path = STAGE_ROOT / "data" / "bfcl_stage1_train_base_100.parquet"
    return [normalize(row) for row in pd.read_parquet(path).to_dict(orient="records")]


@pytest.fixture(scope="session")
def validation_rows():
    path = STAGE_ROOT / "data" / "bfcl_val_400.parquet"
    return [normalize(row) for row in pd.read_parquet(path).to_dict(orient="records")]

