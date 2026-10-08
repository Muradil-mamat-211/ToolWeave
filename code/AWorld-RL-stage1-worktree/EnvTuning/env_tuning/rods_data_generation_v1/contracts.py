"""Direct validators for the frozen Training<->Generator JSON contracts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .config import WORKSPACE
from .models import SeedRecord


SEED_SCHEMA_PATH = WORKSPACE / "stage1_format_rl/schemas/rods_boundary_seed_v1.schema.json"
PLANNER_SCHEMA_PATH = WORKSPACE / "stage1_format_rl/schemas/planner_input_v2.schema.json"


@lru_cache(maxsize=1)
def planner_input_validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(PLANNER_SCHEMA_PATH.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def seed_schema_validator() -> Draft202012Validator:
    schema = json.loads(SEED_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def validate_seed_record(raw: Mapping[str, Any]) -> SeedRecord:
    """Validate against the existing schema, then apply semantic checks."""

    seed_schema_validator().validate(raw)
    return SeedRecord.from_mapping(raw)
