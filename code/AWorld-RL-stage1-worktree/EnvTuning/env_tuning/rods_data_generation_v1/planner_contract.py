"""Task-only Planner contract: source tools once, no policy trajectories."""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

from .function_catalog import CatalogError, FunctionCatalog
from .models import SeedRecord, SUPPORTED_DATA_TYPES
from .validation.parameter_complexity import ARGUMENT_LIMITS

PLANNER_INPUT_VERSION = "toolweave.planner_input.v2"
CATEGORY_PROMPTS = {
    "multi_turn_base": "project/planner_categories/base.txt",
    "multi_turn_miss_func": "project/planner_categories/missing_function.txt",
    "multi_turn_miss_param": "project/planner_categories/missing_parameter.txt",
    "multi_turn_long_context": "project/planner_categories/long_context.txt",
}

def planner_max_turns(data_type: str) -> int:
    if data_type not in SUPPORTED_DATA_TYPES:
        raise ValueError(f"unsupported Planner category: {data_type}")
    return 4 if data_type in {"multi_turn_miss_func", "multi_turn_miss_param"} else 5

def snapshot_view(value: Any, *, depth: int = 0) -> Any:
    """Bound large VM previews, with explicit truncation metadata."""
    if isinstance(value, str) and len(value) > 1000:
        return {"preview": value[:1000], "truncated": True,
                "original_characters": len(value),
                "sha256": hashlib.sha256(value.encode()).hexdigest()}
    if isinstance(value, Mapping):
        if depth > 10:
            return {"truncated": True, "original_keys": len(value)}
        items = list(value.items())
        result = {str(key): snapshot_view(item, depth=depth + 1) for key, item in items[:40]}
        if len(items) > 40:
            result["__planner_preview__"] = {"truncated": True, "original_keys": len(items)}
        return result
    if isinstance(value, (list, tuple)):
        preview = [snapshot_view(item, depth=depth + 1) for item in value[:20]]
        return {"preview": preview, "truncated": True, "original_items": len(value)} if len(value) > 20 else preview
    return value

def original_tool_updates(seed: SeedRecord, catalog: FunctionCatalog) -> list[dict[str, Any]]:
    source = seed.generation_metadata.get("planner_source", {})
    if not isinstance(source, Mapping):
        raise CatalogError("planner_source must be an object")
    # Legacy seeds can recover exact source events from the active training
    # parquet. No event is guessed from shared literals or policy rollouts.
    raw_updates = source.get("tool_updates")
    if raw_updates is None or raw_updates == []:
        raw_updates = catalog.source_tool_updates.get(seed.sample_id, [])
    if not isinstance(raw_updates, list):
        raise CatalogError("source tool_updates must be a list")
    seen = {schema["name"] for schema in seed.available_functions}
    updates = []
    for event in raw_updates:
        if not isinstance(event, Mapping) or type(event.get("turn_id")) is not int:
            raise CatalogError("source tool update needs an integer turn_id")
        if not 0 < event["turn_id"] < len(seed.Q_old):
            raise CatalogError("source tool update turn_id is out of range")
        schemas = event.get("tools")
        if not isinstance(schemas, list) or not schemas:
            raise CatalogError("source tool update needs nonempty tools")
        accepted = []
        for schema in schemas:
            if not isinstance(schema, Mapping) or not isinstance(schema.get("name"), str):
                raise CatalogError("malformed source recovery schema")
            spec = catalog.get(schema["name"])
            if spec.schema != schema:
                raise CatalogError(f"source recovery schema differs from VM catalog: {spec.name}")
            if spec.name in seen:
                raise CatalogError(f"duplicate original tool definition: {spec.name}")
            seen.add(spec.name)
            accepted.append(copy.deepcopy(dict(schema)))
        updates.append({"turn_id": event["turn_id"], "tools": accepted})
    return updates

def build_planner_input(seed: SeedRecord, catalog: FunctionCatalog, *,
                        current_config: dict[str, Any],
                        initial_runtime_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Whitelist fields instead of leaking arbitrary generation_metadata."""
    catalog.with_seed_functions(seed)
    updates = original_tool_updates(seed, catalog)
    return {
        "schema_version": PLANNER_INPUT_VERSION,
        "sample_id": seed.sample_id,
        "data_type": seed.data_type,
        "involved_classes": catalog.infer_seed_classes(seed),
        "original_task": {
            "user_queries": copy.deepcopy(seed.Q_old),
            "reference_ground_truth": copy.deepcopy(seed.GT_old),
            "initial_tool_definitions": copy.deepcopy(seed.available_functions),
            "tool_updates": updates,
        },
        "initial_environment": {
            "config": copy.deepcopy(current_config),
            "runtime_snapshot_view": snapshot_view(initial_runtime_snapshot),
            "long_context": seed.data_type == "multi_turn_long_context",
        },
        "generation_contract": {
            "minimum_planned_turns": 2,
            "maximum_planned_turns": planner_max_turns(seed.data_type),
            "maximum_final_turns": 5,
            "functions_per_turn": [1, 3],
            "one_class_per_turn": True,
            "reserved_recovery_turns": int(planner_max_turns(seed.data_type) == 4),
            "argument_limits": copy.deepcopy(ARGUMENT_LIMITS),
        },
    }

def allowed_source_functions(payload: Mapping[str, Any]) -> list[str]:
    task = payload["original_task"]
    schemas = list(task["initial_tool_definitions"])
    for update in task["tool_updates"]:
        schemas.extend(update["tools"])
    return [schema["name"] for schema in schemas]
