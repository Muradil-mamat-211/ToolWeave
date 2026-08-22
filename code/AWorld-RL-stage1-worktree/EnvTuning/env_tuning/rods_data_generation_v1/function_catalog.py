"""Adapter over the public BFCL function-document catalog.

The public documents contain executable bottom-level API schemas but no RODS
HIGH-LEVEL labels or decomposition mappings.  Optional explicit metadata is
supported; a high-level entry without a deterministic mapping fails closed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import FunctionSpec, SeedRecord


CLASS_BY_FILE = {
    "gorilla_file_system": "GorillaFileSystem",
    "math_api": "MathAPI",
    "message_api": "MessageAPI",
    "posting_api": "TwitterAPI",
    "ticket_api": "TicketAPI",
    "trading_bot": "TradingBot",
    "travel_booking": "TravelAPI",
    "vehicle_control": "VehicleControlAPI",
}


class CatalogError(ValueError):
    pass


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in raw.splitlines() if line.strip()]
    records = parsed if isinstance(parsed, list) else [parsed]
    if not all(isinstance(record, dict) for record in records):
        raise CatalogError(f"catalog file contains non-object records: {path}")
    return records


class FunctionCatalog:
    def __init__(self, specs: Iterable[FunctionSpec]):
        self._specs: dict[str, FunctionSpec] = {}
        for spec in specs:
            existing = self._specs.get(spec.name)
            if existing is not None and existing != spec:
                raise CatalogError(f"conflicting schema for function {spec.name}")
            self._specs[spec.name] = spec

    @classmethod
    def from_bfcl_directory(cls, directory: str | Path) -> "FunctionCatalog":
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(root)
        specs: list[FunctionSpec] = []
        for stem, class_name in CLASS_BY_FILE.items():
            path = root / f"{stem}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            for schema in _read_json_or_jsonl(path):
                name = schema.get("name")
                if not isinstance(name, str) or not name:
                    raise CatalogError(f"function without a name in {path}")
                raw_level = str(schema.get("x-rods-level", "BOTTOM_LEVEL")).upper()
                decomposition = schema.get("x-rods-decomposition", [])
                if not isinstance(decomposition, list):
                    raise CatalogError(f"invalid decomposition metadata for {name}")
                specs.append(
                    FunctionSpec(
                        name=name,
                        class_name=class_name,
                        schema=copy.deepcopy(schema),
                        level=raw_level,
                        decomposition=tuple(copy.deepcopy(decomposition)),
                    )
                )
        return cls(specs)

    @classmethod
    def from_training_parquet(cls, parquet_path: str | Path) -> "FunctionCatalog":
        """Build schemas from the exact BFCL rows used by Training.

        EnvTuning's checked-in ``bfcl_train.parquet`` and the separately
        downloaded BFCL function-document directory are not schema-identical.
        The former is the actor/environment contract: those schemas are
        embedded in each training prompt and their argument lists agree with
        ``bfcl_env``.  Loading them here keeps Generator parameter generation,
        VM execution, and the eventual candidate prompt on one contract.

        A function's BFCL class is recovered deterministically by intersecting
        the row-level ``involved_classes`` sets for every row exposing it.  A
        conflicting schema or a non-unique class mapping fails closed.
        """

        source = Path(parquet_path)
        if not source.is_file():
            raise FileNotFoundError(source)

        # Lazy imports keep the lightweight catalog/unit paths independent of
        # parquet until this production-aligned source is explicitly selected.
        import pandas as pd

        from env_tuning.rods_matchtir_v1.provenance import extract_available_functions

        frame = pd.read_parquet(source)
        required_columns = {"prompt", "extra_info"}
        if not required_columns.issubset(frame.columns):
            raise CatalogError(
                f"training parquet lacks catalog columns: "
                f"{sorted(required_columns - set(frame.columns))}"
            )

        schemas: dict[str, dict[str, Any]] = {}
        possible_classes: dict[str, set[str]] = {}
        observations: dict[str, int] = {}

        for row_index, row in frame.iterrows():
            extra_info = row["extra_info"]
            if not isinstance(extra_info, Mapping):
                raise CatalogError(f"row {row_index} has malformed extra_info")
            interaction_kwargs = extra_info.get("interaction_kwargs")
            if not isinstance(interaction_kwargs, Mapping):
                raise CatalogError(f"row {row_index} lacks interaction_kwargs")
            raw_classes = interaction_kwargs.get("involved_classes")
            if not isinstance(raw_classes, (list, tuple)) and not hasattr(
                raw_classes, "tolist"
            ):
                raise CatalogError(f"row {row_index} has malformed involved_classes")
            classes = {
                str(value)
                for value in (
                    raw_classes.tolist() if hasattr(raw_classes, "tolist") else raw_classes
                )
            }
            if not classes or not classes.issubset(set(CLASS_BY_FILE.values())):
                raise CatalogError(
                    f"row {row_index} has unknown/empty involved_classes: {sorted(classes)}"
                )

            try:
                row_schemas = extract_available_functions(row["prompt"])
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CatalogError(
                    f"row {row_index} has an invalid embedded function catalog: {exc}"
                ) from exc
            if not row_schemas:
                raise CatalogError(f"row {row_index} exposes no functions")

            for schema in row_schemas:
                name = schema.get("name")
                if not isinstance(name, str) or not name:
                    raise CatalogError(f"row {row_index} exposes a nameless function")
                prior = schemas.get(name)
                if prior is not None and prior != schema:
                    raise CatalogError(
                        f"training parquet contains conflicting schemas for function {name}"
                    )
                schemas[name] = copy.deepcopy(schema)
                if name in possible_classes:
                    possible_classes[name].intersection_update(classes)
                else:
                    possible_classes[name] = set(classes)
                observations[name] = observations.get(name, 0) + 1

        if not schemas:
            raise CatalogError("training parquet produced an empty function catalog")

        specs: list[FunctionSpec] = []
        for name, schema in sorted(schemas.items()):
            classes = possible_classes[name]
            if len(classes) != 1:
                raise CatalogError(
                    f"cannot recover a unique BFCL class for {name}: "
                    f"{sorted(classes)} across {observations[name]} rows"
                )
            specs.append(
                FunctionSpec(
                    name=name,
                    class_name=next(iter(classes)),
                    schema=schema,
                )
            )
        return cls(specs)

    def with_seed_functions(self, seed: SeedRecord) -> "FunctionCatalog":
        """Verify a seed against the configured, execution-aligned catalog."""

        for schema in seed.available_functions:
            name = schema.get("name")
            if not isinstance(name, str) or name not in self._specs:
                raise CatalogError(f"seed exposes unknown function: {name!r}")
            official = self._specs[name]
            if official.schema != schema:
                raise CatalogError(f"seed schema differs from public BFCL schema: {name}")
        return self

    def get(self, name: str) -> FunctionSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise CatalogError(f"unknown BFCL function: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def class_for_function(self) -> dict[str, str]:
        return {name: spec.class_name for name, spec in self._specs.items()}

    def functions_for_classes(self, classes: Iterable[str]) -> list[FunctionSpec]:
        class_set = set(classes)
        return sorted(
            (spec for spec in self._specs.values() if spec.class_name in class_set),
            key=lambda spec: (spec.class_name, spec.name),
        )

    def infer_seed_classes(self, seed: SeedRecord) -> list[str]:
        classes: set[str] = set()
        for schema in seed.available_functions:
            name = schema.get("name")
            if isinstance(name, str) and name in self._specs:
                classes.add(self._specs[name].class_name)
        for turns in seed.GT_old:
            if not isinstance(turns, list):
                continue
            for raw_call in turns:
                if isinstance(raw_call, str):
                    name = raw_call.split("(", 1)[0].strip()
                    if name in self._specs:
                        classes.add(self._specs[name].class_name)
        classes.update(
            key for key in seed.initial_config if key in set(CLASS_BY_FILE.values())
        )
        if not classes:
            raise CatalogError("cannot infer any BFCL class from seed contract")
        return sorted(classes)

    def validate_arguments(self, spec: FunctionSpec, arguments: Mapping[str, Any]) -> None:
        params = spec.schema.get("parameters", {})
        properties = params.get("properties", {}) if isinstance(params, Mapping) else {}
        required = params.get("required", []) if isinstance(params, Mapping) else []
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise CatalogError(f"malformed parameter schema for {spec.name}")
        unknown = set(arguments) - set(properties)
        missing = set(required) - set(arguments)
        if unknown:
            raise CatalogError(f"schema-external parameters for {spec.name}: {sorted(unknown)}")
        if missing:
            raise CatalogError(f"missing required parameters for {spec.name}: {sorted(missing)}")
        for key, value in arguments.items():
            self._validate_value(value, properties[key], path=f"{spec.name}.{key}")

    @classmethod
    def _validate_value(cls, value: Any, schema: Any, *, path: str) -> None:
        if not isinstance(schema, Mapping):
            return
        raw_type = schema.get("type")
        if isinstance(raw_type, Mapping):
            # Two malformed public BFCL schemas contain a schema object in the
            # type slot. Validate against that object rather than guessing.
            cls._validate_value(value, raw_type, path=path)
            return
        expected = {
            "dict": dict,
            "object": dict,
            "string": str,
            "integer": int,
            "float": (int, float),
            "number": (int, float),
            "array": list,
            "boolean": bool,
        }.get(raw_type)
        if expected is not None:
            if raw_type == "integer" and isinstance(value, bool):
                raise CatalogError(f"{path} must be integer, not boolean")
            if raw_type in {"float", "number"} and isinstance(value, bool):
                raise CatalogError(f"{path} must be numeric, not boolean")
            if not isinstance(value, expected):
                raise CatalogError(f"{path} has wrong type for {raw_type}")
        if raw_type == "array" and isinstance(value, list):
            item_schema = schema.get("items", {})
            for index, item in enumerate(value):
                cls._validate_value(item, item_schema, path=f"{path}[{index}]")

    def decompose(self, spec: FunctionSpec) -> tuple[dict[str, Any], ...]:
        if spec.level == "BOTTOM_LEVEL":
            return ()
        if spec.level != "HIGH_LEVEL":
            raise CatalogError(f"unknown function level for {spec.name}: {spec.level}")
        if not spec.decomposition:
            raise CatalogError(
                f"HIGH-LEVEL function {spec.name} has no published deterministic decomposition"
            )
        return spec.decomposition

    def tool_schemas(self, names: Iterable[str]) -> list[dict[str, Any]]:
        return [copy.deepcopy(self.get(name).schema) for name in names]
