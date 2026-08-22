"""Deterministic unit-aware grounding for numeric GT arguments.

SOURCE_STATUS = PROJECT_UNIT_SEMANTIC_GUARD

RODS does not publish this precision gate.  Contracts below are audited from
the active 128-function schemas and the public ``bfcl_env`` implementation.
Runtime LLM output never determines a function's unit.  Ambiguous descriptions
remain ``UNIT_UNKNOWN`` and do not acquire a guessed conversion.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from ..function_catalog import FunctionCatalog
from ..models import ConversationDraft, ExecutionRecord, GateResult
from ..result_semantics import (
    ExecutionSemanticOutcome,
    classify_execution_result,
    normalize_execution_result,
)


SOURCE_STATUS = "PROJECT_UNIT_SEMANTIC_GUARD"


@dataclass(frozen=True)
class UnitContract:
    dimension: str
    canonical_unit: str
    evidence: str


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    dimension: str
    source_turn: int
    source_text: str


# Static parameter contracts are restricted to explicit active-schema text.
_STATIC_PARAMETER_CONTRACTS: dict[tuple[str, str], UnitContract] = {
    ("fillFuelTank", "fuelAmount"): UnitContract(
        "volume", "gallon", "schema: amount of fuel to fill in gallons"
    ),
    ("setCruiseControl", "distanceToNextVehicle"): UnitContract(
        "distance", "meter", "schema: distance to the next vehicle in meters"
    ),
    ("estimate_drive_feasibility_by_mileage", "distance"): UnitContract(
        "distance", "mile", "schema: distance to travel in miles"
    ),
    ("liter_to_gallon", "liter"): UnitContract(
        "volume", "liter", "schema/public implementation: amount of liter"
    ),
    ("gallon_to_liter", "gallon"): UnitContract(
        "volume", "gallon", "schema/public implementation: amount of gallon"
    ),
    ("notify_price_change", "threshold"): UnitContract(
        "ratio", "percent", "schema: percentage change threshold"
    ),
    ("set_budget_limit", "budget_limit"): UnitContract(
        "currency", "usd", "schema: budget limit in USD"
    ),
}


# The unit of these numeric parameters is selected by another schema argument.
_DYNAMIC_PARAMETER_UNITS: dict[tuple[str, str], tuple[str, str | None]] = {
    ("adjustClimateControl", "temperature"): ("unit", "celsius"),
    ("imperial_si_conversion", "value"): ("unit_in", None),
    ("si_unit_conversion", "value"): ("unit_in", None),
    ("compute_exchange_rate", "value"): ("base_currency", None),
}


_STATIC_RESULT_CONTRACTS: dict[tuple[str, str], UnitContract] = {
    ("fillFuelTank", "fuelLevel"): UnitContract("volume", "gallon", "schema response"),
    ("liter_to_gallon", "gallon"): UnitContract("volume", "gallon", "schema response"),
    ("gallon_to_liter", "liter"): UnitContract("volume", "liter", "schema response"),
    ("setCruiseControl", "distanceToNextVehicle"): UnitContract(
        "distance", "meter", "schema response"
    ),
    ("setCruiseControl", "currentSpeed"): UnitContract(
        "speed", "kilometer_per_hour", "schema response: km/h"
    ),
    ("estimate_distance", "distance"): UnitContract(
        "distance", "kilometer", "schema response: km"
    ),
    ("get_outside_temperature_from_google", "outsideTemperature"): UnitContract(
        "temperature", "celsius", "schema response"
    ),
    ("get_outside_temperature_from_weather_com", "outsideTemperature"): UnitContract(
        "temperature", "celsius", "schema response"
    ),
    ("adjustClimateControl", "currentTemperature"): UnitContract(
        "temperature", "celsius", "schema response"
    ),
    ("set_budget_limit", "budget_limit"): UnitContract(
        "currency", "usd", "schema response"
    ),
}


_DYNAMIC_RESULT_UNITS: dict[tuple[str, str], str] = {
    ("imperial_si_conversion", "result"): "unit_out",
    ("si_unit_conversion", "result"): "unit_out",
    ("compute_exchange_rate", "exchanged_value"): "target_currency",
}


_CONVERTER_FUNCTIONS = frozenset(
    {
        "liter_to_gallon",
        "gallon_to_liter",
        "imperial_si_conversion",
        "si_unit_conversion",
        "compute_exchange_rate",
    }
)


_UNIT_ALIASES: dict[str, tuple[str, str]] = {
    "liter": ("volume", "liter"),
    "liters": ("volume", "liter"),
    "litre": ("volume", "liter"),
    "litres": ("volume", "liter"),
    "gallon": ("volume", "gallon"),
    "gallons": ("volume", "gallon"),
    "meter": ("distance", "meter"),
    "meters": ("distance", "meter"),
    "metre": ("distance", "meter"),
    "metres": ("distance", "meter"),
    "foot": ("distance", "foot"),
    "feet": ("distance", "foot"),
    "ft": ("distance", "foot"),
    "mile": ("distance", "mile"),
    "miles": ("distance", "mile"),
    "kilometer": ("distance", "kilometer"),
    "kilometers": ("distance", "kilometer"),
    "kilometre": ("distance", "kilometer"),
    "kilometres": ("distance", "kilometer"),
    "km": ("distance", "kilometer"),
    "centimeter": ("distance", "centimeter"),
    "centimeters": ("distance", "centimeter"),
    "cm": ("distance", "centimeter"),
    "inch": ("distance", "inch"),
    "inches": ("distance", "inch"),
    "yard": ("distance", "yard"),
    "yards": ("distance", "yard"),
    "celsius": ("temperature", "celsius"),
    "fahrenheit": ("temperature", "fahrenheit"),
    "percent": ("ratio", "percent"),
    "percentage": ("ratio", "percent"),
    "%": ("ratio", "percent"),
    "usd": ("currency", "usd"),
    "dollar": ("currency", "usd"),
    "dollars": ("currency", "usd"),
    "mph": ("speed", "mile_per_hour"),
    "km/h": ("speed", "kilometer_per_hour"),
    "kph": ("speed", "kilometer_per_hour"),
    "kg": ("mass", "kilogram"),
    "kilogram": ("mass", "kilogram"),
    "kilograms": ("mass", "kilogram"),
    "lb": ("mass", "pound"),
    "lbs": ("mass", "pound"),
    "pound": ("mass", "pound"),
    "pounds": ("mass", "pound"),
}


_DYNAMIC_UNIT_ALIASES = {
    "m": ("distance", "meter"),
    "mm": ("distance", "millimeter"),
    "um": ("distance", "micrometer"),
    "nm": ("distance", "nanometer"),
    "yd": ("distance", "yard"),
    "in": ("distance", "inch"),
    **_UNIT_ALIASES,
}


# Deterministic factors/affine conversions present in active BFCL converter
# implementations.  Currency conversion is deliberately excluded because its
# rate comes from VM execution rather than a project constant.
_LINEAR_TO_BASE: dict[str, tuple[str, float]] = {
    "liter": ("volume", 1.0),
    "gallon": ("volume", 3.78541),
    "meter": ("distance", 1.0),
    "foot": ("distance", 0.3048),
    "mile": ("distance", 1609.34),
    "kilometer": ("distance", 1000.0),
    "centimeter": ("distance", 0.01),
    "millimeter": ("distance", 0.001),
    "micrometer": ("distance", 1e-6),
    "nanometer": ("distance", 1e-9),
    "inch": ("distance", 0.0254),
    "yard": ("distance", 0.9144),
    "kilogram": ("mass", 1.0),
    "pound": ("mass", 0.453592),
    "mile_per_hour": ("speed", 1.0),
    "kilometer_per_hour": ("speed", 0.621371),
    "percent": ("ratio", 1.0),
}


_QUERY_UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>kilometers?|kilometres?|centimeters?|meters?|metres?|"
    r"liters?|litres?|gallons?|fahrenheit|celsius|percentage|percent|"
    r"dollars?|miles?|inches?|yards?|pounds?|kilograms?|feet|foot|"
    r"km/h|kph|mph|cm|km|ft|lbs?|kg|%)(?![A-Za-z])",
    re.IGNORECASE,
)
_USD_PREFIX_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])\$\s*(?P<value>\d[\d,]*(?:\.\d+)?)"
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-8)


def _unit_from_name(raw: Any) -> UnitContract | None:
    if not isinstance(raw, str):
        return None
    normalized = " ".join(raw.strip().casefold().split())
    if normalized in _DYNAMIC_UNIT_ALIASES:
        dimension, unit = _DYNAMIC_UNIT_ALIASES[normalized]
        return UnitContract(dimension, unit, f"explicit unit argument={raw!r}")
    if re.fullmatch(r"[a-z]{3}", normalized):
        # Active TravelAPI currency enums are explicit ISO-style codes.
        return UnitContract("currency", normalized, f"explicit currency argument={raw!r}")
    return None


def convert_unit_value(value: float, source_unit: str, target_unit: str) -> float:
    """Convert across the audited deterministic table.

    This utility verifies quantities; it never rewrites a candidate argument.
    """

    if source_unit == target_unit:
        return float(value)
    if {source_unit, target_unit} <= {"celsius", "fahrenheit"}:
        if source_unit == "celsius":
            return float(value) * 1.8 + 32.0
        return (float(value) - 32.0) * (5.0 / 9.0)
    source = _LINEAR_TO_BASE.get(source_unit)
    target = _LINEAR_TO_BASE.get(target_unit)
    if source is None or target is None or source[0] != target[0]:
        raise ValueError(f"no audited conversion from {source_unit} to {target_unit}")
    return float(value) * source[1] / target[1]


def _parameter_contract(
    function_name: str, parameter: str, arguments: Mapping[str, Any]
) -> UnitContract | None:
    static = _STATIC_PARAMETER_CONTRACTS.get((function_name, parameter))
    if static is not None:
        return static
    dynamic = _DYNAMIC_PARAMETER_UNITS.get((function_name, parameter))
    if dynamic is None:
        return None
    unit_parameter, default = dynamic
    resolved = _unit_from_name(arguments.get(unit_parameter, default))
    if resolved is None:
        return None
    return UnitContract(
        resolved.dimension,
        resolved.canonical_unit,
        f"schema argument {unit_parameter}; {resolved.evidence}",
    )


def _result_contract(record: ExecutionRecord, source_path: str) -> UnitContract | None:
    leaf = re.sub(r"\[\d+\]", "", source_path).rsplit(".", 1)[-1]
    static = _STATIC_RESULT_CONTRACTS.get((record.call.name, leaf))
    if static is not None:
        return static
    dynamic_parameter = _DYNAMIC_RESULT_UNITS.get((record.call.name, leaf))
    if dynamic_parameter is None:
        return None
    resolved = _unit_from_name(record.call.arguments.get(dynamic_parameter))
    if resolved is None:
        return None
    return UnitContract(
        resolved.dimension,
        resolved.canonical_unit,
        f"audited converter output selected by {dynamic_parameter}",
    )


def _iter_scalar_paths(value: Any, *, path: str = "result") -> Iterable[tuple[str, Any]]:
    normalized = normalize_execution_result(value)
    if isinstance(normalized, Mapping):
        for key, child in normalized.items():
            yield from _iter_scalar_paths(child, path=f"{path}.{key}")
    elif isinstance(normalized, (list, tuple)):
        for index, child in enumerate(normalized):
            yield from _iter_scalar_paths(child, path=f"{path}[{index}]")
    else:
        yield path, normalized


def _query_quantities(queries: Sequence[tuple[int, str]]) -> list[Quantity]:
    output: list[Quantity] = []
    for turn_id, text in queries:
        for match in _QUERY_UNIT_PATTERN.finditer(text):
            unit = _unit_from_name(match.group("unit"))
            if unit is None:
                continue
            output.append(
                Quantity(
                    float(match.group("value").replace(",", "")),
                    unit.canonical_unit,
                    unit.dimension,
                    turn_id,
                    match.group(0),
                )
            )
        for match in _USD_PREFIX_PATTERN.finditer(text):
            output.append(
                Quantity(
                    float(match.group("value").replace(",", "")),
                    "usd",
                    "currency",
                    turn_id,
                    match.group(0),
                )
            )
    return output


def _prior_output_sources(
    value: float, records: Sequence[tuple[int, int, ExecutionRecord]]
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for turn_id, call_id, record in reversed(records):
        semantic = classify_execution_result(record.call.name, record.execution_result)
        if semantic.outcome != ExecutionSemanticOutcome.SUCCESS:
            continue
        for source_path, source_value in _iter_scalar_paths(record.execution_result):
            if not _is_number(source_value) or not _close(float(source_value), value):
                continue
            contract = _result_contract(record, source_path)
            matches.append(
                {
                    "source_turn": turn_id,
                    "source_call": call_id,
                    "source_path": source_path,
                    "producer": record.call.name,
                    "contract": contract,
                }
            )
    return matches


def _schema_description(catalog: FunctionCatalog, function: str, parameter: str) -> str:
    properties = catalog.get(function).schema.get("parameters", {}).get("properties", {})
    raw = properties.get(parameter, {}) if isinstance(properties, Mapping) else {}
    return str(raw.get("description", "")) if isinstance(raw, Mapping) else ""


def audit_catalog_unit_contracts(catalog: FunctionCatalog) -> list[dict[str, Any]]:
    """Return the explicit active-catalog unit audit, including unknowns."""

    rows: list[dict[str, Any]] = []
    for function_name in catalog.names():
        spec = catalog.get(function_name)
        properties = spec.schema.get("parameters", {}).get("properties", {})
        if not isinstance(properties, Mapping):
            continue
        for parameter, raw_schema in properties.items():
            if not isinstance(raw_schema, Mapping) or raw_schema.get("type") not in {
                "float",
                "number",
                "integer",
            }:
                continue
            contract = _parameter_contract(function_name, str(parameter), {})
            description = str(raw_schema.get("description", ""))
            if contract is not None:
                rows.append(
                    {
                        "function": function_name,
                        "parameter": str(parameter),
                        "dimension": contract.dimension,
                        "canonical_unit": contract.canonical_unit,
                        "status": "AUDITED_STATIC",
                        "evidence": contract.evidence,
                        "schema_description": description,
                        "source_status": SOURCE_STATUS,
                    }
                )
                continue
            if (function_name, str(parameter)) in _DYNAMIC_PARAMETER_UNITS:
                rows.append(
                    {
                        "function": function_name,
                        "parameter": str(parameter),
                        "dimension": "DYNAMIC",
                        "canonical_unit": "DYNAMIC_FROM_SCHEMA_ARGUMENT",
                        "status": "AUDITED_DYNAMIC",
                        "evidence": str(_DYNAMIC_PARAMETER_UNITS[(function_name, str(parameter))]),
                        "schema_description": description,
                        "source_status": SOURCE_STATUS,
                    }
                )
                continue
            if "m/h" in description.casefold():
                rows.append(
                    {
                        "function": function_name,
                        "parameter": str(parameter),
                        "dimension": "UNIT_UNKNOWN",
                        "canonical_unit": "UNIT_UNKNOWN",
                        "status": "UNIT_UNKNOWN",
                        "evidence": "schema token 'm/h' is ambiguous; no unit guessed",
                        "schema_description": description,
                        "source_status": SOURCE_STATUS,
                    }
                )
    return rows


def unit_semantic_gate(
    draft: ConversationDraft, *, catalog: FunctionCatalog
) -> GateResult:
    """Reject explicit unit mismatches without rewriting query or GT values."""

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    prior_records: list[tuple[int, int, ExecutionRecord]] = []
    visible_queries: list[tuple[int, str]] = []

    for turn_id, turn in enumerate(draft.turns):
        if turn.query.strip():
            visible_queries.append((turn_id, turn.query))
        updated_records: list[ExecutionRecord] = []
        for call_id, record in enumerate(turn.execution_records):
            provenance = dict(record.dependency_provenance)
            record_checks: list[dict[str, Any]] = []
            if not turn.is_intentional_missing:
                for parameter, raw_value in record.call.arguments.items():
                    if not _is_number(raw_value):
                        continue
                    contract = _parameter_contract(
                        record.call.name, parameter, record.call.arguments
                    )
                    if contract is None:
                        continue
                    value = float(raw_value)
                    base = {
                        "turn_id": turn_id,
                        "call_id": call_id,
                        "function": record.call.name,
                        "parameter": parameter,
                        "value": raw_value,
                        "dimension": contract.dimension,
                        "canonical_unit": contract.canonical_unit,
                        "contract_evidence": contract.evidence,
                        "source_status": SOURCE_STATUS,
                    }
                    prior_sources = _prior_output_sources(value, prior_records)
                    known_sources = [
                        item for item in prior_sources if item["contract"] is not None
                    ]
                    if known_sources:
                        source = known_sources[0]
                        source_contract: UnitContract = source["contract"]
                        row = {
                            **base,
                            "source_type": "PRIOR_TOOL_OUTPUT",
                            "source_turn": source["source_turn"],
                            "source_call": source["source_call"],
                            "source_path": source["source_path"],
                            "source_unit": source_contract.canonical_unit,
                            "transformation": (
                                "UNIT_CONVERSION"
                                if source["producer"] in _CONVERTER_FUNCTIONS
                                else "UNIT_PRESERVING_TOOL_OUTPUT"
                            ),
                        }
                        if (
                            source_contract.dimension != contract.dimension
                            or source_contract.canonical_unit != contract.canonical_unit
                        ):
                            row.update(
                                status="FAIL_UNIT_MISMATCH",
                                reason=(
                                    f"prior output unit {source_contract.canonical_unit} "
                                    f"cannot ground {contract.canonical_unit}"
                                ),
                            )
                            failures.append(row)
                        else:
                            row["status"] = "PASS"
                        checks.append(row)
                        record_checks.append(row)
                        continue

                    quantities = _query_quantities(visible_queries)
                    exact = [item for item in quantities if _close(item.value, value)]
                    if exact:
                        mismatches = [
                            item
                            for item in exact
                            if item.dimension != contract.dimension
                            or item.unit != contract.canonical_unit
                        ]
                        if mismatches:
                            source = mismatches[0]
                            row = {
                                **base,
                                "source_type": "USER_CONTEXT",
                                "source_turn": source.source_turn,
                                "source_text": source.source_text,
                                "source_unit": source.unit,
                                "status": "FAIL_UNIT_MISMATCH",
                                "reason": (
                                    f"explicit query quantity uses {source.unit}, "
                                    f"but {record.call.name}.{parameter} requires "
                                    f"{contract.canonical_unit}"
                                ),
                            }
                            failures.append(row)
                        else:
                            source = exact[0]
                            row = {
                                **base,
                                "source_type": "USER_CONTEXT",
                                "source_turn": source.source_turn,
                                "source_text": source.source_text,
                                "source_unit": source.unit,
                                "status": "PASS",
                                "transformation": "NONE",
                            }
                        checks.append(row)
                        record_checks.append(row)
                        continue

                    unaudited_conversion = None
                    for item in quantities:
                        if item.dimension != contract.dimension or item.unit == contract.canonical_unit:
                            continue
                        try:
                            converted = convert_unit_value(
                                item.value, item.unit, contract.canonical_unit
                            )
                        except ValueError:
                            continue
                        if _close(converted, value):
                            unaudited_conversion = item
                            break
                    if unaudited_conversion is not None:
                        row = {
                            **base,
                            "source_type": "USER_CONTEXT",
                            "source_turn": unaudited_conversion.source_turn,
                            "source_text": unaudited_conversion.source_text,
                            "source_unit": unaudited_conversion.unit,
                            "status": "FAIL_UNAUDITED_CONVERSION",
                            "reason": (
                                "numeric conversion has no successful audited conversion "
                                "tool-output provenance"
                            ),
                        }
                        failures.append(row)
                    elif prior_sources:
                        row = {
                            **base,
                            "source_type": "PRIOR_TOOL_OUTPUT",
                            "status": "FAIL_UNIT_SOURCE_UNKNOWN",
                            "reason": "matching prior numeric output has no audited unit contract",
                            "source_paths": [item["source_path"] for item in prior_sources],
                        }
                        failures.append(row)
                    else:
                        row = {
                            **base,
                            "source_type": "DEFER_TO_SEMANTIC_GROUNDING",
                            "status": "UNIT_UNSPECIFIED_NO_CONFLICT",
                            "reason": (
                                "no explicit conflicting unit; scalar provenance remains "
                                "the semantic-grounding gate's responsibility"
                            ),
                        }
                    checks.append(row)
                    record_checks.append(row)
            provenance["unit_semantics"] = record_checks
            updated_records.append(replace(record, dependency_provenance=provenance))
            if not turn.is_intentional_missing:
                prior_records.append((turn_id, call_id, updated_records[-1]))
        turn.execution_records = updated_records

    metadata = {
        "source_status": SOURCE_STATUS,
        "checks": checks,
        "failures": failures,
        "catalog_audit": audit_catalog_unit_contracts(catalog),
        "gt_rewritten": False,
    }
    if failures:
        first = failures[0]
        return GateResult(
            "unit_semantic_gate",
            False,
            f"{first['status']}: {first['reason']}",
            metadata,
        )
    return GateResult(
        "unit_semantic_gate",
        True,
        "all explicit unit-bearing GT quantities are compatible with audited contracts",
        metadata,
    )
