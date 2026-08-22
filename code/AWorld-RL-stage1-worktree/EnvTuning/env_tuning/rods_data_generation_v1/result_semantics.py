"""Contract-aware BFCL execution-result semantics.

SOURCE_STATUS = PROJECT_SEMANTIC_GUARD

EnvTuning's public execution helper distinguishes Python/VM exceptions and
explicit ``{"error": ...}`` payloads.  It does not publish a semantic result
classifier for synthesis.  This project layer therefore audits the concrete
return contracts in ``bfcl_env/func_source_code_wo_aug`` and distinguishes a
successful negative business answer from a hard execution failure.  In
particular, it intentionally contains no generic ``"not found"`` substring
rule.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


SOURCE_STATUS = "PROJECT_SEMANTIC_GUARD"


# Bounded normalization protects the classifier from both untrusted model/API
# payloads and accidental host-memory amplification.  These are engineering
# safety limits, not RODS paper hyperparameters.
_MAX_PARSE_CHARS = 1_048_576
_MAX_NORMALIZATION_DEPTH = 16
_MAX_COLLECTION_ITEMS = 10_000
_MAX_STRING_DECODE_PASSES = 3


class ExecutionSemanticOutcome(str, Enum):
    HARD_ERROR = "HARD_ERROR"
    DOMAIN_NEGATIVE = "DOMAIN_NEGATIVE"
    SUCCESS = "SUCCESS"


@dataclass(frozen=True)
class ResultSemanticClassification:
    outcome: ExecutionSemanticOutcome
    detail: str
    source_path: str | None = None


@dataclass(frozen=True)
class DomainNegativeContract:
    function_name: str
    source_path: tuple[str, ...]
    exact_value: Any
    downstream_parameter_names: frozenset[str] = frozenset()


# These are exact return contracts observed in the public, non-augmented BFCL
# implementations.  Empty search/list results are valid answers and are not
# listed unless the implementation publishes an explicit sentinel.
DOMAIN_NEGATIVE_CONTRACTS: tuple[DomainNegativeContract, ...] = (
    DomainNegativeContract(
        "get_symbol_by_name",
        ("symbol",),
        "Stock not found",
        frozenset({"symbol", "stock", "ticker"}),
    ),
    DomainNegativeContract(
        "get_nearest_airport_by_city",
        ("nearest_airport",),
        "Unknown",
        frozenset({"airport", "travel_from", "travel_to", "departure", "destination"}),
    ),
    DomainNegativeContract(
        "estimate_drive_feasibility_by_mileage",
        ("canDrive",),
        False,
    ),
    DomainNegativeContract("follow_user", ("follow_status",), False),
    DomainNegativeContract("unfollow_user", ("unfollow_status",), False),
    DomainNegativeContract("message_get_login_status", ("login_status",), False),
    DomainNegativeContract("posting_get_login_status", ("login_status",), False),
    DomainNegativeContract("ticket_get_login_status", ("username",), False),
)


# False here means the attempted state transition/verification did not happen;
# unlike a read-only negative answer it cannot support downstream execution.
HARD_FAILURE_FLAG_CONTRACTS: dict[str, tuple[str, ...]] = {
    "add_contact": ("added_status",),
    "authenticate_twitter": ("authentication_status",),
    "book_flight": ("booking_status",),
    "cancel_booking": ("cancel_status",),
    "delete_message": ("deleted_status",),
    "message_login": ("login_status",),
    "purchase_insurance": ("insurance_status",),
    "send_message": ("sent_status",),
    "ticket_login": ("success",),
    "logout": ("success",),
    "verify_traveler_information": ("verification_status",),
}


def normalize_execution_result(value: Any) -> Any:
    """Safely canonicalize structured and stringified BFCL results.

    Public BFCL helpers normally return a Python object or JSON text.  Some
    real traces additionally contain a Python-literal-like container string.
    We decode only bounded strings that visibly represent a container, use
    :func:`ast.literal_eval` rather than ``eval``, and recursively normalize
    nested containers.  Ordinary prose is returned unchanged.
    """

    item_count = 0

    def visit(item: Any, *, depth: int, decode_passes: int) -> Any:
        nonlocal item_count
        if depth > _MAX_NORMALIZATION_DEPTH:
            return item
        if isinstance(item, str):
            stripped = item.strip()
            if (
                decode_passes >= _MAX_STRING_DECODE_PASSES
                or len(stripped) > _MAX_PARSE_CHARS
                or len(stripped) < 2
                or (stripped[0], stripped[-1])
                not in {("{", "}"), ("[", "]"), ("(", ")")}
            ):
                return item
            parsed: Any
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, RecursionError):
                try:
                    parsed = ast.literal_eval(stripped)
                except (ValueError, SyntaxError, MemoryError, RecursionError):
                    return item
            if not isinstance(parsed, (Mapping, list, tuple)):
                return item
            return visit(parsed, depth=depth + 1, decode_passes=decode_passes + 1)
        if isinstance(item, Mapping):
            if len(item) > _MAX_COLLECTION_ITEMS:
                return item
            item_count += len(item)
            if item_count > _MAX_COLLECTION_ITEMS:
                return item
            return {
                str(key): visit(child, depth=depth + 1, decode_passes=decode_passes)
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            if len(item) > _MAX_COLLECTION_ITEMS:
                return item
            item_count += len(item)
            if item_count > _MAX_COLLECTION_ITEMS:
                return item
            return [
                visit(child, depth=depth + 1, decode_passes=decode_passes)
                for child in item
            ]
        return item

    return visit(value, depth=0, decode_passes=0)


def _value_at_path(value: Any, path: Sequence[str]) -> tuple[bool, Any]:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False, None
        current = current[key]
    return True, current


def find_explicit_vm_error(value: Any, *, path: str = "result") -> tuple[str, str] | None:
    """Find official/public explicit error transport, without fuzzy text rules."""

    if isinstance(value, str):
        lowered = value.casefold()
        if lowered.startswith("error during execution:") or lowered.startswith("error:"):
            return value, path
        return None
    if isinstance(value, Mapping):
        if "error" in value:
            return str(value["error"]), f"{path}.error"
        for key, item in value.items():
            nested = find_explicit_vm_error(item, path=f"{path}.{key}")
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested = find_explicit_vm_error(item, path=f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def domain_negative_contract(
    function_name: str, result: Any
) -> DomainNegativeContract | None:
    result = normalize_execution_result(result)
    for contract in DOMAIN_NEGATIVE_CONTRACTS:
        if contract.function_name != function_name:
            continue
        found, value = _value_at_path(result, contract.source_path)
        if found and value == contract.exact_value:
            return contract
    return None


def classify_execution_result(
    function_name: str, result: Any
) -> ResultSemanticClassification:
    """Classify one decoded result from its audited function return contract."""

    normalized = normalize_execution_result(result)
    explicit_error = find_explicit_vm_error(normalized)
    if explicit_error is not None:
        detail, path = explicit_error
        return ResultSemanticClassification(
            ExecutionSemanticOutcome.HARD_ERROR,
            detail,
            path,
        )

    negative = domain_negative_contract(function_name, normalized)
    if negative is not None:
        source_path = "result." + ".".join(negative.source_path)
        return ResultSemanticClassification(
            ExecutionSemanticOutcome.DOMAIN_NEGATIVE,
            f"audited domain-negative contract at {source_path}",
            source_path,
        )

    failure_path = HARD_FAILURE_FLAG_CONTRACTS.get(function_name)
    if failure_path is not None:
        found, flag = _value_at_path(normalized, failure_path)
        if found and flag is False:
            source_path = "result." + ".".join(failure_path)
            return ResultSemanticClassification(
                ExecutionSemanticOutcome.HARD_ERROR,
                f"audited failure flag {source_path}=false",
                source_path,
            )

    return ResultSemanticClassification(
        ExecutionSemanticOutcome.SUCCESS,
        "no explicit VM error, audited negative sentinel, or audited false failure flag",
    )


_SUSPICIOUS_RESULT_TOKENS = frozenset(
    {
        "error",
        "fail",
        "failed",
        "failure",
        "invalid",
        "not authenticated",
        "not found",
        "unavailable",
        "unknown",
    }
)


def find_unclassified_suspicious_results(
    function_name: str, result: Any
) -> list[dict[str, Any]]:
    """Return non-blocking telemetry for unclassified suspicious sentinels.

    This helper deliberately does **not** alter
    :func:`classify_execution_result`.  It runs only for results currently
    classified as SUCCESS and records exact function/path/value evidence for
    contract auditing.  New sentinels remain SUCCESS until their public BFCL
    return contract is reviewed; telemetry cannot reject a candidate.
    """

    normalized_result = normalize_execution_result(result)
    if classify_execution_result(function_name, normalized_result).outcome != ExecutionSemanticOutcome.SUCCESS:
        return []

    observations: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        suspicious = False
        reason = ""
        if value is False:
            suspicious = True
            reason = "unclassified false-like business result"
        elif isinstance(value, str):
            normalized = " ".join(value.casefold().split())
            if any(token in normalized for token in _SUSPICIOUS_RESULT_TOKENS):
                suspicious = True
                reason = "unclassified suspicious string sentinel"
        if suspicious:
            observations.append(
                {
                    "function": function_name,
                    "path": path,
                    "value": value,
                    "reason": reason,
                    "source_status": "PROJECT_SEMANTIC_GUARD_TELEMETRY_ONLY",
                    "changes_execution_semantics": False,
                }
            )
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(normalized_result, "result")
    return observations
