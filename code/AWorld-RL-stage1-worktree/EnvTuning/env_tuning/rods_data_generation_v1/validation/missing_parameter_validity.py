"""Policy-visible validity checks for reconstructed Missing Parameter tasks.

SOURCE_STATUS = PROJECT_MISSING_PARAMETER_VALIDITY_GUARD

The actor may clarify only when a required value is absent or ambiguous in its
visible interaction history.  Hidden raw environment state is deliberately
excluded from this module even though other deterministic GT gates may inspect
that state.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from ..function_catalog import FunctionCatalog
from ..models import ConversationDraft, ExecutionRecord, GateResult
from ..result_semantics import (
    ExecutionSemanticOutcome,
    classify_execution_result,
    normalize_execution_result,
)


SOURCE_STATUS = "PROJECT_MISSING_PARAMETER_VALIDITY_GUARD"
_GENERIC_TOKENS = {
    "id",
    "ids",
    "get",
    "set",
    "list",
    "all",
    "user",
    "current",
    "details",
    "detail",
    "value",
    "values",
    "result",
}
_UNSUPPORTED_STRINGS = {"", "none", "null", "unknown", "undefined", "n/a"}
_ENTITY_TOKEN_ALIASES = {
    "accounts": "account",
    "bookings": "booking",
    "cards": "card",
    "messages": "message",
    "orders": "order",
    "tickets": "ticket",
    "tweets": "tweet",
}


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value).replace("_", " ")
    return {
        _ENTITY_TOKEN_ALIASES.get(token.casefold(), token.casefold())
        for token in re.findall(r"[A-Za-z0-9]+", expanded)
    }


def _entity_tokens(parameter: str) -> set[str]:
    return _tokens(parameter) - _GENERIC_TOKENS


def _source_is_compatible(parameter: str, function_name: str, path: str) -> bool:
    parameter_tokens = _tokens(parameter)
    entities = _entity_tokens(parameter)
    raw_path_tokens = _tokens(path)
    path_tokens = raw_path_tokens - _GENERIC_TOKENS
    if "id" in parameter_tokens:
        if parameter_tokens.issubset(raw_path_tokens):
            return True
        # A schema whose required argument is literally ``id`` may consume a
        # visible result field named ``id`` without inventing an entity type.
        if parameter_tokens == {"id"} and "id" in raw_path_tokens:
            return True
    elif entities & path_tokens:
        return True
    if not entities:
        return False
    # Public BFCL frequently returns a generic ``id`` field from an
    # entity-specific function (for example get_user_tickets -> [{id: 1}]).
    # Function context may disambiguate that generic identity only; it must
    # not turn every scalar returned by place_order into an order ID.
    generic_identity = bool({"id", "ids"} & raw_path_tokens) or path.endswith(
        "#mapping_key"
    )
    function_tokens = _tokens(function_name) - _GENERIC_TOKENS
    return generic_identity and bool(entities & function_tokens)


def _iter_scalars(value: Any, *, path: str) -> Iterable[tuple[str, Any]]:
    normalized = normalize_execution_result(value)
    if isinstance(normalized, Mapping):
        for key, child in normalized.items():
            # A mapping key is an actor-visible identity only for a keyed
            # entity collection (booking_400 -> {...}), not for ordinary
            # response field names such as ``status`` or ``order_id``.
            key_is_entity_identity = (
                isinstance(child, Mapping)
                and isinstance(key, (str, int))
                and not isinstance(key, bool)
                and (
                    isinstance(key, int)
                    or bool(re.search(r"\d", str(key)))
                )
            )
            if key_is_entity_identity:
                yield f"{path}.{key}#mapping_key", key
            yield from _iter_scalars(child, path=f"{path}.{key}")
    elif isinstance(normalized, (list, tuple)):
        for index, child in enumerate(normalized):
            yield from _iter_scalars(child, path=f"{path}[{index}]")
    else:
        yield path, normalized


def _usable_identity(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        return value.strip().casefold() not in _UNSUPPORTED_STRINGS
    return isinstance(value, int)


def _coerce_like(value: Any, target: Any) -> Any:
    if isinstance(target, bool):
        return value
    if isinstance(target, int) and isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if isinstance(target, float) and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _stable_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr)


def _value_mentioned(value: Any, text: str) -> bool:
    if isinstance(value, str):
        normalized = " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
        context = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
        return bool(normalized) and normalized in context
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raw = f"{float(value):g}"
        return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(raw)}(?![A-Za-z0-9])", text))
    return _stable_value(value).casefold() in text.casefold()


def _append_candidate(
    candidates: dict[str, Any],
    sources: list[dict[str, Any]],
    *,
    value: Any,
    target: Any,
    source: dict[str, Any],
) -> None:
    candidate = _coerce_like(value, target)
    if not _usable_identity(candidate):
        return
    key = _stable_value(candidate)
    candidates[key] = candidate
    sources.append({"value": candidate, **source})


def evaluate_missing_parameter_validity(
    draft: ConversationDraft,
    *,
    affected_turn: int,
    parameter: str,
    target_value: Any,
    affected_query: str,
    catalog: FunctionCatalog,
) -> dict[str, Any]:
    """Audit whether ``parameter`` is uniquely recoverable before a turn."""

    del catalog  # Schemas establish requiredness in the caller; no hidden state is read here.
    if affected_turn < 0 or affected_turn >= len(draft.turns):
        raise ValueError("affected Missing Parameter turn is out of range")

    candidates: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []

    # Prior user messages are actor-visible.  Only an exact target marker is
    # extracted deterministically; arbitrary NLP entity inference is not used.
    for turn_id, turn in enumerate(draft.turns[:affected_turn]):
        if _value_mentioned(target_value, turn.query):
            _append_candidate(
                candidates,
                sources,
                value=target_value,
                target=target_value,
                source={
                    "source_type": "USER_CONTEXT",
                    "source_turn": turn_id,
                    "source_path": f"turn[{turn_id}].query",
                },
            )

        # Prior assistant calls and successful observations are visible.  Raw
        # pre/post environment snapshots are intentionally never traversed.
        for call_id, record in enumerate(turn.execution_records):
            semantic = classify_execution_result(
                record.call.name, record.execution_result
            )
            for argument, value in record.call.arguments.items():
                if not _source_is_compatible(parameter, record.call.name, argument):
                    continue
                for source_path, scalar in _iter_scalars(
                    value, path=f"call.arguments.{argument}"
                ):
                    _append_candidate(
                        candidates,
                        sources,
                        value=scalar,
                        target=target_value,
                        source={
                            "source_type": "PRIOR_ASSISTANT_CALL",
                            "source_turn": turn_id,
                            "source_call": call_id,
                            "source_path": source_path,
                            "producer": record.call.name,
                        },
                    )
            if semantic.outcome != ExecutionSemanticOutcome.SUCCESS:
                continue
            for source_path, value in _iter_scalars(
                record.execution_result, path="result"
            ):
                if not _source_is_compatible(
                    parameter, record.call.name, source_path
                ):
                    continue
                _append_candidate(
                    candidates,
                    sources,
                    value=value,
                    target=target_value,
                    source={
                        "source_type": "PRIOR_TOOL_OUTPUT",
                        "source_turn": turn_id,
                        "source_call": call_id,
                        "source_path": source_path,
                        "producer": record.call.name,
                    },
                )

    compatible = [candidates[key] for key in sorted(candidates)]
    uniquely_recoverable = len(compatible) == 1
    target_was_still_exposed = _value_mentioned(target_value, affected_query)
    decision = (
        "REJECT_VALUE_STILL_EXPOSED"
        if target_was_still_exposed
        else "REJECT_UNIQUELY_RECOVERABLE"
        if uniquely_recoverable
        else "GENUINE_MISSING_PARAMETER"
    )
    return {
        "source_status": SOURCE_STATUS,
        "parameter": parameter,
        "target_value": target_value,
        "affected_turn": affected_turn,
        "visible_context_definition": {
            "user_queries_through_affected_turn": True,
            "prior_assistant_calls": True,
            "prior_successful_tool_observations": True,
            "raw_environment_state": False,
        },
        "compatible_visible_candidates": compatible,
        "candidate_sources": sources,
        "uniquely_recoverable": uniquely_recoverable,
        "ambiguity_count": len(compatible),
        "target_was_still_exposed": target_was_still_exposed,
        "decision": decision,
    }


def _recover_missing_parameter_metadata(
    draft: ConversationDraft, *, catalog: FunctionCatalog
) -> dict[str, Any] | None:
    """Recover metadata omitted by an older structural-profile refresh.

    This is intentionally fail closed.  It uses only the final intentional
    missing marker, the immediately following recovery turn, catalog-declared
    *required* parameters, and a value explicitly present in the recovery
    query but absent from the affected query.  Hidden environment state and
    fuzzy entity inference are never consulted.
    """

    affected_indexes = [
        index
        for index, turn in enumerate(draft.turns)
        if turn.is_intentional_missing and turn.missing_kind == "parameter"
    ]
    if len(affected_indexes) != 1:
        return None
    affected_turn = affected_indexes[0]
    recovery_turn = affected_turn + 1
    if recovery_turn >= len(draft.turns):
        return None
    affected = draft.turns[affected_turn]
    recovery = draft.turns[recovery_turn]
    if recovery.is_intentional_missing or not recovery.execution_records:
        return None

    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for record in recovery.execution_records:
        try:
            spec = catalog.get(record.call.name)
        except ValueError:
            return None
        parameters = spec.schema.get("parameters", {})
        required = parameters.get("required", []) if isinstance(parameters, Mapping) else []
        if not isinstance(required, list):
            return None
        for parameter in required:
            if parameter not in record.call.arguments:
                continue
            value = record.call.arguments[parameter]
            if not _value_mentioned(value, recovery.query):
                continue
            if _value_mentioned(value, affected.query):
                continue
            key = (str(parameter), _stable_value(value))
            candidates[key] = {
                "parameter": str(parameter),
                "value": value,
                "function": record.call.name,
            }
    if len(candidates) != 1:
        return None
    recovered = next(iter(candidates.values()))
    return {
        "affected_turn": affected_turn,
        "recovery_turn": recovery_turn,
        "kind": "missing_parameter",
        "missing_parameter": recovered["parameter"],
        "source_status": SOURCE_STATUS,
        "metadata_recovery": {
            "status": "DETERMINISTIC_RECOVERY_FROM_FINAL_TURNS",
            "function": recovered["function"],
            "value": recovered["value"],
            "hidden_environment_state_used": False,
        },
    }


def missing_parameter_validity_gate(
    draft: ConversationDraft, *, catalog: FunctionCatalog
) -> GateResult:
    """Recheck final transformed MP protocol using policy-visible context."""

    if draft.data_type != "multi_turn_miss_param":
        return GateResult(
            "missing_parameter_validity_gate",
            True,
            "not a Missing Parameter candidate",
            {"source_status": SOURCE_STATUS, "status": "NOT_APPLICABLE"},
        )
    adversarial = draft.structural_profile.get("adversarial", {})
    if not isinstance(adversarial, Mapping) or not {
        "affected_turn",
        "recovery_turn",
        "missing_parameter",
    }.issubset(adversarial):
        adversarial = _recover_missing_parameter_metadata(draft, catalog=catalog)
    if not isinstance(adversarial, Mapping):
        return GateResult(
            "missing_parameter_validity_gate",
            False,
            "Missing Parameter metadata is absent and cannot be recovered uniquely",
            {
                "source_status": SOURCE_STATUS,
                "metadata_recovery": "FAILED_CLOSED",
                "hidden_environment_state_used": False,
            },
        )
    try:
        affected_turn = int(adversarial["affected_turn"])
        recovery_turn = int(adversarial["recovery_turn"])
        parameter = str(adversarial["missing_parameter"])
        affected = draft.turns[affected_turn]
        recovery = draft.turns[recovery_turn]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return GateResult(
            "missing_parameter_validity_gate",
            False,
            f"malformed Missing Parameter metadata: {exc}",
            {"source_status": SOURCE_STATUS},
        )
    if not affected.is_intentional_missing or affected.missing_kind != "parameter":
        return GateResult(
            "missing_parameter_validity_gate",
            False,
            "affected turn is not marked as intentional Missing Parameter",
            {"source_status": SOURCE_STATUS},
        )
    values = [
        record.call.arguments[parameter]
        for record in recovery.execution_records
        if parameter in record.call.arguments
    ]
    if not values or any(value != values[0] for value in values[1:]):
        return GateResult(
            "missing_parameter_validity_gate",
            False,
            "recovery GT does not establish one consistent missing value",
            {"source_status": SOURCE_STATUS},
        )
    audit = evaluate_missing_parameter_validity(
        draft,
        affected_turn=affected_turn,
        parameter=parameter,
        target_value=values[0],
        affected_query=affected.query,
        catalog=catalog,
    )
    recovery_supplies_value = _value_mentioned(values[0], recovery.query)
    audit["recovery_turn"] = recovery_turn
    audit["recovery_supplies_value"] = recovery_supplies_value
    if audit["decision"] != "GENUINE_MISSING_PARAMETER":
        return GateResult(
            "missing_parameter_validity_gate",
            False,
            f"missing_parameter_not_genuine: {audit['decision']}",
            audit,
        )
    if not recovery_supplies_value:
        return GateResult(
            "missing_parameter_validity_gate",
            False,
            "recovery turn does not supply the previously missing value",
            audit,
        )
    return GateResult(
        "missing_parameter_validity_gate",
        True,
        "required parameter is absent or ambiguous in policy-visible context and recovered explicitly",
        audit,
    )
