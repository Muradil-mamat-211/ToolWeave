"""Precondition-aware minimality checks for final executable GT calls.

SOURCE_STATUS = PROJECT_ACTION_MINIMALITY_GUARD

This project precision guard requires every final call to be directly requested,
to produce a value consumed by a later required argument, or to satisfy an
audited public-BFCL execution precondition.  It does not ask the Quality Judge
to rationalize extra calls after the fact, and it does not use embeddings.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from ..function_catalog import FunctionCatalog
from ..models import ConversationDraft, ExecutionRecord, GateResult


SOURCE_STATUS = "PROJECT_ACTION_MINIMALITY_GUARD"

# Lexical groups express operation identity, not a similarity score.  They are
# deliberately small and deterministic.  Final Query<->GT verification remains
# an independent, stricter semantic gate.
_ACTION_GROUPS = (
    frozenset({"add", "append", "insert"}),
    frozenset({"authenticate", "login", "logon", "signin"}),
    frozenset({"book", "reserve"}),
    frozenset({"cancel"}),
    frozenset({"close"}),
    frozenset({"convert", "conversion"}),
    frozenset({"copy", "cp", "duplicate"}),
    frozenset({"add", "create", "make", "mkdir", "register", "touch"}),
    frozenset({"delete", "remove", "rm", "rmdir"}),
    frozenset({"cat", "check", "closest", "display", "find", "get", "give", "have", "how", "inspect", "know", "list", "look", "ls", "nearest", "open", "provide", "pull", "read", "retrieve", "search", "see", "show", "tell", "view", "what", "which"}),
    frozenset({"edit", "modify", "update"}),
    frozenset({"estimate", "calculate", "compute"}),
    frozenset({"fill", "put", "refuel"}),
    frozenset({"follow"}),
    frozenset({"activate", "engage", "release", "switch"}),
    frozenset({"lock"}),
    frozenset({"contact", "email", "forward", "message", "notify", "reach", "send", "tell"}),
    frozenset({"move", "mv", "rename"}),
    frozenset({"cd", "enter", "go", "navigate", "open"}),
    frozenset({"place", "order", "buy", "sell", "trade"}),
    frozenset({"post", "publish", "tweet"}),
    frozenset({"press"}),
    frozenset({"resolve", "fix"}),
    frozenset({"set", "adjust", "change"}),
    frozenset({"start", "ignite"}),
    frozenset({"stop"}),
    frozenset({"unlock"}),
    frozenset({"log", "logout", "out", "signout"}),
    frozenset({"add", "echo", "put", "write"}),
    frozenset({"add", "combine", "sum", "plus", "total"}),
    frozenset({"subtract", "minus"}),
    frozenset({"multiply", "times"}),
    frozenset({"divide"}),
    frozenset({"compare", "diff", "difference"}),
    frozenset({"highest", "lowest", "max", "maximum", "min", "minimum"}),
)
_GENERIC_SCHEMA_WORDS = {
    "api", "belong", "belongs", "bot", "control", "function", "method",
    "system", "tool", "vehicle", "user", "allows", "used", "using",
}
_CONVERSION_NAMES = {
    "gallon_to_liter",
    "liter_to_gallon",
    "imperial_si_conversion",
    "si_unit_conversion",
    "compute_exchange_rate",
}
_TOKEN_ALIASES = {
    "adds": "add",
    "books": "book",
    "cancels": "cancel",
    "closes": "close",
    "computes": "compute",
    "converts": "convert",
    "creates": "create",
    "deletes": "delete",
    "displays": "display",
    "edits": "edit",
    "estimates": "estimate",
    "fills": "fill",
    "finds": "find",
    "gets": "get",
    "gallons": "gallon",
    "lists": "list",
    "liters": "liter",
    "looking": "look",
    "bookings": "booking",
    "cards": "card",
    "conversations": "message",
    "details": "info",
    "doors": "door",
    "fast": "speed",
    "files": "file",
    "folders": "folder",
    "information": "info",
    "logs": "log",
    "messages": "message",
    "miles": "mileage",
    "orders": "order",
    "stocks": "stock",
    "tickets": "ticket",
    "locks": "lock",
    "moves": "move",
    "presses": "press",
    "removes": "remove",
    "resolves": "resolve",
    "retrieves": "retrieve",
    "sends": "send",
    "sets": "set",
    "starts": "start",
    "updates": "update",
    "writes": "write",
}
_ACTION_WORDS = frozenset().union(*_ACTION_GROUPS)
_GENERIC_OBJECT_WORDS = {
    "a", "all", "an", "and", "any", "api", "based", "by", "class", "current",
    "detail", "for", "from", "function", "id", "in", "into", "method", "name",
    "new", "of", "option", "result", "selected", "system", "the", "to", "tool",
    "user", "using", "value", "with",
}
_ACTION_NOUNS = {"book", "message", "order", "post", "ticket", "tweet"}
_OBJECT_OPTIONAL_FUNCTIONS = {
    "absolute_value", "add", "authenticate_travel", "authenticate_twitter",
    "divide", "imperial_si_conversion", "logarithm", "max_value", "mean",
    "message_get_login_status", "message_login", "min_value", "multiply",
    "percentage", "place_order", "posting_get_login_status", "power",
    "round_number", "si_unit_conversion", "square_root", "standard_deviation",
    "subtract", "sum_values", "ticket_get_login_status", "ticket_login",
    "trading_get_login_status", "trading_login", "trading_logout",
    "travel_get_login_status",
}
_REFERENTIAL_TOKENS = {"it", "one", "that", "them", "those", "this"}


def _tokens(text: str) -> set[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ")
    normalized = (
        expanded.casefold()
        .replace("log in", "login")
        .replace("log out", "logout")
        .replace("sign in", "signin")
    )
    values: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", normalized):
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        values.add(_TOKEN_ALIASES.get(token, token))
    return values


def _operation_groups(tokens: set[str]) -> set[int]:
    return {
        index
        for index, group in enumerate(_ACTION_GROUPS)
        if tokens & group
    }


def _query_for_turn(draft: ConversationDraft, turn_index: int) -> str:
    query = draft.turns[turn_index].query
    # A Missing Function/Parameter recovery call executes the intent stated in
    # the immediately preceding intentional-missing turn after availability or
    # information is restored.
    if turn_index > 0 and draft.turns[turn_index - 1].is_intentional_missing:
        query = f"{draft.turns[turn_index - 1].query}\n{query}"
    elif turn_index > 0 and _tokens(query) & _REFERENTIAL_TOKENS:
        query = f"{draft.turns[turn_index - 1].query}\n{query}"
    return query


def _argument_object_tokens(value: Any) -> set[str]:
    if isinstance(value, str):
        return _tokens(value)
    if isinstance(value, Mapping):
        tokens: set[str] = set()
        for child in value.values():
            tokens.update(_argument_object_tokens(child))
        return tokens
    if isinstance(value, (list, tuple)):
        tokens = set()
        for child in value:
            tokens.update(_argument_object_tokens(child))
        return tokens
    return set()


def _conversion_is_direct(function_name: str, query_tokens: set[str]) -> bool:
    function_tokens = _tokens(function_name)
    has_convert_intent = bool(query_tokens & _ACTION_GROUPS[5]) or {
        "how",
        "many",
    }.issubset(query_tokens)
    if not has_convert_intent:
        return False
    if "to" in function_tokens:
        units = function_tokens - {"to"} - _ACTION_GROUPS[5]
        return len(units & query_tokens) >= min(2, len(units))
    # Dynamic conversion APIs name units in arguments; the existing unit and
    # semantic-grounding gates verify those arguments independently.
    return function_name in {"imperial_si_conversion", "si_unit_conversion", "compute_exchange_rate"}


def _is_direct_intent(
    record: ExecutionRecord, *, query: str, catalog: FunctionCatalog
) -> tuple[bool, dict[str, Any]]:
    spec = catalog.get(record.call.name)
    query_tokens = _tokens(query)
    function_tokens = _tokens(record.call.name)
    description = str(spec.schema.get("description", ""))
    if "Tool description:" in description:
        description = description.split("Tool description:", 1)[1]
    description_tokens = _tokens(description) - _GENERIC_SCHEMA_WORDS

    if record.call.name == "wc" and {"how", "many"}.issubset(query_tokens):
        return True, {
            "rule": "explicit natural-language count request for wc",
            "query_tokens": sorted(query_tokens),
            "function_tokens": sorted(function_tokens),
        }
    if record.call.name == "set_navigation" and "navigation" in query_tokens:
        return True, {
            "rule": "explicit navigation intent",
            "query_tokens": sorted(query_tokens),
            "function_tokens": sorted(function_tokens),
        }
    if record.call.name == "estimate_drive_feasibility_by_mileage" and (
        "enough" in query_tokens
        or {"can", "drive"}.issubset(query_tokens)
        or {"can", "go"}.issubset(query_tokens)
    ):
        return True, {
            "rule": "explicit drive-feasibility request",
            "query_tokens": sorted(query_tokens),
            "function_tokens": sorted(function_tokens),
        }

    if record.call.name in _CONVERSION_NAMES or "to" in function_tokens and bool(
        function_tokens & {"gallon", "liter"}
    ):
        direct = _conversion_is_direct(record.call.name, query_tokens)
        return direct, {
            "rule": "explicit conversion intent and units",
            "query_tokens": sorted(query_tokens),
            "function_tokens": sorted(function_tokens),
        }

    function_groups = _operation_groups(function_tokens | description_tokens)
    query_groups = _operation_groups(query_tokens)
    operation_matches = function_groups & query_groups
    schema = spec.schema.get("parameters", {})
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    response_schema = spec.schema.get("response", {})
    response_properties = (
        response_schema.get("properties", {})
        if isinstance(response_schema, Mapping)
        else {}
    )
    parameter_tokens = {
        token
        for parameter in properties
        for token in _tokens(str(parameter))
    }
    response_tokens = {
        token
        for parameter in response_properties
        for token in _tokens(str(parameter))
    }
    argument_tokens = {
        token
        for value in record.call.arguments.values()
        for token in _argument_object_tokens(value)
    }
    raw_function_objects = (
        function_tokens
        | description_tokens
        | parameter_tokens
        | response_tokens
        | argument_tokens
    )
    function_objects = (
        raw_function_objects - _ACTION_WORDS - _GENERIC_OBJECT_WORDS
    ) | (raw_function_objects & _ACTION_NOUNS)
    query_objects = (
        query_tokens - _ACTION_WORDS - _GENERIC_OBJECT_WORDS
    ) | (query_tokens & _ACTION_NOUNS)
    object_matches = function_objects & query_objects
    object_optional = (
        record.call.class_name == "MathAPI"
        or record.call.name in _OBJECT_OPTIONAL_FUNCTIONS
        or not function_objects
    )
    direct = bool(operation_matches) and (bool(object_matches) or object_optional)
    return direct, {
        "rule": "deterministic operation-group and object-concept intersection",
        "matching_operation_groups": sorted(operation_matches),
        "function_object_tokens": sorted(function_objects),
        "query_object_tokens": sorted(query_objects),
        "matching_object_tokens": sorted(object_matches),
        "object_match_optional": object_optional,
        "query_tokens": sorted(query_tokens),
        "function_tokens": sorted(function_tokens),
    }


def _flatten_final_calls(
    draft: ConversationDraft,
) -> list[tuple[int, int, ExecutionRecord]]:
    return [
        (turn_index, call_index, record)
        for turn_index, turn in enumerate(draft.turns)
        if not turn.is_intentional_missing
        for call_index, record in enumerate(turn.execution_records)
    ]


def _dependency_consumers(
    calls: list[tuple[int, int, ExecutionRecord]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    consumed: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for turn_index, call_index, record in calls:
        for dependency in record.dependency_provenance.get(
            "resolved_dependency_call_ids", []
        ):
            try:
                source = (int(dependency["turn_id"]), int(dependency["call_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            consumed.setdefault(source, []).append(
                {
                    "consumer_turn": turn_index,
                    "consumer_call": call_index,
                    "consumer_function": record.call.name,
                }
            )
    return consumed


def _nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        if key in current:
            current = current[key]
            continue
        # Environment snapshots may expose internal names with a leading
        # underscore while VM results use the public response field.
        underscored = f"_{key}"
        if underscored not in current:
            return None
        current = current[underscored]
    return current


def _is_audited_prerequisite(
    producer: ExecutionRecord, target: ExecutionRecord
) -> tuple[bool, dict[str, Any]]:
    pair = (producer.call.name, target.call.name)
    if pair == ("pressBrakePedal", "startEngine"):
        result = producer.execution_result
        pressed = isinstance(result, Mapping) and result.get("brakePedalStatus") == "pressed"
        force = result.get("brakePedalForce") if isinstance(result, Mapping) else None
        return bool(pressed and force == 1000.0), {
            "contract": "startEngine requires pressed brake at force 1000",
            "public_source": "bfcl_env/func_source_code_wo_aug/vehicle_control.py:startEngine",
        }
    if pair == ("lockDoors", "startEngine"):
        result = producer.execution_result
        locked = isinstance(result, Mapping) and result.get("remainingUnlockedDoors") == 0
        return locked, {
            "contract": "startEngine requires all doors locked",
            "public_source": "bfcl_env/func_source_code_wo_aug/vehicle_control.py:startEngine",
        }
    if pair == ("fillFuelTank", "startEngine"):
        result = producer.execution_result
        fuel = result.get("fuelLevel") if isinstance(result, Mapping) else None
        return isinstance(fuel, (int, float)) and not isinstance(fuel, bool) and fuel > 0, {
            "contract": "startEngine requires non-empty fuel tank",
            "public_source": "bfcl_env/func_source_code_wo_aug/vehicle_control.py:startEngine",
        }
    if pair == ("cd", "mv"):
        result = producer.execution_result
        directory = (
            result.get("current_working_directory")
            if isinstance(result, Mapping)
            else None
        )
        return isinstance(directory, str) and bool(directory), {
            "contract": "mv source/destination are resolved in the current GorillaFileSystem directory",
            "public_source": "BFCL GorillaFileSystem schema and bfcl_env filesystem implementation",
            "working_directory": directory,
        }

    # The public BFCL implementations require authenticated session state for
    # downstream class actions.  These exact producer names are audited APIs,
    # not names generated by an LLM.
    authenticated_producers = {
        "message_login": "MessageAPI",
        "ticket_login": "TicketAPI",
        "trading_login": "TradingBot",
        "authenticate_twitter": "TwitterAPI",
        "authenticate_travel": "TravelAPI",
    }
    expected_class = authenticated_producers.get(producer.call.name)
    if expected_class == target.call.class_name and target.call.name != producer.call.name:
        return True, {
            "contract": "authenticated session prerequisite for later class action",
            "public_source": "audited bfcl_env class implementation",
        }
    return False, {}


def action_minimality_gate(
    draft: ConversationDraft, *, catalog: FunctionCatalog
) -> GateResult:
    """Classify every final GT call and reject unneeded extras."""

    calls = _flatten_final_calls(draft)
    consumers = _dependency_consumers(calls)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, (turn_index, call_index, record) in enumerate(calls):
        key = (turn_index, call_index)
        row: dict[str, Any] = {
            "turn_id": turn_index,
            "call_id": call_index,
            "function": record.call.name,
            "canonical_call": record.canonical_call,
            "source_status": SOURCE_STATUS,
        }
        if key in consumers:
            row.update(
                classification="DEPENDENCY_PRODUCER",
                evidence={"consumers": consumers[key]},
            )
        else:
            direct, direct_evidence = _is_direct_intent(
                record,
                query=_query_for_turn(draft, turn_index),
                catalog=catalog,
            )
            if direct:
                row.update(
                    classification="DIRECT_INTENT",
                    evidence=direct_evidence,
                )
            else:
                prerequisite_evidence: dict[str, Any] | None = None
                target_ref: dict[str, Any] | None = None
                for later_turn, later_call, later_record in calls[index + 1 :]:
                    required, evidence = _is_audited_prerequisite(record, later_record)
                    if required:
                        prerequisite_evidence = evidence
                        target_ref = {
                            "turn_id": later_turn,
                            "call_id": later_call,
                            "function": later_record.call.name,
                        }
                        break
                if prerequisite_evidence is not None:
                    row.update(
                        classification="REQUIRED_PREREQUISITE",
                        evidence={
                            **prerequisite_evidence,
                            "target": target_ref,
                        },
                    )
                else:
                    row.update(
                        classification="REDUNDANT_EXTRA_CALL",
                        evidence={
                            "direct_intent_check": direct_evidence,
                            "consumed_by_later_required_argument": False,
                            "audited_vm_prerequisite": False,
                        },
                    )
                    failures.append(row)
        rows.append(row)

    metadata = {
        "source_status": SOURCE_STATUS,
        "calls": rows,
        "failures": failures,
        "allowed_classifications": [
            "DIRECT_INTENT",
            "REQUIRED_PREREQUISITE",
            "DEPENDENCY_PRODUCER",
        ],
        "embedding_or_similarity_threshold_used": False,
        "judge_override_allowed": False,
    }
    draft.structural_profile.setdefault("project_semantic_guards", {})[
        "action_minimality"
    ] = metadata
    if failures:
        first = failures[0]
        return GateResult(
            "action_minimality_gate",
            False,
            "REDUNDANT_EXTRA_CALL at "
            f"turn={first['turn_id']} call={first['call_id']}: "
            f"{first['canonical_call']}",
            metadata,
        )
    return GateResult(
        "action_minimality_gate",
        True,
        "every final GT call is direct, prerequisite, or a dependency producer",
        metadata,
    )
