"""Auditable provided-argument grounding for final synthetic GT.

SOURCE_STATUS = PROJECT_SEMANTIC_GUARD

RODS does not publish this deterministic gate.  It is a conservative project
correctness guard: every model-provided required or optional GT argument must
be traceable to final user context, a successful prior tool result, relevant
pre-state, a schema-defined deterministic default, or the explicit Missing
Function/Parameter protocol. Unprovided optional arguments are intentionally
out of scope. No embedding or structural-distance threshold is used.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ..function_catalog import FunctionCatalog
from ..models import ConversationDraft, ExecutionRecord, GateResult
from ..result_semantics import (
    ExecutionSemanticOutcome,
    classify_execution_result,
    domain_negative_contract,
)
from ..structural_profile import draft_structural_profile, structural_alignment_diagnostics


SOURCE_STATUS = "PROJECT_SEMANTIC_GUARD"


class ParameterSourceType(str, Enum):
    USER_CONTEXT = "USER_CONTEXT"
    PRIOR_TOOL_OUTPUT = "PRIOR_TOOL_OUTPUT"
    ENV_STATE = "ENV_STATE"
    INTENTIONAL_MISSING_PROTOCOL = "INTENTIONAL_MISSING_PROTOCOL"
    SCHEMA_DEFAULT = "SCHEMA_DEFAULT"
    UNSUPPORTED = "UNSUPPORTED"


_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}
_UNSUPPORTED_STRINGS = {
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "undefined",
    "placeholder",
    "password123",
}
_FREE_TEXT_PARAMETERS = {
    "content",
    "description",
    "message",
    "query",
    "reason",
    "resolution",
    "text",
    "title",
}
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "been",
    "could",
    "for",
    "from",
    "have",
    "into",
    "need",
    "please",
    "that",
    "the",
    "their",
    "this",
    "those",
    "to",
    "want",
    "with",
    "would",
    "you",
}
_GENERIC_KEY_TOKENS = {"id", "key", "name", "status", "value", "list", "map"}
_GENERIC_HARD_ANCHORS = {"ID"}
_KEY_ALIASES = {
    "ticker": "symbol",
    "stock": "symbol",
    "stocks": "symbol",
    "receiver": "user",
    "sender": "user",
    "username": "user",
    "current": "user",
    "airport": "airport",
    "departure": "airport",
    "destination": "destination",
    "travel": "airport",
    "cardholder": "card",
    "credit": "card",
    "doors": "door",
}
_MONTHS = {
    "01": "january", "02": "february", "03": "march", "04": "april",
    "05": "may", "06": "june", "07": "july", "08": "august",
    "09": "september", "10": "october", "11": "november", "12": "december",
}


def _normal_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _tokens(value: str) -> set[str]:
    # Split snake/camel/dotted names deterministically.
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value).replace("_", " ")
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9]+", expanded)}


def _canonical_key_tokens(value: str) -> set[str]:
    return {_KEY_ALIASES.get(token, token) for token in _tokens(value)}


def _keys_related(parameter: str, source_path: str) -> bool:
    parameter_tokens = _canonical_key_tokens(parameter) - _GENERIC_KEY_TOKENS
    path_tokens = _canonical_key_tokens(source_path) - _GENERIC_KEY_TOKENS
    if parameter_tokens & path_tokens:
        return True
    # ``receiver_id`` and ``user_map`` are a public BFCL relation; both reduce
    # to the entity token ``user`` after aliases even when ID is generic.
    return "user" in parameter_tokens and "user" in path_tokens


def _iter_scalar_paths(value: Any, *, path: str) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_scalar_paths(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_scalar_paths(item, path=f"{path}[{index}]")
    else:
        yield path, value


def _iter_mapping_identity_paths(
    value: Any, *, path: str
) -> Iterable[tuple[str, Any]]:
    """Expose mapping keys as explicit tool-output identities.

    BFCL commonly returns collections keyed by the downstream ID (for example
    ``booking_history[booking_id]``).  The key is part of the actual result,
    even when the nested record does not repeat it as a scalar field.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, (str, int)) and not isinstance(key, bool):
                yield f"{path}.{key}#mapping_key", key
            yield from _iter_mapping_identity_paths(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_mapping_identity_paths(item, path=f"{path}[{index}]")


def _is_degenerate_source_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return not math.isfinite(float(value)) or float(value) in {0.0, 1.0}
    if isinstance(value, str):
        return value.strip().casefold() in _UNSUPPORTED_STRINGS
    return value is None


def _string_mentioned(value: str, text: str) -> bool:
    normalized_value = _normal_text(value)
    normalized_text = _normal_text(text)
    if not normalized_value:
        return False
    if len(normalized_value) <= 2:
        return bool(re.search(rf"\b{re.escape(normalized_value)}\b", normalized_text))
    return normalized_value in normalized_text


def _number_mentioned(value: int | float, text: str) -> bool:
    numeric = float(value)
    candidates = {str(value), f"{numeric:g}"}
    if numeric.is_integer():
        # Natural user requests commonly group thousands (``5,000``) while
        # canonical function arguments contain ``5000.0``.  This is exact
        # numeric formatting equivalence, not fuzzy similarity.
        candidates.add(f"{int(numeric):,}")
    if numeric.is_integer() and int(numeric) in _NUMBER_WORDS:
        candidates.add(_NUMBER_WORDS[int(numeric)])
    raw = text.casefold()
    normalized = _normal_text(text)
    if numeric.is_integer() and re.search(
        rf"(?<![a-z0-9]){int(numeric)}(?:st|nd|rd|th)?(?![a-z0-9])",
        raw,
    ):
        return True
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(item.casefold())}(?![a-z0-9])", raw)
        or re.search(rf"(?<![a-z0-9]){re.escape(item.casefold())}(?![a-z0-9])", normalized)
        for item in candidates
    )


def _boolean_mentioned(
    function_name: str, parameter: str, value: bool, text: str
) -> bool:
    normalized = _normal_text(text)
    parameter_tokens = _canonical_key_tokens(parameter)
    if "unlock" in parameter_tokens:
        return (value and "unlock" in normalized) or (not value and "lock" in normalized and "unlock" not in normalized)
    if (
        function_name == "setCruiseControl"
        and parameter == "activate"
        and value
        and (
            "set my cruise control" in normalized
            or "set cruise control" in normalized
            or "turn on cruise control" in normalized
            or "enable cruise control" in normalized
        )
    ):
        return True
    positive = {"enable", "enabled", "activate", "activated", "on", "yes", "true"}
    negative = {"disable", "disabled", "deactivate", "off", "no", "false"}
    words = set(normalized.split())
    return bool(words & (positive if value else negative))


def _scalar_mentioned(
    parameter: str, value: Any, text: str, *, function_name: str = ""
) -> bool:
    if isinstance(value, bool):
        return _boolean_mentioned(function_name, parameter, value, text)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        normalized = _normal_text(text)
        if (
            function_name == "pressBrakePedal"
            and parameter == "pedalPosition"
            and float(value) == 1.0
            and any(phrase in normalized for phrase in ("start the car", "start the engine", "press the brake", "hit the brake"))
        ):
            return True
        return _number_mentioned(value, text)
    if isinstance(value, str):
        date_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
        if date_match:
            year, month, day = date_match.groups()
            natural = _normal_text(text)
            day_number = str(int(day))
            if year in natural and _MONTHS[month] in natural and re.search(
                rf"\b{re.escape(day_number)}(?:st|nd|rd|th)?\b", natural
            ):
                return True
        if parameter == "mode" and value == "w" and "word" in _normal_text(text):
            # Public GorillaFileSystem.wc contract: mode='w' means word count.
            return True
        return _string_mentioned(value, text)
    return False


def _user_context_evidence(
    function_name: str,
    parameter: str,
    value: Any,
    query_sources: Sequence[tuple[int, str]],
) -> dict[str, Any] | None:
    leaves = list(_iter_scalar_paths(value, path=parameter))
    supporting_turns: set[int] = set()
    for _, leaf in leaves:
        matched_turn = next(
            (
                turn_id
                for turn_id, query in reversed(query_sources)
                if _scalar_mentioned(
                    parameter, leaf, query, function_name=function_name
                )
            ),
            None,
        )
        if matched_turn is None:
            break
        supporting_turns.add(matched_turn)
    else:
        return {
            "source_type": ParameterSourceType.USER_CONTEXT.value,
            "source_turns": sorted(supporting_turns),
            "source_path": [f"turn[{turn_id}].final_query" for turn_id in sorted(supporting_turns)],
            "match": "all scalar leaves explicitly mentioned",
        }

    if parameter not in _FREE_TEXT_PARAMETERS or not isinstance(value, str):
        return None

    # Free-form prose may paraphrase a request.  Keep all audit-sensitive
    # anchors (IDs, numbers, uppercase symbols) exact, then require at least
    # one non-stopword lexical anchor from final user context.  This is not an
    # embedding/similarity threshold.
    context = "\n".join(query for _, query in query_sources)
    hard_anchors = set(
        re.findall(
            r"\b(?:[A-Z]{2,}[A-Z0-9]*|[A-Za-z0-9_-]*\d[A-Za-z0-9_.-]*)\b",
            value,
        )
    ) - _GENERIC_HARD_ANCHORS
    if any(not _string_mentioned(anchor, context) for anchor in hard_anchors):
        return None
    value_words = {
        word for word in _normal_text(value).split() if len(word) >= 4 and word not in _STOPWORDS
    }
    context_words = set(_normal_text(context).split())
    overlap = sorted(value_words & context_words)
    if not overlap:
        return None
    return {
        "source_type": ParameterSourceType.USER_CONTEXT.value,
        "source_turns": [turn_id for turn_id, _ in query_sources],
        "source_path": "final_user_context",
        "match": "free-text lexical anchors",
        "anchors": overlap,
    }


def _exact_value_evidence(
    parameter: str,
    value: Any,
    records: Sequence[tuple[int, int, ExecutionRecord]],
) -> dict[str, Any] | None:
    leaves = list(_iter_scalar_paths(value, path=parameter))
    evidence: list[dict[str, Any]] = []
    for _, leaf in leaves:
        if _is_degenerate_source_value(leaf):
            return None
        found: dict[str, Any] | None = None
        for turn_id, call_index, record in reversed(records):
            semantic = classify_execution_result(record.call.name, record.execution_result)
            if semantic.outcome != ExecutionSemanticOutcome.SUCCESS:
                continue
            result_values = list(
                _iter_scalar_paths(record.execution_result, path="result")
            ) + list(
                _iter_mapping_identity_paths(record.execution_result, path="result")
            )
            for source_path, source_value in result_values:
                if source_value == leaf:
                    found = {
                        "source_type": ParameterSourceType.PRIOR_TOOL_OUTPUT.value,
                        "source_turn": turn_id,
                        "source_call": call_index,
                        "source_path": source_path,
                    }
                    break
            if found is not None:
                break
        if found is None:
            return None
        evidence.append(found)
    return {
        "source_type": ParameterSourceType.PRIOR_TOOL_OUTPUT.value,
        "leaf_sources": evidence,
    }


def _environment_evidence(parameter: str, value: Any, pre_state: Any) -> dict[str, Any] | None:
    leaves = list(_iter_scalar_paths(value, path=parameter))
    evidence: list[dict[str, Any]] = []
    state_values = list(_iter_scalar_paths(pre_state, path="pre_state"))
    for _, leaf in leaves:
        if _is_degenerate_source_value(leaf):
            return None
        match = next(
            (
                source_path
                for source_path, source_value in state_values
                if (
                    source_value == leaf
                    or (
                        isinstance(leaf, str)
                        and leaf.casefold() in source_path.casefold().split(".")
                    )
                )
                and _keys_related(parameter, source_path)
            ),
            None,
        )
        if match is None:
            return None
        evidence.append(
            {
                "source_type": ParameterSourceType.ENV_STATE.value,
                "source_path": match,
            }
        )
    return {"source_type": ParameterSourceType.ENV_STATE.value, "leaf_sources": evidence}


def _domain_negative_conflict(
    parameter: str,
    value: Any,
    current_query: str,
    prior_records: Sequence[tuple[int, int, ExecutionRecord]],
) -> dict[str, Any] | None:
    for turn_id, call_index, record in reversed(prior_records):
        contract = domain_negative_contract(record.call.name, record.execution_result)
        if contract is None or parameter not in contract.downstream_parameter_names:
            continue
        explicit_current_value = all(
            _scalar_mentioned(parameter, leaf, current_query)
            for _, leaf in _iter_scalar_paths(value, path=parameter)
        )
        producer_inputs = list(record.call.arguments.values())
        reuses_failed_lookup_input = any(item == value for item in producer_inputs)
        if explicit_current_value and not reuses_failed_lookup_input:
            continue
        return {
            "source_type": ParameterSourceType.UNSUPPORTED.value,
            "reason": "domain-negative producer cannot ground downstream argument",
            "source_turn": turn_id,
            "source_call": call_index,
            "source_path": "result." + ".".join(contract.source_path),
            "producer": record.canonical_call,
        }
    return None


def _composite_text_evidence(
    parameter: str,
    value: Any,
    query_sources: Sequence[tuple[int, str]],
    records: Sequence[tuple[int, int, ExecutionRecord]],
) -> dict[str, Any] | None:
    if parameter not in _FREE_TEXT_PARAMETERS or not isinstance(value, str):
        return None
    query_context = "\n".join(query for _, query in query_sources)
    result_rows: list[tuple[int, int, str, Any]] = []
    for turn_id, call_id, record in records:
        semantic = classify_execution_result(record.call.name, record.execution_result)
        if semantic.outcome != ExecutionSemanticOutcome.SUCCESS:
            continue
        result_rows.extend(
            (turn_id, call_id, path, item)
            for path, item in _iter_scalar_paths(record.execution_result, path="result")
        )
    prior_context = "\n".join(str(item) for _, _, _, item in result_rows)
    combined = query_context + "\n" + prior_context
    hard_anchors = set(
        re.findall(
            r"\b(?:[A-Z]{2,}[A-Z0-9]*|[A-Za-z0-9_-]*\d[A-Za-z0-9_.-]*)\b",
            value,
        )
    ) - _GENERIC_HARD_ANCHORS
    if any(not _string_mentioned(anchor, combined) for anchor in hard_anchors):
        return None
    value_words = {
        word
        for word in _normal_text(value).split()
        if len(word) >= 4 and word not in _STOPWORDS
    }
    combined_words = set(_normal_text(combined).split())
    overlap = sorted(value_words & combined_words)
    if not overlap:
        return None
    source_calls = sorted(
        {
            (turn_id, call_id)
            for turn_id, call_id, _, item in result_rows
            if any(_string_mentioned(anchor, str(item)) for anchor in hard_anchors)
        }
    )
    return {
        "source_type": (
            ParameterSourceType.PRIOR_TOOL_OUTPUT.value
            if source_calls
            else ParameterSourceType.USER_CONTEXT.value
        ),
        "source_path": "final_user_context+successful_prior_results",
        "source_calls": [
            {"turn_id": turn_id, "call_id": call_id}
            for turn_id, call_id in source_calls
        ],
        "match": "free-text lexical and exact audit anchors",
        "anchors": overlap,
    }


def _replace_structural_diagnostics(draft: ConversationDraft) -> None:
    preserved_seed = draft.structural_profile.get("seed_profile")
    preserved_adversarial = draft.structural_profile.get("adversarial")
    refreshed = draft_structural_profile(draft.turns)
    if preserved_seed is not None:
        refreshed["seed_profile"] = preserved_seed
        refreshed["alignment_diagnostics"] = structural_alignment_diagnostics(
            preserved_seed, refreshed
        )
    refreshed["alignment_mechanism"] = draft.structural_profile.get(
        "alignment_mechanism",
        "Planner-prompt-mediated; deterministic profile is diagnostics only",
    )
    # Missing Function/Parameter metadata is a semantic protocol record, not
    # a derived structural statistic.  Refreshing diagnostics must not erase
    # it before downstream deterministic gates inspect the final task.
    if preserved_adversarial is not None:
        refreshed["adversarial"] = preserved_adversarial
    draft.structural_profile = refreshed


def semantic_grounding_gate(
    draft: ConversationDraft,
    *,
    catalog: FunctionCatalog,
) -> GateResult:
    """Annotate provenance and reject any unsupported provided GT argument."""

    failures: list[dict[str, Any]] = []
    parameter_records: list[dict[str, Any]] = []
    prior_executed: list[tuple[int, int, ExecutionRecord]] = []
    query_sources: list[tuple[int, str]] = []

    for turn_index, turn in enumerate(draft.turns):
        if turn.query.strip():
            query_sources.append((turn_index, turn.query))
        updated_records: list[ExecutionRecord] = []
        for call_index, record in enumerate(turn.execution_records):
            semantic = classify_execution_result(record.call.name, record.execution_result)
            provenance = dict(record.dependency_provenance)
            parameter_schema = catalog.get(record.call.name).schema.get(
                "parameters", {}
            )
            required = parameter_schema.get("required", [])
            required = [str(name) for name in required]
            properties = parameter_schema.get("properties", {})
            if not isinstance(properties, Mapping):
                properties = {}
            # PROJECT_SEMANTIC_GUARD: a model-provided optional parameter is an
            # action choice just like a required parameter.  Therefore every
            # key actually present in the canonical GT call is audited.  An
            # optional key omitted from the call is not materialized here.
            provided_parameters = list(record.call.arguments)
            argument_evidence: list[dict[str, Any]] = []

            if turn.is_intentional_missing:
                for parameter in provided_parameters:
                    argument_evidence.append(
                        {
                            "parameter": parameter,
                            "required": parameter in required,
                            "provided": True,
                            "source_type": ParameterSourceType.INTENTIONAL_MISSING_PROTOCOL.value,
                            "source_turn": turn_index,
                            "source_path": f"turn[{turn_index}].intentional_missing",
                        }
                    )
                status = "INTENTIONAL_MISSING_SKIPPED"
            else:
                for parameter in provided_parameters:
                    value = record.call.arguments.get(parameter)
                    is_required = parameter in required
                    raw_property = properties.get(parameter, {})
                    has_schema_default = (
                        not is_required
                        and isinstance(raw_property, Mapping)
                        and "default" in raw_property
                        and value == raw_property["default"]
                    )
                    current_query = turn.query or "\n".join(
                        query for _, query in query_sources
                    )
                    negative_conflict = _domain_negative_conflict(
                        parameter,
                        value,
                        current_query,
                        prior_executed,
                    )
                    evidence = (
                        {
                            "source_type": ParameterSourceType.SCHEMA_DEFAULT.value,
                            "source_path": (
                                f"catalog.{record.call.name}.parameters."
                                f"properties.{parameter}.default"
                            ),
                            "default_value": value,
                        }
                        if has_schema_default
                        else _user_context_evidence(
                            record.call.name, parameter, value, query_sources
                        )
                    )
                    if evidence is None:
                        evidence = _exact_value_evidence(parameter, value, prior_executed)
                    if evidence is None:
                        evidence = _composite_text_evidence(
                            parameter, value, query_sources, prior_executed
                        )
                    if evidence is None:
                        evidence = _environment_evidence(parameter, value, record.pre_state)
                    if negative_conflict is not None and not (
                        evidence is not None
                        and evidence.get("source_type")
                        == ParameterSourceType.PRIOR_TOOL_OUTPUT.value
                    ):
                        # A failed lookup cannot ground its own downstream
                        # action.  A separate SUCCESS result can establish the
                        # value independently; classify_execution_result makes
                        # the failed lookup ineligible for that evidence.
                        # Merely finding the scalar elsewhere in pre-state is
                        # insufficient to prove the failed entity->value
                        # relation and therefore does not override the conflict.
                        evidence = negative_conflict
                    if evidence is None:
                        evidence = {
                            "source_type": ParameterSourceType.UNSUPPORTED.value,
                            "reason": "no auditable user/prior-output/relevant-state source",
                        }
                    row = {
                        "turn_id": turn_index,
                        "call_id": call_index,
                        "function": record.call.name,
                        "parameter": parameter,
                        "required": is_required,
                        "provided": True,
                        "value": value,
                        **evidence,
                    }
                    argument_evidence.append(row)
                    parameter_records.append(row)
                    if evidence["source_type"] == ParameterSourceType.UNSUPPORTED.value:
                        failures.append(row)
                status = "GROUNDED" if not any(
                    item["source_type"] == ParameterSourceType.UNSUPPORTED.value
                    for item in argument_evidence
                ) else "UNSUPPORTED"

            resolved_dependencies: list[dict[str, int]] = []
            for item in argument_evidence:
                if item.get("source_type") != ParameterSourceType.PRIOR_TOOL_OUTPUT.value:
                    continue
                dependency_sources = list(item.get("leaf_sources", [])) + list(
                    item.get("source_calls", [])
                )
                for source in dependency_sources:
                    key = {
                        "turn_id": int(source.get("source_turn", source.get("turn_id"))),
                        "call_id": int(source.get("source_call", source.get("call_id"))),
                    }
                    if key not in resolved_dependencies:
                        resolved_dependencies.append(key)
            provenance.update(
                {
                    "source_status": SOURCE_STATUS,
                    "semantic_outcome": semantic.outcome.value,
                    "semantic_detail": semantic.detail,
                    "parameter_provenance": argument_evidence,
                    "resolved_dependency_call_ids": resolved_dependencies,
                    "parameter_dependency_status": status,
                }
            )
            updated = replace(
                record,
                dependency_provenance=provenance,
                semantic_outcome=semantic.outcome.value,
                semantic_detail=semantic.detail,
                success=semantic.outcome != ExecutionSemanticOutcome.HARD_ERROR,
                error_detail=(
                    semantic.detail
                    if semantic.outcome == ExecutionSemanticOutcome.HARD_ERROR
                    else None
                ),
            )
            updated_records.append(updated)
            if semantic.outcome == ExecutionSemanticOutcome.HARD_ERROR:
                failures.append(
                    {
                        "turn_id": turn_index,
                        "call_id": call_index,
                        "function": record.call.name,
                        "source_type": ParameterSourceType.UNSUPPORTED.value,
                        "reason": f"final trace contains HARD_ERROR: {semantic.detail}",
                    }
                )
            if not turn.is_intentional_missing:
                prior_executed.append((turn_index, call_index, updated))
        turn.execution_records = updated_records

    _replace_structural_diagnostics(draft)
    outcome_counts = {item.value: 0 for item in ExecutionSemanticOutcome}
    for turn in draft.turns:
        for record in turn.execution_records:
            outcome_counts[record.semantic_outcome] += 1
    metadata = {
        "source_status": SOURCE_STATUS,
        "parameter_provenance": parameter_records,
        "unsupported": failures,
        "semantic_outcome_counts": outcome_counts,
    }
    if failures:
        first = failures[0]
        return GateResult(
            "semantic_grounding_gate",
            False,
            "unsupported final GT semantics at "
            f"turn={first.get('turn_id')} call={first.get('call_id')} "
            f"parameter={first.get('parameter')}: {first.get('reason')}",
            metadata,
        )
    return GateResult(
        "semantic_grounding_gate",
        True,
        "all provided required/optional final GT arguments have auditable provenance",
        metadata,
    )


def semantic_context_for_verifier(draft: ConversationDraft) -> str:
    """Compact final-task context for per-turn and global-coherence review."""

    rows: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(draft.turns):
        records: list[dict[str, Any]] = []
        for call_index, record in enumerate(turn.execution_records):
            changed_paths = []
            pre = dict(_iter_scalar_paths(record.pre_state, path="state"))
            post = dict(_iter_scalar_paths(record.post_state, path="state"))
            for path in sorted(set(pre) | set(post)):
                if pre.get(path) != post.get(path):
                    changed_paths.append(
                        {"path": path, "before": pre.get(path), "after": post.get(path)}
                    )
            records.append(
                {
                    "call_id": call_index,
                    "call": record.canonical_call,
                    "result": record.execution_result,
                    "semantic_outcome": record.semantic_outcome,
                    "parameter_provenance": record.dependency_provenance.get(
                        "parameter_provenance", []
                    ),
                    "relevant_state_changes": changed_paths,
                }
            )
        rows.append(
            {
                "turn_id": turn_index,
                "final_query": turn.query,
                "intentional_missing": turn.is_intentional_missing,
                "missing_kind": turn.missing_kind,
                "recovery_tools": [tool.get("name") for tool in turn.recovery_tools],
                "final_gt": turn.ground_truth,
                "execution": records,
            }
        )
    raw_guard_context = draft.structural_profile.get("project_semantic_guards", {})
    observation_checks = []
    if isinstance(raw_guard_context, Mapping):
        observation = raw_guard_context.get("observation_entailment", {})
        if isinstance(observation, Mapping):
            for check in observation.get("checks", []):
                if not isinstance(check, Mapping):
                    continue
                observation_checks.append(
                    {
                        "turn_id": check.get("turn_id"),
                        "source_phrase": check.get("source_phrase"),
                        "claimed_fact": check.get("claim"),
                        "claim_anchors": check.get("claim_anchors", []),
                        "missing_anchors": check.get("missing_anchors", []),
                        "evidence_relation": check.get("status"),
                        "prior_observation_refs": [
                            {
                                "source_turn": item.get("source_turn"),
                                "source_call": item.get("source_call"),
                                "function": item.get("function"),
                                "semantic_outcome": item.get("semantic_outcome"),
                            }
                            for item in check.get("prior_observations", [])
                            if isinstance(item, Mapping)
                        ],
                    }
                )

    context = {
        "source_status": "PROJECT_SEMANTIC_GUARD/GLOBAL_COHERENCE",
        "planner_latent_narrative": draft.narrative,
        "global_coherence_contract": {
            "single_underlying_goal_required": True,
            "topic_transition_requires_semantic_bridge": True,
            "later_turn_should_use_prior_context_state_or_result_when_promised": True,
            "planner_narrative_must_match_final_queries_and_gt": True,
            "narrative_promises_must_be_executed": True,
            "unrelated_filler_turns_forbidden": True,
            "embedding_threshold_used": False,
        },
        "observation_entailment_contract": {
            "source_status": "PROJECT_OBSERVATION_ENTAILMENT_GUARD",
            "explicit_observation_claims_must_be_supported": True,
            "judge_or_verifier_may_not_override_deterministic_failure": True,
            "deferred_claims_require_fail_closed_semantic_review": True,
            "checks": observation_checks,
        },
        "turns": rows,
    }
    return json.dumps(context, ensure_ascii=False, sort_keys=True, default=repr)
