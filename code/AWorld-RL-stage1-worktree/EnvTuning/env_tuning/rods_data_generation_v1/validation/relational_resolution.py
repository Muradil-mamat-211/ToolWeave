"""Deterministic ambiguity checks for relational object selection.

SOURCE_STATUS = PROJECT_RELATIONAL_GUARD

RODS does not publish a deterministic relational resolver.  This narrow
project guard activates only when final user text explicitly requests an
extremum (for example ``most recent`` or ``lowest``) *and* a later GT argument
selects one object from a recoverable result/state collection.  It verifies
the selected object against the real values and rejects unresolved ties.  It
does not invent a tie-break, embedding score, distance threshold, or general
semantic-similarity rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from ..models import ConversationDraft, ExecutionRecord, GateResult
from ..result_semantics import ExecutionSemanticOutcome, classify_execution_result


SOURCE_STATUS = "PROJECT_RELATIONAL_GUARD"


@dataclass(frozen=True)
class _Relation:
    phrase: str
    direction: str
    family: str


_RELATIONS: tuple[_Relation, ...] = (
    _Relation("most recent", "max", "temporal"),
    _Relation("latest", "max", "temporal"),
    _Relation("newest", "max", "temporal"),
    _Relation("earliest", "min", "temporal"),
    _Relation("oldest", "min", "temporal"),
    _Relation("highest", "max", "magnitude"),
    _Relation("largest", "max", "magnitude"),
    _Relation("maximum", "max", "magnitude"),
    _Relation("lowest", "min", "magnitude"),
    _Relation("smallest", "min", "magnitude"),
    _Relation("minimum", "min", "magnitude"),
    _Relation("closest", "min", "distance"),
    _Relation("farthest", "max", "distance"),
)
_TEMPORAL_TOKENS = {
    "date",
    "time",
    "timestamp",
    "created",
    "creation",
    "updated",
    "modified",
    "year",
}
_DISTANCE_TOKENS = {"distance", "mileage", "range", "proximity"}
_IDENTITY_TOKENS = {
    "id",
    "identifier",
    "name",
    "symbol",
    "code",
    "number",
    "key",
}
_QUERY_STOPWORDS = {
    "a",
    "an",
    "for",
    "get",
    "give",
    "i",
    "me",
    "my",
    "of",
    "please",
    "the",
    "to",
    "want",
}


def _normal_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value).replace("_", " ")
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9]+", expanded)}


def _relations_in_query(query: str) -> list[_Relation]:
    normalized = _normal_text(query)
    return [
        relation
        for relation in _RELATIONS
        if re.search(rf"\b{re.escape(relation.phrase)}\b", normalized)
    ]


def _iter_scalar_paths(value: Any, *, path: str) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_scalar_paths(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_scalar_paths(item, path=f"{path}[{index}]")
    else:
        yield path, value


def _iter_candidate_collections(
    value: Any, *, path: str
) -> Iterable[tuple[str, dict[str, Mapping[str, Any]]]]:
    """Yield only explicit object collections; scalar lists are not guessed."""

    if isinstance(value, Mapping):
        mapped = {
            str(key): item for key, item in value.items() if isinstance(item, Mapping)
        }
        # A single wrapper object such as {"booking_history": {...}} is not
        # itself a candidate collection. Relational ambiguity requires at
        # least two recoverable alternatives; recurse to the nested mapping.
        if len(mapped) >= 2 and len(mapped) == len(value):
            yield path, mapped
        for key, item in value.items():
            yield from _iter_candidate_collections(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(item, Mapping) for item in value):
            collection: dict[str, Mapping[str, Any]] = {}
            for index, item in enumerate(value):
                identity = _object_identity(item, fallback=str(index))
                if identity in collection:
                    # Duplicate identities make the collection non-recoverable.
                    collection = {}
                    break
                collection[identity] = item
            if collection:
                yield path, collection
        for index, item in enumerate(value):
            yield from _iter_candidate_collections(item, path=f"{path}[{index}]")


def _object_identity(value: Mapping[str, Any], *, fallback: str) -> str:
    for path, item in _iter_scalar_paths(value, path="candidate"):
        if not isinstance(item, (str, int)) or isinstance(item, bool):
            continue
        if _tokens(path) & _IDENTITY_TOKENS:
            return str(item)
    return fallback


def _selected_identity(
    argument_value: Any, collection: Mapping[str, Mapping[str, Any]]
) -> str | None:
    if not isinstance(argument_value, (str, int)) or isinstance(argument_value, bool):
        return None
    if str(argument_value) in collection:
        return str(argument_value)
    for identity, attributes in collection.items():
        for path, item in _iter_scalar_paths(attributes, path="candidate"):
            if not (_tokens(path) & _IDENTITY_TOKENS):
                continue
            if item == argument_value:
                return identity
    return None


def _candidate_scalar_fields(
    collection: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for identity, attributes in collection.items():
        output[identity] = {
            path.removeprefix("candidate."): value
            for path, value in _iter_scalar_paths(attributes, path="candidate")
        }
    return output


def _comparable(value: Any, *, family: str) -> tuple[str, Any] | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return "number", float(value)
    if family != "temporal" or not isinstance(value, str):
        return None
    raw = value.strip()
    for candidate in (raw, raw.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
            return "datetime", (
                parsed.year,
                parsed.month,
                parsed.day,
                parsed.hour,
                parsed.minute,
                parsed.second,
                parsed.microsecond,
            )
        except ValueError:
            pass
    for date_format in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            parsed = datetime.strptime(raw, date_format)
            return "datetime", (
                parsed.year,
                parsed.month,
                parsed.day,
                0,
                0,
                0,
                0,
            )
        except ValueError:
            pass
    return None


def _relation_field(
    relation: _Relation,
    query: str,
    fields: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, tuple[str, Any]]] | None:
    if not fields:
        return None
    common = set.intersection(*(set(values) for values in fields.values()))
    eligible: list[tuple[str, dict[str, tuple[str, Any]], int]] = []
    query_tokens = _tokens(query) - _QUERY_STOPWORDS
    relation_tokens = _tokens(relation.phrase)
    for path in sorted(common):
        converted: dict[str, tuple[str, Any]] = {}
        for identity, values in fields.items():
            item = _comparable(values[path], family=relation.family)
            if item is None:
                break
            converted[identity] = item
        else:
            kinds = {item[0] for item in converted.values()}
            if len(kinds) != 1:
                continue
            path_tokens = _tokens(path)
            if relation.family == "temporal" and not (path_tokens & _TEMPORAL_TOKENS):
                continue
            if relation.family == "distance" and not (path_tokens & _DISTANCE_TOKENS):
                continue
            score = len(path_tokens & (query_tokens - relation_tokens))
            eligible.append((path, converted, score))
    if not eligible:
        return None
    best_score = max(item[2] for item in eligible)
    best = [item for item in eligible if item[2] == best_score]
    if len(best) != 1:
        # Multiple equally plausible fields are not deterministically resolved.
        return None
    path, converted, _ = best[0]
    return path, converted


def _value_explicitly_mentioned(value: Any, query: str) -> bool:
    normalized = _normal_text(query)
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(
            re.search(rf"(?<![a-z0-9]){re.escape(f'{float(value):g}')}(?![a-z0-9])", normalized)
        )
    if isinstance(value, str):
        needle = _normal_text(value)
        return bool(needle) and bool(
            re.search(rf"\b{re.escape(needle)}\b", normalized)
        )
    return False


def _tie_disambiguation(
    *,
    query: str,
    selected: str,
    tied: Sequence[str],
    fields: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if _value_explicitly_mentioned(selected, query):
        return {"source": "candidate_identity", "value": selected}
    selected_fields = fields[selected]
    for path, value in selected_fields.items():
        if not _value_explicitly_mentioned(value, query):
            continue
        if all(fields[other].get(path) != value for other in tied if other != selected):
            return {"source": path, "value": value}
    return None


def _collection_sources(
    records: Sequence[ExecutionRecord], pre_state: Mapping[str, Any]
) -> Iterable[tuple[str, dict[str, Mapping[str, Any]]]]:
    # Real successful tool results have priority over broad environment state.
    for record in reversed(records):
        semantic = classify_execution_result(record.call.name, record.execution_result)
        if semantic.outcome != ExecutionSemanticOutcome.SUCCESS:
            continue
        yield from _iter_candidate_collections(
            record.execution_result,
            path=f"turn[{record.turn_id}].call[{record.call_id}].result",
        )
    yield from _iter_candidate_collections(pre_state, path="pre_state")


def relational_resolution_gate(draft: ConversationDraft) -> GateResult:
    """Fail closed on a recoverable, ambiguous/non-extremal GT selection."""

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    prior_records: list[ExecutionRecord] = []

    for turn_index, turn in enumerate(draft.turns):
        relations = _relations_in_query(turn.query)
        turn_records: list[ExecutionRecord] = []
        if turn.is_intentional_missing:
            prior_records.extend(turn.execution_records)
            continue
        for record in turn.execution_records:
            if not relations:
                turn_records.append(record)
                prior_records.append(record)
                continue
            matched_selection = False
            history = prior_records + turn_records
            for parameter, argument_value in record.call.arguments.items():
                for source_path, collection in _collection_sources(
                    history, record.pre_state
                ):
                    selected = _selected_identity(argument_value, collection)
                    if selected is None:
                        continue
                    matched_selection = True
                    fields = _candidate_scalar_fields(collection)
                    for relation in relations:
                        resolved = _relation_field(relation, turn.query, fields)
                        base = {
                            "turn_id": turn_index,
                            "call_id": record.call_id,
                            "function": record.call.name,
                            "parameter": parameter,
                            "selected": selected,
                            "relation": relation.phrase,
                            "source_path": source_path,
                        }
                        if resolved is None:
                            failure = {
                                **base,
                                "status": "NOT_RECOVERABLE",
                                "reason": "relation field is not uniquely recoverable",
                            }
                            checks.append(failure)
                            failures.append(failure)
                            continue
                        field_path, comparable = resolved
                        raw_values = {
                            identity: fields[identity][field_path]
                            for identity in comparable
                        }
                        extreme = (
                            max(item[1] for item in comparable.values())
                            if relation.direction == "max"
                            else min(item[1] for item in comparable.values())
                        )
                        tied = sorted(
                            identity
                            for identity, item in comparable.items()
                            if item[1] == extreme
                        )
                        row = {
                            **base,
                            "field_path": field_path,
                            "candidate_values": raw_values,
                            "extreme_candidates": tied,
                        }
                        if selected not in tied:
                            failure = {
                                **row,
                                "status": "NON_EXTREMUM_SELECTION",
                                "reason": "GT-selected object is not the requested extremum",
                            }
                            checks.append(failure)
                            failures.append(failure)
                            continue
                        if len(tied) > 1:
                            disambiguation = _tie_disambiguation(
                                query=turn.query,
                                selected=selected,
                                tied=tied,
                                fields=fields,
                            )
                            if disambiguation is None:
                                failure = {
                                    **row,
                                    "status": "AMBIGUOUS_TIE",
                                    "reason": "multiple extrema tie and final query does not disambiguate",
                                }
                                checks.append(failure)
                                failures.append(failure)
                                continue
                            row["disambiguation"] = disambiguation
                        row["status"] = "RESOLVED"
                        checks.append(row)
                    # The first real source that contains the selected object
                    # is authoritative; duplicated snapshots are not re-counted.
                    break
            if relations and not matched_selection:
                checks.append(
                    {
                        "turn_id": turn_index,
                        "call_id": record.call_id,
                        "function": record.call.name,
                        "relations": [item.phrase for item in relations],
                        "status": "NO_DOWNSTREAM_OBJECT_SELECTION",
                        "reason": (
                            "no GT argument selects an object from a recoverable "
                            "collection; function contract/final verifier remains responsible"
                        ),
                    }
                )
            turn_records.append(record)
            prior_records.append(record)

    metadata = {
        "source_status": SOURCE_STATUS,
        "checks": checks,
        "failures": failures,
        "arbitrary_tie_break_used": False,
        "embedding_or_distance_threshold_used": False,
    }
    if failures:
        first = failures[0]
        return GateResult(
            "relational_resolution_gate",
            False,
            f"{first['status']} at turn={first['turn_id']} "
            f"function={first['function']} parameter={first['parameter']}: "
            f"{first['reason']}",
            metadata,
        )
    return GateResult(
        "relational_resolution_gate",
        True,
        "all recoverable relational GT selections are unique and correct",
        metadata,
    )
