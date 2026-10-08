"""Project guards for deterministic text requests and observable history reuse.

These guards constrain new tasks rather than changing frozen BFCL/MatchTIR
rewards. They do not prove every possible tool trajectory equivalent.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Any

from ..models import ConversationDraft, GateResult

SOURCE_STATUS = "PROJECT_QUERY_CONTRACT_GUARD"
# Audited text-producing arguments, not every schema string (IDs/paths/enums).
TEXT_ARGUMENTS = {"send_message": "message", "echo": "content"}
PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")
QUOTED = re.compile(r"`([^`\n]+)`|\"([^\"\n]+)\"|'([^'\n]+)'")
FORMAT_INTENT = re.compile(r"\b(?:exact|fixed)\s+(?:message\s+)?(?:format|template)\b", re.I)
READ_FUNCTIONS = {"cat", "grep", "wc", "ls", "find", "tail"}
WORD_COUNTS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
               "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _leaves(value: Any, path: str = "result"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        yield path + ".length", str(len(value)), "ARRAY_LENGTH"
        for index, child in enumerate(value):
            yield from _leaves(child, f"{path}[{index}]")
    elif value is not None:
        yield path, str(value), "EXACT_VISIBLE_TEXT"


def _selection_matches(slot: str, bound: str, leaf: str, context: str, record,
                       path: str, transform: str) -> bool:
    normalized = slot.casefold()
    matching_count = normalized in {"update_lines", "matching_lines", "matched_lines", "match_count"}
    if normalized in {"lines", "line_count"} and re.search(r"\bnumber of matching lines\b", context, re.I):
        matching_count = True
    if matching_count:
        return (record.call.name == "grep" and path == "result.matching_lines.length"
                and transform == "ARRAY_LENGTH" and bound == leaf)
    count_units = {"characters": "c", "character_count": "c", "chars": "c",
                   "lines": "l", "line_count": "l", "words": "w", "word_count": "w"}
    if normalized in count_units:
        return (record.call.name == "wc" and path == "result.count" and bound == leaf
                and record.call.arguments.get("mode", "l") == count_units[normalized])
    if normalized in {"function", "function_name", "declared_function", "fn"}:
        if record.call.name not in {"cat", "grep"} or not str(record.call.arguments.get("file_name", "")).endswith(".py"):
            return False
        names = re.findall(r"\b(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", leaf)
        if names:
            if re.search(r"\bfirst\s+(?:declared\s+)?function\b", context, re.I):
                return bound == names[0]
            if re.search(r"\blast\s+(?:declared\s+)?function\b", context, re.I):
                return bound == names[-1]
            return len(names) == 1 and bound == names[0]
        return False
    if normalized in {"ending", "tail", "closing", "suffix", "last_words", "final_words"}:
        count = re.search(r"\b(?:last|final)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:nonempty\s+)?(?:whitespace[- ]separated\s+)?(?:words|tokens)\b", context, re.I)
        if count:
            token = count.group(1).lower()
            number = int(token) if token.isdigit() else WORD_COUNTS[token]
            return 0 < number <= len(leaf.split()) and bound == " ".join(leaf.split()[-number:])
    return True


def _occurrences(value: str, text: str):
    if not value:
        return []
    # Avoid accidental partial identifiers/numbers, preserving exact case/text.
    left = r"(?<!\w)" if value[0].isalnum() else ""
    right = r"(?!\w)" if value[-1].isalnum() else ""
    return list(re.finditer(left + re.escape(value) + right, text))


def _templates(context: str):
    for quote in QUOTED.finditer(context):
        text = next(group for group in quote.groups() if group is not None)
        slots = list(PLACEHOLDER.finditer(text))
        if not slots or not FORMAT_INTENT.search(context[max(0, quote.start() - 100):quote.start()]):
            continue
        names = [slot.group(1) for slot in slots]
        if len(names) != len(set(names)):
            continue
        chunks, cursor = [], 0
        for slot in slots:
            chunks.extend([re.escape(text[cursor:slot.start()]), "(.+?)"])
            cursor = slot.end()
        chunks.append(re.escape(text[cursor:]))
        # Adjacent placeholders cannot provide a unique segmentation.
        if any(a.end() == b.start() for a, b in zip(slots, slots[1:])):
            continue
        yield text, names, re.compile("".join(chunks), re.S)


def bind_exact_text(value: str, context: str, successful) -> dict[str, Any] | None:
    """Bind a text argument to exact visible facts and supported computations.

    This is value provenance, not a verdict on the whole task or LC policy.
    The admission gate independently checks answer leakage/history coverage.
    """
    for template, names, regex in _templates(context):
        match = regex.fullmatch(value)
        if match is None:
            continue
        bindings = []
        for name, bound in zip(names, match.groups()):
            sources = []
            for record in successful:
                if not record.success:
                    continue
                for path, leaf, transform in _leaves(record.execution_result):
                    if not _selection_matches(name, bound, leaf, context, record, path, transform):
                        continue
                    for occurrence in _occurrences(bound, leaf):
                        sources.append({"turn_id": record.turn_id, "call_id": record.call_id,
                            "path": path, "start": occurrence.start(), "end": occurrence.end(),
                            "observation_characters": len(leaf), "value_transform": transform,
                            "span_measurement": "ACTOR_VISIBLE_SCALAR_TEXT"})
            bindings.append({"slot": name, "value": bound, "sources": sources})
        if all(binding["sources"] for binding in bindings):
            return {"template": template, "bindings": bindings}
    return None


def query_contract_gate(
    draft: ConversationDraft, *, min_long_observation_chars: int = 2048,
    minimum_span_fraction: float = 0.5, minimum_late_fraction: float = 0.8,
) -> GateResult:
    checks, failures, cross_turn_bindings, long_pairs = [], [], [], []
    successful = []
    context_queries = []
    has_text_call = any(
        call.name in TEXT_ARGUMENTS
        for turn in draft.turns if not turn.is_intentional_missing for call in turn.calls
    )
    for turn in draft.turns:
        context_queries.append(turn.query)
        if turn.is_intentional_missing:
            continue
        context = "\n".join(context_queries)
        for call_id, call in enumerate(turn.calls):
            if call.name == "grep":
                pattern = call.arguments.get("pattern")
                quoted_values = [next(g for g in match.groups() if g is not None)
                                 for match in QUOTED.finditer(turn.query)]
                if pattern not in quoted_values:
                    failures.append({"turn_id": turn.turn_id, "call_id": call_id,
                        "function": call.name, "reason": "Search text must be explicitly quoted, preserving whitespace."})
            parameter = TEXT_ARGUMENTS.get(call.name)
            if parameter is not None:
                value = call.arguments.get(parameter)
                if not isinstance(value, str):
                    failures.append({"turn_id": turn.turn_id, "function": call.name,
                                     "reason": "Text argument is not a string."})
                    continue
                prior_reads = [record for record in successful
                               if record.turn_id < turn.turn_id and record.call.name in READ_FUNCTIONS]
                check = {"turn_id": turn.turn_id, "call_id": call_id,
                         "function": call.name, "parameter": parameter}
                if prior_reads and value in context:
                    failures.append({**check, "reason": "The fully instantiated historical answer was pasted into user context."})
                    continue
                # An explicitly supplied user literal is deterministic, but cannot
                # replace history in a file-inspection -> derived-text task.
                if value in context and not prior_reads:
                    check["mode"] = "EXPLICIT_USER_LITERAL"
                    checks.append(check)
                else:
                    bound_text = bind_exact_text(value, context, successful)
                    accepted = ({**check, "mode": "EXACT_HISTORY_TEMPLATE", **bound_text}
                                if bound_text is not None else None)
                    if accepted is None:
                        failures.append({**check, "reason": "Text must be a user literal or match an explicit exact format with observable slot values; inspected facts must not be pasted as a resolved answer."})
                    else:
                        checks.append(accepted)
                        earlier = [source for binding in accepted["bindings"] for source in binding["sources"]
                                   if source["turn_id"] < turn.turn_id]
                        cross_turn_bindings.extend(earlier)
                        if prior_reads and not earlier:
                            failures.append({**check, "reason": "Derived text does not actually reuse an earlier turn."})
                        for first, second in combinations(accepted["bindings"], 2):
                            if first["value"] == second["value"]:
                                continue
                            for a in first["sources"]:
                                for b in second["sources"]:
                                    size = a["observation_characters"]
                                    if (a["turn_id"], a["call_id"], a["path"]) != (b["turn_id"], b["call_id"], b["path"]) or a["turn_id"] >= turn.turn_id or size < min_long_observation_chars:
                                        continue
                                    span = (max(a["end"], b["end"]) - min(a["start"], b["start"])) / size
                                    late = max(a["end"], b["end"]) / size
                                    if span >= minimum_span_fraction and late >= minimum_late_fraction:
                                        long_pairs.append({"consumer_turn": turn.turn_id, "template": accepted["template"],
                                            "first": a, "second": b, "span_fraction": span, "late_fraction": late})
            # Only completed success records, never hidden snapshots, enter evidence.
            record = next((record for record in turn.execution_records if record.call_id == call_id), None)
            if record is not None and record.success:
                successful.append(record)
    if draft.data_type == "multi_turn_long_context" and not long_pairs:
        failures.append({"reason": "LC needs an exact later output using two distinct facts from widely separated positions of a large earlier observation."})
    return GateResult(
        name="query_contract_gate", passed=not failures,
        detail="exact text, quoted search conditions and observable history contracts passed" if not failures else failures[0]["reason"],
        metadata={"source_status": SOURCE_STATUS, "checks": checks, "failures": failures,
                  "cross_turn_binding_count": len(cross_turn_bindings), "long_context_pairs": long_pairs,
                  "policy": {"minimum_span_fraction": minimum_span_fraction,
                             "minimum_late_fraction": minimum_late_fraction,
                             "threshold_source": "PROJECT_POLICY_NOT_BFCL_OFFICIAL"},
                  "text_argument_scope": TEXT_ARGUMENTS, "has_audited_text_call": has_text_call,
                  "hidden_snapshot_used": False, "training_reward_changed": False},
    )
