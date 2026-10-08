"""Gate 3: conservative recursive Appendix G complexity bounds."""

from __future__ import annotations

from typing import Any

from ..models import ConversationDraft, GateResult

MAX_ARGUMENT_STRING_CHARACTERS = 200
MAX_ARGUMENT_COLLECTION_ITEMS = 5
ARGUMENT_LIMITS = {
    "max_string_characters": MAX_ARGUMENT_STRING_CHARACTERS,
    "max_collection_items": MAX_ARGUMENT_COLLECTION_ITEMS,
}


def argument_complexity_violation(value: Any, path: str = "arguments") -> str | None:
    if isinstance(value, str) and len(value) > MAX_ARGUMENT_STRING_CHARACTERS:
        return f"string exceeds {MAX_ARGUMENT_STRING_CHARACTERS} characters at {path}"
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_ARGUMENT_COLLECTION_ITEMS:
            return f"list/tuple exceeds {MAX_ARGUMENT_COLLECTION_ITEMS} elements at {path}"
        for index, item in enumerate(value):
            issue = argument_complexity_violation(item, f"{path}[{index}]")
            if issue:
                return issue
    if isinstance(value, dict):
        for key, item in value.items():
            issue = argument_complexity_violation(item, f"{path}.{key}")
            if issue:
                return issue
    return None


def parameter_complexity_gate(draft: ConversationDraft) -> GateResult:
    checked = 0
    for turn in draft.turns:
        if turn.is_intentional_missing:
            continue
        for call in turn.calls:
            checked += 1
            issue = argument_complexity_violation(call.arguments, f"turn[{turn.turn_id}].{call.name}.arguments")
            if issue:
                return GateResult("parameter_complexity_gate", False, issue)
    return GateResult(
        "parameter_complexity_gate",
        True,
        "recursive list/tuple and string limits satisfied",
        {"checked_call_count": checked, "interpretation": "recursive conservative engineering interpretation"},
    )
