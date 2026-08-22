"""Gate 3: conservative recursive Appendix G complexity bounds."""

from __future__ import annotations

from typing import Any

from ..models import ConversationDraft, GateResult


def _violation(value: Any, path: str) -> str | None:
    if isinstance(value, str) and len(value) > 200:
        return f"string exceeds 200 characters at {path}"
    if isinstance(value, (list, tuple)):
        if len(value) > 5:
            return f"list/tuple exceeds 5 elements at {path}"
        for index, item in enumerate(value):
            issue = _violation(item, f"{path}[{index}]")
            if issue:
                return issue
    if isinstance(value, dict):
        for key, item in value.items():
            issue = _violation(item, f"{path}.{key}")
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
            issue = _violation(call.arguments, f"turn[{turn.turn_id}].{call.name}.arguments")
            if issue:
                return GateResult("parameter_complexity_gate", False, issue)
    return GateResult(
        "parameter_complexity_gate",
        True,
        "recursive list/tuple and string limits satisfied",
        {"checked_call_count": checked, "interpretation": "recursive conservative engineering interpretation"},
    )
