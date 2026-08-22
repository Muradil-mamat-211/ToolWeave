"""Gate 2: turn-by-turn tool availability, including MF recovery."""

from __future__ import annotations

from ..models import ConversationDraft, GateResult


def tool_availability_gate(draft: ConversationDraft) -> GateResult:
    available = {
        tool.get("name") for tool in draft.initial_tools if isinstance(tool.get("name"), str)
    }
    adversarial = draft.structural_profile.get("adversarial", {})
    withheld = adversarial.get("withheld_function") if isinstance(adversarial, dict) else None
    affected_turn = adversarial.get("affected_turn") if isinstance(adversarial, dict) else None
    recovery_turn = adversarial.get("recovery_turn") if isinstance(adversarial, dict) else None
    trace: list[dict] = []
    for turn in draft.turns:
        restored = {
            tool.get("name")
            for tool in turn.recovery_tools
            if isinstance(tool.get("name"), str)
        }
        available.update(restored)
        call_names = [] if turn.is_intentional_missing else [call.name for call in turn.calls]
        missing = sorted(set(call_names) - available)
        trace.append(
            {
                "turn_id": turn.turn_id,
                "restored": sorted(restored),
                "available_count": len(available),
                "calls": call_names,
            }
        )
        if missing:
            return GateResult(
                "tool_availability_gate",
                False,
                f"GT functions unavailable at turn {turn.turn_id}: {missing}",
                {"trace": trace},
            )
        if withheld and turn.turn_id == affected_turn and withheld in available:
            return GateResult(
                "tool_availability_gate",
                False,
                "MF withheld function is available during affected turn",
                {"trace": trace},
            )
        if withheld and turn.turn_id == recovery_turn and withheld not in available:
            return GateResult(
                "tool_availability_gate",
                False,
                "MF withheld function was not restored on recovery turn",
                {"trace": trace},
            )
    return GateResult(
        "tool_availability_gate",
        True,
        "every final GT call is available in its turn",
        {"trace": trace},
    )
