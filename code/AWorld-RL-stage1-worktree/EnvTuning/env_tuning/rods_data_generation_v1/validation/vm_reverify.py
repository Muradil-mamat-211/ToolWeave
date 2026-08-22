"""Gate 1: replay final GT in a fresh real BFCL VM."""

from __future__ import annotations

from ..environment_adapter import EnvironmentFactory
from ..models import ConversationDraft, GateResult
from ..result_semantics import (
    ExecutionSemanticOutcome,
    classify_execution_result,
    normalize_execution_result,
)


def fresh_vm_reverify_gate(
    draft: ConversationDraft,
    *,
    environment_factory: EnvironmentFactory,
    seed_id: str,
) -> GateResult:
    session = environment_factory.create(
        initial_config=draft.initial_config,
        involved_classes=draft.involved_classes,
        seed_id=seed_id,
        long_context=draft.data_type == "multi_turn_long_context",
        purpose="fresh_final_reverify",
    )
    fresh_id = session.environment_id
    if fresh_id == draft.synthesis_environment_id:
        session.close()
        return GateResult("fresh_vm_gate", False, "fresh VM reused synthesis instance")
    executed = 0
    try:
        for turn in draft.turns:
            if turn.is_intentional_missing:
                continue
            for call in turn.calls:
                result = session.execute(call)
                executed += 1
                # PROJECT_SEMANTIC_GUARD: the fresh gate shares the exact
                # classifier used during synthesis.  A stale/custom session
                # ``success=True`` flag cannot override a structured error.
                normalized = normalize_execution_result(result.result)
                semantic = classify_execution_result(call.name, normalized)
                if (
                    not result.success
                    or semantic.outcome == ExecutionSemanticOutcome.HARD_ERROR
                ):
                    return GateResult(
                        "fresh_vm_gate",
                        False,
                        result.error_detail
                        or semantic.detail
                        or "fresh VM execution failed",
                        {
                            "environment_id": fresh_id,
                            "turn_id": turn.turn_id,
                            "call": call.canonical(),
                            "result": normalized,
                            "semantic_outcome": semantic.outcome.value,
                            "semantic_detail": semantic.detail,
                        },
                    )
    finally:
        session.close()
    return GateResult(
        "fresh_vm_gate",
        True,
        "all final GT calls executed on a fresh instance",
        {
            "environment_id": fresh_id,
            "synthesis_environment_id": draft.synthesis_environment_id,
            "executed_call_count": executed,
        },
    )
