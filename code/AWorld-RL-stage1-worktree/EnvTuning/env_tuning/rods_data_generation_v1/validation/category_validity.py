"""Category protocol checks; LC character threshold is a project policy."""
from __future__ import annotations

import json

from ..models import ConversationDraft, GateResult


def category_validity_gate(draft: ConversationDraft, *, min_long_observation_chars: int = 2048) -> GateResult:
    name = "category_validity_gate"
    missing = [i for i, turn in enumerate(draft.turns) if turn.is_intentional_missing]
    metadata = {"data_type": draft.data_type, "final_turn_count": len(draft.turns)}
    if not 2 <= len(draft.turns) <= 5:
        return GateResult(name, False, "final conversation must have 2-5 turns including recovery", metadata)
    expected = {"multi_turn_miss_func": "function", "multi_turn_miss_param": "parameter"}
    if draft.data_type in expected:
        if len(missing) != 1:
            return GateResult(name, False, "missing category requires exactly one affected turn", metadata)
        index = missing[0]
        affected = draft.turns[index]
        if affected.execution_records:
            return GateResult(name, False, "intentional missing turn cannot contain executed observations", metadata)
        if affected.missing_kind != expected[draft.data_type] or index + 1 >= len(draft.turns):
            return GateResult(name, False, "missing kind/recovery position is invalid", metadata)
        recovery = draft.turns[index + 1]
        if recovery.is_intentional_missing or not recovery.calls:
            return GateResult(name, False, "missing turn requires executable immediate recovery", metadata)
        if draft.data_type == "multi_turn_miss_func" and not recovery.recovery_tools:
            return GateResult(name, False, "missing-function recovery must restore a schema", metadata)
        if draft.data_type == "multi_turn_miss_param" and not recovery.query.strip():
            return GateResult(name, False, "missing-parameter recovery needs a user clarification", metadata)
    elif draft.data_type in {"multi_turn_base", "multi_turn_long_context"}:
        if missing:
            return GateResult(name, False, "normal categories cannot contain intentional missing turns", metadata)
    else:
        return GateResult(name, False, "unsupported category", metadata)
    if draft.data_type == "multi_turn_long_context":
        exposures = []
        for index, turn in enumerate(draft.turns[:-1]):
            for record in turn.execution_records:
                if record.success:
                    size = len(json.dumps(record.execution_result, ensure_ascii=False))
                    if size >= min_long_observation_chars:
                        exposures.append({"turn_id": index, "call": record.canonical_call, "characters": size})
        metadata.update({"large_prior_observations": exposures,
                         "minimum_observation_characters": min_long_observation_chars,
                         "threshold_source": "PROJECT_POLICY_NOT_BFCL_OFFICIAL",
                         "dependency_verification": "required from final semantic verifier"})
        if not exposures:
            return GateResult(name, False, "long-context task lacks a large prior actor-visible observation", metadata)
    return GateResult(name, True, "category protocol and measured exposure passed", metadata)
