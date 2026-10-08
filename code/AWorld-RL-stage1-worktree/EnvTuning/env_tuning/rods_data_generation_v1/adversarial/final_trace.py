"""Place synthesis evidence on final executable turns, without duplicate calls."""
from __future__ import annotations

import copy
from dataclasses import replace

from ..models import ConversationDraft


def finalize_recovery_trace(draft: ConversationDraft) -> None:
    previous = None
    for turn_id, turn in enumerate(draft.turns):
        turn.turn_id = turn_id
        if turn.is_intentional_missing:
            # calls retain the intended operation for omission diagnostics;
            # this final actor turn has GT=[] and no executed observation.
            turn.execution_records = []
            continue
        records = []
        for record in turn.execution_records:
            provenance = copy.deepcopy(record.dependency_provenance)
            provenance["synthesis_turn_id"] = record.turn_id
            provenance["state_predecessor"] = None if previous is None else {
                "turn_id": previous.turn_id,
                "call_id": previous.call_id,
                "exact_state_continuity": previous.post_state == record.pre_state,
            }
            updated = replace(record, turn_id=turn_id, dependency_provenance=provenance)
            records.append(updated)
            previous = updated
        turn.execution_records = records


def planner_scaffold_alignment(draft: ConversationDraft) -> dict:
    """Map original executable scaffold steps to final execution evidence."""
    mapping = []
    for turn in draft.turns:
        if turn.is_intentional_missing:
            continue
        sources = sorted({record.dependency_provenance.get("synthesis_turn_id", record.turn_id)
                          for record in turn.execution_records})
        for source in sources:
            mapping.append({"scaffold_turn_id": source, "final_executable_turn_id": turn.turn_id,
                            "functions": [record.call.name for record in turn.execution_records
                                          if record.dependency_provenance.get("synthesis_turn_id", record.turn_id) == source]})
    return {"source": "ACTUAL_EXECUTION_RECORD_SYNTHESIS_TURN_ID", "index_base": 0,
            "narrative_numbering": "PRE_TRANSFORM_EXECUTABLE_SCAFFOLD",
            "intentional_missing_turn_ids": [turn.turn_id for turn in draft.turns if turn.is_intentional_missing],
            "mapping": mapping}
