"""Read-only reconstruction of historical candidates for semantic revalidation.

SOURCE_STATUS = PROJECT_SEMANTIC_GUARD

Historical artifacts are never edited.  This adapter rebuilds the internal
draft from their durable execution trace and frozen Training row so the final
semantic gates can be applied in a new artifact directory.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from .candidate_builder import FUNCTION_MARKER, UPDATE_MESSAGE
from .function_catalog import FunctionCatalog
from .models import (
    ConversationDraft,
    ExecutionRecord,
    FunctionCall,
    SynthesizedTurn,
)
from .result_semantics import classify_execution_result
from .validation.semantic_grounding import semantic_grounding_gate


SOURCE_STATUS = "PROJECT_SEMANTIC_GUARD"


def _query_text(raw: Any) -> str:
    if not isinstance(raw, list) or not raw:
        return ""
    message = raw[0]
    return str(message.get("content", "")) if isinstance(message, Mapping) else ""


def _recovery_tools(processed: Any) -> list[dict[str, Any]]:
    if not isinstance(processed, str) or UPDATE_MESSAGE not in processed:
        return []
    raw = processed.split("\n" + UPDATE_MESSAGE, 1)[0]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list) or not all(isinstance(item, Mapping) for item in parsed):
        return []
    return [dict(item) for item in parsed]


def candidate_to_draft(candidate: Mapping[str, Any]) -> ConversationDraft:
    sample = candidate["sample"]
    metadata = candidate["generation_metadata"]
    kwargs = sample["extra_info"]["interaction_kwargs"]
    questions = list(kwargs["question"])
    processed = list(kwargs.get("processed_question", []))
    trace = list(metadata["execution_trace"])
    if len(trace) != len(questions):
        raise ValueError("historical execution trace and question count differ")

    turns: list[SynthesizedTurn] = []
    for turn_index, trace_turn in enumerate(trace):
        records: list[ExecutionRecord] = []
        for call_index, raw_record in enumerate(trace_turn.get("records", [])):
            raw_call = raw_record["call"]
            call = FunctionCall(
                name=str(raw_call["name"]),
                arguments=dict(raw_call["arguments"]),
                class_name=str(raw_call["class_name"]),
            )
            semantic = classify_execution_result(call.name, raw_record.get("execution_result"))
            records.append(
                ExecutionRecord(
                    turn_id=turn_index,
                    call_id=call_index,
                    call=call,
                    canonical_call=str(raw_record.get("canonical_call", call.canonical())),
                    pre_state=copy.deepcopy(raw_record.get("pre_state", {})),
                    execution_result=copy.deepcopy(raw_record.get("execution_result")),
                    post_state=copy.deepcopy(raw_record.get("post_state", {})),
                    dependency_provenance=copy.deepcopy(
                        raw_record.get("dependency_provenance", {})
                    ),
                    success=semantic.outcome.value != "HARD_ERROR",
                    semantic_outcome=semantic.outcome.value,
                    semantic_detail=semantic.detail,
                    error_detail=(
                        semantic.detail if semantic.outcome.value == "HARD_ERROR" else None
                    ),
                )
            )
        recovery = _recovery_tools(processed[turn_index - 1]) if turn_index > 0 else []
        turns.append(
            SynthesizedTurn(
                turn_id=turn_index,
                class_name=(records[0].call.class_name if records else ""),
                calls=[record.call for record in records],
                execution_records=records,
                raw_query=_query_text(questions[turn_index]),
                query=_query_text(questions[turn_index]),
                query_verification_reason="historical infrastructure-clean candidate",
                recovery_tools=recovery,
                is_intentional_missing=bool(trace_turn.get("intentional_missing", False)),
                missing_kind=trace_turn.get("missing_kind"),
            )
        )

    system_prompt = sample["prompt"][0]["content"]
    if FUNCTION_MARKER not in system_prompt:
        raise ValueError("candidate system prompt has no function marker")
    initial_tools = json.loads(system_prompt.split(FUNCTION_MARKER, 1)[1])
    raw_initial_config = kwargs["initial_config"]
    initial_config = (
        json.loads(raw_initial_config)
        if isinstance(raw_initial_config, str)
        else copy.deepcopy(raw_initial_config)
    )
    return ConversationDraft(
        narrative=str(
            metadata.get(
                "latent_narrative",
                "NOT_RECOVERABLE_FROM_HISTORICAL_CANDIDATE_PAYLOAD",
            )
        ),
        data_type=str(sample["data_source"]),
        initial_config=initial_config,
        initial_tools=initial_tools,
        involved_classes=list(kwargs["involved_classes"]),
        turns=turns,
        synthesis_environment_id=str(
            metadata.get("synthesis_environment_id", "historical")
        ),
        structural_profile=copy.deepcopy(metadata.get("structural_profile", {})),
    )


def revalidate_candidate_grounding(
    candidate: Mapping[str, Any], *, catalog: FunctionCatalog
) -> tuple[ConversationDraft, Any]:
    draft = candidate_to_draft(candidate)
    return draft, semantic_grounding_gate(draft, catalog=catalog)
