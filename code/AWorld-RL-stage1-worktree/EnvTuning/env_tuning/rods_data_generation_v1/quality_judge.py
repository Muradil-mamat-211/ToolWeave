"""Appendix C.4 quality judge with deterministic leakage rejection."""

from __future__ import annotations

import json
import re
from typing import Any

from .llm_backend import LLMBackend
from .function_catalog import CatalogError, FunctionCatalog
from .metrics import GeneratorMetrics
from .models import ConversationDraft, JudgeResult
from .parsing import parse_judge_response
from .prompts import load_prompt
from .query_generator import GENERATION_LEAKAGE, query_leaks_function_name
from .adversarial.final_trace import planner_scaffold_alignment


AUTOMATIC_LEAKAGE = re.compile(
    r"(?:\bthought\s+process\b|\bconstruct\s+query\b|\bstep\s+[12]\s*:)",
    re.IGNORECASE,
)
RAW_TOOL_JSON_LEAKAGE = re.compile(
    r"[\"'](?:name|parameters)[\"']\s*:", re.IGNORECASE
)


def response_contract_diagnostics(records, definitions) -> list[dict[str, Any]]:
    """Audit source return-type mismatches without changing real VM returns."""
    diagnostics = []

    def inspect(value, schema, path, record):
        if not isinstance(schema, dict):
            return
        try:
            FunctionCatalog._validate_value(value, {key: child for key, child in schema.items() if key != "items"}, path=path)
        except CatalogError as exc:
            diagnostics.append({"turn_id": record.turn_id, "call_id": record.call_id,
                "function": record.call.name, "path": path, "declared_type": schema.get("type"),
                "actual_type": type(value).__name__, "detail": str(exc),
                "source": "ACTUAL_FROZEN_VM_RETURN_VS_SOURCE_SCHEMA"})
            return
        if isinstance(value, dict):
            for key, child_schema in schema.get("properties", {}).items():
                if key in value:
                    inspect(value[key], child_schema, f"{path}.{key}", record)
        elif isinstance(value, list) and schema.get("type") == "array":
            for index, child in enumerate(value):
                inspect(child, schema.get("items", {}), f"{path}[{index}]", record)

    for record in records:
        schema = definitions.get(record.call.name, {}).get("response")
        if schema is not None:
            inspect(record.execution_result, schema, "result", record)
    return diagnostics


def conversation_summary(draft: ConversationDraft) -> dict[str, Any]:
    """Return the final, auditable conversation presented to quality agents."""

    used_names = {call.name for turn in draft.turns for call in turn.calls}
    actor_definitions = {
        tool["name"]: tool
        for tool in draft.initial_tools
        if tool.get("name")
    }
    for turn in draft.turns:
        for tool in turn.recovery_tools:
            if tool.get("name"):
                actor_definitions[tool["name"]] = tool
    definitions = {name: tool for name, tool in actor_definitions.items() if name in used_names}
    available = {tool.get("name") for tool in draft.initial_tools if tool.get("name")}
    turns = []
    for turn in draft.turns:
        added = [tool.get("name") for tool in turn.recovery_tools if tool.get("name")]
        available.update(added)
        turns.append({
            "turn": turn.turn_id + 1,
            "query": turn.query,
            "ground_truth": turn.ground_truth,
            "intentional_missing": turn.is_intentional_missing,
            "missing_kind": turn.missing_kind,
            "tools_added_before_turn": added,
            "actor_visible_tool_names": sorted(available),
            "execution_results": [record.execution_result for record in turn.execution_records],
        })
    return {
        "data_type": draft.data_type,
        "narrative": draft.narrative,
        "planner_scaffold_alignment": planner_scaffold_alignment(draft),
        "initial_config": draft.initial_config,
        "initial_tool_names": [tool.get("name") for tool in draft.initial_tools],
        "turns": turns,
        "structural_profile": draft.structural_profile,
        "verification_evidence": {
            "source_status": "PROJECT_EXECUTION_EVIDENCE",
            "visibility_note": (
                "Pre/post snapshots are verifier evidence, not extra actor-visible history. "
                "Use the initial tool names and recovery timeline for actor availability. "
                "Reconcile returned descriptions with source schemas and actual state effects; "
                "hidden snapshots do not supply an actor's missing parameter."
            ),
            "used_tool_definitions": [definitions[name] for name in sorted(definitions)],
            # MF review needs alternatives that were visible but not chosen by
            # the scaffold. Names alone cannot establish their capabilities.
            "other_actor_tool_definitions": [
                actor_definitions[name] for name in sorted(actor_definitions)
                if name not in used_names
            ],
            "response_contract_diagnostics": response_contract_diagnostics(
                [record for turn in draft.turns for record in turn.execution_records], definitions),
            "execution_trace": [
                {
                    "turn_id": record.turn_id,
                    "call_id": record.call_id,
                    "call": record.canonical_call,
                    "success": record.success,
                    "pre_state": record.pre_state,
                    "result": record.execution_result,
                    "post_state": record.post_state,
                }
                for turn in draft.turns
                for record in turn.execution_records
            ],
        },
    }


def deterministic_leakage_reason(draft: ConversationDraft) -> str | None:
    """Detect explicit function/data-generation leakage before an LLM verdict.

    The paper also asks the Judge to detect parameter-name leakage.  That
    semantic check stays in the official Judge prompt because many public BFCL
    parameter identifiers are ordinary words (for example ``number``), making
    a blanket substring rule unsound.
    """

    for turn in draft.turns:
        query = turn.query
        if (
            GENERATION_LEAKAGE.search(query)
            or AUTOMATIC_LEAKAGE.search(query)
            or RAW_TOOL_JSON_LEAKAGE.search(query)
        ):
            return f"Turn {turn.turn_id + 1} contains generation/prompt leakage"
        for call in turn.calls:
            if query_leaks_function_name(query, call.name):
                return f"Turn {turn.turn_id + 1} leaks function name {call.name}"
    return None


class QualityJudgeAgent:
    """Strictly parse Appendix C.4 decisions; malformed output never accepts."""

    def __init__(self, backend: LLMBackend, metrics: GeneratorMetrics, *, validation_policy: str = "strict"):
        self.backend = backend
        self.metrics = metrics
        self.validation_policy = validation_policy

    async def evaluate(self, draft: ConversationDraft, *, pass_index: int = 1) -> JudgeResult:
        leakage = deterministic_leakage_reason(draft)
        if leakage is not None:
            self.metrics.increment("validation/judge_reject")
            return JudgeResult(
                reason="Deterministic automatic-rejection pattern matched.",
                decision="reject",
                fail_reason=leakage,
            )

        system = load_prompt("official_rods/quality_judge_system.txt")
        guidance = ("project/rods_quality_guidance.txt" if self.validation_policy == "rods"
                    else "project/quality_review_contract.txt")
        system += "\n\n" + load_prompt(guidance)
        system += "\n\n" + load_prompt("project/response_presentation.txt")
        user = load_prompt(
            "official_rods/quality_judge_user.txt",
            {
                "sample_summary": json.dumps(
                    conversation_summary(draft), ensure_ascii=False, indent=2, default=repr
                )
            },
        )
        response = await self.backend.complete(
            role="quality_judge",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            metadata={"pass_index": pass_index, "num_turns": len(draft.turns)},
        )
        self.metrics.increment("latency/quality_judge_seconds_sum", response.latency_seconds)
        self.metrics.increment("latency/quality_judge_count")
        result = parse_judge_response(response.text)
        self.metrics.increment(
            "validation/judge_accept" if result.accepted else "validation/judge_reject"
        )
        if pass_index == 2:
            self.metrics.increment(
                "validation/second_judge_accept"
                if result.accepted
                else "validation/second_judge_reject"
            )
        return result
