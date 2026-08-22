"""Appendix C.4 quality judge with deterministic leakage rejection."""

from __future__ import annotations

import json
import re
from typing import Any

from .llm_backend import LLMBackend
from .metrics import GeneratorMetrics
from .models import ConversationDraft, JudgeResult
from .parsing import parse_judge_response
from .prompts import load_prompt
from .query_generator import GENERATION_LEAKAGE, query_leaks_function_name


AUTOMATIC_LEAKAGE = re.compile(
    r"(?:\bthought\s+process\b|\bconstruct\s+query\b|\bstep\s+[12]\s*:)",
    re.IGNORECASE,
)
RAW_TOOL_JSON_LEAKAGE = re.compile(
    r"[\"'](?:name|parameters)[\"']\s*:", re.IGNORECASE
)


def conversation_summary(draft: ConversationDraft) -> dict[str, Any]:
    """Return the final, auditable conversation presented to quality agents."""

    return {
        "data_type": draft.data_type,
        "narrative": draft.narrative,
        "initial_config": draft.initial_config,
        "initial_tool_names": [tool.get("name") for tool in draft.initial_tools],
        "turns": [
            {
                "turn": turn.turn_id + 1,
                "query": turn.query,
                "ground_truth": turn.ground_truth,
                "intentional_missing": turn.is_intentional_missing,
                "missing_kind": turn.missing_kind,
                "tools_added_before_turn": [tool.get("name") for tool in turn.recovery_tools],
                "execution_results": [record.execution_result for record in turn.execution_records],
            }
            for turn in draft.turns
        ],
        "structural_profile": draft.structural_profile,
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

    def __init__(self, backend: LLMBackend, metrics: GeneratorMetrics):
        self.backend = backend
        self.metrics = metrics

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
