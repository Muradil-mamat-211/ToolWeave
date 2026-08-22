"""Appendix C.5/C.6 one-cycle query-only refinement."""

from __future__ import annotations

import json
import re

from .llm_backend import LLMBackend
from .metrics import GeneratorMetrics
from .models import ConversationDraft, JudgeResult
from .parsing import parse_refine_classification, parse_refine_rewrite
from .prompts import load_prompt
from .quality_judge import conversation_summary
from .query_generator import QueryGenerator


TURN_REFERENCE = re.compile(r"\bturn\s*(?:#|number)?\s*(\d+)\b", re.IGNORECASE)


def locate_rejected_turn(fail_reason: str, *, num_turns: int) -> int:
    """Resolve an explicitly numbered, one-based turn or fail closed."""

    matches = {int(value) for value in TURN_REFERENCE.findall(fail_reason)}
    if len(matches) != 1:
        raise ValueError("query-fixable rejection must identify exactly one turn")
    one_based = matches.pop()
    if not 1 <= one_based <= num_turns:
        raise ValueError("rejection references a turn outside the conversation")
    return one_based - 1


class RefineAgent:
    def __init__(self, backend: LLMBackend, metrics: GeneratorMetrics):
        self.backend = backend
        self.metrics = metrics

    async def classify(
        self, draft: ConversationDraft, rejected: JudgeResult
    ) -> tuple[str, str]:
        system = load_prompt("official_rods/refine_classify_system.txt")
        user = load_prompt(
            "official_rods/refine_classify_user.txt",
            {
                "fail_reason": rejected.fail_reason,
                "data_summary": json.dumps(
                    conversation_summary(draft), ensure_ascii=False, indent=2, default=repr
                ),
            },
        )
        response = await self.backend.complete(
            role="refine_classify",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            metadata={"fail_reason": rejected.fail_reason},
        )
        self.metrics.increment("latency/refine_classify_seconds_sum", response.latency_seconds)
        self.metrics.increment("latency/refine_classify_count")
        reason, classification = parse_refine_classification(response.text)
        self.metrics.increment(f"validation/refine_{classification}")
        return reason, classification

    async def rewrite_query(
        self, draft: ConversationDraft, rejected: JudgeResult
    ) -> tuple[ConversationDraft, int, str]:
        turn_index = locate_rejected_turn(
            rejected.fail_reason, num_turns=len(draft.turns)
        )
        turn = draft.turns[turn_index]
        system = load_prompt("official_rods/refine_rewrite_system.txt")
        user = load_prompt(
            "official_rods/refine_rewrite_user.txt",
            {
                "fail_reason": rejected.fail_reason,
                "old_query": turn.query,
                "gt_str": json.dumps(turn.ground_truth, ensure_ascii=False),
            },
        )
        response = await self.backend.complete(
            role="refine_rewrite",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            metadata={"turn_index": turn_index, "fail_reason": rejected.fail_reason},
        )
        self.metrics.increment("latency/refine_rewrite_seconds_sum", response.latency_seconds)
        self.metrics.increment("latency/refine_rewrite_count")
        rewritten = parse_refine_rewrite(response.text)
        QueryGenerator.validate_no_leakage(rewritten, turn.execution_records)
        old_query = turn.query
        turn.query = rewritten
        return draft, turn_index, old_query
