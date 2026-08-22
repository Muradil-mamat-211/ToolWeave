"""RODS Stage III whole-conversation coherence rewrite."""

from __future__ import annotations

import json

from .llm_backend import LLMBackend
from .metrics import GeneratorMetrics
from .models import ConversationDraft
from .parsing import StructuredParseError, parse_rewrite_response
from .prompts import load_prompt
from .query_generator import QueryGenerator


class CoherenceRewriteAgent:
    def __init__(self, backend: LLMBackend, metrics: GeneratorMetrics):
        self.backend = backend
        self.metrics = metrics
        self.calls = 0

    async def rewrite(self, draft: ConversationDraft) -> ConversationDraft:
        system = load_prompt("official_rods/coherence_rewrite_system.txt")
        turns = [
            {
                "turn": index + 1,
                "raw_query": turn.raw_query,
                "ground_truth": [call.canonical() for call in turn.calls],
                "execution_results": [record.execution_result for record in turn.execution_records],
            }
            for index, turn in enumerate(draft.turns)
        ]
        user = load_prompt(
            "official_rods/coherence_rewrite_user.txt",
            {
                "narrative": draft.narrative,
                "turns_for_rewrite": json.dumps(turns, ensure_ascii=False, indent=2, default=repr),
                "num_turns": len(draft.turns),
            },
        )
        self.calls += 1
        response = await self.backend.complete(
            role="coherence_rewrite",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            metadata={"num_turns": len(draft.turns)},
        )
        self.metrics.increment("latency/coherence_rewrite_seconds_sum", response.latency_seconds)
        self.metrics.increment("latency/coherence_rewrite_count")
        try:
            queries = parse_rewrite_response(response.text, expected_count=len(draft.turns))
            # Apply the same leakage gate as per-turn generation after the
            # holistic rewrite.  This does not perform an extra LLM call.
            for turn, query in zip(draft.turns, queries):
                QueryGenerator.validate_no_leakage(query, turn.execution_records)
                turn.query = query
        except StructuredParseError:
            self.metrics.increment("rewrite/rewrite_failure")
            raise
        self.metrics.increment("rewrite/rewrite_success")
        return draft
