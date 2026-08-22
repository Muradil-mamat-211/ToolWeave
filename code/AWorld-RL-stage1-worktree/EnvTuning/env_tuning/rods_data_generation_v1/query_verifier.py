"""Reconstructed query-to-executed-GT semantic verifier transport."""

from __future__ import annotations

import json
from typing import Sequence

from .llm_backend import LLMBackend
from .metrics import GeneratorMetrics
from .models import ConversationDraft, ExecutionRecord, GateResult
from .parsing import StructuredParseError, parse_verifier_response
from .prompts import load_prompt
from .validation.semantic_grounding import semantic_context_for_verifier


class QueryVerifier:
    SOURCE_STATUS = "RECONSTRUCTED_FROM_RODS_SPEC"
    FINAL_SOURCE_STATUS = "PROJECT_SEMANTIC_GUARD"

    def __init__(self, backend: LLMBackend, metrics: GeneratorMetrics):
        self.backend = backend
        self.metrics = metrics

    async def verify(
        self,
        *,
        query: str,
        turn_records: Sequence[ExecutionRecord],
        execution_context: dict,
    ) -> tuple[str, bool]:
        prompt = load_prompt(
            "reconstructed/query_verification.txt",
            {
                "query": query,
                "ground_truth_context": json.dumps(
                    [
                        {"call": record.canonical_call, "result": record.execution_result}
                        for record in turn_records
                    ],
                    ensure_ascii=False,
                    default=repr,
                ),
                "execution_context": json.dumps(execution_context, ensure_ascii=False, default=repr),
            },
        )
        response = await self.backend.complete(
            role="query_verifier",
            messages=[{"role": "user", "content": prompt}],
            metadata={"turn_id": turn_records[0].turn_id},
        )
        self.metrics.increment("latency/query_verifier_seconds_sum", response.latency_seconds)
        self.metrics.increment("latency/query_verifier_count")
        return parse_verifier_response(response.text)

    async def verify_final_conversation(self, draft: ConversationDraft) -> GateResult:
        """Re-verify final post-rewrite/post-transform queries before Judge."""

        prompt = load_prompt(
            "reconstructed/final_semantic_verification.txt",
            {"final_semantic_context": semantic_context_for_verifier(draft)},
        )
        response = await self.backend.complete(
            role="final_query_verifier",
            messages=[{"role": "user", "content": prompt}],
            metadata={
                "num_turns": len(draft.turns),
                "source_status": self.FINAL_SOURCE_STATUS,
            },
        )
        self.metrics.increment(
            "latency/final_query_verifier_seconds_sum", response.latency_seconds
        )
        self.metrics.increment("latency/final_query_verifier_count")
        try:
            reason, accepted = parse_verifier_response(response.text)
        except StructuredParseError as exc:
            self.metrics.increment("queries/final_query_verify_parse_failures")
            return GateResult(
                "final_query_semantic_gate",
                False,
                f"fail-closed final verifier parse error: {exc}",
                {"source_status": self.FINAL_SOURCE_STATUS},
            )
        if not accepted:
            self.metrics.increment("queries/final_query_verify_failures")
        return GateResult(
            "final_query_semantic_gate",
            accepted,
            reason,
            {
                "source_status": self.FINAL_SOURCE_STATUS,
                "global_coherence_source_status": (
                    "PROJECT_SEMANTIC_GUARD/GLOBAL_COHERENCE"
                ),
                "verdict": "accept" if accepted else "reject",
                "request_id": response.request_id,
            },
        )
