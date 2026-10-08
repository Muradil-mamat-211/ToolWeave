"""Reconstructed query-to-executed-GT semantic verifier transport."""

from __future__ import annotations

import json
from typing import Sequence

from .llm_backend import LLMBackend
from .function_catalog import FunctionCatalog
from .metrics import GeneratorMetrics
from .models import ConversationDraft, ExecutionRecord, GateResult
from .parsing import StructuredParseError, parse_verifier_response
from .prompts import load_prompt
from .validation.semantic_grounding import semantic_context_for_verifier


class QueryVerifier:
    SOURCE_STATUS = "RECONSTRUCTED_FROM_RODS_SPEC"
    FINAL_SOURCE_STATUS = "PROJECT_SEMANTIC_GUARD"

    def __init__(self, backend: LLMBackend, metrics: GeneratorMetrics, *, catalog: FunctionCatalog | None = None):
        self.backend = backend
        self.metrics = metrics
        self.catalog = catalog

    async def verify(
        self,
        *,
        query: str,
        turn_records: Sequence[ExecutionRecord],
        execution_context: dict,
        prior_records: Sequence[ExecutionRecord] = (),
    ) -> tuple[str, bool]:
        evidence = {
            "visibility_note": "Pre/post states are verifier evidence, not extra actor-visible history. Reconcile return prose with actual state effects and tool contracts; hidden state does not supply missing user information.",
            "used_tool_definitions": ([self.catalog.get(name).schema for name in sorted({record.call.name for record in turn_records})]
                                      if self.catalog is not None else []),
            "execution_trace": [{"call": record.canonical_call, "success": record.success,
                                 "pre_state": record.pre_state, "result": record.execution_result,
                                 "post_state": record.post_state} for record in turn_records],
        }
        if prior_records:
            evidence["actor_visible_prior_trace"] = [{"turn_id": record.turn_id,
                "call": record.canonical_call, "result": record.execution_result} for record in prior_records]
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
                "execution_context": json.dumps({"environment_snapshot": execution_context,
                                                "verification_evidence": evidence}, ensure_ascii=False, default=repr),
            },
        )
        prompt += "\n\n" + load_prompt("project/response_presentation.txt", {})
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
        prompt += "\n\n" + load_prompt("project/response_presentation.txt", {})
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
