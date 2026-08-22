"""Reconstructed Missing Function transform based on Appendix P semantics."""

from __future__ import annotations

import copy
import json

from ..function_catalog import FunctionCatalog
from ..llm_backend import LLMBackend
from ..metrics import GeneratorMetrics
from ..models import ConversationDraft, SynthesizedTurn
from ..parsing import StructuredParseError, parse_missing_function_response
from ..prompts import load_prompt


class MissingFunctionTransformer:
    SOURCE_STATUS = "RECONSTRUCTED_FROM_RODS_SPEC"

    def __init__(
        self,
        backend: LLMBackend,
        catalog: FunctionCatalog,
        metrics: GeneratorMetrics | None = None,
    ):
        self.backend = backend
        self.catalog = catalog
        self.metrics = metrics

    async def transform(self, draft: ConversationDraft) -> ConversationDraft:
        prompt = load_prompt(
            "reconstructed/missing_function.txt",
            {
                "conversation": json.dumps(
                    [
                        {
                            "turn": turn.turn_id,
                            "query": turn.query,
                            "calls": [call.canonical() for call in turn.calls],
                        }
                        for turn in draft.turns
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                "eligible_functions": json.dumps(
                    sorted({call.name for turn in draft.turns for call in turn.calls}),
                    ensure_ascii=False,
                ),
            },
        )
        response = await self.backend.complete(
            role="missing_function",
            messages=[{"role": "user", "content": prompt}],
            metadata={"num_turns": len(draft.turns)},
        )
        if self.metrics is not None:
            self.metrics.increment("latency/missing_function_seconds_sum", response.latency_seconds)
            self.metrics.increment("latency/missing_function_count")
        choice = parse_missing_function_response(response.text)
        if not 0 <= choice.affected_turn < len(draft.turns):
            raise StructuredParseError("missing-function affected turn is out of range")
        affected = draft.turns[choice.affected_turn]
        if choice.function_name not in {call.name for call in affected.calls}:
            raise StructuredParseError("withheld function is not required by affected turn")
        initial_names = {tool.get("name") for tool in draft.initial_tools}
        if choice.function_name not in initial_names:
            raise StructuredParseError("withheld function was not initially available")
        if any(
            choice.function_name in {call.name for call in prior.calls}
            for prior in draft.turns[: choice.affected_turn]
        ):
            raise StructuredParseError(
                "withheld function is required before the selected affected turn"
            )
        missing_schema = self.catalog.get(choice.function_name).schema

        output = copy.deepcopy(draft)
        output.initial_tools = [
            tool for tool in output.initial_tools if tool.get("name") != choice.function_name
        ]
        if not output.initial_tools:
            raise StructuredParseError("Missing Function transform would remove every initial tool")
        affected = output.turns[choice.affected_turn]
        recovery = copy.deepcopy(affected)
        affected.is_intentional_missing = True
        affected.missing_kind = "function"
        recovery.is_intentional_missing = False
        recovery.missing_kind = None
        recovery.raw_query = ""
        recovery.query = ""
        recovery.recovery_tools = [copy.deepcopy(missing_schema)]
        recovery.query_verification_reason = (
            "Appendix-P recovery turn restores the withheld tool for the unresolved request."
        )
        output.turns.insert(choice.affected_turn + 1, recovery)
        for turn_id, turn in enumerate(output.turns):
            turn.turn_id = turn_id
        output.structural_profile["adversarial"] = {
            "kind": "missing_function",
            "affected_turn": choice.affected_turn,
            "recovery_turn": choice.affected_turn + 1,
            "withheld_function": choice.function_name,
            "source_status": self.SOURCE_STATUS,
        }
        return output
