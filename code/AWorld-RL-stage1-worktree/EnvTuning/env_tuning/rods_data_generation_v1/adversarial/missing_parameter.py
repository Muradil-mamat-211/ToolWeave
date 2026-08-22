"""Reconstructed Missing Parameter clarification transform."""

from __future__ import annotations

import copy
import json
from typing import Any

from ..function_catalog import FunctionCatalog
from ..llm_backend import LLMBackend
from ..metrics import GeneratorMetrics
from ..models import ConversationDraft
from ..parsing import StructuredParseError, parse_missing_parameter_response
from ..prompts import load_prompt
from ..validation.missing_parameter_validity import (
    evaluate_missing_parameter_validity,
)


def _value_markers(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.casefold()]
    if isinstance(value, (int, float, bool)):
        return [str(value).casefold()]
    if isinstance(value, (list, tuple)):
        markers: list[str] = []
        for item in value:
            markers.extend(_value_markers(item))
        return markers or [json.dumps(value, ensure_ascii=False).casefold()]
    return [json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()]


class MissingParameterTransformer:
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
            "reconstructed/missing_parameter.txt",
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
                )
            },
        )
        response = await self.backend.complete(
            role="missing_parameter",
            messages=[{"role": "user", "content": prompt}],
            metadata={"num_turns": len(draft.turns)},
        )
        if self.metrics is not None:
            self.metrics.increment("latency/missing_parameter_seconds_sum", response.latency_seconds)
            self.metrics.increment("latency/missing_parameter_count")
        choice = parse_missing_parameter_response(response.text)
        if not 0 <= choice.affected_turn < len(draft.turns):
            raise StructuredParseError("missing-parameter affected turn is out of range")
        source_turn = draft.turns[choice.affected_turn]
        values: list[Any] = []
        for call in source_turn.calls:
            required = set(
                self.catalog.get(call.name).schema.get("parameters", {}).get("required", [])
            )
            if choice.parameter_name in call.arguments and choice.parameter_name in required:
                values.append(call.arguments[choice.parameter_name])
        if not values:
            raise StructuredParseError("selected missing parameter is not required by affected GT")
        if any(value != values[0] for value in values[1:]):
            raise StructuredParseError("selected parameter has ambiguous values across calls")
        markers = [marker for marker in _value_markers(values[0]) if marker]
        affected_lower = choice.affected_query.casefold()
        recovery_lower = choice.recovery_query.casefold()
        if any(marker in affected_lower for marker in markers):
            raise StructuredParseError("affected query still exposes the deliberately missing value")
        if markers and not all(marker in recovery_lower for marker in markers):
            raise StructuredParseError("recovery query does not supply the missing value")

        # PROJECT_MISSING_PARAMETER_VALIDITY_GUARD: current-query omission is
        # insufficient.  A unique ID/value already visible in prior assistant
        # calls or successful tool observations does not require clarification.
        validity = evaluate_missing_parameter_validity(
            draft,
            affected_turn=choice.affected_turn,
            parameter=choice.parameter_name,
            target_value=values[0],
            affected_query=choice.affected_query,
            catalog=self.catalog,
        )
        if validity["decision"] == "REJECT_UNIQUELY_RECOVERABLE":
            raise StructuredParseError(
                "selected missing parameter is uniquely recoverable from "
                "policy-visible context"
            )
        if validity["decision"] == "REJECT_VALUE_STILL_EXPOSED":
            raise StructuredParseError(
                "affected query still exposes the deliberately missing value"
            )

        output = copy.deepcopy(draft)
        affected = output.turns[choice.affected_turn]
        recovery = copy.deepcopy(affected)
        affected.raw_query = choice.affected_query
        affected.query = choice.affected_query
        affected.is_intentional_missing = True
        affected.missing_kind = "parameter"
        recovery.raw_query = choice.recovery_query
        recovery.query = choice.recovery_query
        recovery.is_intentional_missing = False
        recovery.missing_kind = None
        recovery.recovery_tools = []
        recovery.query_verification_reason = (
            "Appendix-P recovery turn explicitly supplies the omitted parameter value."
        )
        output.turns.insert(choice.affected_turn + 1, recovery)
        for turn_id, turn in enumerate(output.turns):
            turn.turn_id = turn_id
        output.structural_profile["adversarial"] = {
            "kind": "missing_parameter",
            "affected_turn": choice.affected_turn,
            "recovery_turn": choice.affected_turn + 1,
            "missing_parameter": choice.parameter_name,
            "source_status": self.SOURCE_STATUS,
            "missing_parameter_validity": validity,
        }
        return output
