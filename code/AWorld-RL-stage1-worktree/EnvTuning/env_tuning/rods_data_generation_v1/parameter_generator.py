"""Reconstructed parameter-generation agent constrained by BFCL schemas."""

from __future__ import annotations

import json
from typing import Any, Sequence

from .function_catalog import CatalogError, FunctionCatalog
from .llm_backend import LLMBackend
from .metrics import GeneratorMetrics
from .models import ExecutionRecord, FunctionCall, FunctionSpec
from .parsing import StructuredParseError, parse_arguments_response
from .prompts import load_prompt


class ParameterGenerator:
    def __init__(
        self,
        backend: LLMBackend,
        catalog: FunctionCatalog,
        metrics: GeneratorMetrics,
        *,
        max_parse_attempts: int = 1,
    ) -> None:
        self.backend = backend
        self.catalog = catalog
        self.metrics = metrics
        self.max_parse_attempts = max_parse_attempts

    async def generate(
        self,
        *,
        spec: FunctionSpec,
        environment_state: dict[str, Any],
        execution_history: Sequence[ExecutionRecord],
        narrative: str,
        turn_id: int,
    ) -> FunctionCall:
        required = spec.schema.get("parameters", {}).get("required", [])
        example = json.dumps({str(name): "value" for name in required}, ensure_ascii=False)
        prompt = load_prompt(
            "reconstructed/parameter_generation.txt",
            {
                "function_schema": json.dumps(spec.schema, ensure_ascii=False, indent=2),
                "environment_state": json.dumps(environment_state, ensure_ascii=False, default=repr),
                "execution_history": json.dumps(
                    [
                        {
                            "call": record.canonical_call,
                            "result": record.execution_result,
                            "turn_id": record.turn_id,
                            "call_id": record.call_id,
                        }
                        for record in execution_history
                    ],
                    ensure_ascii=False,
                    default=repr,
                ),
                "turn_intent": f"Narrative: {narrative}\nTurn index: {turn_id}",
                "arguments_example": example,
            },
        )
        last_error: Exception | None = None
        for parse_attempt in range(self.max_parse_attempts):
            response = await self.backend.complete(
                role="parameter_generator",
                messages=[{"role": "user", "content": prompt}],
                metadata={"function": spec.name, "turn_id": turn_id, "parse_attempt": parse_attempt + 1},
            )
            self.metrics.increment("latency/parameter_generator_seconds_sum", response.latency_seconds)
            self.metrics.increment("latency/parameter_generator_count")
            try:
                _, arguments = parse_arguments_response(response.text)
                self.catalog.validate_arguments(spec, arguments)
                return FunctionCall(spec.name, arguments, spec.class_name)
            except (StructuredParseError, CatalogError) as exc:
                last_error = exc
        raise StructuredParseError(f"parameter generation failed for {spec.name}: {last_error}")
