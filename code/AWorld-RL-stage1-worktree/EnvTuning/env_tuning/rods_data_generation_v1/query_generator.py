"""GT-first natural query generation with leakage checks."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .function_catalog import FunctionCatalog
from .llm_backend import LLMBackend
from .metrics import GeneratorMetrics
from .models import ExecutionRecord
from .parsing import StructuredParseError, parse_query_response
from .query_prompt_registry import QueryPromptRegistry


GENERATION_LEAKAGE = re.compile(
    r"\b(?:data generation|ground truth|function schema|construct query|thought process)\b",
    re.IGNORECASE,
)


def query_leaks_function_name(query: str, function_name: str) -> bool:
    """Return true only for deterministic function-identifier leakage.

    BFCL contains function names that are also ordinary user verbs, including
    ``find`` and ``comment``.  A bare occurrence of such a word is not proof
    that the API identifier leaked.  Names with implementation syntax
    (snake_case/camelCase) remain unambiguous, while ordinary lowercase words
    are rejected only when introduced as a function/tool invocation.
    Semantic or technical-jargon leakage remains a Quality Judge criterion.
    """

    if not function_name:
        return False
    exact_name = re.escape(function_name)
    if not re.search(rf"\b{exact_name}\b", query, flags=re.IGNORECASE):
        return False
    identifier_like = "_" in function_name or any(char.isupper() for char in function_name)
    if identifier_like:
        return True
    return bool(
        re.search(
            rf"\b(?:call|invoke|run|use)\s+(?:the\s+)?(?:function\s+|tool\s+)?{exact_name}\b"
            rf"|\b(?:function|tool)\s+(?:named\s+)?{exact_name}\b",
            query,
            flags=re.IGNORECASE,
        )
    )


class QueryGenerator:
    def __init__(
        self,
        backend: LLMBackend,
        catalog: FunctionCatalog,
        metrics: GeneratorMetrics,
        *,
        max_parse_attempts: int = 1,
        prompt_registry: QueryPromptRegistry | None = None,
    ) -> None:
        self.backend = backend
        self.catalog = catalog
        self.metrics = metrics
        self.max_parse_attempts = max_parse_attempts
        self.prompt_registry = prompt_registry or QueryPromptRegistry()

    @staticmethod
    def validate_no_leakage(query: str, records: Sequence[ExecutionRecord]) -> None:
        if GENERATION_LEAKAGE.search(query):
            raise StructuredParseError("query contains data-generation leakage")
        for record in records:
            if query_leaks_function_name(query, record.call.name):
                raise StructuredParseError(f"query leaks function name {record.call.name}")

    async def generate(
        self,
        *,
        class_name: str,
        narrative: str,
        turn_records: Sequence[ExecutionRecord],
        prior_queries: Sequence[str],
    ) -> tuple[str, str]:
        if not turn_records:
            raise StructuredParseError("query generation requires executed GT calls")
        record_classes = {record.call.class_name for record in turn_records}
        if record_classes != {class_name}:
            raise StructuredParseError(
                f"query class {class_name!r} does not match executed calls {sorted(record_classes)!r}"
            )
        prompt = self.prompt_registry.render(
            class_name,
            {
                "narrative": narrative,
                "executed_calls": json.dumps(
                    [
                        {"call": record.canonical_call, "result": record.execution_result}
                        for record in turn_records
                    ],
                    ensure_ascii=False,
                    default=repr,
                ),
                "prior_context": json.dumps(list(prior_queries), ensure_ascii=False),
            },
        )
        last_error: Exception | None = None
        for parse_attempt in range(self.max_parse_attempts):
            response = await self.backend.complete(
                role="query_generator",
                messages=[{"role": "user", "content": prompt}],
                metadata={
                    "turn_id": turn_records[0].turn_id,
                    "class_name": class_name,
                    "prompt_source_status": self.prompt_registry.SOURCE_STATUS,
                    "parse_attempt": parse_attempt + 1,
                },
            )
            self.metrics.increment("latency/query_generator_seconds_sum", response.latency_seconds)
            self.metrics.increment("latency/query_generator_count")
            try:
                reason, query = parse_query_response(response.text)
                self.validate_no_leakage(query, turn_records)
                return reason, query
            except StructuredParseError as exc:
                last_error = exc
        self.metrics.increment("queries/query_gen_failures")
        raise StructuredParseError(f"query generation failed: {last_error}")
