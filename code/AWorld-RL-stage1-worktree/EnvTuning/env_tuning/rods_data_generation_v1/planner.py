"""Project task-only, category-aware Planner transport and strict parser."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Callable, Sequence

from .function_catalog import FunctionCatalog
from .error_taxonomy import ERROR_GUIDANCE
from .llm_backend import LLMBackend
from .metrics import GeneratorMetrics
from .models import ErrorRecord, PlannerResult, SeedRecord
from .parsing import StructuredParseError, parse_planner_response
from .prompts import load_prompt
from .contracts import planner_input_validator
from .planner_contract import (
    CATEGORY_PROMPTS, allowed_source_functions, build_planner_input, planner_max_turns,
)


class NoValidFunctionsError(ValueError):
    """The reconstructed catalog adapter has no remaining executable choice."""


def _mapping_delta(base: Any, current: Any) -> Any:
    """Return accumulated patch changes for the compact retry-feedback block.

    The current config itself is supplied in initial_environment; forensic
    per-call pre/post state is intentionally excluded from retry feedback.
    """

    if isinstance(base, Mapping) and isinstance(current, Mapping):
        delta: dict[str, Any] = {}
        for key, value in current.items():
            if key not in base:
                delta[str(key)] = value
                continue
            nested = _mapping_delta(base[key], value)
            if nested is not None:
                delta[str(key)] = nested
        return delta or None
    return None if base == current else current


def _planner_failure_view(error: ErrorRecord) -> dict[str, Any]:
    """Keep complete failure semantics while excluding forensic state blobs."""

    context = {
        key: error.context[key]
        for key in ("call", "result")
        if key in error.context
    }
    return {
        "attempt_id": error.attempt_id,
        "error_type": error.error_type.value,
        "turn_id": error.turn_id,
        "function_names": list(error.function_names),
        "detail": error.detail,
        "context": context,
    }


class PlannerAgent:
    def __init__(
        self,
        backend: LLMBackend,
        catalog: FunctionCatalog,
        metrics: GeneratorMetrics,
        *,
        max_parse_retries: int = 3,
        environment_factory: Any = None,
        validation_policy: str = "strict",
    ) -> None:
        self.backend = backend
        self.catalog = catalog
        self.metrics = metrics
        self.max_parse_retries = max_parse_retries
        self.environment_factory = environment_factory
        self.validation_policy = validation_policy
        self.rendered_prompts: list[str] = []
        self.rendered_inputs: list[dict[str, Any]] = []

    def _render(
        self,
        seed: SeedRecord,
        *,
        failure_history: Sequence[ErrorRecord],
        blocked_functions: set[str],
        current_config: dict[str, Any],
    ) -> tuple[str, list[str]]:
        snapshot = None
        if self.environment_factory is not None:
            session = self.environment_factory.create(
                initial_config=current_config,
                involved_classes=self.catalog.infer_seed_classes(seed),
                seed_id=seed.sample_id,
                long_context=seed.data_type == "multi_turn_long_context",
                purpose="planner_initial_state",
            )
            try:
                snapshot = session.snapshot()
            finally:
                session.close()
        payload = build_planner_input(
            seed, self.catalog, current_config=current_config,
            initial_runtime_snapshot=snapshot,
        )
        names = [name for name in allowed_source_functions(payload) if name not in blocked_functions]
        if not names:
            raise NoValidFunctionsError(
                "no unblocked functions remain for inferred seed classes"
            )
        if failure_history:
            # Compact new-attempt feedback follows the existing retry policy;
            # the v2 prompt and its transport are project substitutions.
            payload["retry_feedback"] = {
                "failures": [_planner_failure_view(error) for error in failure_history],
                "blocked_functions": sorted(blocked_functions),
                "guidance": [ERROR_GUIDANCE[error.error_type] for error in failure_history],
                "config_patch_delta": _mapping_delta(seed.initial_config, current_config) or {},
                "instruction": "Generate a COMPLETELY DIFFERENT plan using different functions.",
            }
        elif blocked_functions:
            payload["retry_feedback"] = {"blocked_functions": sorted(blocked_functions)}
        planner_input_validator().validate(payload)
        common_prompt = "project/planner_common.txt"
        category_prompt = CATEGORY_PROMPTS[seed.data_type]
        if self.validation_policy == "rods":
            common_prompt = "project/planner_common_rods.txt"
            if seed.data_type == "multi_turn_long_context":
                category_prompt = "project/planner_categories/long_context_rods.txt"
        prompt = load_prompt(common_prompt, {
            "category_rules": load_prompt(category_prompt),
            "min_turns": 2,
            "max_turns": planner_max_turns(seed.data_type),
            "planner_input": json.dumps(payload, ensure_ascii=False, indent=2),
        })
        self.rendered_inputs.append(payload)
        self.rendered_prompts.append(prompt)
        return prompt, names

    async def plan(
        self,
        seed: SeedRecord,
        *,
        failure_history: Sequence[ErrorRecord],
        blocked_functions: set[str],
        current_config: dict[str, Any],
        call_observer: Callable[[], None] | None = None,
    ) -> PlannerResult:
        prompt, names = self._render(
            seed,
            failure_history=failure_history,
            blocked_functions=blocked_functions,
            current_config=current_config,
        )
        last_error: Exception | None = None
        for parse_attempt in range(self.max_parse_retries):
            self.metrics.increment("planner/planner_calls")
            if call_observer is not None:
                call_observer()
            response = await self.backend.complete(
                role="planner",
                messages=[{"role": "user", "content": prompt}],
                metadata={"seed_id": seed.sample_id, "parse_attempt": parse_attempt + 1},
            )
            self.metrics.increment("latency/planner_seconds_sum", response.latency_seconds)
            self.metrics.increment("latency/planner_count")
            try:
                return parse_planner_response(
                    response.text,
                    allowed_functions=names,
                    class_for_function=self.catalog.class_for_function(),
                    blocked_functions=blocked_functions,
                    max_turns=planner_max_turns(seed.data_type),
                )
            except StructuredParseError as exc:
                self.metrics.increment("planner/planner_parse_failures")
                last_error = exc
        raise StructuredParseError(f"planner exhausted parse retries: {last_error}")
