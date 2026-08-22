"""RODS Appendix C.1 Planner prompt transport and strict parser."""

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
from .structural_profile import seed_structural_profile


class NoValidFunctionsError(ValueError):
    """The reconstructed catalog adapter has no remaining executable choice."""


def _mapping_delta(base: Any, current: Any) -> Any:
    """Return the deterministic merged-patch delta without copying base state.

    Retry Planner context must contain every accumulated patch, but copying an
    unchanged Long Context VM snapshot into every retry can exceed the serving
    model's context window. Deep merge never deletes keys, so a recursive
    current-vs-seed delta preserves every effective patch exactly.
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
    ) -> None:
        self.backend = backend
        self.catalog = catalog
        self.metrics = metrics
        self.max_parse_retries = max_parse_retries
        self.rendered_prompts: list[str] = []

    def _render(
        self,
        seed: SeedRecord,
        *,
        failure_history: Sequence[ErrorRecord],
        blocked_functions: set[str],
        current_config: dict[str, Any],
    ) -> tuple[str, list[str]]:
        classes = self.catalog.infer_seed_classes(seed)
        available = [
            spec
            for spec in self.catalog.functions_for_classes(classes)
            if spec.name not in blocked_functions
        ]
        names = [spec.name for spec in available]
        if not names:
            raise NoValidFunctionsError(
                "no unblocked functions remain for inferred seed classes"
            )
        function_rows = [
            {
                "name": spec.name,
                "class": spec.class_name,
                "level": spec.level,
                "description": spec.schema.get("description", ""),
            }
            for spec in available
        ]
        prompt = load_prompt(
            "official_rods/planner_user.txt",
            {
                "classes_str": ", ".join(classes),
                "queries_text": json.dumps(seed.Q_old, ensure_ascii=False),
                "gt_summary": json.dumps(seed.GT_old, ensure_ascii=False),
                "func_list": json.dumps(function_rows, ensure_ascii=False, indent=2),
            },
        )
        # PROJECT_STRUCTURAL_GUIDANCE: Appendix C.1 already asks the Planner
        # to preserve capability/dependency structure.  Public sources do not
        # publish a deterministic Phi extractor or acceptance threshold.  This
        # block only makes recoverable seed facts explicit to the Planner; the
        # same profile remains diagnostics-only downstream.
        prompt += (
            "\n\n# PROJECT_STRUCTURAL_GUIDANCE (NON-GATING)\n"
            "Recoverable seed structure:\n"
            + json.dumps(
                seed_structural_profile(seed, self.catalog),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\nPreserve the recoverable topology while generating a different task. "
            "No unpublished numeric structural threshold is applied."
        )
        if failure_history:
            # Appendix F requires this context on every re-plan; the exact
            # transport appendix is not published, so this additive block is
            # explicitly reconstruction around the unchanged C.1 template.
            prompt += (
                "\n\n# Feedback From Previous Failed Attempts\n"
                + "Complete failure history (compact Planner view; full VM forensics remain durable):\n"
                + json.dumps(
                    [_planner_failure_view(error) for error in failure_history],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\nBlocked functions:\n"
                + json.dumps(sorted(blocked_functions), ensure_ascii=False)
                + "\nError-specific guidance:\n"
                + json.dumps(
                    [ERROR_GUIDANCE[error.error_type] for error in failure_history],
                    ensure_ascii=False,
                )
                + "\nAccumulated effective config-patch delta from the seed config:\n"
                + json.dumps(
                    _mapping_delta(seed.initial_config, current_config) or {},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\nGenerate a COMPLETELY DIFFERENT plan using different functions."
            )
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
                )
            except StructuredParseError as exc:
                self.metrics.increment("planner/planner_parse_failures")
                last_error = exc
        raise StructuredParseError(f"planner exhausted parse retries: {last_error}")
