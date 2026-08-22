"""RODS Stage II: sample -> parameters -> VM -> query -> verification."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .environment_adapter import EnvironmentFactory
from .error_taxonomy import ErrorType, PATCHABLE_ERRORS
from .function_catalog import CatalogError, FunctionCatalog
from .metrics import GeneratorMetrics
from .models import (
    ConversationDraft,
    ErrorRecord,
    ExecutionRecord,
    FunctionSpec,
    PlannerResult,
    SeedRecord,
    SynthesizedTurn,
)
from .parameter_generator import ParameterGenerator
from .parsing import StructuredParseError
from .query_generator import QueryGenerator
from .query_verifier import QueryVerifier
from .result_semantics import find_unclassified_suspicious_results
from .structural_profile import draft_structural_profile


class StageFailure(RuntimeError):
    def __init__(self, error: ErrorRecord):
        super().__init__(error.detail)
        self.error = error


class ExecutionOrchestrator:
    def __init__(
        self,
        *,
        catalog: FunctionCatalog,
        environment_factory: EnvironmentFactory,
        parameter_generator: ParameterGenerator,
        query_generator: QueryGenerator,
        query_verifier: QueryVerifier,
        metrics: GeneratorMetrics,
        suspicious_result_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.catalog = catalog
        self.environment_factory = environment_factory
        self.parameter_generator = parameter_generator
        self.query_generator = query_generator
        self.query_verifier = query_verifier
        self.metrics = metrics
        self.suspicious_result_sink = suspicious_result_sink

    def _failure(
        self,
        error_type: ErrorType,
        *,
        seed: SeedRecord,
        attempt_id: int,
        turn_id: int | None,
        function_names: Sequence[str],
        detail: str,
        context: dict[str, Any] | None = None,
    ) -> StageFailure:
        return StageFailure(
            ErrorRecord(
                error_type=error_type,
                seed_id=seed.sample_id,
                attempt_id=attempt_id,
                turn_id=turn_id,
                function_names=tuple(function_names),
                detail=detail,
                patchable=error_type in PATCHABLE_ERRORS,
                context=context or {},
            )
        )

    def _expand_functions(
        self,
        names: Sequence[str],
        *,
        seed: SeedRecord,
        attempt_id: int,
        turn_id: int,
    ) -> list[FunctionSpec]:
        expanded: list[FunctionSpec] = []
        for name in names:
            try:
                spec = self.catalog.get(name)
            except CatalogError as exc:
                raise self._failure(
                    ErrorType.FUNC_SAMPLE_FAILED,
                    seed=seed,
                    attempt_id=attempt_id,
                    turn_id=turn_id,
                    function_names=[name],
                    detail=str(exc),
                ) from exc
            if spec.level == "BOTTOM_LEVEL":
                expanded.append(spec)
                continue
            try:
                decomposition = self.catalog.decompose(spec)
                for entry in decomposition:
                    child_name = entry.get("name")
                    if not isinstance(child_name, str):
                        raise CatalogError("decomposition entry has no function name")
                    expanded.append(self.catalog.get(child_name))
            except CatalogError as exc:
                raise self._failure(
                    ErrorType.DECOMPOSE_FAILED,
                    seed=seed,
                    attempt_id=attempt_id,
                    turn_id=turn_id,
                    function_names=[name],
                    detail=str(exc),
                ) from exc
        return expanded

    async def execute(
        self,
        *,
        seed: SeedRecord,
        plan: PlannerResult,
        initial_config: dict[str, Any],
        attempt_id: int,
    ) -> ConversationDraft:
        involved_classes = sorted({turn.class_name for turn in plan.turns})
        session = self.environment_factory.create(
            initial_config=copy.deepcopy(initial_config),
            involved_classes=involved_classes,
            seed_id=seed.sample_id,
            long_context=seed.data_type == "multi_turn_long_context",
            purpose=f"synthesis_attempt_{attempt_id}",
        )
        environment_id = session.environment_id
        all_records: list[ExecutionRecord] = []
        turns: list[SynthesizedTurn] = []
        prior_queries: list[str] = []
        try:
            for plan_turn in plan.turns:
                self.metrics.increment("execution/turns_planned")
                self.metrics.increment("execution/functions_planned", len(plan_turn.function_names))
                specs = self._expand_functions(
                    plan_turn.function_names,
                    seed=seed,
                    attempt_id=attempt_id,
                    turn_id=plan_turn.turn_id,
                )
                turn_records: list[ExecutionRecord] = []
                seen_calls: set[str] = set()
                for spec in specs:
                    try:
                        call = await self.parameter_generator.generate(
                            spec=spec,
                            environment_state=session.snapshot(),
                            execution_history=all_records,
                            narrative=plan.narrative,
                            turn_id=plan_turn.turn_id,
                        )
                    except (StructuredParseError, CatalogError) as exc:
                        raise self._failure(
                            ErrorType.PARAM_GEN_FAILED,
                            seed=seed,
                            attempt_id=attempt_id,
                            turn_id=plan_turn.turn_id,
                            function_names=[spec.name],
                            detail=str(exc),
                        ) from exc
                    canonical = call.canonical()
                    if canonical in seen_calls:
                        raise self._failure(
                            ErrorType.DUPLICATE_FUNC,
                            seed=seed,
                            attempt_id=attempt_id,
                            turn_id=plan_turn.turn_id,
                            function_names=[call.name],
                            detail=f"duplicate call in turn: {canonical}",
                        )
                    seen_calls.add(canonical)
                    vm_result = session.execute(call)
                    suspicious_results = find_unclassified_suspicious_results(
                        call.name, vm_result.result
                    )
                    if suspicious_results:
                        self.metrics.increment(
                            "execution/unclassified_suspicious_result",
                            len(suspicious_results),
                        )
                        if self.suspicious_result_sink is not None:
                            for observation in suspicious_results:
                                self.suspicious_result_sink(
                                    {
                                        **observation,
                                        "seed_id": seed.sample_id,
                                        "attempt_id": attempt_id,
                                        "turn_id": plan_turn.turn_id,
                                        "call_id": len(turn_records),
                                        "canonical_call": canonical,
                                    }
                                )
                    prior_record = all_records[-1] if all_records else None
                    record = ExecutionRecord(
                        turn_id=plan_turn.turn_id,
                        call_id=len(turn_records),
                        call=call,
                        canonical_call=canonical,
                        pre_state=vm_result.pre_state,
                        execution_result=vm_result.result,
                        post_state=vm_result.post_state,
                        dependency_provenance={
                            "available_previous_calls": [r.canonical_call for r in all_records],
                            "parameter_source": "LLM proposed; schema checked; VM executed",
                            "resolved_dependency_call_ids": [],
                            "parameter_dependency_status": "NOT_RECOVERABLE",
                            # Telemetry only: these exact observations remain
                            # SUCCESS until a public return contract is audited.
                            "unclassified_suspicious_results": suspicious_results,
                            "state_predecessor": (
                                {
                                    "turn_id": prior_record.turn_id,
                                    "call_id": prior_record.call_id,
                                    "exact_state_continuity": (
                                        prior_record.post_state == vm_result.pre_state
                                    ),
                                }
                                if prior_record is not None
                                else None
                            ),
                        },
                        success=vm_result.success,
                        semantic_outcome=vm_result.semantic_outcome,
                        semantic_detail=vm_result.semantic_detail,
                        error_detail=vm_result.error_detail,
                    )
                    turn_records.append(record)
                    all_records.append(record)
                    if not vm_result.success:
                        self.metrics.increment("execution/VM_failure")
                        raise self._failure(
                            ErrorType.VM_EXEC_FAILED,
                            seed=seed,
                            attempt_id=attempt_id,
                            turn_id=plan_turn.turn_id,
                            function_names=[call.name],
                            detail=vm_result.error_detail or "BFCL VM rejected call",
                            context={
                                "call": canonical,
                                "arguments": copy.deepcopy(call.arguments),
                                "function_schema": copy.deepcopy(spec.schema),
                                "pre_state": copy.deepcopy(vm_result.pre_state),
                                "result": copy.deepcopy(vm_result.result),
                                "post_state": copy.deepcopy(vm_result.post_state),
                                "previous_successful_calls": [
                                    record.canonical_call
                                    for record in all_records[:-1]
                                    if record.success
                                ],
                                "attempt_initial_config": copy.deepcopy(initial_config),
                            },
                        )
                    self.metrics.increment("execution/VM_success")
                    if vm_result.semantic_outcome == "DOMAIN_NEGATIVE":
                        self.metrics.increment("execution/domain_negative")
                    self.metrics.increment("execution/GT_calls_executed")

                try:
                    _, query = await self.query_generator.generate(
                        class_name=plan_turn.class_name,
                        narrative=plan.narrative,
                        turn_records=turn_records,
                        prior_queries=prior_queries,
                    )
                except FileNotFoundError as exc:
                    raise self._failure(
                        ErrorType.NO_PROMPTS,
                        seed=seed,
                        attempt_id=attempt_id,
                        turn_id=plan_turn.turn_id,
                        function_names=[spec.name for spec in specs],
                        detail=str(exc),
                    ) from exc
                except StructuredParseError as exc:
                    raise self._failure(
                        ErrorType.QUERY_GEN_FAILED,
                        seed=seed,
                        attempt_id=attempt_id,
                        turn_id=plan_turn.turn_id,
                        function_names=[spec.name for spec in specs],
                        detail=str(exc),
                    ) from exc
                try:
                    verify_reason, accepted = await self.query_verifier.verify(
                        query=query,
                        turn_records=turn_records,
                        execution_context=session.snapshot(),
                    )
                except StructuredParseError as exc:
                    self.metrics.increment("queries/query_verify_failures")
                    raise self._failure(
                        ErrorType.QUERY_VERIFY_NO_TAG,
                        seed=seed,
                        attempt_id=attempt_id,
                        turn_id=plan_turn.turn_id,
                        function_names=[spec.name for spec in specs],
                        detail=str(exc),
                    ) from exc
                if not accepted:
                    self.metrics.increment("queries/query_verify_failures")
                    raise self._failure(
                        ErrorType.QUERY_VERIFY_FAILED,
                        seed=seed,
                        attempt_id=attempt_id,
                        turn_id=plan_turn.turn_id,
                        function_names=[spec.name for spec in specs],
                        detail=verify_reason,
                    )
                prior_queries.append(query)
                turns.append(
                    SynthesizedTurn(
                        turn_id=plan_turn.turn_id,
                        class_name=plan_turn.class_name,
                        calls=[record.call for record in turn_records],
                        execution_records=turn_records,
                        raw_query=query,
                        query=query,
                        query_verification_reason=verify_reason,
                    )
                )
        finally:
            session.close()

        initial_tools = [
            spec.schema for spec in self.catalog.functions_for_classes(involved_classes)
        ]
        return ConversationDraft(
            narrative=plan.narrative,
            data_type=seed.data_type,
            initial_config=copy.deepcopy(initial_config),
            initial_tools=copy.deepcopy(initial_tools),
            involved_classes=involved_classes,
            turns=turns,
            synthesis_environment_id=environment_id,
            structural_profile=draft_structural_profile(turns),
        )
