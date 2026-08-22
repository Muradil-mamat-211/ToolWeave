"""Feedback-driven RODS Data-Generation Branch V1 state machine."""

from __future__ import annotations

import inspect
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .adversarial.missing_function import MissingFunctionTransformer
from .adversarial.missing_parameter import MissingParameterTransformer
from .candidate_builder import CandidateBuilder
from .coherence_rewrite import CoherenceRewriteAgent
from .config import GeneratorConfig
from .contracts import validate_seed_record
from .config_patch import ConfigPatchAgent
from .environment_adapter import EnvironmentFactory, SynthesisEnvironmentAdapter
from .error_taxonomy import ErrorType, PATCHABLE_ERRORS
from .execution_orchestrator import ExecutionOrchestrator, StageFailure
from .feedback import FeedbackState
from .function_catalog import CatalogError, FunctionCatalog
from .llm_backend import LLMBackend, pop_request_metadata, push_request_metadata
from .metrics import GeneratorMetrics
from .models import ErrorRecord, GateResult, PipelineResult, SeedRecord, stable_id, utc_now
from .parameter_generator import ParameterGenerator
from .parsing import StructuredParseError
from .planner import NoValidFunctionsError, PlannerAgent
from .quality_judge import QualityJudgeAgent
from .query_generator import QueryGenerator
from .query_verifier import QueryVerifier
from .queue import LockedJsonlQueue
from .refine import RefineAgent
from .structural_profile import (
    seed_structural_profile,
    structural_alignment_diagnostics,
)
from .validation.parameter_complexity import parameter_complexity_gate
from .validation.missing_parameter_validity import missing_parameter_validity_gate
from .validation.observation_entailment import observation_entailment_gate
from .validation.action_minimality import action_minimality_gate
from .validation.semantic_grounding import semantic_grounding_gate
from .validation.relational_resolution import relational_resolution_gate
from .validation.tool_availability import tool_availability_gate
from .validation.unit_semantics import unit_semantic_gate
from .validation.vm_reverify import fresh_vm_reverify_gate


CheckpointCallback = Callable[[dict[str, Any]], Any | Awaitable[Any]]


def _restore_error(raw: Mapping[str, Any]) -> ErrorRecord:
    return ErrorRecord(
        error_type=ErrorType(str(raw["error_type"])),
        seed_id=str(raw["seed_id"]),
        attempt_id=int(raw["attempt_id"]),
        turn_id=(None if raw.get("turn_id") is None else int(raw["turn_id"])),
        function_names=tuple(str(value) for value in raw.get("function_names", [])),
        detail=str(raw.get("detail", "")),
        patchable=bool(raw.get("patchable", False)),
        timestamp=str(raw.get("timestamp", "")),
        context=dict(raw.get("context", {})),
    )


class RODSDataGenerationPipeline:
    """Consume an already-selected seed and emit only a fully validated row.

    Boundary selection is intentionally absent.  This class trusts the frozen
    Training Branch to decide which records enter the seed queue.
    """

    def __init__(
        self,
        *,
        config: GeneratorConfig,
        backend: LLMBackend,
        catalog: FunctionCatalog | None = None,
        environment_factory: EnvironmentFactory | None = None,
        candidate_builder: CandidateBuilder | None = None,
        metrics: GeneratorMetrics | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.metrics = metrics or GeneratorMetrics()
        if catalog is not None:
            self.catalog = catalog
        elif config.function_schema_parquet:
            self.catalog = FunctionCatalog.from_training_parquet(
                config.function_schema_parquet
            )
        else:
            # Explicit compatibility/test path. Production configs should use
            # the active Training parquet so Generator and bfcl_env share one
            # executable schema contract.
            self.catalog = FunctionCatalog.from_bfcl_directory(
                config.function_catalog_dir
            )
        self.environment_factory = environment_factory or SynthesisEnvironmentAdapter(
            is_augmented=config.use_augmented_environment
        )
        self.candidate_builder = candidate_builder or CandidateBuilder()

        self.planner = PlannerAgent(
            backend,
            self.catalog,
            self.metrics,
            max_parse_retries=config.planner_retries,
        )
        self.parameter_generator = ParameterGenerator(
            backend,
            self.catalog,
            self.metrics,
            max_parse_attempts=config.agent_parse_retries,
        )
        self.query_generator = QueryGenerator(
            backend,
            self.catalog,
            self.metrics,
            max_parse_attempts=config.agent_parse_retries,
        )
        self.query_verifier = QueryVerifier(backend, self.metrics)
        suspicious_queue = (
            None
            if config.dry_run
            else LockedJsonlQueue(
                Path(config.queues.expanded_log_dir)
                / "unclassified_suspicious_results.jsonl",
                key_field="observation_id",
            )
        )

        def record_suspicious_result(observation: Mapping[str, Any]) -> None:
            if suspicious_queue is None:
                return
            record = {
                "observation_id": stable_id(
                    "suspicious_result", observation
                ),
                "timestamp": utc_now(),
                **dict(observation),
            }
            suspicious_queue.append([record])

        self.execution = ExecutionOrchestrator(
            catalog=self.catalog,
            environment_factory=self.environment_factory,
            parameter_generator=self.parameter_generator,
            query_generator=self.query_generator,
            query_verifier=self.query_verifier,
            metrics=self.metrics,
            suspicious_result_sink=record_suspicious_result,
        )
        self.rewrite = CoherenceRewriteAgent(backend, self.metrics)
        self.missing_function = MissingFunctionTransformer(
            backend, self.catalog, self.metrics
        )
        self.missing_parameter = MissingParameterTransformer(
            backend, self.catalog, self.metrics
        )
        self.patch_agent = ConfigPatchAgent(backend, self.metrics)
        self.judge = QualityJudgeAgent(backend, self.metrics)
        self.refine = RefineAgent(backend, self.metrics)

    @staticmethod
    def _error(
        error_type: ErrorType,
        *,
        seed: SeedRecord,
        attempt_id: int,
        detail: str,
        function_names: Sequence[str] = (),
        turn_id: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ErrorRecord:
        return ErrorRecord(
            error_type=error_type,
            seed_id=seed.sample_id,
            attempt_id=attempt_id,
            turn_id=turn_id,
            function_names=tuple(function_names),
            detail=detail,
            patchable=error_type in PATCHABLE_ERRORS,
            context=dict(context or {}),
        )

    async def _checkpoint(
        self,
        callback: CheckpointCallback | None,
        feedback: FeedbackState,
        *,
        completed_failed_attempts: int,
        planner_calls: int,
    ) -> None:
        if callback is None:
            return
        result = callback(
            feedback.checkpoint(
                completed_failed_attempts=completed_failed_attempts,
                planner_calls=planner_calls,
            )
        )
        if inspect.isawaitable(result):
            await result

    def _restore_feedback(
        self, seed: SeedRecord, resume_state: Mapping[str, Any] | None
    ) -> tuple[FeedbackState, int]:
        if not resume_state:
            return FeedbackState.from_initial_config(seed.initial_config), 0
        failures = [
            _restore_error(raw)
            for raw in resume_state.get("failures", [])
            if isinstance(raw, Mapping)
        ]
        explicit_count = resume_state.get("completed_failed_attempts")
        if explicit_count is None:
            # Backward-compatible recovery for tracker.v1.  Deriving from
            # durable failure records fixes the historical state where a
            # successful third attempt was incorrectly persisted as
            # ``attempts=3`` before its candidate reached the queue.
            failure_attempt_ids = [
                error.attempt_id for error in failures if error.attempt_id > 0
            ]
            completed_failed_attempts = (
                max(failure_attempt_ids)
                if failure_attempt_ids
                else int(resume_state.get("attempts", 0))
            )
        else:
            completed_failed_attempts = int(explicit_count)
        if (
            completed_failed_attempts < 0
            or completed_failed_attempts > self.config.max_pipeline_attempts
        ):
            raise ValueError("resume-state attempt count is invalid")
        return (
            FeedbackState.from_resume(
                seed.initial_config,
                failures=failures,
                blocked_functions={str(value) for value in resume_state.get("blocklist", [])},
                patch_history=[
                    dict(value)
                    for value in resume_state.get("patches", [])
                    if isinstance(value, Mapping)
                ],
                blocklist_history=[
                    [str(item) for item in values]
                    for values in resume_state.get("blocklist_history", [])
                    if isinstance(values, list)
                ],
                current_config=(
                    dict(resume_state["current_config"])
                    if isinstance(resume_state.get("current_config"), Mapping)
                    else None
                ),
            ),
            completed_failed_attempts,
        )

    def _result(
        self,
        *,
        seed: SeedRecord,
        status: str,
        candidate: dict[str, Any] | None,
        feedback: FeedbackState,
        attempts: int,
        reason: str,
        started: float,
        planner_calls: int,
    ) -> PipelineResult:
        self.metrics.increment("latency/per_seed_total_seconds_sum", time.perf_counter() - started)
        self.metrics.increment("latency/per_seed_total_count")
        self.metrics.ensure_error_keys()
        completed_failed_attempts = max(
            (error.attempt_id for error in feedback.failures if error.attempt_id > 0),
            default=0,
        )
        return PipelineResult(
            seed_id=seed.sample_id,
            status=status,
            candidate=candidate,
            errors=list(feedback.failures),
            attempts=attempts,
            planner_calls=planner_calls,
            blocklist_history=list(feedback.blocklist_history),
            config_patch_history=list(feedback.patch_history),
            metrics=self.metrics.snapshot(),
            reason=reason,
            checkpoint=feedback.checkpoint(
                completed_failed_attempts=completed_failed_attempts,
                planner_calls=planner_calls,
            ),
        )

    async def _register_attempt_failure(
        self,
        error: ErrorRecord,
        *,
        feedback: FeedbackState,
        completed_failed_attempts: int,
        planner_calls: int,
        checkpoint_callback: CheckpointCallback | None,
    ) -> None:
        self.metrics.record_error(error.error_type)
        await feedback.register_failure(error, patch_agent=self.patch_agent)
        await self._checkpoint(
            checkpoint_callback,
            feedback,
            completed_failed_attempts=completed_failed_attempts,
            planner_calls=planner_calls,
        )

    async def generate(
        self,
        raw_seed: SeedRecord | Mapping[str, Any],
        *,
        resume_state: Mapping[str, Any] | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> PipelineResult:
        started = time.perf_counter()
        seed = raw_seed if isinstance(raw_seed, SeedRecord) else validate_seed_record(raw_seed)
        feedback, completed_failed_attempts = self._restore_feedback(seed, resume_state)
        planner_calls = int((resume_state or {}).get("planner_calls", 0))

        def observe_planner_call() -> None:
            nonlocal planner_calls
            planner_calls += 1
        try:
            self.catalog.with_seed_functions(seed)
        except CatalogError as exc:
            error = self._error(
                ErrorType.FUNC_SAMPLE_FAILED,
                seed=seed,
                attempt_id=completed_failed_attempts,
                detail=f"seed/catalog contract rejected: {exc}",
            )
            feedback.failures.append(error)
            self.metrics.record_error(error.error_type)
            self.metrics.increment("queue/seeds_dropped")
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=completed_failed_attempts,
                reason=error.detail,
                started=started,
                planner_calls=planner_calls,
            )
        draft = None
        attempts = completed_failed_attempts

        for attempt_id in range(
            completed_failed_attempts + 1, self.config.max_pipeline_attempts + 1
        ):
            attempts = attempt_id
            self.metrics.increment("planner/attempt_count")
            plan = None
            attempt_context = push_request_metadata(attempt_id=attempt_id)
            try:
                try:
                    plan = await self.planner.plan(
                        seed,
                        failure_history=feedback.failures,
                        blocked_functions=feedback.blocked_functions,
                        current_config=feedback.current_config,
                        call_observer=observe_planner_call,
                    )
                except FileNotFoundError as exc:
                    raise StageFailure(
                        self._error(
                            ErrorType.NO_PROMPTS,
                            seed=seed,
                            attempt_id=attempt_id,
                            detail=str(exc),
                        )
                    ) from exc
                except (NoValidFunctionsError, CatalogError) as exc:
                    raise StageFailure(
                        self._error(
                            ErrorType.FUNC_SAMPLE_FAILED,
                            seed=seed,
                            attempt_id=attempt_id,
                            detail=str(exc),
                        )
                    ) from exc
                except StructuredParseError as exc:
                    raise StageFailure(
                        self._error(
                            ErrorType.NO_PATTERN,
                            seed=seed,
                            attempt_id=attempt_id,
                            detail=str(exc),
                        )
                    ) from exc

                draft = await self.execution.execute(
                    seed=seed,
                    plan=plan,
                    initial_config=feedback.current_config,
                    attempt_id=attempt_id,
                )
                seed_profile = seed_structural_profile(seed, self.catalog)
                draft.structural_profile["seed_profile"] = seed_profile
                draft.structural_profile["alignment_diagnostics"] = (
                    structural_alignment_diagnostics(seed_profile, draft.structural_profile)
                )
                draft.structural_profile["alignment_mechanism"] = (
                    "Appendix C.1 Planner prompt; deterministic profile is diagnostics only"
                )
                draft = await self.rewrite.rewrite(draft)
                if seed.data_type == "multi_turn_miss_func":
                    draft = await self.missing_function.transform(draft)
                elif seed.data_type == "multi_turn_miss_param":
                    draft = await self.missing_parameter.transform(draft)
                break
            except StageFailure as exc:
                error = exc.error
            except FileNotFoundError as exc:
                error = self._error(
                    ErrorType.NO_PROMPTS,
                    seed=seed,
                    attempt_id=attempt_id,
                    detail=str(exc),
                )
            except StructuredParseError as exc:
                function_names = (
                    [name for turn in plan.turns for name in turn.function_names]
                    if plan is not None
                    else []
                )
                error = self._error(
                    ErrorType.CONVERSATION_CONSTRUCT_FAILED,
                    seed=seed,
                    attempt_id=attempt_id,
                    function_names=function_names,
                    detail=str(exc),
                )
            except Exception as exc:
                function_names = (
                    [name for turn in plan.turns for name in turn.function_names]
                    if plan is not None
                    else []
                )
                error = self._error(
                    ErrorType.PIPELINE_EXCEPTION,
                    seed=seed,
                    attempt_id=attempt_id,
                    function_names=function_names,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            finally:
                pop_request_metadata(attempt_context)
            await self._register_attempt_failure(
                error,
                feedback=feedback,
                completed_failed_attempts=attempt_id,
                planner_calls=planner_calls,
                checkpoint_callback=checkpoint_callback,
            )
        else:
            draft = None

        if draft is None:
            self.metrics.increment("queue/seeds_dropped")
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=attempts,
                reason="three feedback-conditioned pipeline attempts failed",
                started=started,
                planner_calls=planner_calls,
            )

        gates: list[GateResult] = []
        unit_semantics = unit_semantic_gate(draft, catalog=self.catalog)
        gates.append(unit_semantics)
        self.metrics.increment(
            f"validation/{unit_semantics.name}",
            1.0 if unit_semantics.passed else 0.0,
        )
        if not unit_semantics.passed:
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=attempts,
                reason=(
                    f"deterministic gate rejected: {unit_semantics.name}: "
                    f"{unit_semantics.detail}"
                ),
                started=started,
                planner_calls=planner_calls,
            )

        grounding = semantic_grounding_gate(draft, catalog=self.catalog)
        gates.append(grounding)
        self.metrics.increment(
            f"validation/{grounding.name}", 1.0 if grounding.passed else 0.0
        )
        semantic_counts = grounding.metadata.get("semantic_outcome_counts", {})
        if isinstance(semantic_counts, Mapping):
            self.metrics.increment(
                "execution/final_domain_negative",
                float(semantic_counts.get("DOMAIN_NEGATIVE", 0)),
            )
            self.metrics.increment(
                "execution/final_hard_error",
                float(semantic_counts.get("HARD_ERROR", 0)),
            )
        if not grounding.passed:
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=attempts,
                reason=(
                    f"deterministic gate rejected: {grounding.name}: "
                    f"{grounding.detail}"
                ),
                started=started,
                planner_calls=planner_calls,
            )

        missing_parameter_validity = missing_parameter_validity_gate(
            draft, catalog=self.catalog
        )
        gates.append(missing_parameter_validity)
        self.metrics.increment(
            f"validation/{missing_parameter_validity.name}",
            1.0 if missing_parameter_validity.passed else 0.0,
        )
        if not missing_parameter_validity.passed:
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=attempts,
                reason=(
                    "deterministic gate rejected: "
                    f"{missing_parameter_validity.name}: "
                    f"{missing_parameter_validity.detail}"
                ),
                started=started,
                planner_calls=planner_calls,
            )

        observation_entailment = observation_entailment_gate(draft)
        gates.append(observation_entailment)
        self.metrics.increment(
            f"validation/{observation_entailment.name}",
            1.0 if observation_entailment.passed else 0.0,
        )
        if not observation_entailment.passed:
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=attempts,
                reason=(
                    "deterministic gate rejected: "
                    f"{observation_entailment.name}: "
                    f"{observation_entailment.detail}"
                ),
                started=started,
                planner_calls=planner_calls,
            )

        relational = relational_resolution_gate(draft)
        gates.append(relational)
        self.metrics.increment(
            f"validation/{relational.name}", 1.0 if relational.passed else 0.0
        )
        if not relational.passed:
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=attempts,
                reason=(
                    f"deterministic gate rejected: {relational.name}: "
                    f"{relational.detail}"
                ),
                started=started,
                planner_calls=planner_calls,
            )

        try:
            final_semantic = await self.query_verifier.verify_final_conversation(draft)
        except Exception as exc:
            error = self._error(
                ErrorType.PIPELINE_EXCEPTION,
                seed=seed,
                attempt_id=attempts,
                detail=f"final semantic verifier transport failure: {type(exc).__name__}: {exc}",
            )
            feedback.failures.append(error)
            self.metrics.record_error(error.error_type)
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=attempts,
                reason=error.detail,
                started=started,
                planner_calls=planner_calls,
            )
        gates.append(final_semantic)
        self.metrics.increment(
            f"validation/{final_semantic.name}",
            1.0 if final_semantic.passed else 0.0,
        )
        if not final_semantic.passed:
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=attempts,
                reason=(
                    f"final semantic verifier rejected: {final_semantic.detail}"
                ),
                started=started,
                planner_calls=planner_calls,
            )

        # This deterministic gate runs after the semantic verifier so an
        # already-invalid rewritten Query/GT pair retains its precise original
        # failure category.  It still runs before Fresh VM, Judge, and candidate
        # construction, so neither Judge nor refinement can override it.
        action_minimality = action_minimality_gate(draft, catalog=self.catalog)
        gates.append(action_minimality)
        self.metrics.increment(
            f"validation/{action_minimality.name}",
            1.0 if action_minimality.passed else 0.0,
        )
        if not action_minimality.passed:
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=attempts,
                reason=(
                    "deterministic gate rejected: "
                    f"{action_minimality.name}: {action_minimality.detail}"
                ),
                started=started,
                planner_calls=planner_calls,
            )

        gate_functions = (
            lambda: fresh_vm_reverify_gate(
                draft, environment_factory=self.environment_factory, seed_id=seed.sample_id
            ),
            lambda: tool_availability_gate(draft),
            lambda: parameter_complexity_gate(draft),
        )
        for gate_function in gate_functions:
            gate = gate_function()
            gates.append(gate)
            self.metrics.increment(f"validation/{gate.name}", 1.0 if gate.passed else 0.0)
            if not gate.passed:
                self.metrics.increment("lifecycle/candidates_dropped")
                return self._result(
                    seed=seed,
                    status="DROPPED",
                    candidate=None,
                    feedback=feedback,
                    attempts=attempts,
                    reason=f"deterministic gate rejected: {gate.name}: {gate.detail}",
                    started=started,
                    planner_calls=planner_calls,
                )

        refinement_used = False
        refinement_metadata: dict[str, Any] = {}
        try:
            judge = await self.judge.evaluate(draft, pass_index=1)
            if not judge.accepted:
                classify_reason, classification = await self.refine.classify(draft, judge)
                refinement_metadata = {
                    "classification_reason": classify_reason,
                    "classification": classification,
                    "first_judge": asdict(judge),
                }
                if classification == "gt_unfixable":
                    self.metrics.increment("lifecycle/candidates_dropped")
                    return self._result(
                        seed=seed,
                        status="DROPPED",
                        candidate=None,
                        feedback=feedback,
                        attempts=attempts,
                        reason="Quality Judge rejection classified as gt_unfixable",
                        started=started,
                        planner_calls=planner_calls,
                    )
                draft, turn_index, old_query = await self.refine.rewrite_query(draft, judge)
                refinement_used = True
                refinement_metadata.update(
                    {"turn_index": turn_index, "old_query": old_query, "new_query": draft.turns[turn_index].query}
                )
                # Refine Rewrite changes final actor-visible text.  It must not
                # bypass the same semantic gates applied after whole-conversation
                # rewrite and adversarial transformation.
                refined_unit = unit_semantic_gate(draft, catalog=self.catalog)
                if not refined_unit.passed:
                    self.metrics.increment("lifecycle/candidates_dropped")
                    return self._result(
                        seed=seed,
                        status="DROPPED",
                        candidate=None,
                        feedback=feedback,
                        attempts=attempts,
                        reason=(
                            "refined query failed unit semantics: "
                            f"{refined_unit.detail}"
                        ),
                        started=started,
                        planner_calls=planner_calls,
                    )
                refined_grounding = semantic_grounding_gate(draft, catalog=self.catalog)
                if not refined_grounding.passed:
                    self.metrics.increment("lifecycle/candidates_dropped")
                    return self._result(
                        seed=seed,
                        status="DROPPED",
                        candidate=None,
                        feedback=feedback,
                        attempts=attempts,
                        reason=(
                            "refined query failed semantic grounding: "
                            f"{refined_grounding.detail}"
                        ),
                        started=started,
                        planner_calls=planner_calls,
                    )
                refined_missing_parameter = missing_parameter_validity_gate(
                    draft, catalog=self.catalog
                )
                if not refined_missing_parameter.passed:
                    self.metrics.increment("lifecycle/candidates_dropped")
                    return self._result(
                        seed=seed,
                        status="DROPPED",
                        candidate=None,
                        feedback=feedback,
                        attempts=attempts,
                        reason=(
                            "refined query failed Missing Parameter validity: "
                            f"{refined_missing_parameter.detail}"
                        ),
                        started=started,
                        planner_calls=planner_calls,
                    )
                refined_observation_entailment = observation_entailment_gate(draft)
                if not refined_observation_entailment.passed:
                    self.metrics.increment("lifecycle/candidates_dropped")
                    return self._result(
                        seed=seed,
                        status="DROPPED",
                        candidate=None,
                        feedback=feedback,
                        attempts=attempts,
                        reason=(
                            "refined query failed observation entailment: "
                            f"{refined_observation_entailment.detail}"
                        ),
                        started=started,
                        planner_calls=planner_calls,
                    )
                refined_relational = relational_resolution_gate(draft)
                if not refined_relational.passed:
                    self.metrics.increment("lifecycle/candidates_dropped")
                    return self._result(
                        seed=seed,
                        status="DROPPED",
                        candidate=None,
                        feedback=feedback,
                        attempts=attempts,
                        reason=(
                            "refined query failed relational resolution: "
                            f"{refined_relational.detail}"
                        ),
                        started=started,
                        planner_calls=planner_calls,
                    )
                try:
                    refined_semantic = await self.query_verifier.verify_final_conversation(draft)
                except Exception as exc:
                    error = self._error(
                        ErrorType.PIPELINE_EXCEPTION,
                        seed=seed,
                        attempt_id=attempts,
                        detail=(
                            "refined final semantic verifier transport failure: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                    feedback.failures.append(error)
                    self.metrics.record_error(error.error_type)
                    self.metrics.increment("lifecycle/candidates_dropped")
                    return self._result(
                        seed=seed,
                        status="DROPPED",
                        candidate=None,
                        feedback=feedback,
                        attempts=attempts,
                        reason=error.detail,
                        started=started,
                        planner_calls=planner_calls,
                    )
                if not refined_semantic.passed:
                    self.metrics.increment("lifecycle/candidates_dropped")
                    return self._result(
                        seed=seed,
                        status="DROPPED",
                        candidate=None,
                        feedback=feedback,
                        attempts=attempts,
                        reason=(
                            "refined query failed final semantic verification: "
                            f"{refined_semantic.detail}"
                        ),
                        started=started,
                        planner_calls=planner_calls,
                    )
                refined_action_minimality = action_minimality_gate(
                    draft, catalog=self.catalog
                )
                if not refined_action_minimality.passed:
                    self.metrics.increment("lifecycle/candidates_dropped")
                    return self._result(
                        seed=seed,
                        status="DROPPED",
                        candidate=None,
                        feedback=feedback,
                        attempts=attempts,
                        reason=(
                            "refined query failed action minimality: "
                            f"{refined_action_minimality.detail}"
                        ),
                        started=started,
                        planner_calls=planner_calls,
                    )
                gates = [
                    refined_unit if gate.name == "unit_semantic_gate" else
                    refined_grounding if gate.name == "semantic_grounding_gate" else
                    refined_missing_parameter if gate.name == "missing_parameter_validity_gate" else
                    refined_observation_entailment if gate.name == "observation_entailment_gate" else
                    refined_action_minimality if gate.name == "action_minimality_gate" else
                    refined_relational if gate.name == "relational_resolution_gate" else
                    refined_semantic if gate.name == "final_query_semantic_gate" else gate
                    for gate in gates
                ]
                judge = await self.judge.evaluate(draft, pass_index=2)
                refinement_metadata["second_judge"] = asdict(judge)
                if not judge.accepted:
                    self.metrics.increment("lifecycle/candidates_dropped")
                    return self._result(
                        seed=seed,
                        status="DROPPED",
                        candidate=None,
                        feedback=feedback,
                        attempts=attempts,
                        reason="Quality Judge rejected after the single permitted refinement",
                        started=started,
                        planner_calls=planner_calls,
                    )

            candidate = self.candidate_builder.build(
                seed=seed,
                draft=draft,
                gates=gates,
                final_judge=judge,
                generator_backend=self.config.llm.backend,
                generator_model=self.config.llm.model,
                pipeline_attempts=attempts,
                planner_calls=planner_calls,
                failures=feedback.failures,
                blocklist_history=feedback.blocklist_history,
                config_patch_history=feedback.patch_history,
                refinement_used=refinement_used,
                refinement_metadata=refinement_metadata,
            )
        except Exception as exc:
            error = self._error(
                ErrorType.PIPELINE_EXCEPTION,
                seed=seed,
                attempt_id=attempts,
                detail=f"final validation/judge/candidate failure: {exc}",
            )
            feedback.failures.append(error)
            self.metrics.record_error(error.error_type)
            self.metrics.increment("lifecycle/candidates_dropped")
            return self._result(
                seed=seed,
                status="DROPPED",
                candidate=None,
                feedback=feedback,
                attempts=attempts,
                reason=error.detail,
                started=started,
                planner_calls=planner_calls,
            )

        self.metrics.increment("queue/seeds_succeeded")
        self.metrics.increment("lifecycle/candidates_validated")
        self.metrics.increment("lifecycle/generated_epoch", seed.source_epoch)
        return self._result(
            seed=seed,
            status="SUCCEEDED",
            candidate=candidate,
            feedback=feedback,
            attempts=attempts,
            reason="all deterministic gates and final Quality Judge passed",
            started=started,
            planner_calls=planner_calls,
        )
