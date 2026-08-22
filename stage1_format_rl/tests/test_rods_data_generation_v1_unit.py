from __future__ import annotations

import asyncio
import json
import multiprocessing
from pathlib import Path

import httpx
import pytest

from env_tuning.rods_data_generation_v1.config_patch import (
    ConfigPatchAgent,
    UnsafePatchError,
    apply_patch_operations,
    deep_merge,
)
from env_tuning.rods_data_generation_v1.config import LLMConfig
from env_tuning.rods_data_generation_v1.contracts import validate_seed_record
from env_tuning.rods_data_generation_v1.error_taxonomy import ErrorType, PATCHABLE_ERRORS
from env_tuning.rods_data_generation_v1.environment_adapter import VMCallResult
from env_tuning.rods_data_generation_v1.execution_orchestrator import (
    ExecutionOrchestrator,
    StageFailure,
)
from env_tuning.rods_data_generation_v1.function_catalog import CatalogError, FunctionCatalog
from env_tuning.rods_data_generation_v1.llm_backend import (
    BackendError,
    FakeLLMBackend,
    ReplayLLMBackend,
    VLLMOpenAIBackend,
    pop_request_metadata,
    push_request_metadata,
)
from env_tuning.rods_data_generation_v1.metrics import GeneratorMetrics
from env_tuning.rods_data_generation_v1.models import (
    ErrorRecord,
    ExecutionRecord,
    FunctionSpec,
    FunctionCall,
    PatchOperation,
    PlannerResult,
    PlanTurn,
    SeedRecord,
    SeedStatus,
)
from env_tuning.rods_data_generation_v1.query_generator import QueryGenerator
from env_tuning.rods_data_generation_v1.query_prompt_registry import (
    PUBLIC_BFCL_CLASSES,
    QueryPromptRegistry,
)
from env_tuning.rods_data_generation_v1.parsing import (
    StructuredParseError,
    parse_arguments_response,
    parse_config_patch_response,
    parse_judge_response,
    parse_planner_response,
    parse_refine_classification,
    parse_refine_rewrite,
    parse_rewrite_response,
    parse_verifier_response,
)
from env_tuning.rods_data_generation_v1.planner import PlannerAgent
from env_tuning.rods_data_generation_v1.prompts import PROMPT_ROOT, load_prompt
from env_tuning.rods_data_generation_v1.queue import (
    LockedJsonlQueue,
    ProductionQueueGuardError,
)
from env_tuning.rods_data_generation_v1.source_audit import SOURCE_PROVENANCE, SourceStatus
from env_tuning.rods_data_generation_v1.tracker import PromptTracker

from rods_data_generation_v1_fixtures import make_catalog, make_seed


def _queue_writer(path: str, worker: int, count: int) -> None:
    queue = LockedJsonlQueue(path, key_field="candidate_id")
    queue.append(
        {"candidate_id": f"worker-{worker}-{index}", "worker": worker, "index": index}
        for index in range(count)
    )


class _StubSession:
    environment_id = "stub-synthesis"

    def __init__(self, *, vm_success: bool = True):
        self.vm_success = vm_success
        self.calls = []

    def snapshot(self):
        return {"stub": {"call_count": len(self.calls)}}

    def execute(self, call):
        self.calls.append(call)
        return VMCallResult(
            result={"ok": self.vm_success},
            success=self.vm_success,
            error_detail=None if self.vm_success else "controlled VM failure",
            pre_state={"count": len(self.calls) - 1},
            post_state={"count": len(self.calls)},
        )

    def close(self):
        return None


class _StubFactory:
    created_environment_ids = []

    def __init__(self, *, vm_success: bool = True):
        self.vm_success = vm_success

    def create(self, **kwargs):
        return _StubSession(vm_success=self.vm_success)


class _StubParameterGenerator:
    def __init__(self, *, fail: bool = False):
        self.fail = fail

    async def generate(self, *, spec, **kwargs):
        if self.fail:
            raise StructuredParseError("controlled parameter parse failure")
        arguments = {"a": 1.0, "b": 2.0} if spec.name == "add" else {}
        return FunctionCall(spec.name, arguments, spec.class_name)


class _StubQueryGenerator:
    def __init__(self, *, fail: bool = False, no_prompt: bool = False):
        self.fail = fail
        self.no_prompt = no_prompt

    async def generate(self, **kwargs):
        if self.no_prompt:
            raise FileNotFoundError("controlled missing per-class query prompt")
        if self.fail:
            raise StructuredParseError("controlled query parse failure")
        return "reason", "Please calculate the result."


class _StubVerifier:
    def __init__(self, *, malformed: bool = False, accepted: bool = True):
        self.malformed = malformed
        self.accepted = accepted

    async def verify(self, **kwargs):
        if self.malformed:
            raise StructuredParseError("controlled missing verdict tag")
        return "controlled verifier", self.accepted


def _orchestrator(
    *,
    catalog=None,
    vm_success=True,
    parameter_fail=False,
    query_fail=False,
    query_no_prompt=False,
    verifier_malformed=False,
    verifier_accepted=True,
):
    metrics = GeneratorMetrics()
    catalog = catalog or make_catalog()
    return ExecutionOrchestrator(
        catalog=catalog,
        environment_factory=_StubFactory(vm_success=vm_success),
        parameter_generator=_StubParameterGenerator(fail=parameter_fail),
        query_generator=_StubQueryGenerator(
            fail=query_fail, no_prompt=query_no_prompt
        ),
        query_verifier=_StubVerifier(
            malformed=verifier_malformed, accepted=verifier_accepted
        ),
        metrics=metrics,
    )


def test_seed_contract_is_strict_and_preserves_source_epoch():
    seed = validate_seed_record(make_seed(epoch=9))
    assert seed.source_epoch == 9
    assert seed.source_global_step == 70
    invalid = make_seed()
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="seed fields differ"):
        SeedRecord.from_mapping(invalid)


def test_source_provenance_never_labels_reconstruction_as_official_code():
    assert SOURCE_PROVENANCE["bfcl_vm_execution"] == SourceStatus.OFFICIAL_REUSED_CODE
    for component in (
        "parameter_generation",
        "query_generation",
        "query_verification",
        "missing_function_transform",
        "missing_parameter_transform",
    ):
        assert SOURCE_PROVENANCE[component] == SourceStatus.RECONSTRUCTED


def test_all_generator_prompts_use_reason_not_think_and_have_source_headers():
    paths = sorted(PROMPT_ROOT.glob("**/*.txt"))
    assert len(paths) >= 11
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        assert "SOURCE_STATUS" in raw
        assert "RODS arXiv:2606.19047v1" in raw
        body = load_prompt(str(path.relative_to(PROMPT_ROOT)), {}) if "{" not in raw.split("---PROMPT---", 1)[1] else None
        if body is not None:
            assert "<think>" not in body
        assert "<think>" not in raw.split("---PROMPT---", 1)[1]


def test_strict_parsers_accept_contracts_and_reject_malformed_tags():
    plan = parse_planner_response(
        "<reason>x</reason><narrative>n</narrative>"
        "<turn>MathAPI: add</turn><turn>MathAPI: multiply</turn>",
        allowed_functions={"add", "multiply"},
        class_for_function={"add": "MathAPI", "multiply": "MathAPI"},
    )
    assert [turn.function_names for turn in plan.turns] == [("add",), ("multiply",)]
    with pytest.raises(StructuredParseError, match="blocked"):
        parse_planner_response(
            "<reason>x</reason><narrative>n</narrative>"
            "<turn>MathAPI: add</turn><turn>MathAPI: multiply</turn>",
            allowed_functions={"add", "multiply"},
            class_for_function={"add": "MathAPI", "multiply": "MathAPI"},
            blocked_functions={"add"},
        )
    _, args = parse_arguments_response(
        '<reason>x</reason><arguments>{"a": 1, "b": 2}</arguments>'
    )
    assert args == {"a": 1, "b": 2}
    assert parse_verifier_response(
        "<reason>x</reason><verdict>accept</verdict>"
    )[1]
    with pytest.raises(StructuredParseError):
        parse_verifier_response("OK")
    assert parse_rewrite_response("<query>a</query><query>b</query>", expected_count=2) == ["a", "b"]
    with pytest.raises(StructuredParseError):
        parse_rewrite_response("<query>a</query>", expected_count=2)
    assert parse_judge_response(
        "<reason>x</reason><decision>accept</decision><fail_reason></fail_reason>"
    ).accepted
    assert parse_judge_response(
        "<reason>x</reason><decision>accept</decision>"
    ).accepted
    with pytest.raises(StructuredParseError, match="requires fail_reason"):
        parse_judge_response("<reason>x</reason><decision>reject</decision>")
    with pytest.raises(StructuredParseError, match="cannot contain fail_reason"):
        parse_judge_response(
            "<reason>x</reason><decision>accept</decision>"
            "<fail_reason>contradiction</fail_reason>"
        )
    assert parse_refine_classification(
        "<reason>x</reason><answer>query_fixable</answer>"
    )[1] == "query_fixable"
    assert parse_refine_rewrite("<answer>fixed</answer>") == "fixed"


def test_patch_parser_and_deep_merge_are_deterministic_and_type_safe():
    _, operations = parse_config_patch_response(
        "<reason>open market</reason>"
        "<patch><class>TradingBot</class><field>market.status</field><value>\"Open\"</value></patch>"
        "<patch><class>TradingBot</class><field>authenticated</field><value>true</value></patch>"
    )
    merged, patch = apply_patch_operations(
        {"TradingBot": {"market": {"status": "Closed", "timezone": "UTC"}}},
        operations,
    )
    assert merged == {
        "TradingBot": {
            "market": {"status": "Open", "timezone": "UTC"},
            "authenticated": True,
        }
    }
    assert patch["TradingBot"]["market"]["status"] == "Open"
    assert deep_merge({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"x": 1, "y": 2}}
    assert deep_merge({"a": {"x": 1}}, {"a": {"x": 2}}) == {"a": {"x": 2}}
    accumulated = deep_merge(
        deep_merge({"a": {"x": 1}}, {"a": {"y": 2}}), {"a": {"x": 3}}
    )
    assert accumulated == {"a": {"x": 3, "y": 2}}
    with pytest.raises(UnsafePatchError, match="scalar cannot replace dict"):
        deep_merge({"a": {"x": 1}}, {"a": 2})
    with pytest.raises(UnsafePatchError, match="type-incompatible"):
        deep_merge({"a": 1}, {"a": "one"})


def test_filesystem_patch_rejects_dot_split_corruption_and_accepts_escaped_key():
    initial = {
        "GorillaFileSystem": {
            "root": {
                "home": {"type": "directory", "contents": {}}
            }
        }
    }
    file_value = {"type": "file", "content": "audit"}
    with pytest.raises(UnsafePatchError, match="invalid/missing type"):
        apply_patch_operations(
            initial,
            [
                PatchOperation(
                    "GorillaFileSystem",
                    "root.home.contents.report.txt",
                    file_value,
                )
            ],
        )
    merged, _ = apply_patch_operations(
        initial,
        [
            PatchOperation(
                "GorillaFileSystem",
                r"root.home.contents.report\.txt",
                file_value,
            )
        ],
    )
    assert merged["GorillaFileSystem"]["root"]["home"]["contents"] == {
        "report.txt": file_value
    }


def test_only_four_environment_errors_can_trigger_config_patch():
    assert PATCHABLE_ERRORS == {
        ErrorType.PARAM_GEN_FAILED,
        ErrorType.DECOMPOSE_FAILED,
        ErrorType.FUNC_SAMPLE_FAILED,
        ErrorType.VM_EXEC_FAILED,
    }
    backend = FakeLLMBackend({})
    agent = ConfigPatchAgent(backend, GeneratorMetrics())
    error = ErrorRecord(
        error_type=ErrorType.QUERY_VERIFY_FAILED,
        seed_id="s",
        attempt_id=1,
        turn_id=0,
        function_names=("add",),
        detail="mismatch",
        patchable=False,
    )
    with pytest.raises(ValueError, match="non-patchable"):
        asyncio.run(agent.propose(error, current_config={}))
    assert backend.calls == []


def test_error_taxonomy_has_exactly_the_twelve_appendix_e_types():
    assert {item.value for item in ErrorType} == {
        "param_gen_failed",
        "decompose_failed",
        "func_sample_failed",
        "vm_exec_failed",
        "duplicate_func",
        "query_gen_failed",
        "query_verify_failed",
        "query_verify_no_tag",
        "conversation_construct_failed",
        "no_prompts",
        "no_pattern",
        "pipeline_exception",
    }


def test_execution_orchestrator_maps_all_eight_stage_errors_to_appendix_e_types():
    seed = SeedRecord.from_mapping(make_seed())
    one_add = PlannerResult("reason", "narrative", (PlanTurn(0, "MathAPI", ("add",)),))

    with pytest.raises(StageFailure) as caught:
        _orchestrator()._expand_functions(
            ["not_a_function"], seed=seed, attempt_id=1, turn_id=0
        )
    assert caught.value.error.error_type == ErrorType.FUNC_SAMPLE_FAILED

    high_catalog = FunctionCatalog(
        [FunctionSpec("high", "MathAPI", {"name": "high"}, level="HIGH_LEVEL")]
    )
    with pytest.raises(StageFailure) as caught:
        _orchestrator(catalog=high_catalog)._expand_functions(
            ["high"], seed=seed, attempt_id=1, turn_id=0
        )
    assert caught.value.error.error_type == ErrorType.DECOMPOSE_FAILED

    scenarios = [
        ({"parameter_fail": True}, one_add, ErrorType.PARAM_GEN_FAILED),
        ({"vm_success": False}, one_add, ErrorType.VM_EXEC_FAILED),
        (
            {},
            PlannerResult("reason", "narrative", (PlanTurn(0, "MathAPI", ("add", "add")),)),
            ErrorType.DUPLICATE_FUNC,
        ),
        ({"query_fail": True}, one_add, ErrorType.QUERY_GEN_FAILED),
        ({"query_no_prompt": True}, one_add, ErrorType.NO_PROMPTS),
        ({"verifier_malformed": True}, one_add, ErrorType.QUERY_VERIFY_NO_TAG),
        ({"verifier_accepted": False}, one_add, ErrorType.QUERY_VERIFY_FAILED),
    ]
    for options, plan, expected in scenarios:
        with pytest.raises(StageFailure) as caught:
            asyncio.run(
                _orchestrator(**options).execute(
                    seed=seed, plan=plan, initial_config={}, attempt_id=1
                )
            )
        assert caught.value.error.error_type == expected


def test_vm_failure_records_schema_state_arguments_and_config_for_forensics():
    seed = SeedRecord.from_mapping(make_seed())
    plan = PlannerResult(
        "reason", "narrative", (PlanTurn(0, "MathAPI", ("add",)),)
    )
    with pytest.raises(StageFailure) as caught:
        asyncio.run(
            _orchestrator(vm_success=False).execute(
                seed=seed,
                plan=plan,
                initial_config={"MathAPI": {"fixture": True}},
                attempt_id=2,
            )
        )
    context = caught.value.error.context
    assert context["arguments"] == {"a": 1.0, "b": 2.0}
    assert context["function_schema"]["name"] == "add"
    assert context["pre_state"] == {"count": 0}
    assert context["post_state"] == {"count": 1}
    assert context["attempt_initial_config"] == {"MathAPI": {"fixture": True}}
    assert context["previous_successful_calls"] == []


def test_unpublished_high_level_decomposition_fails_closed():
    catalog = FunctionCatalog(
        [FunctionSpec("high", "MathAPI", {"name": "high"}, level="HIGH_LEVEL")]
    )
    with pytest.raises(CatalogError, match="no published deterministic decomposition"):
        catalog.decompose(catalog.get("high"))


def test_fake_backend_is_role_scripted_and_replay_backend_reads_jsonl(tmp_path):
    fake = FakeLLMBackend({"planner": ["one", "two"]})
    first = asyncio.run(fake.complete(role="planner", messages=[], metadata={}))
    second = asyncio.run(fake.complete(role="planner", messages=[], metadata={}))
    assert [first.text, second.text] == ["one", "two"]
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        json.dumps({"role": "planner", "text": "recorded"}) + "\n",
        encoding="utf-8",
    )
    replay = ReplayLLMBackend.from_jsonl(replay_path)
    assert asyncio.run(replay.complete(role="planner", messages=[])).text == "recorded"


def test_async_request_context_adds_seed_and_attempt_to_every_agent_call():
    backend = FakeLLMBackend({"quality_judge": ["fixture"]})
    seed_context = push_request_metadata(seed_id="seed-17")
    attempt_context = push_request_metadata(attempt_id=3)
    try:
        asyncio.run(
            backend.complete(
                role="quality_judge",
                messages=[],
                metadata={"pass_index": 1},
            )
        )
    finally:
        pop_request_metadata(attempt_context)
        pop_request_metadata(seed_context)
    assert backend.calls[0]["metadata"] == {
        "seed_id": "seed-17",
        "attempt_id": 3,
        "pass_index": 1,
    }


def test_retry_planner_keeps_complete_compact_feedback_and_only_patch_delta():
    raw_seed = make_seed()
    unchanged_blob = "UNCHANGED_STATE_BLOB_" + ("x" * 5000)
    raw_seed["initial_config"] = {
        "MathAPI": {"unchanged": unchanged_blob, "nested": {"base": 1}}
    }
    seed = SeedRecord.from_mapping(raw_seed)
    current_config = {
        "MathAPI": {
            "unchanged": unchanged_blob,
            "nested": {"base": 1},
            "fixture_enabled": True,
        }
    }
    error = ErrorRecord(
        error_type=ErrorType.VM_EXEC_FAILED,
        seed_id=seed.sample_id,
        attempt_id=1,
        turn_id=0,
        function_names=("add",),
        detail="controlled VM failure",
        patchable=True,
        context={
            "call": "add(a=1.0, b=2.0)",
            "result": {"error": "controlled VM failure"},
            "pre_state": {"blob": "FORENSIC_PRE_STATE_" + ("y" * 5000)},
            "post_state": {"blob": "FORENSIC_POST_STATE_" + ("z" * 5000)},
        },
    )
    planner = PlannerAgent(
        FakeLLMBackend({}), make_catalog(), GeneratorMetrics()
    )
    prompt, _ = planner._render(
        seed,
        failure_history=[error],
        blocked_functions={"add"},
        current_config=current_config,
    )
    assert "controlled VM failure" in prompt
    assert "add(a=1.0, b=2.0)" in prompt
    assert '"add"' in prompt
    assert '"fixture_enabled": true' in prompt
    assert "UNCHANGED_STATE_BLOB_" not in prompt
    assert "FORENSIC_PRE_STATE_" not in prompt
    assert "FORENSIC_POST_STATE_" not in prompt


def test_vllm_http_error_durably_logs_response_body_and_provenance(tmp_path):
    async def exercise() -> None:
        raw_log = tmp_path / "raw.jsonl"
        backend = VLLMOpenAIBackend(
            LLMConfig(
                backend="vllm_openai",
                endpoint="http://fixture.invalid/v1",
                model="fixture-model",
                transport_retries=0,
                raw_response_log_path=str(raw_log),
            )
        )
        await backend._client.aclose()

        def reject(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": {"message": "maximum context length exceeded"}},
                request=request,
            )

        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
        context = push_request_metadata(seed_id="long-seed", attempt_id=3)
        try:
            with pytest.raises(BackendError, match="maximum context length exceeded"):
                await backend.complete(
                    role="planner",
                    messages=[{"role": "user", "content": "oversized"}],
                )
        finally:
            pop_request_metadata(context)
            await backend.aclose()
        records = [json.loads(line) for line in raw_log.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["status"] == "ERROR"
        assert "maximum context length exceeded" in records[0]["error"]
        assert records[0]["metadata"] == {"seed_id": "long-seed", "attempt_id": 3}

    asyncio.run(exercise())


def test_locked_jsonl_queue_is_multiprocess_safe_and_idempotent(tmp_path):
    path = str(tmp_path / "shared.jsonl")
    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=_queue_writer, args=(path, worker, 25)) for worker in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=20)
        assert worker.exitcode == 0
    queue = LockedJsonlQueue(path, key_field="candidate_id")
    assert len(queue.read()) == 100
    accepted, duplicate = queue.append([{"candidate_id": "worker-0-0"}])
    assert (accepted, duplicate) == (0, 1)


def test_test_mode_cannot_target_production_candidate_queue(tmp_path):
    production = tmp_path / "production.jsonl"
    with pytest.raises(ProductionQueueGuardError):
        LockedJsonlQueue(
            production,
            test_mode=True,
            production_path=production,
        )


def test_tracker_recovers_running_seed_and_preserves_checkpoint(tmp_path):
    state = tmp_path / "tracker.json"
    events = tmp_path / "events.jsonl"
    tracker = PromptTracker(state, events)
    assert tracker.register("seed") == SeedStatus.PENDING
    assert tracker.try_claim("seed")
    tracker.update_running(
        "seed",
        completed_failed_attempts=1,
        planner_calls=2,
        failures=[{"error_type": "vm_exec_failed"}],
        patches=[{"applied": True}],
        blocklist=["divide"],
        blocklist_history=[["divide"]],
        current_config={"MathAPI": {"fixture": True}},
    )
    # A second tracker in the same live process must not steal RUNNING work.
    concurrent = PromptTracker(state, events)
    assert concurrent.snapshot()["seeds"]["seed"]["status"] == SeedStatus.RUNNING.value

    # Simulate a process crash by replacing the owner with a nonexistent PID.
    raw_state = json.loads(state.read_text(encoding="utf-8"))
    raw_state["seeds"]["seed"]["worker_pid"] = 99999999
    raw_state["seeds"]["seed"]["worker_process_start"] = "gone"
    state.write_text(json.dumps(raw_state), encoding="utf-8")
    recovered = PromptTracker(state, events)
    item = recovered.snapshot()["seeds"]["seed"]
    assert item["status"] == SeedStatus.PENDING.value
    assert item["recovered_after_crash"] is True
    assert recovered.resume_state("seed")["completed_failed_attempts"] == 1
    assert recovered.resume_state("seed")["blocklist"] == ["divide"]
    assert recovered.try_claim("seed")


def _execution_record(name: str, class_name: str, arguments: dict) -> ExecutionRecord:
    call = FunctionCall(name, arguments, class_name)
    return ExecutionRecord(
        turn_id=0,
        call_id=0,
        call=call,
        canonical_call=call.canonical(),
        pre_state={},
        execution_result={"ok": True},
        post_state={},
        dependency_provenance={},
        success=True,
    )


def test_query_class_1_registry_routes_all_eight_public_bfcl_classes():
    registry = QueryPromptRegistry()
    expected_fragments = {
        "GorillaFileSystem": "file or directory work",
        "MathAPI": "everyday calculation request",
        "MessageAPI": "natural messaging activity",
        "TwitterAPI": "ordinary social-post activity",
        "TicketAPI": "support ticket",
        "TradingBot": "portfolio, market, or order-management",
        "TravelAPI": "itinerary, search, reservation",
        "VehicleControlAPI": "driver-facing vehicle language",
    }
    assert set(expected_fragments) == set(PUBLIC_BFCL_CLASSES)
    for class_name, marker in expected_fragments.items():
        prompt = registry.render(
            class_name,
            {
                "narrative": "n",
                "executed_calls": "[]",
                "prior_context": "[]",
            },
        )
        assert class_name in prompt
        assert marker in prompt


def test_query_class_2_and_3_backend_receives_only_selected_class_conditioning():
    backend = FakeLLMBackend(
        {
            "query_generator": [
                "<reason>The computation is represented naturally.</reason>"
                "<query>What is two plus three?</query>"
            ]
        }
    )
    generator = QueryGenerator(backend, make_catalog(), GeneratorMetrics())
    record = _execution_record("add", "MathAPI", {"a": 2.0, "b": 3.0})
    asyncio.run(
        generator.generate(
            class_name="MathAPI",
            narrative="A calculation",
            turn_records=[record],
            prior_queries=[],
        )
    )
    prompt = backend.calls[0]["messages"][0]["content"]
    assert "everyday calculation request" in prompt
    assert "portfolio, market, or order-management" not in prompt
    assert "for THIS TURN" in prompt
    assert "no action that is merely planned elsewhere" in prompt
    assert backend.calls[0]["metadata"]["class_name"] == "MathAPI"


def test_query_class_4_unknown_class_fails_closed_without_generic_fallback():
    with pytest.raises(FileNotFoundError, match="no reconstructed per-class"):
        QueryPromptRegistry().render(
            "UnknownBFCLClass",
            {"narrative": "n", "executed_calls": "[]", "prior_context": "[]"},
        )


def test_query_class_5_class_conditioning_preserves_function_leakage_rejection():
    backend = FakeLLMBackend(
        {
            "query_generator": [
                "<reason>bad</reason><query>Please call add for these values.</query>"
            ]
        }
    )
    generator = QueryGenerator(backend, make_catalog(), GeneratorMetrics())
    record = _execution_record("add", "MathAPI", {"a": 2.0, "b": 3.0})
    with pytest.raises(StructuredParseError, match="query generation failed"):
        asyncio.run(
            generator.generate(
                class_name="MathAPI",
                narrative="A calculation",
                turn_records=[record],
                prior_queries=[],
            )
        )


def test_natural_verb_function_names_are_not_false_positive_leakage():
    find_record = _execution_record("find", "GorillaFileSystem", {"path": "."})
    comment_record = _execution_record("comment", "TwitterAPI", {"text": "nice"})
    QueryGenerator.validate_no_leakage(
        "Find the annual report in my documents.", [find_record]
    )
    QueryGenerator.validate_no_leakage(
        "Please comment on the launch update.", [comment_record]
    )
    with pytest.raises(StructuredParseError, match="leaks function name find"):
        QueryGenerator.validate_no_leakage(
            "Run the find function for me.", [find_record]
        )
    snake_case = _execution_record("get_stock_info", "TradingBot", {"symbol": "NVDA"})
    with pytest.raises(StructuredParseError, match="get_stock_info"):
        QueryGenerator.validate_no_leakage(
            "Use get_stock_info for Nvidia.", [snake_case]
        )


def test_planner_allows_same_function_name_with_distinct_future_arguments():
    plan = parse_planner_response(
        "<reason>Resolve both cities.</reason><narrative>Compare two endpoints.</narrative>"
        "<turn>TravelAPI: get_nearest_airport_by_city, get_nearest_airport_by_city</turn>"
        "<turn>TravelAPI: list_all_airports</turn>",
        allowed_functions={"get_nearest_airport_by_city", "list_all_airports"},
        class_for_function={
            "get_nearest_airport_by_city": "TravelAPI",
            "list_all_airports": "TravelAPI",
        },
    )
    assert plan.turns[0].function_names == (
        "get_nearest_airport_by_city",
        "get_nearest_airport_by_city",
    )


def test_query_class_6_multi_call_turn_uses_one_query_covering_all_gt_calls():
    backend = FakeLLMBackend(
        {
            "query_generator": [
                "<reason>The request covers both calculations.</reason>"
                "<query>Please find the sum of two and three and the product of four and five.</query>"
            ]
        }
    )
    generator = QueryGenerator(backend, make_catalog(), GeneratorMetrics())
    records = [
        _execution_record("add", "MathAPI", {"a": 2.0, "b": 3.0}),
        _execution_record("multiply", "MathAPI", {"a": 4.0, "b": 5.0}),
    ]
    _, query = asyncio.run(
        generator.generate(
            class_name="MathAPI",
            narrative="Two calculations",
            turn_records=records,
            prior_queries=[],
        )
    )
    assert "sum" in query and "product" in query
    assert len(backend.calls) == 1
    prompt = backend.calls[0]["messages"][0]["content"]
    assert "add(a=2.0, b=3.0)" in prompt
    assert "multiply(a=4.0, b=5.0)" in prompt


def test_catalog_validates_real_math_schema_and_arguments():
    catalog = make_catalog()
    catalog.validate_arguments(catalog.get("add"), {"a": 1.0, "b": 2.0})
    with pytest.raises(CatalogError, match="schema-external"):
        catalog.validate_arguments(catalog.get("add"), {"a": 1.0, "b": 2.0, "c": 3.0})
    with pytest.raises(CatalogError, match="missing required"):
        catalog.validate_arguments(catalog.get("add"), {"a": 1.0})


def test_training_parquet_catalog_matches_active_envtuning_contract_and_fails_closed():
    active_catalog = FunctionCatalog.from_training_parquet(
        "/root/autodl-tmp/rods-workspace/stage1_format_rl/data/"
        "bfcl_stage3_train_all_400_shuffled_seed42.parquet"
    )

    # Regression for the online smoke failure: the separately downloaded
    # function docs require travel_cost, while EnvTuning's actual VM method and
    # active actor prompt do not accept it.
    book_flight = active_catalog.get("book_flight")
    assert book_flight.class_name == "TravelAPI"
    assert book_flight.schema["parameters"]["required"] == [
        "access_token",
        "card_id",
        "travel_date",
        "travel_from",
        "travel_to",
        "travel_class",
    ]
    assert "travel_cost" not in book_flight.schema["parameters"]["properties"]

    raw_seed = make_seed()
    raw_seed["available_functions"] = [book_flight.schema]
    seed = SeedRecord.from_mapping(raw_seed)
    assert active_catalog.with_seed_functions(seed) is active_catalog

    altered = make_seed()
    altered_schema = json.loads(json.dumps(book_flight.schema))
    altered_schema["parameters"]["properties"]["invented_parameter"] = {
        "type": "string"
    }
    altered["available_functions"] = [altered_schema]
    with pytest.raises(CatalogError, match="schema differs"):
        active_catalog.with_seed_functions(SeedRecord.from_mapping(altered))

    unknown = make_seed()
    unknown["available_functions"] = [
        {
            "name": "invented_function",
            "description": "not executable",
            "parameters": {"type": "dict", "properties": {}, "required": []},
        }
    ]
    with pytest.raises(CatalogError, match="unknown function"):
        active_catalog.with_seed_functions(SeedRecord.from_mapping(unknown))
