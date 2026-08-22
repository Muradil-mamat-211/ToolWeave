"""Deterministic regressions for Generator semantic quality hardening."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

from env_tuning.rods_data_generation_v1.daemon import GeneratorDaemon
from env_tuning.rods_data_generation_v1.function_catalog import FunctionCatalog
from env_tuning.rods_data_generation_v1.llm_backend import FakeLLMBackend
from env_tuning.rods_data_generation_v1.metrics import GeneratorMetrics
from env_tuning.rods_data_generation_v1.models import (
    ConversationDraft,
    ExecutionRecord,
    FunctionCall,
    PipelineResult,
    SynthesizedTurn,
)
from env_tuning.rods_data_generation_v1.pipeline import RODSDataGenerationPipeline
from env_tuning.rods_data_generation_v1.revalidation import (
    revalidate_candidate_grounding,
)
from env_tuning.rods_data_generation_v1.result_semantics import (
    ExecutionSemanticOutcome,
    classify_execution_result,
)
from env_tuning.rods_data_generation_v1.structural_profile import (
    draft_structural_profile,
    structural_alignment_diagnostics,
)
from env_tuning.rods_data_generation_v1.terminal_journal import TerminalResultJournal
from env_tuning.rods_data_generation_v1.validation.semantic_grounding import (
    semantic_grounding_gate,
)
from env_tuning.rods_matchtir_v1.lifecycle import validate_candidate_record

from rods_data_generation_v1_fixtures import (
    VERIFY_ACCEPT,
    make_catalog,
    make_config,
    make_seed,
    success_script,
)


WORKSPACE = Path("/root/autodl-tmp/rods-workspace")
RECOVERY_CANDIDATES = (
    WORKSPACE
    / "stage1_format_rl/artifacts/"
    "stage3_generator_incident_recovery_20260812T071318Z/"
    "queues/validated_candidates.jsonl"
)
ACTIVE_CATALOG_PARQUET = (
    WORKSPACE / "stage1_format_rl/data/bfcl_stage3_train_all_400_shuffled_seed42.parquet"
)
KNOWN_SEMANTIC_REGRESSIONS = {
    "multi_turn_base_113",
    "multi_turn_miss_param_101",
    "multi_turn_miss_func_145",
    "multi_turn_base_187",
    "multi_turn_base_66",
}


def _recovery_candidates() -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line in RECOVERY_CANDIDATES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            candidate = json.loads(line)
            records[candidate["generation_metadata"]["source_seed_id"]] = candidate
    return records


def _record(
    name: str,
    arguments: dict,
    result: object,
    *,
    class_name: str,
    turn_id: int,
    call_id: int,
    pre_state: dict | None = None,
    post_state: dict | None = None,
) -> ExecutionRecord:
    call = FunctionCall(name, arguments, class_name)
    semantic = classify_execution_result(name, result)
    return ExecutionRecord(
        turn_id=turn_id,
        call_id=call_id,
        call=call,
        canonical_call=call.canonical(),
        pre_state=copy.deepcopy(pre_state or {}),
        execution_result=copy.deepcopy(result),
        post_state=copy.deepcopy(post_state if post_state is not None else (pre_state or {})),
        dependency_provenance={},
        success=semantic.outcome != ExecutionSemanticOutcome.HARD_ERROR,
        semantic_outcome=semantic.outcome.value,
        semantic_detail=semantic.detail,
        error_detail=(
            semantic.detail
            if semantic.outcome == ExecutionSemanticOutcome.HARD_ERROR
            else None
        ),
    )


def _draft(turns: list[SynthesizedTurn], data_type: str = "multi_turn_base") -> ConversationDraft:
    return ConversationDraft(
        narrative="semantic fixture",
        data_type=data_type,
        initial_config={},
        initial_tools=[],
        involved_classes=sorted({turn.class_name for turn in turns}),
        turns=turns,
        synthesis_environment_id="semantic-fixture",
        structural_profile={},
    )


def _turn(
    turn_id: int,
    class_name: str,
    query: str,
    records: list[ExecutionRecord],
    *,
    intentional_missing: bool = False,
    missing_kind: str | None = None,
) -> SynthesizedTurn:
    return SynthesizedTurn(
        turn_id=turn_id,
        class_name=class_name,
        calls=[record.call for record in records],
        execution_records=records,
        raw_query=query,
        query=query,
        query_verification_reason="fixture",
        is_intentional_missing=intentional_missing,
        missing_kind=missing_kind,
    )


def test_hard_error_is_not_semantic_success() -> None:
    classified = classify_execution_result("get_stock_info", {"error": "bad symbol"})
    assert classified.outcome == ExecutionSemanticOutcome.HARD_ERROR


def test_domain_negative_is_not_a_vm_hard_error() -> None:
    classified = classify_execution_result(
        "get_symbol_by_name", {"symbol": "Stock not found"}
    )
    assert classified.outcome == ExecutionSemanticOutcome.DOMAIN_NEGATIVE


def test_domain_negative_cannot_ground_downstream_dependency() -> None:
    catalog = make_catalog()
    first = _record(
        "get_symbol_by_name",
        {"name": "Tesla Inc."},
        {"symbol": "Stock not found"},
        class_name="TradingBot",
        turn_id=0,
        call_id=0,
    )
    second = _record(
        "add_to_watchlist",
        {"stock": "TSLA"},
        {"symbol": ["TSLA"]},
        class_name="TradingBot",
        turn_id=0,
        call_id=1,
        pre_state={"TradingBot": {"stocks": {"TSLA": {"price": 1.0}}}},
    )
    gate = semantic_grounding_gate(
        _draft([_turn(0, "TradingBot", "Find Tesla Inc. and add it to my watchlist.", [first, second])]),
        catalog=catalog,
    )
    assert gate.passed is False
    assert "domain-negative producer" in gate.detail


def test_independent_success_output_can_ground_after_domain_negative() -> None:
    negative = _record(
        "get_symbol_by_name",
        {"name": "Tesla Inc."},
        {"symbol": "Stock not found"},
        class_name="TradingBot",
        turn_id=0,
        call_id=0,
    )
    independent = _record(
        "get_order_details",
        {"order_id": 12345},
        {"id": 12345, "symbol": "TSLA"},
        class_name="TradingBot",
        turn_id=0,
        call_id=1,
    )
    action = _record(
        "add_to_watchlist",
        {"stock": "TSLA"},
        {"symbol": ["TSLA"]},
        class_name="TradingBot",
        turn_id=0,
        call_id=2,
    )
    draft = _draft(
        [
            _turn(
                0,
                "TradingBot",
                "Check Tesla Inc., inspect order 12345, and add that order's stock to my watchlist.",
                [negative, independent, action],
            )
        ]
    )
    gate = semantic_grounding_gate(draft, catalog=make_catalog())
    assert gate.passed is True
    stock = next(
        item
        for item in draft.turns[0].execution_records[2].dependency_provenance.get(
            "parameter_provenance", []
        )
        if item["parameter"] == "stock"
    )
    assert stock["source_type"] == "PRIOR_TOOL_OUTPUT"


def test_standalone_negative_lookup_is_legal_when_query_aligned() -> None:
    record = _record(
        "get_symbol_by_name",
        {"name": "Not Listed Corp"},
        {"symbol": "Stock not found"},
        class_name="TradingBot",
        turn_id=0,
        call_id=0,
    )
    gate = semantic_grounding_gate(
        _draft([_turn(0, "TradingBot", "Is Not Listed Corp available as a stock symbol?", [record])]),
        catalog=make_catalog(),
    )
    assert gate.passed is True
    assert gate.metadata["semantic_outcome_counts"]["DOMAIN_NEGATIVE"] == 1


def test_explicit_query_argument_grounding_passes() -> None:
    record = _record(
        "add_to_watchlist",
        {"stock": "TSLA"},
        {"symbol": ["TSLA"]},
        class_name="TradingBot",
        turn_id=0,
        call_id=0,
    )
    draft = _draft([_turn(0, "TradingBot", "Add stock symbol TSLA to my watchlist.", [record])])
    gate = semantic_grounding_gate(draft, catalog=make_catalog())
    assert gate.passed is True
    evidence = draft.turns[0].execution_records[0].dependency_provenance[
        "parameter_provenance"
    ][0]
    assert evidence["source_type"] == "USER_CONTEXT"


def test_exact_natural_formatting_and_explicit_activation_grounding_pass() -> None:
    exchange = _record(
        "compute_exchange_rate",
        {"base_currency": "EUR", "target_currency": "USD", "value": 5000.0},
        {"exchanged_value": 6250.0},
        class_name="TravelAPI",
        turn_id=0,
        call_id=0,
    )
    exchange_gate = semantic_grounding_gate(
        _draft(
            [
                _turn(
                    0,
                    "TravelAPI",
                    "Convert my 5,000 EUR travel budget to USD.",
                    [exchange],
                )
            ]
        ),
        catalog=make_catalog(),
    )
    assert exchange_gate.passed is True

    cruise = _record(
        "setCruiseControl",
        {"activate": True, "distanceToNextVehicle": 100.0, "speed": 110.0},
        {"cruiseStatus": "active"},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=0,
    )
    cruise_gate = semantic_grounding_gate(
        _draft(
            [
                _turn(
                    0,
                    "VehicleControlAPI",
                    "Set my cruise control to 110 and keep a following distance of 100.",
                    [cruise],
                )
            ]
        ),
        catalog=make_catalog(),
    )
    assert cruise_gate.passed is True

    tweet = _record(
        "mention",
        {"mentioned_usernames": ["alice"], "tweet_id": 9},
        {"mention_status": "Users mentioned successfully"},
        class_name="TwitterAPI",
        turn_id=0,
        call_id=0,
    )
    ordinal_gate = semantic_grounding_gate(
        _draft(
            [
                _turn(
                    0,
                    "TwitterAPI",
                    "Mention @alice in my 9th tweet.",
                    [tweet],
                )
            ]
        ),
        catalog=make_catalog(),
    )
    assert ordinal_gate.passed is True


def test_prior_successful_tool_output_grounding_passes() -> None:
    first = _record(
        "get_user_id",
        {"user": "Alice"},
        {"user_id": "USR003"},
        class_name="MessageAPI",
        turn_id=0,
        call_id=0,
    )
    second = _record(
        "send_message",
        {"receiver_id": "USR003", "message": "Hello Alice"},
        {"sent_status": True},
        class_name="MessageAPI",
        turn_id=0,
        call_id=1,
    )
    draft = _draft(
        [_turn(0, "MessageAPI", "Find Alice and send her the message Hello Alice.", [first, second])]
    )
    gate = semantic_grounding_gate(draft, catalog=make_catalog())
    assert gate.passed is True
    evidence = draft.turns[0].execution_records[1].dependency_provenance[
        "parameter_provenance"
    ]
    receiver = next(item for item in evidence if item["parameter"] == "receiver_id")
    assert receiver["source_type"] == "PRIOR_TOOL_OUTPUT"
    assert receiver["leaf_sources"][0]["source_path"] == "result.user_id"
    profile = draft_structural_profile(draft.turns)
    assert profile["recoverable_dependency_depth"] == 2
    assert profile["explicit_dependency_edges"] == [
        {
            "from": {"turn_id": 0, "call_id": 0},
            "to": {"turn_id": 0, "call_id": 1},
        }
    ]


def test_structural_diagnostics_are_non_gating() -> None:
    diagnostics = structural_alignment_diagnostics(
        {
            "num_user_turns": 2,
            "gt_call_count_per_turn": [1, 1],
            "tool_classes_per_turn": [["MessageAPI"], ["MessageAPI"]],
            "recoverable_nested_dependency_depth": 1,
        },
        {
            "num_user_turns": 2,
            "gt_call_count_per_turn": [1, 2],
            "tool_classes_per_turn": [["MessageAPI"], ["MessageAPI"]],
            "recoverable_dependency_depth": 2,
        },
    )
    assert diagnostics["recoverable_dependency_depth_delta"] == 1
    assert diagnostics["used_for_acceptance"] is False
    assert diagnostics["acceptance_threshold"] == "NOT_DEFINED_BY_PUBLIC_RODS_SOURCES"


def test_relevant_environment_state_grounding_passes() -> None:
    state = {"MessageAPI": {"user_map": {"Alice": "USR001"}}}
    record = _record(
        "message_login",
        {"user_id": "USR001"},
        {"login_status": True},
        class_name="MessageAPI",
        turn_id=0,
        call_id=0,
        pre_state=state,
    )
    draft = _draft([_turn(0, "MessageAPI", "Log me in to my messaging account.", [record])])
    gate = semantic_grounding_gate(draft, catalog=make_catalog())
    assert gate.passed is True
    evidence = draft.turns[0].execution_records[0].dependency_provenance[
        "parameter_provenance"
    ][0]
    assert evidence["source_type"] == "ENV_STATE"
    assert "user_map" in evidence["leaf_sources"][0]["source_path"]


def test_unsupported_invented_parameter_fails() -> None:
    record = _record(
        "ticket_login",
        {"username": "Sarah", "password": "password123"},
        {"success": True},
        class_name="TicketAPI",
        turn_id=0,
        call_id=0,
    )
    gate = semantic_grounding_gate(
        _draft([_turn(0, "TicketAPI", "Log in as Sarah.", [record])]),
        catalog=make_catalog(),
    )
    assert gate.passed is False
    assert gate.metadata["unsupported"][0]["parameter"] == "password"


def test_unrelated_state_zero_one_unknown_do_not_ground_distance() -> None:
    state = {
        "VehicleControlAPI": {
            "unrelated_zero": 0.0,
            "unrelated_one": 1,
            "label": "unknown",
        }
    }
    record = _record(
        "estimate_drive_feasibility_by_mileage",
        {"distance": 0.0},
        {"canDrive": True},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=0,
        pre_state=state,
    )
    gate = semantic_grounding_gate(
        _draft([_turn(0, "VehicleControlAPI", "Can I reach my destination?", [record])]),
        catalog=make_catalog(),
    )
    assert gate.passed is False
    assert gate.metadata["unsupported"][0]["parameter"] == "distance"


def test_missing_function_and_parameter_protocols_preserve_recovery_grounding() -> None:
    add = _record(
        "add",
        {"a": 2.0, "b": 3.0},
        5.0,
        class_name="MathAPI",
        turn_id=0,
        call_id=0,
    )
    affected = _turn(
        0,
        "MathAPI",
        "Please add two and three.",
        [add],
        intentional_missing=True,
        missing_kind="function",
    )
    recovery = _turn(1, "MathAPI", "", [copy.deepcopy(add)])
    gate = semantic_grounding_gate(
        _draft([affected, recovery], "multi_turn_miss_func"), catalog=make_catalog()
    )
    assert gate.passed is True
    assert affected.execution_records[0].dependency_provenance[
        "parameter_dependency_status"
    ] == "INTENTIONAL_MISSING_SKIPPED"

    affected_mp = _turn(
        0,
        "MathAPI",
        "Please add an unspecified value to three.",
        [copy.deepcopy(add)],
        intentional_missing=True,
        missing_kind="parameter",
    )
    recovery_mp = _turn(1, "MathAPI", "Use two as the missing value.", [copy.deepcopy(add)])
    gate_mp = semantic_grounding_gate(
        _draft([affected_mp, recovery_mp], "multi_turn_miss_param"),
        catalog=make_catalog(),
    )
    assert gate_mp.passed is True


def test_post_rewrite_semantic_drift_fails_before_quality_judge() -> None:
    script = success_script()
    script["coherence_rewrite"] = [
        "<query>Please subtract three from two.</query>"
        "<query>Now, what is four times five?</query>"
    ]
    script["final_query_verifier"] = [
        "<reason>Turn zero asks for subtraction but GT performs addition.</reason>"
        "<verdict>reject</verdict>"
    ]
    backend = FakeLLMBackend(script)
    pipeline = RODSDataGenerationPipeline(
        config=make_config(), backend=backend, catalog=make_catalog()
    )
    result = asyncio.run(pipeline.generate(make_seed()))
    assert result.status == "DROPPED"
    assert "final semantic verifier rejected" in result.reason
    assert not [call for call in backend.calls if call["role"] == "quality_judge"]


def test_exact_cross_seed_training_content_is_deduplicated_with_terminal_provenance(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path=tmp_path, dry_run=False, test_mode=True)
    daemon = GeneratorDaemon(config=config, backend=FakeLLMBackend({}))
    base = {
        "candidate_id": "candidate_same_content",
        "validated": True,
        "validation": {"passed": True},
        "generation_metadata": {
            "source_seed_id": "seed_A",
            "generated_epoch": 1,
            "content_fingerprint": "candidate_content_v2_same",
        },
        "sample": {"training": "identical"},
    }
    second = copy.deepcopy(base)
    second["generation_metadata"]["source_seed_id"] = "seed_B"
    daemon._append_candidate_exact(base)
    daemon._append_candidate_exact(second)
    assert daemon.candidate_queue.read() == [base]

    journal = TerminalResultJournal(tmp_path / "terminal_dedup.jsonl")
    for seed_id, candidate in (("seed_A", base), ("seed_B", second)):
        journal.commit(
            PipelineResult(
                seed_id=seed_id,
                status="SUCCEEDED",
                candidate=candidate,
                errors=[],
                attempts=1,
                planner_calls=1,
                blocklist_history=[],
                config_patch_history=[],
                metrics={},
                checkpoint={
                    "completed_failed_attempts": 0,
                    "planner_calls": 1,
                    "failures": [],
                    "patches": [],
                    "blocklist": [],
                    "blocklist_history": [],
                    "current_config": {},
                },
            )
        )
    assert {item["seed_id"] for item in journal.read()} == {"seed_A", "seed_B"}


def test_candidate_identity_depends_on_training_content_not_source_seed() -> None:
    first_seed = make_seed()
    second_seed = copy.deepcopy(first_seed)
    first_seed["sample_id"] = "semantic_identity_seed_A"
    second_seed["sample_id"] = "semantic_identity_seed_B"

    first = asyncio.run(
        RODSDataGenerationPipeline(
            config=make_config(),
            backend=FakeLLMBackend(success_script()),
            catalog=make_catalog(),
        ).generate(first_seed)
    )
    second = asyncio.run(
        RODSDataGenerationPipeline(
            config=make_config(),
            backend=FakeLLMBackend(success_script()),
            catalog=make_catalog(),
        ).generate(second_seed)
    )
    assert first.status == second.status == "SUCCEEDED"
    assert first.candidate is not None and second.candidate is not None
    assert first.candidate["candidate_id"] == second.candidate["candidate_id"]
    assert (
        first.candidate["generation_metadata"]["content_fingerprint"]
        == second.candidate["generation_metadata"]["content_fingerprint"]
    )
    assert (
        first.candidate["generation_metadata"]["source_seed_id"]
        != second.candidate["generation_metadata"]["source_seed_id"]
    )


def test_real_recovery_candidates_are_reclassified_by_evidence_not_seed_hardcode() -> None:
    candidates = _recovery_candidates()
    catalog = FunctionCatalog.from_training_parquet(ACTIVE_CATALOG_PARQUET)
    assert KNOWN_SEMANTIC_REGRESSIONS <= set(candidates)
    results = {
        seed_id: revalidate_candidate_grounding(candidates[seed_id], catalog=catalog)[1]
        for seed_id in KNOWN_SEMANTIC_REGRESSIONS
    }
    assert all(not gate.passed for gate in results.values())
    assert "domain-negative producer" in results["multi_turn_base_113"].detail
    assert "domain-negative producer" in results["multi_turn_miss_param_101"].detail
    assert "domain-negative producer" in results["multi_turn_miss_func_145"].detail
    assert "distance" in results["multi_turn_base_66"].detail
    assert results["multi_turn_base_187"].metadata["unsupported"]
    # Their frozen schema validity is unchanged; semantic quarantine is an
    # additional Generator gate, not a weakened Training contract.
    assert all(validate_candidate_record(candidates[seed_id])["validated"] for seed_id in results)
