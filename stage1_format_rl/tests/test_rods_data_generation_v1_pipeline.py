from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

from env_tuning.rods_data_generation_v1.daemon import (
    GenerationGuardError,
    GeneratorDaemon,
)
from env_tuning.rods_data_generation_v1.environment_adapter import SynthesisEnvironmentAdapter
from env_tuning.rods_data_generation_v1.error_taxonomy import ErrorType
from env_tuning.rods_data_generation_v1.llm_backend import FakeLLMBackend, ReplayLLMBackend
from env_tuning.rods_data_generation_v1.models import FunctionCall
from env_tuning.rods_data_generation_v1.pipeline import RODSDataGenerationPipeline
from env_tuning.rods_data_generation_v1.queue import LockedJsonlQueue
from env_tuning.rods_matchtir_v1.lifecycle import (
    LifecycleConfig,
    RODSStage3Lifecycle,
    validate_candidate_record,
)

from rods_data_generation_v1_fixtures import (
    JUDGE_ACCEPT,
    PARAM_ADD,
    PARAM_MULTIPLY,
    PLANNER_ADD_MULTIPLY,
    QUERY_ADD,
    QUERY_MULTIPLY,
    REWRITE_TWO,
    VERIFY_ACCEPT,
    make_backend,
    make_catalog,
    make_config,
    make_seed,
    success_script,
)


def _run_pipeline(data_type: str, *, backend=None, tmp_path: Path | None = None):
    config = make_config(tmp_path=tmp_path)
    backend = backend or make_backend(data_type)
    factory = SynthesisEnvironmentAdapter(is_augmented=False)
    pipeline = RODSDataGenerationPipeline(
        config=config,
        backend=backend,
        catalog=make_catalog(),
        environment_factory=factory,
    )
    result = asyncio.run(pipeline.generate(make_seed(data_type)))
    return result, backend, factory, pipeline


@pytest.mark.parametrize(
    "data_type",
    [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    ],
)
def test_real_cpu_bfcl_e2e_for_all_four_types(data_type):
    result, backend, factory, _ = _run_pipeline(data_type)
    assert result.status == "SUCCEEDED", result.reason
    candidate = validate_candidate_record(result.candidate)
    assert candidate["validated"] is True
    assert candidate["validation"]["passed"] is True
    assert candidate["generation_metadata"]["generated_epoch"] == 7
    assert candidate["generation_metadata"]["source_seed_id"] == f"seed-{data_type}"
    assert len(factory.created_environment_ids) == 2
    assert len(set(factory.created_environment_ids)) == 2
    gates = candidate["generation_metadata"]["deterministic_gate_results"]
    assert [gate["name"] for gate in gates] == [
        "unit_semantic_gate",
        "semantic_grounding_gate",
        "missing_parameter_validity_gate",
        "observation_entailment_gate",
        "relational_resolution_gate",
        "final_query_semantic_gate",
        "action_minimality_gate",
        "fresh_vm_gate",
        "tool_availability_gate",
        "parameter_complexity_gate",
    ]
    assert all(gate["passed"] for gate in gates)
    trace = candidate["generation_metadata"]["execution_trace"]
    assert trace and trace[0]["records"]
    first_record = trace[0]["records"][0]
    assert {"call", "pre_state", "execution_result", "post_state", "dependency_provenance"}.issubset(first_record)
    assert candidate["generation_metadata"]["structural_profile"]["seed_profile"]["used_for_acceptance"] is False

    kwargs = candidate["sample"]["extra_info"]["interaction_kwargs"]
    system = candidate["sample"]["prompt"][0]["content"]
    assert "<think>" in system and "<tool_call>" in system and "<answer>" in system
    assert "<reason>" not in system
    assert all(
        "<think>" not in message["content"]
        for call in backend.calls
        for message in call["messages"]
    )

    if data_type == "multi_turn_miss_func":
        assert kwargs["ground_truth"][0] == []
        assert kwargs["question"][0]
        assert kwargs["question"][1] == []
        assert kwargs["ground_truth"][1]
        assert "I have updated some more functions" in kwargs["processed_question"][0]
        visible_initial_tools = json.loads(system.split(
            "Here is a list of functions in JSON format that you can invoke.\n", 1
        )[1])
        assert "add" not in {tool["name"] for tool in visible_initial_tools}
    elif data_type == "multi_turn_miss_param":
        assert kwargs["ground_truth"][0] == []
        assert kwargs["ground_truth"][1]
        assert kwargs["question"][0][0]["content"].startswith("Please combine")
        assert "2.0" in kwargs["question"][1][0]["content"]
    else:
        assert all(kwargs["ground_truth"])
    if data_type == "multi_turn_long_context":
        assert "long_context" in kwargs["id"]


def test_replay_backend_runs_complete_real_vm_pipeline(tmp_path):
    replay_path = tmp_path / "replay.jsonl"
    records = []
    for role, values in success_script().items():
        records.extend({"role": role, "text": value} for value in values)
    replay_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    replay = ReplayLLMBackend.from_jsonl(replay_path)
    result, _, factory, _ = _run_pipeline("multi_turn_base", backend=replay)
    assert result.status == "SUCCEEDED"
    assert len(factory.created_environment_ids) == 2
    assert replay.remaining() == 0


def _official_jsonl_record(path: Path, record_id: str) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("id") == record_id:
            return record
    raise AssertionError(f"official BFCL record not found: {record_id}")


def test_real_stateful_gorilla_filesystem_chain_and_fresh_vm_replay(tmp_path):
    """Official base_12 state: touch -> echo dependency across user turns."""

    asset_root = Path(os.environ.get("TOOLWEAVE_ASSET_ROOT", Path(__file__).resolve().parents[2]))
    bfcl_root = asset_root / "data/Berkeley-Function-Calling-Leaderboard"
    task = _official_jsonl_record(
        bfcl_root / "BFCL_v3_multi_turn_base.json", "multi_turn_base_12"
    )
    answer = _official_jsonl_record(
        bfcl_root / "possible_answer/BFCL_v3_multi_turn_base.json",
        "multi_turn_base_12",
    )
    assert answer["ground_truth"][0] == [
        "cd(folder='Documents')",
        "touch(file_name='summary.txt')",
    ]

    catalog = make_catalog()
    seed = make_seed()
    seed.update(
        {
            "sample_id": "official-stateful-multi_turn_base_12",
            "Q_old": task["question"],
            "GT_old": answer["ground_truth"],
            "available_functions": [
                copy.deepcopy(spec.schema)
                for spec in catalog.functions_for_classes(["GorillaFileSystem"])
            ],
            "initial_config": task["initial_config"],
            "generation_metadata": {
                "source": "local official BFCL_v3_multi_turn_base_12",
                "progress_source": "R_P_only",
            },
        }
    )
    backend = FakeLLMBackend(
        {
            "planner": [
                "<reason>Preserve the official create-write-inspect state chain.</reason>"
                "<narrative>A user prepares and checks a short research summary.</narrative>"
                "<turn>GorillaFileSystem: cd, touch</turn>"
                "<turn>GorillaFileSystem: echo</turn>"
                "<turn>GorillaFileSystem: wc</turn>"
            ],
            "parameter_generator": [
                '<reason>Enter the official folder.</reason><arguments>{"folder":"Documents"}</arguments>',
                '<reason>Create the requested file.</reason><arguments>{"file_name":"summary.txt"}</arguments>',
                '<reason>Write the requested content.</reason><arguments>{"content":"quantum computing","file_name":"summary.txt"}</arguments>',
                '<reason>Count words in the created file.</reason><arguments>{"file_name":"summary.txt","mode":"w"}</arguments>',
            ],
            "query_generator": [
                "<reason>Describe navigation and creation naturally.</reason>"
                "<query>Open my Documents folder and create a new summary.txt file.</query>",
                "<reason>Describe the write intent.</reason>"
                "<query>Put only quantum computing into that summary file.</query>",
                "<reason>Describe the dependent inspection.</reason>"
                "<query>How many words are in that summary file now?</query>",
            ],
            "query_verifier": [VERIFY_ACCEPT, VERIFY_ACCEPT, VERIFY_ACCEPT],
            "coherence_rewrite": [
                "<query>Open my Documents folder and create a new summary.txt file.</query>"
                "<query>Now put only quantum computing into that summary file.</query>"
                "<query>Finally, how many words are in it?</query>"
            ],
            "final_query_verifier": [VERIFY_ACCEPT],
            "quality_judge": [JUDGE_ACCEPT],
        }
    )
    factory = SynthesisEnvironmentAdapter(is_augmented=False)
    pipeline = RODSDataGenerationPipeline(
        config=make_config(tmp_path=tmp_path),
        backend=backend,
        catalog=catalog,
        environment_factory=factory,
    )
    result = asyncio.run(pipeline.generate(seed))
    assert result.status == "SUCCEEDED", result.reason
    candidate = validate_candidate_record(result.candidate)

    trace = candidate["generation_metadata"]["execution_trace"]
    records = [record for turn in trace for record in turn["records"]]
    touch = next(record for record in records if record["call"]["name"] == "touch")
    echo = next(record for record in records if record["call"]["name"] == "echo")
    assert touch["pre_state"] != touch["post_state"]
    assert echo["pre_state"] == touch["post_state"]
    assert echo["pre_state"] != echo["post_state"]
    assert echo["dependency_provenance"]["state_predecessor"] == {
        "turn_id": 0,
        "call_id": 1,
        "exact_state_continuity": True,
    }

    gates = candidate["generation_metadata"]["deterministic_gate_results"]
    fresh = next(gate for gate in gates if gate["name"] == "fresh_vm_gate")
    assert fresh["passed"] is True
    assert fresh["metadata"]["environment_id"] != fresh["metadata"]["synthesis_environment_id"]
    assert fresh["metadata"]["executed_call_count"] == 4
    assert len(factory.created_environment_ids) == 2

    # Counterfactual on a third real BFCL instance: writing the file without
    # the preceding touch mutation fails, proving the second turn depends on C1.
    counterfactual = factory.create(
        initial_config=task["initial_config"],
        involved_classes=["GorillaFileSystem"],
        seed_id="counterfactual",
        long_context=False,
        purpose="stateful_counterfactual",
    )
    try:
        assert counterfactual.execute(
            FunctionCall("cd", {"folder": "Documents"}, "GorillaFileSystem")
        ).success
        missing_file_write = counterfactual.execute(
            FunctionCall(
                "echo",
                {"content": "quantum computing", "file_name": "summary.txt"},
                "GorillaFileSystem",
            )
        )
        assert missing_file_write.success is False
        assert "No such file" in (missing_file_write.error_detail or "")
    finally:
        counterfactual.close()

    structural = candidate["generation_metadata"]["structural_profile"]
    assert structural["used_for_acceptance"] is False
    assert structural["alignment_diagnostics"]["used_for_acceptance"] is False
    assert structural["alignment_diagnostics"]["acceptance_threshold"] == (
        "NOT_DEFINED_BY_PUBLIC_RODS_SOURCES"
    )
    query_calls = [call for call in backend.calls if call["role"] == "query_generator"]
    assert len(query_calls) == 3
    assert all(
        "file or directory work" in call["messages"][0]["content"]
        for call in query_calls
    )


def test_vm_failure_patches_blocks_and_replans_with_feedback():
    first_plan = (
        "<reason>Try a division workflow.</reason><narrative>Check arithmetic.</narrative>"
        "<turn>MathAPI: divide</turn><turn>MathAPI: subtract</turn>"
    )
    patch = (
        "<reason>Record a compatible fixture setting.</reason>"
        "<patch><class>MathAPI</class><field>fixture_enabled</field><value>true</value></patch>"
    )
    backend = FakeLLMBackend(
        {
            "planner": [first_plan, PLANNER_ADD_MULTIPLY],
            "parameter_generator": [
                '<reason>Exercise VM failure.</reason><arguments>{"a": 1.0, "b": 0.0}</arguments>',
                PARAM_ADD,
                PARAM_MULTIPLY,
            ],
            "config_patch": [patch],
            "query_generator": [QUERY_ADD, QUERY_MULTIPLY],
            "query_verifier": [VERIFY_ACCEPT, VERIFY_ACCEPT],
            "coherence_rewrite": [REWRITE_TWO],
            "final_query_verifier": [VERIFY_ACCEPT],
            "quality_judge": [JUDGE_ACCEPT],
        }
    )
    result, backend, factory, pipeline = _run_pipeline("multi_turn_base", backend=backend)
    assert result.status == "SUCCEEDED", result.reason
    assert result.attempts == 2
    assert [error.error_type for error in result.errors] == [ErrorType.VM_EXEC_FAILED]
    assert result.blocklist_history == [["divide"]]
    assert result.config_patch_history[0]["applied"] is True
    assert result.config_patch_history[0]["patch"] == {
        "MathAPI": {"fixture_enabled": True}
    }
    planner_calls = [call for call in backend.calls if call["role"] == "planner"]
    assert len(planner_calls) == 2
    second_prompt = planner_calls[1]["messages"][0]["content"]
    assert "vm_exec_failed" in second_prompt
    assert '"divide"' in second_prompt
    assert "fixture_enabled" in second_prompt
    assert "COMPLETELY DIFFERENT" in second_prompt
    assert len(factory.created_environment_ids) == 3


def test_three_malformed_planner_attempts_each_use_three_parser_retries():
    backend = FakeLLMBackend({"planner": ["malformed"] * 9})
    result, backend, _, _ = _run_pipeline("multi_turn_base", backend=backend)
    assert result.status == "DROPPED"
    assert result.attempts == 3
    assert result.planner_calls == 9
    assert [error.error_type for error in result.errors] == [ErrorType.NO_PATTERN] * 3
    assert backend.remaining() == 0


def test_attempt3_success_checkpoint_contains_only_two_completed_failures():
    script = success_script()
    script["planner"] = ["malformed"] * 6 + [PLANNER_ADD_MULTIPLY]
    result, _, _, _ = _run_pipeline(
        "multi_turn_base", backend=FakeLLMBackend(script)
    )
    assert result.status == "SUCCEEDED"
    assert result.attempts == 3
    assert result.checkpoint["completed_failed_attempts"] == 2
    assert "attempts" not in result.checkpoint

    resumed_backend = make_backend()
    resumed_pipeline = RODSDataGenerationPipeline(
        config=make_config(),
        backend=resumed_backend,
        catalog=make_catalog(),
    )
    resumed = asyncio.run(
        resumed_pipeline.generate(make_seed(), resume_state=result.checkpoint)
    )
    assert resumed.status == "SUCCEEDED"
    assert resumed.attempts == 3
    assert any(call["role"] == "planner" for call in resumed_backend.calls)


def test_pipeline_maps_conversation_catalog_exhaustion_and_unhandled_exception_errors(monkeypatch):
    conversation_script = success_script()
    conversation_script["coherence_rewrite"] = ["<query>wrong count</query>"]
    result, _, _, _ = _run_pipeline(
        "multi_turn_base", backend=FakeLLMBackend(conversation_script)
    )
    assert result.errors[0].error_type == ErrorType.CONVERSATION_CONSTRUCT_FAILED

    catalog = make_catalog()
    all_math = [spec.name for spec in catalog.functions_for_classes(["MathAPI"])]
    backend = FakeLLMBackend({})
    config = make_config()
    pipeline = RODSDataGenerationPipeline(config=config, backend=backend, catalog=catalog)
    result = asyncio.run(
        pipeline.generate(
            make_seed(),
            resume_state={
                "attempts": 0,
                "failures": [],
                "patches": [],
                "blocklist": all_math,
                "blocklist_history": [all_math],
                "current_config": {},
            },
        )
    )
    assert result.errors[0].error_type == ErrorType.FUNC_SAMPLE_FAILED
    assert {call["role"] for call in backend.calls} == {"config_patch"}

    missing_prompt_pipeline = RODSDataGenerationPipeline(
        config=config, backend=FakeLLMBackend({}), catalog=make_catalog()
    )

    async def missing_prompt(*args, **kwargs):
        raise FileNotFoundError("controlled missing class prompt")

    monkeypatch.setattr(missing_prompt_pipeline.planner, "plan", missing_prompt)
    result = asyncio.run(missing_prompt_pipeline.generate(make_seed()))
    assert [error.error_type for error in result.errors] == [ErrorType.NO_PROMPTS] * 3

    result, _, _, _ = _run_pipeline(
        "multi_turn_base",
        backend=FakeLLMBackend({"planner": [RuntimeError("controlled")] * 3}),
    )
    assert [error.error_type for error in result.errors] == [
        ErrorType.PIPELINE_EXCEPTION,
        ErrorType.PIPELINE_EXCEPTION,
        ErrorType.PIPELINE_EXCEPTION,
    ]


def test_query_verification_failures_drive_three_different_plans_then_drop():
    plans = [
        (
            "<reason>x</reason><narrative>n</narrative>"
            "<turn>MathAPI: add</turn><turn>MathAPI: multiply</turn>"
        ),
        (
            "<reason>x</reason><narrative>n</narrative>"
            "<turn>MathAPI: subtract</turn><turn>MathAPI: multiply</turn>"
        ),
        (
            "<reason>x</reason><narrative>n</narrative>"
            "<turn>MathAPI: divide</turn><turn>MathAPI: multiply</turn>"
        ),
    ]
    params = [
        '<reason>x</reason><arguments>{"a": 2.0, "b": 3.0}</arguments>',
        '<reason>x</reason><arguments>{"a": 5.0, "b": 2.0}</arguments>',
        '<reason>x</reason><arguments>{"a": 6.0, "b": 2.0}</arguments>',
    ]
    queries = [
        "<reason>x</reason><query>What is two plus three?</query>",
        "<reason>x</reason><query>What is five minus two?</query>",
        "<reason>x</reason><query>What is six divided by two?</query>",
    ]
    reject = "<reason>The semantics do not align.</reason><verdict>reject</verdict>"
    backend = FakeLLMBackend(
        {
            "planner": plans,
            "parameter_generator": params,
            "query_generator": queries,
            "query_verifier": [reject] * 3,
        }
    )
    result, backend, _, _ = _run_pipeline("multi_turn_base", backend=backend)
    assert result.status == "DROPPED"
    assert [error.error_type for error in result.errors] == [
        ErrorType.QUERY_VERIFY_FAILED,
        ErrorType.QUERY_VERIFY_FAILED,
        ErrorType.QUERY_VERIFY_FAILED,
    ]
    assert result.blocklist_history == [
        ["add"],
        ["add", "subtract"],
        ["add", "divide", "subtract"],
    ]
    planner_prompts = [
        call["messages"][0]["content"]
        for call in backend.calls
        if call["role"] == "planner"
    ]
    assert "query_verify_failed" in planner_prompts[1]
    assert '"add"' in planner_prompts[1]
    assert '"subtract"' in planner_prompts[2]


def test_judge_query_fixable_allows_exactly_one_rewrite_and_second_judge():
    script = success_script()
    script["quality_judge"] = [
        "<reason>The first query is awkward.</reason><decision>reject</decision>"
        "<fail_reason>Turn 1 is unnatural.</fail_reason>",
        JUDGE_ACCEPT,
    ]
    script["final_query_verifier"] = [VERIFY_ACCEPT, VERIFY_ACCEPT]
    script["refine_classify"] = [
        "<reason>Only wording is wrong.</reason><answer>query_fixable</answer>"
    ]
    script["refine_rewrite"] = [
        "<answer>Could you calculate two plus three for me?</answer>"
    ]
    backend = FakeLLMBackend(script)
    result, backend, _, _ = _run_pipeline("multi_turn_base", backend=backend)
    assert result.status == "SUCCEEDED"
    metadata = result.candidate["generation_metadata"]
    assert metadata["refinement_used"] is True
    assert metadata["refinement_metadata"]["turn_index"] == 0
    kwargs = result.candidate["sample"]["extra_info"]["interaction_kwargs"]
    assert kwargs["question"][0][0]["content"] == "Could you calculate two plus three for me?"
    assert len([call for call in backend.calls if call["role"] == "quality_judge"]) == 2
    assert len([call for call in backend.calls if call["role"] == "refine_rewrite"]) == 1


def test_judge_gt_unfixable_drops_without_rewrite():
    script = success_script()
    script["quality_judge"] = [
        "<reason>The GT is invalid.</reason><decision>reject</decision>"
        "<fail_reason>Turn 1 has incorrect GT state.</fail_reason>"
    ]
    script["refine_classify"] = [
        "<reason>The GT cannot be repaired by wording.</reason><answer>gt_unfixable</answer>"
    ]
    backend = FakeLLMBackend(script)
    result, backend, _, _ = _run_pipeline("multi_turn_base", backend=backend)
    assert result.status == "DROPPED"
    assert "gt_unfixable" in result.reason
    assert not [call for call in backend.calls if call["role"] == "refine_rewrite"]


def test_deterministic_parameter_gate_cannot_be_overridden_by_judge():
    plan = (
        "<reason>x</reason><narrative>n</narrative>"
        "<turn>MathAPI: mean</turn><turn>MathAPI: add</turn>"
    )
    backend = FakeLLMBackend(
        {
            "planner": [plan],
            "parameter_generator": [
                '<reason>x</reason><arguments>{"numbers": [1, 2, 3, 4, 5, 6]}</arguments>',
                PARAM_ADD,
            ],
            "query_generator": [
                "<reason>x</reason><query>What is the average of 1, 2, 3, 4, 5, and 6?</query>",
                QUERY_ADD,
            ],
            "query_verifier": [VERIFY_ACCEPT, VERIFY_ACCEPT],
            "coherence_rewrite": [
                "<query>Could you find the average of 1, 2, 3, 4, 5, and 6?</query>"
                "<query>Could you work out two plus three?</query>"
            ],
            "final_query_verifier": [VERIFY_ACCEPT],
            "quality_judge": [JUDGE_ACCEPT],
        }
    )
    result, backend, _, _ = _run_pipeline("multi_turn_base", backend=backend)
    assert result.status == "DROPPED"
    assert "parameter_complexity_gate" in result.reason
    assert not [call for call in backend.calls if call["role"] == "quality_judge"]


def test_daemon_writes_once_tracks_success_and_never_uses_production_queue(tmp_path):
    config = make_config(tmp_path=tmp_path, dry_run=False, test_mode=True)
    backend = make_backend()
    factory = SynthesisEnvironmentAdapter(is_augmented=False)
    pipeline = RODSDataGenerationPipeline(
        config=config,
        backend=backend,
        catalog=make_catalog(),
        environment_factory=factory,
    )
    LockedJsonlQueue(config.queues.seed_path).append([make_seed()])
    daemon = GeneratorDaemon(config=config, backend=backend, pipeline=pipeline)
    metrics = asyncio.run(daemon.run_once(allow_generation=True))
    candidates = daemon.candidate_queue.read()
    assert len(candidates) == 1
    assert metrics["lifecycle/candidates_written"] == 1.0
    item = daemon.tracker.snapshot()["seeds"]["seed-multi_turn_base"]
    assert item["status"] == "SUCCEEDED"
    assert item["candidate_id"] == candidates[0]["candidate_id"]
    asyncio.run(daemon.run_once(allow_generation=True))
    assert len(daemon.candidate_queue.read()) == 1
    assert daemon.metrics.snapshot()["queue/duplicate_skipped"] >= 1.0
    assert Path(config.queues.candidate_path).resolve() != Path(
        config.queues.production_candidate_path
    ).resolve()


def test_daemon_dry_run_does_not_claim_or_write(tmp_path):
    config = make_config(tmp_path=tmp_path, dry_run=True, test_mode=True)
    backend = FakeLLMBackend({})
    LockedJsonlQueue(config.queues.seed_path).append([make_seed()])
    daemon = GeneratorDaemon(config=config, backend=backend)
    metrics = asyncio.run(daemon.run_once())
    assert metrics["queue/seeds_seen"] == 1.0
    assert "queue/seeds_claimed" not in metrics
    assert daemon.candidate_queue.read() == []
    assert backend.calls == []


def test_non_test_generation_requires_double_guard(tmp_path, monkeypatch):
    config = make_config(tmp_path=tmp_path, dry_run=False, test_mode=False)
    daemon = GeneratorDaemon(config=config, backend=FakeLLMBackend({}))
    monkeypatch.delenv("RODS_ALLOW_DATA_GENERATION", raising=False)
    with pytest.raises(GenerationGuardError):
        asyncio.run(daemon.run_once(allow_generation=False))
    with pytest.raises(GenerationGuardError):
        asyncio.run(daemon.run_once(allow_generation=True))


def test_generated_epoch_n_is_deferred_until_training_epoch_n_plus_one(tmp_path):
    result, _, _, _ = _run_pipeline("multi_turn_base")
    candidate = result.candidate
    assert candidate["generation_metadata"]["generated_epoch"] == 7

    originals = []
    for index in range(5):
        sample = copy.deepcopy(candidate["sample"])
        sample_id = f"original-{index}"
        sample["extra_info"]["original_id"] = sample_id
        sample["extra_info"]["interaction_kwargs"]["id"] = sample_id
        originals.append(sample)
    dataset = SimpleNamespace(dataframe=Dataset.from_list(originals))

    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "training_seeds.jsonl"),
            candidate_input_path=str(tmp_path / "training_candidates.jsonl"),
            state_path=str(tmp_path / "training_state.json"),
            injection_ratio=0.20,
            generated_pool_cap=400,
        )
    )
    lifecycle.candidate_queue.append([candidate])
    same_epoch = lifecycle.on_epoch_boundary(dataset, epoch=7, global_step=70)
    assert same_epoch["rods_stage3_lifecycle/ingested_candidate_count"] == 0.0
    assert same_epoch["rods_stage3_lifecycle/deferred_candidate_count"] == 1.0
    next_epoch = lifecycle.on_epoch_boundary(dataset, epoch=8, global_step=80)
    assert next_epoch["rods_stage3_lifecycle/ingested_candidate_count"] == 1.0
    assert lifecycle.active_generated_count == 1
    ids = [
        row["extra_info"]["interaction_kwargs"]["id"]
        for row in dataset.dataframe
    ]
    assert candidate["sample"]["extra_info"]["interaction_kwargs"]["id"] in ids
