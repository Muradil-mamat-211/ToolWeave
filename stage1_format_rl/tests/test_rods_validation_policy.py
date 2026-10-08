"""Admission-policy regression tests; fake LLM responses, real CPU BFCL VM."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json

import pytest

from env_tuning.rods_data_generation_v1 import pipeline as pipeline_module
from env_tuning.rods_data_generation_v1.config import GeneratorConfig
from env_tuning.rods_data_generation_v1.llm_backend import BackendQuotaExceeded, FakeLLMBackend
from env_tuning.rods_data_generation_v1.models import GateResult
from env_tuning.rods_data_generation_v1.pipeline import RODSDataGenerationPipeline
from rods_data_generation_v1_fixtures import (
    JUDGE_ACCEPT, make_catalog, make_config, make_seed, success_script,
)


CORE_GATES = ["tool_availability_gate", "parameter_complexity_gate"]
REJECT = ("<reason>The request is awkward.</reason><decision>reject</decision>"
          "<fail_reason>Turn 1 is unnatural.</fail_reason>")


def make_pipeline(data_type="multi_turn_base", *, script=None, policy="rods", **overrides):
    config = replace(make_config(), validation_policy=policy, **overrides)
    responses = success_script(data_type) if script is None else script
    if policy == "rods":
        responses.pop("final_query_verifier", None)
    backend = FakeLLMBackend(responses)
    pipeline = RODSDataGenerationPipeline(config=config, backend=backend, catalog=make_catalog())
    return pipeline, backend


def run(pipeline, data_type="multi_turn_base"):
    return asyncio.run(pipeline.generate(make_seed(data_type)))


def test_policy_default_mapping_and_invalid_value():
    assert GeneratorConfig().validation_policy == "rods"
    assert GeneratorConfig.from_mapping({}).validation_policy == "rods"
    assert GeneratorConfig.from_mapping({"validation_policy": "strict"}).validation_policy == "strict"
    assert GeneratorConfig().replay_final_gt is False
    assert GeneratorConfig.from_mapping({}).replay_final_gt is False
    assert GeneratorConfig.from_mapping({"replay_final_gt": True}).replay_final_gt is True
    with pytest.raises(ValueError, match="validation_policy"):
        GeneratorConfig(validation_policy="accept_everything")


@pytest.mark.parametrize("data_type", [
    "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context",
])
def test_rods_four_types_reuse_execution_and_review_final_queries_once(data_type, monkeypatch):
    def unexpected_replay(*args, **kwargs):
        raise AssertionError("default final review must not replay GT")

    monkeypatch.setattr(pipeline_module, "fresh_vm_reverify_gate", unexpected_replay)
    pipeline, backend = make_pipeline(data_type, long_context_min_observation_chars=999999)
    result = run(pipeline, data_type)
    assert result.status == "SUCCEEDED", result.reason
    metadata = result.candidate["generation_metadata"]
    assert metadata["validation_policy"] == "rods"
    assert metadata["final_gt_replay_performed"] is False
    assert metadata["execution_validation_source"] == "synthesis_vm_trace"
    assert [g["name"] for g in metadata["deterministic_gate_results"]] == CORE_GATES
    assert all(g["passed"] for g in metadata["deterministic_gate_results"])
    assert all(g["blocking"] is False for g in metadata["project_diagnostics"])
    assert len(metadata["project_diagnostics"]) == 8
    if data_type == "multi_turn_long_context":
        category = next(g for g in metadata["project_diagnostics"] if g["name"] == "category_validity_gate")
        assert category["passed"] is False
    roles = [call["role"] for call in backend.calls]
    assert "final_query_verifier" not in roles
    assert roles.count("quality_judge") == 1
    assert roles.count("query_verifier") == 2
    assert len(pipeline.environment_factory.created_environment_ids) == 2
    trace = metadata["execution_trace"]
    for turn in trace:
        if turn["intentional_missing"]:
            assert turn["records"] == []
        else:
            assert turn["records"] and all(record["success"] for record in turn["records"])
    kwargs = result.candidate["sample"]["extra_info"]["interaction_kwargs"]
    if data_type in {"multi_turn_miss_func", "multi_turn_miss_param"}:
        assert kwargs["ground_truth"][0] == []
        assert kwargs["ground_truth"][1] == ["add(a=2.0, b=3.0)"]
        assert kwargs["ground_truth"][2] == ["multiply(a=4.0, b=5.0)"]
    prompts = "\n".join(m["content"] for c in backend.calls for m in c["messages"])
    assert "at least half" not in prompts
    assert "final fifth" not in prompts
    assert 'immediately introduced as "exact format"' not in prompts


def test_fresh_gt_replay_remains_explicitly_available():
    pipeline, backend = make_pipeline(replay_final_gt=True)
    result = run(pipeline)
    assert result.status == "SUCCEEDED", result.reason
    metadata = result.candidate["generation_metadata"]
    assert [g["name"] for g in metadata["deterministic_gate_results"]] == ["fresh_vm_gate", *CORE_GATES]
    assert metadata["final_gt_replay_performed"] is True
    assert metadata["execution_validation_source"] == "fresh_vm_replay"
    fresh = metadata["deterministic_gate_results"][0]
    assert fresh["metadata"]["environment_id"] != fresh["metadata"]["synthesis_environment_id"]
    assert len(pipeline.environment_factory.created_environment_ids) == 3
    assert sum(call["role"] == "quality_judge" for call in backend.calls) == 1


@pytest.mark.parametrize("data_type", ["multi_turn_miss_func", "multi_turn_miss_param"])
def test_final_judge_gets_missing_and_recovery_evidence(data_type):
    pipeline, backend = make_pipeline(data_type)
    result = run(pipeline, data_type)
    assert result.status == "SUCCEEDED", result.reason
    call = next(call for call in backend.calls if call["role"] == "quality_judge")
    summary = json.JSONDecoder().raw_decode(
        call["messages"][1]["content"].split("# Sample Data\n", 1)[1].lstrip()
    )[0]
    affected, recovery = summary["turns"][:2]
    assert affected["ground_truth"] == []
    assert affected["execution_results"] == []
    assert recovery["ground_truth"] == ["add(a=2.0, b=3.0)"]
    assert recovery["execution_results"]
    definitions = summary["verification_evidence"]
    names = {tool["name"] for tool in (
        definitions["used_tool_definitions"] + definitions["other_actor_tool_definitions"]
    )}
    assert names == {tool["name"] for tool in make_seed(data_type)["available_functions"]}
    if data_type == "multi_turn_miss_func":
        assert "add" not in affected["actor_visible_tool_names"]
        assert "add" in recovery["actor_visible_tool_names"]
        assert recovery["query"] == ""
        assert recovery["tools_added_before_turn"] == ["add"]
    else:
        assert "2.0" not in affected["query"]
        assert "2.0" in recovery["query"]
    guidance = call["messages"][0]["content"]
    assert "never the future recovery, latent narrative or hidden snapshots" in guidance


def test_missing_function_without_restoration_cannot_reach_judge():
    pipeline, backend = make_pipeline("multi_turn_miss_func")
    transform = pipeline.missing_function.transform

    async def omit_restoration(draft):
        output = await transform(draft)
        output.turns[1].recovery_tools = []
        return output

    pipeline.missing_function.transform = omit_restoration
    result = run(pipeline, "multi_turn_miss_func")
    assert result.status == "DROPPED"
    assert "tool_availability_gate" in result.reason
    assert not any(call["role"] == "quality_judge" for call in backend.calls)


def test_unsupported_diagnostic_cannot_mutate_judge_or_gt(monkeypatch):
    def diagnostic(draft, **kwargs):
        draft.turns[0].calls[0].arguments["a"] = 999999.0
        draft.structural_profile["diagnostic_mutation"] = True
        return GateResult("semantic_grounding_gate", False, "unsupported derivation")

    monkeypatch.setattr(pipeline_module, "semantic_grounding_gate", diagnostic)
    pipeline, backend = make_pipeline()
    result = run(pipeline)
    assert result.status == "SUCCEEDED", result.reason
    metadata = result.candidate["generation_metadata"]
    grounding = next(g for g in metadata["project_diagnostics"] if g["name"] == "semantic_grounding_gate")
    assert grounding["passed"] is False
    assert grounding["blocking"] is False
    assert "diagnostic_mutation" not in metadata["structural_profile"]
    gt = result.candidate["sample"]["extra_info"]["interaction_kwargs"]["ground_truth"]
    assert "a=2.0" in gt[0][0]
    judge_call = next(c for c in backend.calls if c["role"] == "quality_judge")
    assert "999999" not in str(judge_call)


def test_diagnostic_failure_is_recorded_without_blocking(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("diagnostic bug")

    monkeypatch.setattr(pipeline_module, "semantic_grounding_gate", broken)
    pipeline, _ = make_pipeline()
    result = run(pipeline)
    assert result.status == "SUCCEEDED", result.reason
    record = next(g for g in result.candidate["generation_metadata"]["project_diagnostics"]
                  if g["name"] == "semantic_grounding_gate")
    assert record["passed"] is None
    assert "diagnostic bug" in record["detail"]


@pytest.mark.parametrize("function,name", [
    ("fresh_vm_reverify_gate", "fresh_vm_gate"),
    ("tool_availability_gate", "tool_availability_gate"),
    ("parameter_complexity_gate", "parameter_complexity_gate"),
])
def test_each_core_rejection_blocks_judge(monkeypatch, function, name):
    monkeypatch.setattr(pipeline_module, function, lambda *a, **kw: GateResult(name, False, "invalid"))
    pipeline, backend = make_pipeline(replay_final_gt=function == "fresh_vm_reverify_gate")
    result = run(pipeline)
    assert result.status == "DROPPED"
    assert name in result.reason
    assert not any(c["role"] == "quality_judge" for c in backend.calls)


@pytest.mark.parametrize("second_accept", [True, False])
def test_judge_has_one_refinement_and_no_duplicate_final_verifier(second_accept):
    script = success_script()
    script["quality_judge"] = [REJECT, JUDGE_ACCEPT if second_accept else REJECT]
    script["refine_classify"] = ["<reason>Wording only.</reason><answer>query_fixable</answer>"]
    script["refine_rewrite"] = ["<answer>Could you calculate two plus three?</answer>"]
    pipeline, backend = make_pipeline(script=script)
    result = run(pipeline)
    assert result.status == ("SUCCEEDED" if second_accept else "DROPPED"), result.reason
    roles = [c["role"] for c in backend.calls]
    assert roles.count("quality_judge") == 2
    assert roles.count("refine_classify") == roles.count("refine_rewrite") == 1
    assert "final_query_verifier" not in roles
    if second_accept:
        metadata = result.candidate["generation_metadata"]
        assert metadata["refinement_used"] is True
        assert {g["phase"] for g in metadata["project_diagnostics"]} == {"before_judge", "after_refinement"}


def test_gt_unfixable_cannot_be_accepted_or_rewritten():
    script = success_script()
    script["quality_judge"] = [REJECT]
    script["refine_classify"] = ["<reason>Wrong GT.</reason><answer>gt_unfixable</answer>"]
    pipeline, backend = make_pipeline(script=script)
    result = run(pipeline)
    assert result.status == "DROPPED"
    assert "gt_unfixable" in result.reason
    assert not any(c["role"] == "refine_rewrite" for c in backend.calls)


def test_malformed_judge_response_cannot_be_accepted():
    script = success_script()
    script["quality_judge"] = ["looks good"]
    pipeline, _ = make_pipeline(script=script)
    result = run(pipeline)
    assert result.status == "DROPPED"
    assert result.candidate is None


def test_rods_judge_quota_defers_without_spending_construction_attempt():
    script = success_script()
    script["quality_judge"] = [BackendQuotaExceeded("account quota exhausted")]
    pipeline, backend = make_pipeline(script=script)
    checkpoints = []
    with pytest.raises(BackendQuotaExceeded):
        asyncio.run(pipeline.generate(make_seed(), checkpoint_callback=checkpoints.append))
    assert checkpoints[-1]["completed_failed_attempts"] == 0
    assert checkpoints[-1]["failures"] == []
    assert not any(c["role"] == "refine_classify" for c in backend.calls)


def test_strict_policy_retains_blocking_project_gate(monkeypatch):
    monkeypatch.setattr(pipeline_module, "semantic_grounding_gate",
                        lambda *a, **kw: GateResult("semantic_grounding_gate", False, "unsupported"))
    pipeline, backend = make_pipeline(policy="strict")
    result = run(pipeline)
    assert result.status == "DROPPED"
    assert "semantic_grounding_gate" in result.reason
    assert not any(c["role"] == "quality_judge" for c in backend.calls)
