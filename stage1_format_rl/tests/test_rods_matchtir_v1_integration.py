from __future__ import annotations

import asyncio
import numpy as np
import torch
from omegaconf import OmegaConf

from env_tuning.interaction.response_handler import ResponseHandler
from env_tuning.interaction.data_models import ExecutionResult, InstanceState
from env_tuning.interaction.new_multi_turn_fc import MultiTurnFunctionCallInteraction
from env_tuning.interaction.utils import parse_tool_call_objects, parse_tool_calls
from env_tuning.rods_matchtir_v1.provenance import response_relative_step
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import compute_advantage


def _step(call_name: str) -> dict:
    return {
        "user_turn_id": 0,
        "policy_step_id": 0,
        "response_type": "tool_call",
        "provenance_reliable": True,
        "actor_span": {"start": 0, "end": 2},
        "calls": [
            {
                "call_idx": 0,
                "name": call_name,
                "arguments": {"a": 1},
                "valid": True,
            }
        ],
    }


def _data(include_local: bool, weight: float = 1.0) -> tuple[DataProto, OmegaConf]:
    batch_size, response_length = 2, 6
    loss_mask = torch.zeros((batch_size, response_length))
    loss_mask[:, 0:2] = 1
    token_rewards = torch.zeros((batch_size, response_length))
    token_rewards[0, -1] = 1.0
    tensors = {
        "responses": torch.zeros((batch_size, response_length), dtype=torch.long),
        "response_mask": torch.ones((batch_size, response_length)),
        "loss_mask": loss_mask,
        "token_level_rewards": token_rewards,
    }
    provenances = np.array(
        [
            {"ground_truth": [["f(a=1)"]], "policy_steps": [_step("f")]},
            {"ground_truth": [["f(a=1)"]], "policy_steps": [_step("wrong")]},
        ],
        dtype=object,
    )
    non_tensors = {
        "uid": np.array(["q", "q"], dtype=object),
        "matchtir_provenance": provenances,
        "data_source": np.array(["multi_turn_base", "multi_turn_base"], dtype=object),
    }
    config_dict = {"use_kl_in_reward": False}
    if include_local:
        config_dict["matchtir_local"] = {
            "enabled": True,
            "weight": weight,
            "gamma": 0.9,
            "matching": "hard",
            "unmatched_penalty": 0.0,
            "min_group_size": 2,
        }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors), OmegaConf.create(config_dict)


def test_structured_parser_preserves_multi_call_grouping_for_execution_and_credit():
    raw = '[{"name":"f","arguments":{"a":1}}, {"name":"h","arguments":{"b":2}}]'
    calls = parse_tool_call_objects(raw)
    assert [(item["call_idx"], item["name"], item["arguments"]) for item in calls] == [
        (0, "f", {"a": 1}),
        (1, "h", {"b": 2}),
    ]
    assert parse_tool_calls(raw) == "[f(a=1), h(b=2)]"
    response = ResponseHandler().parse_and_validate(
        [{"role": "assistant", "content": f"<think>x</think><tool_call>{raw}</tool_call>"}]
    )
    assert response.is_valid and len(response.tool_calls) == 2


def test_absolute_actor_span_is_converted_without_touching_environment_tokens():
    step = {
        "provenance_reliable": True,
        "actor_token_start_absolute": 12,
        "actor_token_end_absolute": 16,
    }
    converted = response_relative_step(step, prompt_length=10, response_length=8)
    assert converted["actor_span"] == {"start": 2, "end": 6}
    assert converted["provenance_reliable"] is True


def test_trainer_computes_pure_rods_first_then_adds_local_residual():
    baseline_data, baseline_config = _data(include_local=False)
    baseline = compute_advantage(
        baseline_data,
        adv_estimator=AdvantageEstimator.GRPO,
        multi_turn=True,
        config=baseline_config,
    )
    baseline_adv = baseline.batch["advantages"].clone()

    local_data, local_config = _data(include_local=True)
    fused = compute_advantage(
        local_data,
        adv_estimator=AdvantageEstimator.GRPO,
        multi_turn=True,
        config=local_config,
    )
    local_adv = fused.batch["matchtir_local_advantages"]
    assert torch.equal(fused.batch["advantages"], baseline_adv + local_adv)
    assert torch.count_nonzero(local_adv[:, 2:]) == 0
    assert "rods_matchtir_v1_metrics" in fused.meta_info


def test_trainer_weight_zero_is_exact_original_grpo_fallback():
    baseline_data, baseline_config = _data(include_local=False)
    baseline = compute_advantage(
        baseline_data,
        adv_estimator=AdvantageEstimator.GRPO,
        multi_turn=True,
        config=baseline_config,
    )
    zero_data, zero_config = _data(include_local=True, weight=0.0)
    zero = compute_advantage(
        zero_data,
        adv_estimator=AdvantageEstimator.GRPO,
        multi_turn=True,
        config=zero_config,
    )
    assert torch.equal(zero.batch["advantages"], baseline.batch["advantages"])
    assert torch.equal(zero.batch["returns"], baseline.batch["returns"])


def test_interaction_provenance_resets_after_missing_then_recovers_normal_turn(monkeypatch):
    interaction = MultiTurnFunctionCallInteraction(
        {"name": "multi_turn_tool_call", "is_augmented": False}
    )
    state = InstanceState(
        initial_config={},
        involved_classes=[],
        ground_truth=[[], ["f(a=1)"]],
        processed_question=["next question"],
        question=["missing question", "normal question"],
        involved_instances={},
        total_turns=2,
    )
    interaction._instance_dict["request"] = state
    missing_result = asyncio.run(
        interaction.generate_response(
            "request",
            [{"role": "assistant", "content": "<think>x</think><answer>clarify</answer>"}],
            id="sample-id",
        )
    )
    missing_step = missing_result[3]["rods_matchtir_v1_step"]
    assert missing_step["user_turn_id"] == 0
    assert missing_step["policy_step_id"] == 0
    assert missing_step["is_missing_ground_truth"] is True
    assert state.current_turn_index == 1

    monkeypatch.setattr(
        interaction,
        "_execute_function_calls",
        lambda *args, **kwargs: ExecutionResult([], {}, False, True, []),
    )
    monkeypatch.setattr(
        interaction,
        "_determine_next_action",
        lambda *args, **kwargs: (False, "observation", -1.0, {}),
    )
    normal_result = asyncio.run(
        interaction.generate_response(
            "request",
            [
                {
                    "role": "assistant",
                    "content": '<think>x</think><tool_call>{"name":"f","arguments":{"a":1}}</tool_call>',
                }
            ],
            id="sample-id",
        )
    )
    normal_step = normal_result[3]["rods_matchtir_v1_step"]
    assert normal_step["user_turn_id"] == 1
    assert normal_step["policy_step_id"] == 0
    assert normal_step["is_missing_ground_truth"] is False
    assert normal_step["calls"] == [
        {"call_idx": 0, "name": "f", "arguments": {"a": 1}, "valid": True}
    ]
