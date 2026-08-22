from __future__ import annotations

import pytest
import torch

from env_tuning.rods_matchtir_v1.advantage import (
    LocalCreditConfig,
    fuse_rods_and_local_advantages,
)


def tool_step(user_turn: int, policy_step: int, start: int, calls: list[tuple[str, dict]]) -> dict:
    return {
        "user_turn_id": user_turn,
        "policy_step_id": policy_step,
        "response_type": "tool_call",
        "provenance_reliable": True,
        "actor_span": {"start": start, "end": start + 2},
        "calls": [
            {"call_idx": index, "name": name, "arguments": arguments, "valid": True}
            for index, (name, arguments) in enumerate(calls)
        ],
    }


def answer_step(user_turn: int, policy_step: int, start: int) -> dict:
    return {
        "user_turn_id": user_turn,
        "policy_step_id": policy_step,
        "response_type": "answer",
        "provenance_reliable": True,
        "actor_span": {"start": start, "end": start + 2},
        "calls": [],
    }


def run_batch(
    provenances: list[dict],
    *,
    rods_scalars: list[float] | None = None,
    response_length: int = 16,
    config: LocalCreditConfig | None = None,
):
    batch_size = len(provenances)
    rods_scalars = rods_scalars or [0.25] * batch_size
    actor_mask = torch.zeros((batch_size, response_length), dtype=torch.float32)
    for row, provenance in enumerate(provenances):
        for step in provenance["policy_steps"]:
            span = step["actor_span"]
            actor_mask[row, span["start"] : span["end"]] = 1
    rods = torch.zeros_like(actor_mask)
    for row, scalar in enumerate(rods_scalars):
        rods[row][actor_mask[row].bool()] = scalar
    rewards = torch.zeros_like(rods)
    for row in range(batch_size):
        rewards[row, -1] = float(row % 2)
    rewards_before = rewards.clone()
    result = fuse_rods_and_local_advantages(
        rods_advantages=rods,
        rods_returns=rods.clone(),
        token_level_rewards=rewards,
        actor_response_mask=actor_mask,
        uids=["same-prompt"] * batch_size,
        rollout_provenance=provenances,
        data_sources=["multi_turn_base"] * batch_size,
        config=config or LocalCreditConfig(),
    )
    assert torch.equal(rewards, rewards_before), "local credit must never mutate global rewards"
    return result, rods, actor_mask


def test_07_ragged_depth_uses_only_existing_steps_and_zeros_singleton():
    gt = [["f(a=1)", "h(b=2)", "i(c=3)"]]
    provenances = [
        {
            "ground_truth": gt,
            "policy_steps": [
                tool_step(0, 0, 0, [("f", {"a": 1})]),
                tool_step(0, 1, 2, [("h", {"b": 2})]),
                tool_step(0, 2, 4, [("i", {"c": 3})]),
            ],
        },
        {
            "ground_truth": gt,
            "policy_steps": [
                tool_step(0, 0, 0, [("f", {"a": 1})]),
                tool_step(0, 1, 2, [("wrong", {})]),
            ],
        },
        {
            "ground_truth": gt,
            "policy_steps": [tool_step(0, 0, 0, [("wrong", {})])],
        },
    ]
    result, _, _ = run_batch(provenances)
    by_batch_depth = {(r["batch_index"], r["depth"]): r for r in result.step_records}
    assert all(by_batch_depth[(row, 0)]["local_active"] for row in range(3))
    assert all(by_batch_depth[(row, 1)]["local_active"] for row in range(2))
    assert by_batch_depth[(0, 2)]["local_active"] is False
    assert by_batch_depth[(0, 2)]["local_advantage"] == 0.0
    assert result.metrics["rods_matchtir_v1/normalization/local_support_depth_0"] == 3.0
    assert result.metrics["rods_matchtir_v1/normalization/local_support_depth_1"] == 2.0
    assert result.metrics["rods_matchtir_v1/normalization/local_support_depth_2"] == 1.0


def test_08_zero_variance_local_group_is_safely_zero():
    provenance = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [tool_step(0, 0, 0, [("f", {"a": 1})])],
    }
    result, _, _ = run_batch([provenance, provenance])
    assert torch.isfinite(result.local_advantages).all()
    assert torch.count_nonzero(result.local_advantages) == 0
    assert result.metrics["rods_matchtir_v1/normalization/zero_variance_local_count"] == 1.0
    assert all(record["local_active"] is False for record in result.step_records)


def test_09_missing_parameter_turn_has_no_local_and_preserves_rods():
    provenances = [
        {
            "ground_truth": [[]],
            "policy_steps": [tool_step(0, 0, 0, [("invented", {"x": 1})])],
        },
        {
            "ground_truth": [[]],
            "policy_steps": [answer_step(0, 0, 0)],
        },
    ]
    result, rods, _ = run_batch(provenances, rods_scalars=[0.7, -0.2])
    assert result.step_records == ()
    assert torch.equal(result.advantages, rods)
    assert result.metrics["rods_matchtir_v1/missing/missing_turn_local_coverage"] == 0.0


def test_10_missing_mask_does_not_leak_to_next_normal_turn():
    gt = [[], ["f(a=1)"]]
    provenances = [
        {
            "ground_truth": gt,
            "policy_steps": [
                answer_step(0, 0, 0),
                tool_step(1, 0, 2, [("f", {"a": 1})]),
            ],
        },
        {
            "ground_truth": gt,
            "policy_steps": [
                answer_step(0, 0, 0),
                tool_step(1, 0, 2, [("wrong", {})]),
            ],
        },
    ]
    result, rods, _ = run_batch(provenances, rods_scalars=[0.4, -0.4])
    assert all(record["user_turn_id"] == 1 for record in result.step_records)
    assert all(record["local_active"] for record in result.step_records)
    assert torch.equal(result.advantages[:, 0:2], rods[:, 0:2])
    assert result.metrics["rods_matchtir_v1/missing/missing_turn_local_coverage"] == 0.0
    assert result.metrics["rods_matchtir_v1/missing/followup_normal_turn_local_coverage"] == 1.0


def test_11_discounted_return_is_truncated_at_user_turn_boundary():
    provenance = {
        "ground_truth": [["f(a=1)"], ["h(b=2)"]],
        "policy_steps": [
            tool_step(0, 0, 0, [("f", {"a": 1})]),
            tool_step(1, 0, 2, [("h", {"b": 2})]),
        ],
    }
    result, _, _ = run_batch([provenance])
    assert [record["local_return"] for record in result.step_records] == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    "config",
    [
        LocalCreditConfig(enabled=False),
        LocalCreditConfig(enabled=True, weight=0.0),
    ],
)
def test_12_rods_exact_fallback(config):
    provenance = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [tool_step(0, 0, 0, [("f", {"a": 1})])],
    }
    result, rods, _ = run_batch([provenance, provenance], config=config)
    assert result.advantages is rods
    assert torch.equal(result.advantages, rods)
    assert torch.count_nonzero(result.local_advantages) == 0


def test_14_environment_tokens_never_receive_local_advantage():
    # Deliberately make the provenance span wider than the actor mask.  Fusion
    # must intersect with the existing EnvTuning loss mask.
    exact = tool_step(0, 0, 0, [("f", {"a": 1})])
    wrong = tool_step(0, 0, 0, [("wrong", {})])
    exact["actor_span"]["end"] = 4
    wrong["actor_span"]["end"] = 4
    provenances = [
        {"ground_truth": [["f(a=1)"]], "policy_steps": [exact]},
        {"ground_truth": [["f(a=1)"]], "policy_steps": [wrong]},
    ]
    actor_mask = torch.zeros((2, 8), dtype=torch.float32)
    actor_mask[:, 0:2] = 1  # tokens 2:4 model an environment/tool observation
    rods = actor_mask.clone()
    rewards = torch.zeros_like(rods)
    result = fuse_rods_and_local_advantages(
        rods_advantages=rods,
        rods_returns=rods.clone(),
        token_level_rewards=rewards,
        actor_response_mask=actor_mask,
        uids=["q", "q"],
        rollout_provenance=provenances,
        data_sources=["multi_turn_base", "multi_turn_base"],
        config=LocalCreditConfig(),
    )
    assert torch.count_nonzero(result.local_advantages[:, 2:]) == 0
    assert torch.count_nonzero(result.local_token_mask[:, 2:]) == 0
    assert torch.equal(result.advantages[:, 2:], rods[:, 2:])


def test_15_multi_call_provenance_flattens_once_and_maps_back_to_steps():
    gt = [["f(a=1)", "h(b=2)"]]
    provenances = [
        {
            "ground_truth": gt,
            "policy_steps": [
                tool_step(0, 0, 0, [("f", {"a": 1}), ("h", {"b": 2})]),
                tool_step(0, 1, 2, [("f", {"a": 1})]),  # duplicate, must be unmatched
            ],
        },
        {
            "ground_truth": gt,
            "policy_steps": [
                tool_step(0, 0, 0, [("wrong", {}), ("h", {"b": 2})]),
                tool_step(0, 1, 2, [("f", {"a": 1})]),
            ],
        },
    ]
    result, _, _ = run_batch(provenances)
    first_rollout = [record for record in result.step_records if record["batch_index"] == 0]
    assert first_rollout[0]["call_count"] == 2
    assert first_rollout[0]["call_rewards"] == pytest.approx([1.0, 1.0])
    assert first_rollout[0]["step_reward"] == 1.0
    assert first_rollout[1]["call_rewards"] == [0.0]
    assert first_rollout[1]["step_reward"] == 0.0


def test_unreliable_tool_step_fails_closed_for_whole_user_turn():
    reliable = tool_step(0, 0, 0, [("f", {"a": 1})])
    unreliable = tool_step(0, 1, 2, [("f", {"a": 1})])
    unreliable["provenance_reliable"] = False
    provenances = [
        {
            "ground_truth": [["f(a=1)"]],
            "policy_steps": [reliable, unreliable],
        },
        {
            "ground_truth": [["f(a=1)"]],
            "policy_steps": [tool_step(0, 0, 0, [("wrong", {})])],
        },
    ]
    result, rods, _ = run_batch(provenances)
    assert all(record["batch_index"] == 1 for record in result.step_records)
    assert torch.equal(result.advantages, rods)  # remaining support is singleton
    assert result.metrics["rods_matchtir_v1/provenance/unreliable_tool_turn_count"] == 1.0


def test_required_diagnostic_metric_schema_is_emitted():
    provenances = [
        {
            "ground_truth": [["f(a=1)"]],
            "policy_steps": [tool_step(0, 0, 0, [("f", {"a": 1})])],
        },
        {
            "ground_truth": [["f(a=1)"]],
            "policy_steps": [tool_step(0, 0, 0, [("wrong", {})])],
        },
    ]
    result, _, _ = run_batch(provenances)
    required_suffixes = {
        "global/progress_reward_mean",
        "global/progress_reward_std",
        "global/A_RODS_mean",
        "global/A_RODS_std",
        "call/num_predicted_calls",
        "call/num_gt_calls",
        "call/matched_call_count",
        "call/unmatched_call_count",
        "call/match_rate",
        "call/similarity_mean",
        "call/similarity_max",
        "step/policy_steps_per_user_turn",
        "step/calls_per_policy_step",
        "step/step_reward_mean",
        "step/local_return_mean",
        "step/local_return_std",
        "interaction/non_answer_runtime_interactions",
        "interaction/tool_attempt_interactions",
        "interaction/parsed_tool_interactions",
        "interaction/parse_error_tool_attempts",
        "interaction/parse_error_runtime_interactions",
        "interaction/unparsed_runtime_interactions",
        "interaction/peer_supported_interactions",
        "interaction/unsupported_local_interactions",
        "interaction/nonzero_local_advantages",
        "interaction/parse_error_interactions_receiving_local_residual",
        "interaction/parse_error_tokens_receiving_local_residual",
        "normalization/local_support_size",
        "normalization/singleton_local_count",
        "normalization/singleton_local_rate",
        "normalization/zero_variance_local_count",
        "normalization/local_adv_mean",
        "normalization/local_adv_std",
        "normalization/local_coverage",
        "fusion/A_new_mean",
        "fusion/A_new_std",
        "fusion/RMS_A_RODS",
        "fusion/RMS_A_local",
        "fusion/RMS_A_new",
        "fusion/sign_flip_rate",
    }
    assert {
        key.removeprefix("rods_matchtir_v1/") for key in result.metrics
    }.issuperset(required_suffixes)
