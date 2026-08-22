from __future__ import annotations

import asyncio

import pytest
import torch

from env_tuning.interaction.data_models import (
    AttemptedActionType,
    ExecutionResult,
    InstanceState,
)
from env_tuning.interaction.new_multi_turn_fc import MultiTurnFunctionCallInteraction
from env_tuning.interaction.response_handler import ResponseHandler
from env_tuning.rods_matchtir_v1.advantage import (
    LocalCreditConfig,
    fuse_rods_and_local_advantages,
)
from env_tuning.rods_matchtir_v1.provenance import response_relative_step
from verl.workers.rollout.schemas import AsyncRolloutRequest


def runtime_tool_interaction(
    user_turn: int,
    runtime_index: int,
    tool_attempt_index: int,
    start: int,
    calls: list[tuple[str, dict]] | None = None,
    *,
    parse_error: bool = False,
    temporal_reliable: bool = True,
    actor_span_reliable: bool = True,
) -> dict:
    calls = calls or []
    return {
        "local_credit_semantics": "runtime_interaction_final",
        "user_turn_id": user_turn,
        "policy_step_id": runtime_index,
        "runtime_interaction_index": runtime_index,
        "tool_attempt_index": tool_attempt_index,
        "response_type": "parse_error" if parse_error else "tool_call",
        "attempted_action_type": "tool_call",
        "action_classification_reliable": True,
        "temporal_provenance_reliable": temporal_reliable,
        "actor_span_reliable": actor_span_reliable,
        "call_parse_reliable": not parse_error,
        "provenance_reliable": not parse_error,
        "actor_span": {"start": start, "end": start + 2},
        "calls": [
            {
                "call_idx": index,
                "name": name,
                "arguments": arguments,
                "valid": True,
            }
            for index, (name, arguments) in enumerate(calls)
        ],
    }


def runtime_answer(user_turn: int, runtime_index: int, start: int) -> dict:
    return {
        "local_credit_semantics": "runtime_interaction_final",
        "user_turn_id": user_turn,
        "policy_step_id": runtime_index,
        "runtime_interaction_index": runtime_index,
        "tool_attempt_index": None,
        "response_type": "answer",
        "attempted_action_type": "answer",
        "action_classification_reliable": True,
        "temporal_provenance_reliable": True,
        "actor_span_reliable": True,
        "call_parse_reliable": False,
        "provenance_reliable": True,
        "actor_span": {"start": start, "end": start + 2},
        "calls": [],
    }


def run_runtime_batch(
    provenances: list[dict],
    *,
    rods_scalars: list[float] | None = None,
    response_length: int = 24,
):
    batch_size = len(provenances)
    rods_scalars = rods_scalars or [0.25] * batch_size
    actor_mask = torch.zeros((batch_size, response_length), dtype=torch.float32)
    for row, provenance in enumerate(provenances):
        for step in provenance["policy_steps"]:
            span = step.get("actor_span")
            if span:
                actor_mask[row, span["start"] : span["end"]] = 1
    rods = torch.zeros_like(actor_mask)
    for row, scalar in enumerate(rods_scalars):
        rods[row][actor_mask[row].bool()] = scalar
    token_rewards = torch.zeros_like(rods)
    result = fuse_rods_and_local_advantages(
        rods_advantages=rods,
        rods_returns=rods.clone(),
        token_level_rewards=token_rewards,
        actor_response_mask=actor_mask,
        uids=["same-prompt"] * batch_size,
        rollout_provenance=provenances,
        data_sources=["multi_turn_base"] * batch_size,
        config=LocalCreditConfig(),
    )
    return result, rods, actor_mask


def test_a_one_valid_interaction_one_correct_call_has_reward_one():
    provenance = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1})])],
    }
    result, _, _ = run_runtime_batch([provenance])
    assert len(result.step_records) == 1
    assert result.step_records[0]["call_rewards"] == [1.0]
    assert result.step_records[0]["step_reward"] == 1.0


def test_b_two_calls_in_one_action_are_one_temporal_step_and_average_once():
    provenance = {
        "ground_truth": [["f(a=1)", "h(b=2)"]],
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1}), ("h", {"b": 2})])
        ],
    }
    result, _, _ = run_runtime_batch([provenance])
    assert len(result.step_records) == 1
    assert result.step_records[0]["call_count"] == 2
    assert result.step_records[0]["call_rewards"] == [1.0, 1.0]
    assert result.step_records[0]["step_reward"] == 1.0


def test_c_parsed_extra_call_is_matched_once_and_receives_unmatched_zero():
    provenance = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1}), ("extra", {"x": 2})])
        ],
    }
    result, _, _ = run_runtime_batch([provenance])
    assert result.step_records[0]["call_rewards"] == [1.0, 0.0]
    assert result.step_records[0]["step_reward"] == 0.5


def test_d_parse_error_does_not_enter_matching_but_remains_zero_reward_event():
    provenance = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [runtime_tool_interaction(0, 0, 0, 0, parse_error=True)],
    }
    result, _, _ = run_runtime_batch([provenance])
    assert len(result.step_records) == 1
    record = result.step_records[0]
    assert record["response_type"] == "parse_error"
    assert record["call_count"] == 0
    assert record["call_rewards"] == []
    assert record["step_reward"] == 0.0
    assert result.metrics["rods_matchtir_v1/call/num_predicted_calls"] == 0.0


def test_e_parse_parse_valid_sequence_discounts_over_zero_reward_attempts():
    provenance = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, parse_error=True),
            runtime_tool_interaction(0, 1, 1, 2, parse_error=True),
            runtime_tool_interaction(0, 2, 2, 4, [("f", {"a": 1})]),
        ],
    }
    result, _, _ = run_runtime_batch([provenance])
    assert [record["step_reward"] for record in result.step_records] == [0.0, 0.0, 1.0]
    assert [record["local_return"] for record in result.step_records] == pytest.approx(
        [0.81, 0.9, 1.0]
    )


def test_e2_valid_parse_error_valid_uses_real_runtime_discount_distance():
    provenance = {
        "ground_truth": [["f(a=1)", "h(b=2)"]],
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1})]),
            runtime_tool_interaction(0, 1, 1, 2, parse_error=True),
            runtime_tool_interaction(0, 2, 2, 4, [("h", {"b": 2})]),
        ],
    }
    result, _, _ = run_runtime_batch([provenance])
    assert [record["step_reward"] for record in result.step_records] == [1.0, 0.0, 1.0]
    assert [record["local_return"] for record in result.step_records] == pytest.approx(
        [1.81, 0.9, 1.0]
    )


def test_f_discounted_return_resets_at_bfcl_user_turn_boundary():
    provenance = {
        "ground_truth": [["f(a=1)"], ["h(b=2)"]],
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, parse_error=True),
            runtime_tool_interaction(1, 0, 0, 2, [("h", {"b": 2})]),
        ],
    }
    result, _, _ = run_runtime_batch([provenance])
    assert [record["local_return"] for record in result.step_records] == [0.0, 1.0]


def test_g_ragged_peers_never_zero_pad_missing_late_runtime_interactions():
    gt = [["f(a=1)"]]
    long_rollout = {
        "ground_truth": gt,
        "policy_steps": [
            *[runtime_tool_interaction(0, index, index, index * 2, parse_error=True) for index in range(5)],
            runtime_tool_interaction(0, 5, 5, 10, [("f", {"a": 1})]),
        ],
    }
    short_rollout = {
        "ground_truth": gt,
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1})]),
            runtime_tool_interaction(0, 1, 1, 2, parse_error=True),
        ],
    }
    result, _, _ = run_runtime_batch([long_rollout, short_rollout])
    records = {
        (r["batch_index"], r["runtime_interaction_index"]): r
        for r in result.step_records
    }
    assert records[(0, 0)]["peer_support"] == 2
    assert records[(0, 1)]["peer_support"] == 2
    for depth in range(2, 6):
        assert records[(0, depth)]["peer_support"] == 1
        assert records[(0, depth)]["local_advantage"] == 0.0


def test_h_singleton_support_abstains_from_local_credit():
    provenance = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1})])],
    }
    result, rods, _ = run_runtime_batch([provenance])
    assert result.step_records[0]["peer_support"] == 1
    assert result.step_records[0]["local_advantage"] == 0.0
    assert torch.equal(result.advantages, rods)


def test_i_zero_sample_std_abstains_from_local_credit():
    provenance = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1})])],
    }
    result, _, _ = run_runtime_batch([provenance, provenance])
    assert all(record["peer_sample_std"] == 0.0 for record in result.step_records)
    assert all(record["local_advantage"] == 0.0 for record in result.step_records)


def test_j_answer_is_not_a_trailing_local_runtime_interaction():
    provenance = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1})]),
            runtime_answer(0, 1, 2),
        ],
    }
    result, _, _ = run_runtime_batch([provenance])
    assert len(result.step_records) == 1
    assert result.step_records[0]["tool_attempt_index"] == 0
    assert torch.count_nonzero(result.local_advantages[:, 2:4]) == 0


def test_answer_tokens_remain_global_only_when_local_credit_is_active():
    exact = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1})]),
            runtime_answer(0, 1, 2),
        ],
    }
    wrong = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, [("wrong", {})]),
            runtime_answer(0, 1, 2),
        ],
    }
    result, rods, _ = run_runtime_batch([exact, wrong], rods_scalars=[0.4, -0.2])
    assert bool(result.local_token_mask[:, 0:2].all())
    assert not bool(result.local_token_mask[:, 2:4].any())
    assert torch.equal(result.advantages[:, 2:4], rods[:, 2:4])


def test_unknown_malformed_non_answer_remains_a_zero_reward_runtime_step():
    unknown = {
        "local_credit_semantics": "runtime_interaction_final",
        "user_turn_id": 0,
        "policy_step_id": 0,
        "runtime_interaction_index": 0,
        "tool_attempt_index": None,
        "response_type": "parse_error",
        "attempted_action_type": "unknown",
        "action_classification_reliable": False,
        "temporal_provenance_reliable": True,
        "actor_span_reliable": True,
        "call_parse_reliable": False,
        "provenance_reliable": False,
        "actor_span": {"start": 0, "end": 2},
        "calls": [],
    }
    provenance = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [
            unknown,
            runtime_tool_interaction(0, 1, 0, 2, [("f", {"a": 1})]),
        ],
    }
    result, _, _ = run_runtime_batch([provenance])
    assert len(result.step_records) == 2
    assert [record["runtime_interaction_index"] for record in result.step_records] == [0, 1]
    assert [record["tool_attempt_index"] for record in result.step_records] == [None, 0]
    assert [record["step_reward"] for record in result.step_records] == [0.0, 1.0]
    assert [record["local_return"] for record in result.step_records] == pytest.approx(
        [0.9, 1.0]
    )
    assert torch.count_nonzero(result.local_advantages[:, 0:2]) == 0


def test_normalization_uses_runtime_index_not_tool_attempt_index():
    exact = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [runtime_tool_interaction(0, 0, 17, 0, [("f", {"a": 1})])],
    }
    wrong = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [runtime_tool_interaction(0, 0, 91, 0, [("wrong", {})])],
    }
    result, _, _ = run_runtime_batch([exact, wrong])
    assert [record["peer_support"] for record in result.step_records] == [2, 2]
    assert result.step_records[0]["local_advantage"] > 0
    assert result.step_records[1]["local_advantage"] < 0


def test_execution_failed_but_successfully_parsed_call_still_enters_matching():
    step = runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1})])
    step["environment_execution_failed"] = True
    result, _, _ = run_runtime_batch(
        [{"ground_truth": [["f(a=1)"]], "policy_steps": [step]}]
    )
    assert result.step_records[0]["call_rewards"] == [1.0]
    assert result.step_records[0]["step_reward"] == 1.0


def test_k_empty_ground_truth_is_global_only():
    provenance = {
        "ground_truth": [[]],
        "policy_steps": [runtime_tool_interaction(0, 0, 0, 0, [("invented", {"x": 1})])],
    }
    result, rods, _ = run_runtime_batch([provenance])
    assert result.step_records == ()
    assert torch.equal(result.advantages, rods)


def test_l_unreliable_temporal_provenance_fails_closed():
    unreliable = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [
            runtime_tool_interaction(
                0, 0, 0, 0, [("f", {"a": 1})], temporal_reliable=False
            )
        ],
    }
    reliable = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [runtime_tool_interaction(0, 0, 0, 0, [("wrong", {})])],
    }
    result, rods, _ = run_runtime_batch([unreliable, reliable])
    assert all(record["batch_index"] == 1 for record in result.step_records)
    assert result.metrics["rods_matchtir_v1/provenance/unreliable_tool_turn_count"] == 1.0
    assert torch.equal(result.advantages, rods)


def test_m_unreliable_actor_span_keeps_timeline_but_writes_no_token_residual():
    exact = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [
            runtime_tool_interaction(
                0, 0, 0, 0, [("f", {"a": 1})], actor_span_reliable=False
            )
        ],
    }
    wrong = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [runtime_tool_interaction(0, 0, 0, 0, [("wrong", {})])],
    }
    result, rods, _ = run_runtime_batch([exact, wrong])
    first = next(record for record in result.step_records if record["batch_index"] == 0)
    assert first["local_active"] is True
    assert first["local_advantage"] > 0
    assert first["token_residual_applied"] is False
    assert torch.equal(result.advantages[0], rods[0])
    assert torch.count_nonzero(result.local_token_mask[0]) == 0


def test_n_parse_error_actor_tokens_receive_negative_supported_local_residual():
    delayed = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, parse_error=True),
            runtime_tool_interaction(0, 1, 1, 2, [("f", {"a": 1})]),
        ],
    }
    direct = {
        "ground_truth": [["f(a=1)"]],
        "policy_steps": [
            runtime_tool_interaction(0, 0, 0, 0, [("f", {"a": 1})]),
            runtime_answer(0, 1, 2),
        ],
    }
    result, _, _ = run_runtime_batch([delayed, direct])
    parse_record = next(
        record
        for record in result.step_records
        if record["batch_index"] == 0 and record["runtime_interaction_index"] == 0
    )
    assert parse_record["response_type"] == "parse_error"
    assert parse_record["local_advantage"] < 0
    assert parse_record["token_residual_applied"] is True
    assert bool((result.local_advantages[0, 0:2] < 0).all())
    assert (
        result.metrics[
            "rods_matchtir_v1/interaction/parse_error_interactions_receiving_local_residual"
        ]
        == 1.0
    )


@pytest.mark.parametrize(
    "raw,error_code",
    [
        (
            '<tool_call>{"name":"f","arguments":{}}</tool_call>',
            "missing_think",
        ),
        (
            '<think>x</think><tool_call>{"name":"f","arguments":{}}</tool_call>'
            '<tool_call>{"name":"h","arguments":{}}</tool_call>',
            "multiple_tool_call_blocks",
        ),
        ("<think>x</think><tool_call>{bad}</tool_call>", "invalid_tool_json"),
        (
            '<think>x</think><tool_call>{"name":"f","arguments":{}}</tool_call>outside',
            "outside_required_tags",
        ),
    ],
)
def test_parser_classifies_rejected_complete_tool_blocks_without_substring_heuristic(
    raw, error_code
):
    parsed = ResponseHandler().parse_and_validate(
        [{"role": "assistant", "content": raw}]
    )
    assert parsed.attempted_action_type is AttemptedActionType.TOOL_CALL
    assert parsed.action_classification_reliable is True
    assert parsed.call_parse_reliable is False
    assert parsed.parser_error_code == error_code


def test_parser_fails_closed_for_mixed_action_kinds():
    parsed = ResponseHandler().parse_and_validate(
        [
            {
                "role": "assistant",
                "content": (
                    '<think>x</think><tool_call>{"name":"f","arguments":{}}</tool_call>'
                    "<answer>done</answer>"
                ),
            }
        ]
    )
    assert parsed.attempted_action_type is AttemptedActionType.UNKNOWN
    assert parsed.action_classification_reliable is False


def test_response_relative_span_does_not_collapse_parse_failure_into_temporal_failure():
    converted = response_relative_step(
        {
            "temporal_provenance_reliable": True,
            "action_classification_reliable": True,
            "call_parse_reliable": False,
            "actor_span_reliable": True,
            "provenance_reliable": False,
            "actor_token_start_absolute": 12,
            "actor_token_end_absolute": 16,
        },
        prompt_length=10,
        response_length=8,
    )
    assert converted["actor_span"] == {"start": 2, "end": 6}
    assert converted["temporal_provenance_reliable"] is True
    assert converted["actor_span_reliable"] is True
    assert converted["call_parse_reliable"] is False
    assert converted["provenance_reliable"] is False


def test_rollout_schema_binds_parse_error_to_latest_actor_span_without_changing_temporal_semantics():
    request = AsyncRolloutRequest.model_construct(
        request_id="rollout",
        rollout_offset=3,
        assistant_token_spans=[{"start": 12, "end": 18}],
        matchtir_policy_steps=[],
    )
    request.add_matchtir_policy_step(
        {
            "response_type": "parse_error",
            "attempted_action_type": "tool_call",
            "temporal_provenance_reliable": True,
            "call_parse_reliable": False,
            "provenance_reliable": False,
        }
    )
    step = request.matchtir_policy_steps[0]
    assert step["actor_span_reliable"] is True
    assert step["actor_token_start_absolute"] == 12
    assert step["actor_token_end_absolute"] == 18
    assert step["temporal_provenance_reliable"] is True
    assert step["provenance_reliable"] is False


def test_runtime_separates_generation_index_from_tool_attempt_index_and_resets(monkeypatch):
    interaction = MultiTurnFunctionCallInteraction(
        {"name": "multi_turn_tool_call", "is_augmented": False}
    )
    state = InstanceState(
        initial_config={},
        involved_classes=[],
        ground_truth=[["f(a=1)"], ["h(b=2)"]],
        processed_question=["next"],
        question=["first", "second"],
        involved_instances={},
        total_turns=2,
    )
    interaction._instance_dict["request"] = state

    malformed = asyncio.run(
        interaction.generate_response(
            "request",
            [{"role": "assistant", "content": "<think>x</think><tool_call>{bad}</tool_call>"}],
            id="sample",
        )
    )[3]["rods_matchtir_v1_step"]
    assert malformed["runtime_interaction_index"] == 0
    assert malformed["tool_attempt_index"] == 0
    assert malformed["temporal_provenance_reliable"] is True
    assert malformed["call_parse_reliable"] is False

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
    valid = asyncio.run(
        interaction.generate_response(
            "request",
            [
                {
                    "role": "assistant",
                    "content": '<think>x</think><tool_call>{"name":"f","arguments":{"a":1}}</tool_call>',
                }
            ],
            id="sample",
        )
    )[3]["rods_matchtir_v1_step"]
    assert valid["runtime_interaction_index"] == 1
    assert valid["tool_attempt_index"] == 1

    # A valid answer is another runtime generation but is not a local attempt;
    # advancing the BFCL turn resets both counters for the next user turn.
    monkeypatch.setattr(
        interaction,
        "_handle_special_cases",
        lambda *args, **kwargs: interaction.turn_manager.advance_to_next_turn(state, "sample"),
    )
    answer = asyncio.run(
        interaction.generate_response(
            "request",
            [{"role": "assistant", "content": "<think>x</think><answer>done</answer>"}],
            id="sample",
        )
    )[3]["rods_matchtir_v1_step"]
    assert answer["runtime_interaction_index"] == 2
    assert answer["tool_attempt_index"] is None
    assert state.current_turn_policy_step_count == 0
    assert state.current_turn_tool_attempt_count == 0
