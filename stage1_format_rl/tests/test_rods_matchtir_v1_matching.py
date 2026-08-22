from __future__ import annotations

import math

import pytest
import torch

from env_tuning.rods_matchtir_v1.advantage import (
    LocalCreditConfig,
    fuse_rods_and_local_advantages,
)
from env_tuning.rods_matchtir_v1.matching import (
    CanonicalToolCall,
    hard_match_calls,
    matchtir_similarity,
)


def call(name: str, arguments: dict) -> CanonicalToolCall:
    return CanonicalToolCall(name=name, arguments=arguments)


def policy_step(user_turn: int, policy_step: int, calls: list[dict], start: int) -> dict:
    return {
        "user_turn_id": user_turn,
        "policy_step_id": policy_step,
        "response_type": "tool_call",
        "provenance_reliable": True,
        "actor_span": {"start": start, "end": start + 2},
        "calls": [
            {"call_idx": index, "name": item["name"], "arguments": item["arguments"], "valid": True}
            for index, item in enumerate(calls)
        ],
    }


def run_single(provenance: dict, response_length: int = 12):
    actor_mask = torch.zeros((1, response_length), dtype=torch.float32)
    for step in provenance["policy_steps"]:
        span = step["actor_span"]
        actor_mask[0, span["start"] : span["end"]] = 1
    rods = actor_mask.clone() * 0.25
    rewards = torch.zeros_like(rods)
    rewards[0, -1] = 0.5
    return fuse_rods_and_local_advantages(
        rods_advantages=rods,
        rods_returns=rods.clone(),
        token_level_rewards=rewards,
        actor_response_mask=actor_mask,
        uids=["prompt-0"],
        rollout_provenance=[provenance],
        data_sources=["multi_turn_base"],
        config=LocalCreditConfig(),
    )


def test_01_exact_call_match():
    assert matchtir_similarity(call("lookup", {"query": "x"}), call("lookup", {"query": "x"})) == 1.0


def test_02_wrong_tool_name_blocks_positive_reward():
    similarity = matchtir_similarity(call("search", {"query": "x"}), call("lookup", {"query": "x"}))
    assert similarity == 0.0
    result = hard_match_calls([call("search", {"query": "x"})], [call("lookup", {"query": "x"})])
    assert result.rewards == (0.0,)
    assert result.assignments == (None,)


def test_03_partial_parameter_match_uses_name_set_and_exact_values():
    # Same two parameter names, but neither value matches:
    # (tool-name 1 + parameter-name 1 + content 0) / (2 + |GT|=2) = 0.5.
    similarity = matchtir_similarity(
        call("trade", {"stock": "MSFT", "shares": 99}),
        call("trade", {"stock": "AAPL", "shares": 10}),
    )
    assert similarity == pytest.approx(0.5)


def test_official_corner_semantics_case_insensitive_names_and_empty_arguments():
    assert matchtir_similarity(call("LOOKUP", {}), call("lookup", {})) == 1.0
    malformed = CanonicalToolCall.from_prediction(
        {"call_idx": 0, "name": "", "arguments": {}, "valid": False}, 0
    )
    assert matchtir_similarity(malformed, call("lookup", {})) == 0.0


def test_04_duplicate_calls_share_one_hungarian_match():
    exact = call("lookup", {"query": "x"})
    result = hard_match_calls([exact, exact], [exact], unmatched_penalty=0.0)
    assert result.rewards == (1.0, 0.0)
    assert result.matched_count == 1
    assert sum(index is None for index in result.assignments) == 1


def test_04b_assignment_is_true_global_hungarian_not_greedy(monkeypatch):
    import env_tuning.rods_matchtir_v1.matching as matching_module

    # Greedy takes 0.90 first and gets total 0.90.  The global optimum is
    # 0.80 + 0.85 = 1.65.
    matrix = ((0.90, 0.80), (0.85, 0.00))
    monkeypatch.setattr(
        matching_module,
        "matchtir_similarity",
        lambda pred, gt: matrix[pred.call_idx][gt.call_idx],
    )
    predictions = [
        CanonicalToolCall("p0", {}, call_idx=0),
        CanonicalToolCall("p1", {}, call_idx=1),
    ]
    ground_truth = [
        CanonicalToolCall("g0", {}, call_idx=0),
        CanonicalToolCall("g1", {}, call_idx=1),
    ]
    result = hard_match_calls(predictions, ground_truth)
    assert result.assignments == (1, 0)
    assert result.rewards == pytest.approx((0.80, 0.85))


def test_05_multi_call_policy_step_averages_call_rewards():
    provenance = {
        "ground_truth": [["f(a=1)", "h(a=1, b=2)"]],
        "policy_steps": [
            policy_step(
                0,
                0,
                [
                    {"name": "f", "arguments": {"a": 1}},       # 1.0
                    {"name": "h", "arguments": {"a": 9, "b": 8}},  # 0.5
                    {"name": "wrong", "arguments": {}},         # 0.0
                ],
                0,
            )
        ],
    }
    result = run_single(provenance)
    record = result.step_records[0]
    assert record["call_rewards"] == pytest.approx([1.0, 0.5, 0.0])
    assert record["step_reward"] == pytest.approx(0.5)


def test_06_discounted_return_uses_raw_step_rewards_before_normalization():
    provenance = {
        "ground_truth": [["f(a=1)", "h(a=1, b=2)"]],
        "policy_steps": [
            policy_step(0, 0, [{"name": "f", "arguments": {"a": 1}}], 0),
            policy_step(0, 1, [{"name": "h", "arguments": {"a": 9, "b": 8}}], 2),
            policy_step(0, 2, [{"name": "wrong", "arguments": {}}], 4),
        ],
    }
    result = run_single(provenance)
    records = list(result.step_records)
    assert [item["step_reward"] for item in records] == pytest.approx([1.0, 0.5, 0.0])
    assert [item["local_return"] for item in records] == pytest.approx([1.45, 0.5, 0.0])
    # Singleton normalization is the deliberate V1 zero-local fallback.
    assert all(item["local_active"] is False for item in records)
    assert torch.count_nonzero(result.local_advantages) == 0
