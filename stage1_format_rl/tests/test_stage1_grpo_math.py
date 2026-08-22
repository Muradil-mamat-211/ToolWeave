from __future__ import annotations

import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_grpo_outcome_advantage,
    compute_policy_loss,
)


def test_k16_rollout_level_advantage_uses_adapted_verl_sample_std():
    rewards = torch.arange(16, dtype=torch.float32) / 15
    token_rewards = torch.zeros(16, 4)
    token_rewards[:, -1] = rewards
    mask = torch.tensor([[1, 1, 1, 0]] * 16, dtype=torch.float32)
    advantages, returns = compute_grpo_outcome_advantage(
        token_rewards, mask, np.array(["prompt"] * 16), epsilon=1e-6
    )
    expected = (rewards - rewards.mean()) / (rewards.std(unbiased=True) + 1e-6)
    assert torch.allclose(advantages[:, 0], expected)
    assert torch.allclose(advantages[:, 1], expected)
    assert torch.allclose(advantages[:, 2], expected)
    assert torch.equal(advantages[:, 3], torch.zeros(16))
    assert torch.equal(returns, advantages)


def test_all_equal_group_has_zero_advantage():
    token_rewards = torch.zeros(16, 2)
    token_rewards[:, -1] = 1.5
    mask = torch.ones_like(token_rewards)
    advantages, _ = compute_grpo_outcome_advantage(
        token_rewards, mask, np.array(["same"] * 16), epsilon=1e-6
    )
    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_seq_mean_token_mean_exact_aggregation():
    loss = torch.tensor([[2.0, 4.0, 100.0], [1.0, 3.0, 5.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    actual = agg_loss(loss, mask, "seq-mean-token-mean")
    expected = torch.tensor(((2 + 4) / 2 + (1 + 3 + 5) / 3) / 2)
    assert torch.isclose(actual, expected)


def test_asymmetric_clipped_policy_loss_is_finite():
    old = torch.zeros(2, 3)
    current = torch.tensor([[0.5, -0.5, 0.0], [0.2, -0.2, 0.0]])
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    mask = torch.ones_like(old)
    values = compute_policy_loss(
        old,
        current,
        advantages,
        mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.28,
        clip_ratio_c=10.0,
        loss_agg_mode="seq-mean-token-mean",
    )
    assert all(torch.isfinite(value) for value in values)

