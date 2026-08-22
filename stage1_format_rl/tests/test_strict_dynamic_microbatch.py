from __future__ import annotations

import torch
from tensordict import TensorDict

from verl.utils.seqlen_balancing import rearrange_micro_batches


def _batch(lengths: list[int], padded_length: int) -> TensorDict:
    attention_mask = torch.zeros((len(lengths), padded_length), dtype=torch.long)
    for index, length in enumerate(lengths):
        attention_mask[index, :length] = 1
    trajectory_id = torch.arange(len(lengths), dtype=torch.long).unsqueeze(-1)
    return TensorDict(
        {"attention_mask": attention_mask, "trajectory_id": trajectory_id},
        batch_size=[len(lengths)],
    )


def test_legacy_balancing_can_exceed_nominal_token_target() -> None:
    batch = _batch([12, 11, 10, 9, 8, 7], padded_length=16)

    micro_batches, _ = rearrange_micro_batches(
        batch=batch,
        max_token_len=16,
        same_micro_num_in_dp=False,
    )

    assert max(int(item["attention_mask"].sum()) for item in micro_batches) > 16


def test_strict_dynamic_packing_preserves_rows_and_enforces_cap() -> None:
    batch = _batch([12, 11, 10, 9, 8, 7], padded_length=16)

    micro_batches, partitions = rearrange_micro_batches(
        batch=batch,
        max_token_len=16,
        same_micro_num_in_dp=False,
        enforce_max_token_len=True,
    )

    token_sums = [int(item["attention_mask"].sum()) for item in micro_batches]
    recovered_ids = sorted(
        int(index)
        for item in micro_batches
        for index in item["trajectory_id"].flatten().tolist()
    )
    assert max(token_sums) <= 16
    assert recovered_ids == list(range(6))
    assert sorted(index for partition in partitions for index in partition) == list(range(6))


def test_strict_dynamic_packing_uses_effective_not_padded_length() -> None:
    batch = _batch([12, 3], padded_length=17)

    micro_batches, _ = rearrange_micro_batches(
        batch=batch,
        max_token_len=12,
        same_micro_num_in_dp=False,
        enforce_max_token_len=True,
    )

    assert max(int(item["attention_mask"].sum()) for item in micro_batches) <= 12


def test_strict_dynamic_packing_fails_if_one_trajectory_exceeds_cap() -> None:
    batch = _batch([17, 3], padded_length=17)

    try:
        rearrange_micro_batches(
            batch=batch,
            max_token_len=16,
            same_micro_num_in_dp=False,
            enforce_max_token_len=True,
        )
    except ValueError as error:
        assert "single trajectory exceeds" in str(error)
    else:
        raise AssertionError("oversized trajectory must fail closed")
