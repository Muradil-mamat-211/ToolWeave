from types import SimpleNamespace

import numpy as np
import yaml

from verl.trainer.ppo.ray_trainer import _copy_generation_context_metadata


def test_generation_receives_canonical_bfcl_data_source_without_removing_training_copy():
    sources = np.array(
        [
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
        ],
        dtype=object,
    )
    batch = SimpleNamespace(non_tensor_batch={"data_source": sources})
    gen_batch = SimpleNamespace(non_tensor_batch={})

    _copy_generation_context_metadata(batch, gen_batch)

    assert gen_batch.non_tensor_batch["data_source"].tolist() == sources.tolist()
    assert batch.non_tensor_batch["data_source"].tolist() == sources.tolist()
    assert gen_batch.non_tensor_batch["data_source"] is not sources


def test_online_smoke_uses_multistage_sglang_wakeup_for_weight_sync():
    config_path = (
        "/root/autodl-tmp/rods-workspace/stage1_format_rl/configs/"
        "stage3_online_smoke_2x_rtxpro6000_two_updates.yaml"
    )
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["actor_rollout_ref"]["rollout"]["multi_stage_wake_up"] is True
