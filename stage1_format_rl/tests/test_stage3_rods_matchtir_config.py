from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pandas as pd

from env_tuning.rods_matchtir_v1.lifecycle import LifecycleConfig
from stage1_format_rl.infrastructure.resolver import resolve_profile


SOURCE_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get("TOOLWEAVE_ASSET_ROOT", SOURCE_ROOT)).expanduser().resolve()
PROFILE = (
    SOURCE_ROOT
    / "stage1_format_rl/configs/layers/profiles/stage3_reference.yaml"
)
FSDP_WORKERS = (
    SOURCE_ROOT
    / "code/AWorld-RL-stage1-worktree/EnvTuning/verl/verl/workers/fsdp_workers.py"
)


def _resolved():
    return resolve_profile(
        PROFILE,
        environ={"TOOLWEAVE_ASSET_ROOT": str(ASSET_ROOT)},
    )


def test_stage3_config_preserves_rods_runtime_and_enables_runtime_interaction_credit():
    resolved = _resolved()
    config = resolved.effective_verl
    assert config["actor_rollout_ref"]["rollout"]["n"] == 16
    assert config["actor_rollout_ref"]["actor"]["optim"]["lr"] == 1.0e-6
    assert config["actor_rollout_ref"]["actor"]["kl_loss_coef"] == 0.01
    assert config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 20
    assert config["data"]["train_batch_size"] == 20
    assert config["trainer"]["total_epochs"] == 5
    assert config["algorithm"]["use_kl_in_reward"] is False
    assert config["algorithm"]["matchtir_local"] == {
        "mode": "runtime_interaction_final",
        "enabled": True,
        "weight": 1.0,
        "gamma": 0.9,
        "matching": "hard",
        "unmatched_penalty": 0.0,
        "min_group_size": 2,
        "epsilon": 1.0e-6,
    }
    assert resolved.assets["stage2_step25_model"].path.name == "global_step_25"
    lifecycle = config["trainer"]["rods_stage3_lifecycle"]
    assert lifecycle["require_seed_selection_config"] is True
    assert lifecycle["max_seeds_per_selection"] == 16
    assert lifecycle["seed_type_quotas"] == {
        "multi_turn_base": 4,
        "multi_turn_miss_func": 4,
        "multi_turn_miss_param": 4,
        "multi_turn_long_context": 4,
    }
    assert lifecycle["seed_cooldown_steps"] == 13
    assert lifecycle["injection_ratio"] == 0.20
    assert lifecycle["generated_pool_cap"] == 400
    assert lifecycle["stale_after_steps"] is None
    resolved_lifecycle = LifecycleConfig.from_mapping(lifecycle)
    assert resolved_lifecycle.enabled is True
    assert resolved_lifecycle.require_seed_selection_config is True
    assert resolved_lifecycle.seed_selection_configured is True


def test_stage3_training_data_has_four_original_types_and_current_protocol():
    path = _resolved().assets["stage3_train_400"].path
    frame = pd.read_parquet(path)
    assert len(frame) == 400
    assert Counter(frame["data_source"]) == {
        "multi_turn_base": 100,
        "multi_turn_miss_func": 100,
        "multi_turn_miss_param": 100,
        "multi_turn_long_context": 100,
    }
    ids = [row["interaction_kwargs"]["id"] for row in frame["extra_info"]]
    assert len(set(ids)) == 400
    for prompt in frame["prompt"]:
        system = prompt[0]["content"]
        assert "<think>" in system and "<answer>" in system
        assert "<thinking>" not in system


def test_stage3_uses_fixed_denominator_progress_reward():
    reward_path = _resolved().assets["progress_reward"].path
    namespace: dict = {}
    exec(compile(reward_path.read_text(encoding="utf-8"), str(reward_path), "exec"), namespace)
    score = namespace["compute_score"](
        reward_scores={"user_turn_rewards": [1]},
        ground_truth=[["a()"], ["b()"], ["c()"]],
    )
    assert score["expected_user_turns"] == 3
    assert score["missing_terminal_turns"] == 2
    assert score["score"] == 1 / 3


def test_reference_fsdp_offload_obeys_explicit_role_config():
    source = FSDP_WORKERS.read_text(encoding="utf-8")
    assert 'if fsdp_config.get("param_offload", False)' in source
    assert 'None if role == "actor" else CPUOffload(offload_params=True)' not in source
    assert 'None if role == "actor" else CPUOffloadPolicy(pin_memory=True)' not in source
