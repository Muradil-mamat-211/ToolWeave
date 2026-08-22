from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import yaml

from env_tuning.rods_matchtir_v1.lifecycle import LifecycleConfig


CONFIG = Path(
    "/root/autodl-tmp/rods-workspace/stage1_format_rl/configs/"
    "stage3_rods_matchtir_v1_training_branch.yaml"
)
FSDP_WORKERS = Path(
    "/root/autodl-tmp/rods-workspace/code/AWorld-RL-stage1-worktree/"
    "EnvTuning/verl/verl/workers/fsdp_workers.py"
)


def test_stage3_config_preserves_rods_runtime_and_enables_v1():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["actor_rollout_ref"]["rollout"]["n"] == 16
    assert config["actor_rollout_ref"]["actor"]["optim"]["lr"] == 1.0e-6
    assert config["actor_rollout_ref"]["actor"]["kl_loss_coef"] == 0.01
    assert config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 20
    assert config["data"]["train_batch_size"] == 20
    assert config["trainer"]["total_epochs"] == 5
    assert config["algorithm"]["use_kl_in_reward"] is False
    local = config["algorithm"]["matchtir_local"]
    assert local == {
        "enabled": True,
        "weight": 1.0,
        "gamma": 0.9,
        "matching": "hard",
        "unmatched_penalty": 0.0,
        "min_group_size": 2,
        "epsilon": 1.0e-6,
    }
    model_path = Path(config["actor_rollout_ref"]["model"]["path"])
    assert model_path.name == "global_step_25"
    assert (model_path / "model.safetensors.index.json").is_file()
    lifecycle = config["trainer"]["rods_stage3_lifecycle"]
    assert lifecycle["require_seed_selection_config"] is True
    assert lifecycle["max_seeds_per_selection"] is None
    assert lifecycle["seed_type_quotas"] == {
        "multi_turn_base": None,
        "multi_turn_miss_func": None,
        "multi_turn_miss_param": None,
        "multi_turn_long_context": None,
    }
    assert lifecycle["seed_cooldown_steps"] is None
    assert lifecycle["injection_ratio"] == 0.20
    assert lifecycle["generated_pool_cap"] == 400
    assert lifecycle["stale_after_steps"] is None
    with pytest.raises(ValueError, match="paper-unspecified project hyperparameters"):
        LifecycleConfig.from_mapping(lifecycle)
    disabled = dict(lifecycle)
    disabled["enabled"] = False
    resolved_lifecycle = LifecycleConfig.from_mapping(disabled)
    assert resolved_lifecycle.enabled is False
    assert resolved_lifecycle.seed_selection_configured is False


def test_stage3_training_data_has_four_original_types_and_current_protocol():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    path = Path(config["data"]["train_files"])
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
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    reward_path = Path(config["custom_reward_function"]["path"])
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
