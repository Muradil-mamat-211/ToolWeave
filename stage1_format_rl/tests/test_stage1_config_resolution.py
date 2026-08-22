from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def load(path):
    return yaml.safe_load(path.read_text())


def nested(config, *keys):
    value = config
    for key in keys:
        value = value[key]
    return value


def test_resolved_configs_have_required_contract(asset_stage_root):
    for profile in ("repo_aligned", "paper_aligned"):
        path = (
            asset_stage_root
            / "artifacts"
            / f"resolved_stage1_qwen3_4b_k16_{profile}.yaml"
        )
        config = load(path)
        assert nested(config, "algorithm", "adv_estimator") == "grpo"
        assert nested(config, "actor_rollout_ref", "rollout", "name") == "sglang"
        assert nested(config, "actor_rollout_ref", "rollout", "n") == 16
        assert nested(
            config, "actor_rollout_ref", "rollout", "multi_turn", "enable"
        ) is True
        assert nested(
            config,
            "actor_rollout_ref",
            "rollout",
            "multi_turn",
            "max_assistant_turns",
        ) == 100
        assert nested(
            config,
            "actor_rollout_ref",
            "rollout",
            "multi_turn",
            "max_user_turns",
        ) == 100
        assert nested(config, "trainer", "n_gpus_per_node") == 2
        assert nested(config, "trainer", "nnodes") == 1
        assert nested(config, "reward_model", "reward_manager") == "bfcl"
        assert nested(config, "algorithm", "use_kl_in_reward") is False
        assert nested(config, "actor_rollout_ref", "actor", "use_kl_loss") is True
        assert nested(config, "data", "train_batch_size") * 16 == 64
        assert 64 % 2 == 0
        assert nested(
            config, "actor_rollout_ref", "actor", "ppo_mini_batch_size"
        ) == 4
        for key_path in (
            ("actor_rollout_ref", "model", "path"),
            ("data", "train_files"),
            ("data", "val_files"),
            ("custom_reward_function", "path"),
            (
                "actor_rollout_ref",
                "rollout",
                "multi_turn",
                "interaction_config_path",
            ),
        ):
            assert Path(nested(config, *key_path)).exists()


def test_repo_and_paper_differences_are_explicit(asset_stage_root):
    repo = load(
        asset_stage_root
        / "artifacts"
        / "resolved_stage1_qwen3_4b_k16_repo_aligned.yaml"
    )
    paper = load(
        asset_stage_root
        / "artifacts"
        / "resolved_stage1_qwen3_4b_k16_paper_aligned.yaml"
    )
    assert nested(repo, "actor_rollout_ref", "actor", "kl_loss_coef") == 0.1
    assert nested(paper, "actor_rollout_ref", "actor", "kl_loss_coef") == 0.01
    assert nested(repo, "trainer", "total_epochs") == 20
    assert nested(paper, "trainer", "total_epochs") == 5


def test_official_reward_and_interaction_import(asset_stage_root):
    config = load(
        asset_stage_root
        / "artifacts"
        / "resolved_stage1_qwen3_4b_k16_repo_aligned.yaml"
    )
    reward_path = Path(config["custom_reward_function"]["path"])
    spec = importlib.util.spec_from_file_location("resolved_reward", reward_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    assert callable(module.compute_score)

    from env_tuning.interaction.new_multi_turn_fc import (
        MultiTurnFunctionCallInteraction,
    )

    assert MultiTurnFunctionCallInteraction.__name__ == "MultiTurnFunctionCallInteraction"
