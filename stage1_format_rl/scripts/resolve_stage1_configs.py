#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


EXPECTED = {
    "algorithm.adv_estimator": "grpo",
    "actor_rollout_ref.rollout.name": "sglang",
    "actor_rollout_ref.rollout.n": 16,
    "actor_rollout_ref.rollout.multi_turn.enable": True,
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns": 100,
    "actor_rollout_ref.rollout.multi_turn.max_user_turns": 100,
    "trainer.n_gpus_per_node": 2,
    "trainer.nnodes": 1,
    "reward_model.reward_manager": "bfcl",
}


def select(config, key):
    value = OmegaConf.select(config, key)
    if value is None:
        raise KeyError(key)
    return value


def resolve_one(config_dir: Path, config_name: str, output: Path) -> None:
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(config_name=config_name)
    OmegaConf.resolve(config)
    for key, expected in EXPECTED.items():
        actual = select(config, key)
        if actual != expected:
            raise AssertionError(f"{key}: expected {expected!r}, got {actual!r}")

    for key in (
        "actor_rollout_ref.model.path",
        "data.train_files",
        "data.val_files",
        "custom_reward_function.path",
        "actor_rollout_ref.rollout.multi_turn.interaction_config_path",
    ):
        path = Path(str(select(config, key))).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")

    model_path = Path(str(select(config, "actor_rollout_ref.model.path")))
    if "base" in model_path.name.lower():
        raise AssertionError(f"Base model is forbidden: {model_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    print(f"RESOLVED {config_name} -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapted-verl", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.adapted_verl))
    for stem in (
        "stage1_qwen3_4b_k16_repo_aligned",
        "stage1_qwen3_4b_k16_paper_aligned",
    ):
        resolve_one(args.config_dir, stem, args.output_dir / f"resolved_{stem}.yaml")


if __name__ == "__main__":
    main()
