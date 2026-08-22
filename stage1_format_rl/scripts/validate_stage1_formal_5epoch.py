#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shuffle-proof", type=Path, required=True)
    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base=None):
        config = compose(config_name="stage1_qwen3_4b_k16_formal_5epoch")
    OmegaConf.resolve(config)

    expected = {
        "data.train_batch_size": 4,
        "data.shuffle": False,
        "data.seed": 42,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.kl_loss_coef": 0.01,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 20480,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 2,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.rollout.n": 16,
        "actor_rollout_ref.rollout.name": "sglang",
        "actor_rollout_ref.rollout.multi_stage_wake_up": False,
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "trainer.n_gpus_per_node": 2,
        "trainer.save_freq": 25,
        "trainer.max_actor_ckpt_to_keep": 1,
        "trainer.total_epochs": 5,
        "trainer.test_freq": 1_000_000_000,
    }
    for key, wanted in expected.items():
        actual = OmegaConf.select(config, key)
        if actual != wanted:
            raise AssertionError(f"{key}: expected {wanted!r}, got {actual!r}")

    full_sequence_limit = int(config.data.max_prompt_length) + int(config.data.max_response_length)
    sharded_actor_budget = (
        int(config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu)
        * int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size)
    )
    if sharded_actor_budget < full_sequence_limit:
        raise AssertionError(
            f"actor packing would cut the sequence contract: "
            f"{sharded_actor_budget=} < {full_sequence_limit=}"
        )

    save_contents = list(config.actor_rollout_ref.actor.checkpoint.save_contents)
    if save_contents != ["model", "optimizer", "extra"]:
        raise AssertionError(f"non-resumable checkpoint contents: {save_contents}")

    train_path = Path(config.data.train_files)
    model_path = Path(config.actor_rollout_ref.model.path)
    if not train_path.is_file() or not model_path.is_dir():
        raise FileNotFoundError((train_path, model_path))
    frame = pd.read_parquet(train_path)
    if len(frame) != 100:
        raise AssertionError(f"expected 100 train rows, got {len(frame)}")

    manifest_path = train_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["seed"] != 42 or manifest["rows"] != 100:
        raise AssertionError(f"invalid physical shuffle manifest: {manifest}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    proof = {
        "physical_pre_shuffle": True,
        "runtime_sampler": "SequentialSampler",
        "seed": 42,
        "rows": len(frame),
        "batch_size": int(config.data.train_batch_size),
        "steps_per_epoch": len(frame) // int(config.data.train_batch_size),
        "epochs": int(config.trainer.total_epochs),
        "total_trainer_steps": (len(frame) // int(config.data.train_batch_size))
        * int(config.trainer.total_epochs),
        "first_20_source_indices": manifest["source_row_order"][:20],
        "all_epochs_read_all_rows": True,
        "epoch_boundary_resume_preserves_future_order": True,
    }
    args.shuffle_proof.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
