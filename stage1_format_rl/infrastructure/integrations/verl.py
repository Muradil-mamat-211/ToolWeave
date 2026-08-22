"""The sole project adapter from layered config to veRL/Hydra fields."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..loader import dotted_set, expand_templates
from ..models import AssetSpec, ConfigError, MachineConfig, RuntimeConfig, TopologyPlan


_FORBIDDEN_EXPERIMENT_FIELDS = (
    "trainer.nnodes",
    "trainer.n_gpus_per_node",
    "trainer.ray_placement_strategy",
    "trainer.ray_cpus_per_worker",
    "ray_init.num_cpus",
    "data.dataloader_num_workers",
    "actor_rollout_ref.model.enable_activation_offload",
    "actor_rollout_ref.model.enable_gradient_checkpointing",
    "actor_rollout_ref.model.use_remove_padding",
    "actor_rollout_ref.model.use_fused_kernels",
    "actor_rollout_ref.actor.use_dynamic_bsz",
    "actor_rollout_ref.actor.entropy_checkpointing",
    "actor_rollout_ref.actor.entropy_from_logits_with_chunking",
    "actor_rollout_ref.rollout.name",
    "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "actor_rollout_ref.rollout.gpu_memory_utilization",
    "actor_rollout_ref.rollout.multi_stage_wake_up",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz",
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz",
    "actor_rollout_ref.ref.entropy_from_logits_with_chunking",
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size",
    "actor_rollout_ref.actor.fsdp_config",
    "actor_rollout_ref.ref.fsdp_config",
)


def _contains_dotted(mapping: Mapping[str, Any], key: str) -> bool:
    current: Any = mapping
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def build_verl_config(
    experiment: Mapping[str, Any],
    assets: Mapping[str, AssetSpec],
    machine: MachineConfig,
    runtime: RuntimeConfig,
    plan: TopologyPlan,
) -> dict[str, Any]:
    """Build the effective veRL config without inspecting host hardware."""

    raw_verl = experiment.get("verl")
    if not isinstance(raw_verl, Mapping):
        raise ConfigError("experiment must contain a verl mapping")
    for key in _FORBIDDEN_EXPERIMENT_FIELDS:
        if _contains_dotted(raw_verl, key):
            raise ConfigError(
                f"experiment layer illegally owns runtime/topology field {key}"
            )
    config: dict[str, Any] = deepcopy(dict(raw_verl))

    bindings = experiment.get("asset_bindings", {})
    if not isinstance(bindings, Mapping):
        raise ConfigError("experiment.asset_bindings must be a mapping")
    for destination, asset_name in bindings.items():
        if asset_name not in assets:
            raise ConfigError(f"unknown logical asset {asset_name!r} for {destination}")
        dotted_set(config, str(destination), str(assets[str(asset_name)].path))

    output_bindings = experiment.get("output_bindings", {})
    if not isinstance(output_bindings, Mapping):
        raise ConfigError("experiment.output_bindings must be a mapping")
    expanded_outputs = expand_templates(output_bindings, machine.template_values())
    for destination, value in expanded_outputs.items():
        dotted_set(config, str(destination), value)

    # Topology: all values below are derived from one TopologyPlan.
    dotted_set(config, "trainer.nnodes", plan.nnodes)
    dotted_set(config, "trainer.n_gpus_per_node", plan.uniform_learner_gpus_per_node)
    dotted_set(config, "trainer.ray_placement_strategy", plan.ray_placement_strategy)
    dotted_set(config, "trainer.ray_cpus_per_worker", runtime.ray_cpus_per_worker)
    dotted_set(config, "ray_init.num_cpus", plan.ray_num_cpus)
    dotted_set(config, "ray_init.address", runtime.ray_address)

    # Backend/runtime performance and memory behavior.
    dotted_set(config, "data.dataloader_num_workers", runtime.dataloader_num_workers)
    dotted_set(config, "actor_rollout_ref.model.enable_gradient_checkpointing", runtime.model_gradient_checkpointing)
    dotted_set(config, "actor_rollout_ref.model.enable_activation_offload", runtime.model_activation_offload)
    dotted_set(config, "actor_rollout_ref.model.use_remove_padding", runtime.model_use_remove_padding)
    dotted_set(config, "actor_rollout_ref.model.use_fused_kernels", runtime.model_use_fused_kernels)
    dotted_set(config, "actor_rollout_ref.actor.use_dynamic_bsz", runtime.dynamic_batching)
    dotted_set(config, "actor_rollout_ref.actor.entropy_checkpointing", runtime.entropy_checkpointing)
    dotted_set(config, "actor_rollout_ref.actor.entropy_from_logits_with_chunking", runtime.entropy_from_logits_with_chunking)
    dotted_set(config, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", runtime.actor_micro_batch_size_per_gpu)
    dotted_set(config, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu", runtime.actor_max_token_len_per_gpu)
    dotted_set(config, "actor_rollout_ref.actor.ulysses_sequence_parallel_size", runtime.sequence_parallel_size)
    dotted_set(config, "actor_rollout_ref.actor.fsdp_config.param_offload", runtime.actor_param_offload)
    dotted_set(config, "actor_rollout_ref.actor.fsdp_config.optimizer_offload", runtime.actor_optimizer_offload)
    dotted_set(config, "actor_rollout_ref.rollout.name", runtime.rollout_backend)
    dotted_set(config, "actor_rollout_ref.rollout.tensor_model_parallel_size", plan.rollout_tp)
    dotted_set(config, "actor_rollout_ref.rollout.gpu_memory_utilization", runtime.rollout_gpu_memory_utilization)
    dotted_set(config, "actor_rollout_ref.rollout.multi_stage_wake_up", runtime.rollout_multi_stage_wake_up)
    dotted_set(config, "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", runtime.dynamic_batching)
    dotted_set(config, "actor_rollout_ref.rollout.max_num_batched_tokens", runtime.rollout_max_num_batched_tokens)
    dotted_set(config, "actor_rollout_ref.rollout.max_model_len", runtime.rollout_max_model_len)
    dotted_set(config, "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", runtime.rollout_logprob_micro_batch_size_per_gpu)
    dotted_set(config, "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu", runtime.rollout_logprob_max_token_len_per_gpu)
    dotted_set(config, "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", runtime.reference_logprob_micro_batch_size_per_gpu)
    dotted_set(config, "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu", runtime.reference_logprob_max_token_len_per_gpu)
    dotted_set(config, "actor_rollout_ref.ref.log_prob_use_dynamic_bsz", runtime.dynamic_batching)
    dotted_set(config, "actor_rollout_ref.ref.entropy_from_logits_with_chunking", runtime.entropy_from_logits_with_chunking)
    dotted_set(config, "actor_rollout_ref.ref.fsdp_config.param_offload", runtime.reference_param_offload)
    return config
