"""The sole project adapter from layered config to veRL/Hydra fields."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..loader import dotted_set, expand_templates
from ..models import AssetSpec, ConfigError, MachineConfig, RuntimeConfig, TopologyPlan


@dataclass(frozen=True)
class _OwnedVerlField:
    owner: str
    resolve: Callable[[RuntimeConfig, TopologyPlan], Any]


# This table is the single source for both adapter injection and ownership
# validation. Adding a new RuntimeConfig/TopologyPlan destination therefore
# makes it fail-closed in experiment and binding layers automatically.
_AUTHORITATIVE_VERL_FIELDS = {
    "trainer.nnodes": _OwnedVerlField("topology", lambda _runtime, plan: plan.nnodes),
    "trainer.n_gpus_per_node": _OwnedVerlField(
        "topology", lambda _runtime, plan: plan.uniform_learner_gpus_per_node
    ),
    "trainer.ray_placement_strategy": _OwnedVerlField(
        "topology", lambda _runtime, plan: plan.ray_placement_strategy
    ),
    "trainer.ray_cpus_per_worker": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.ray_cpus_per_worker
    ),
    "ray_init.num_cpus": _OwnedVerlField(
        "topology", lambda _runtime, plan: plan.ray_num_cpus
    ),
    "ray_init.address": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.ray_address
    ),
    "data.dataloader_num_workers": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.dataloader_num_workers
    ),
    "actor_rollout_ref.model.enable_gradient_checkpointing": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.model_gradient_checkpointing
    ),
    "actor_rollout_ref.model.enable_activation_offload": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.model_activation_offload
    ),
    "actor_rollout_ref.model.use_remove_padding": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.model_use_remove_padding
    ),
    "actor_rollout_ref.model.use_fused_kernels": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.model_use_fused_kernels
    ),
    "actor_rollout_ref.actor.use_dynamic_bsz": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.dynamic_batching
    ),
    "actor_rollout_ref.actor.entropy_checkpointing": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.entropy_checkpointing
    ),
    "actor_rollout_ref.actor.entropy_from_logits_with_chunking": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.entropy_from_logits_with_chunking
    ),
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.actor_micro_batch_size_per_gpu
    ),
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.actor_max_token_len_per_gpu
    ),
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.sequence_parallel_size
    ),
    "actor_rollout_ref.actor.fsdp_config.param_offload": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.actor_param_offload
    ),
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.actor_optimizer_offload
    ),
    "actor_rollout_ref.rollout.name": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.rollout_backend
    ),
    "actor_rollout_ref.rollout.tensor_model_parallel_size": _OwnedVerlField(
        "topology", lambda _runtime, plan: plan.rollout_tp
    ),
    "actor_rollout_ref.rollout.gpu_memory_utilization": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.rollout_gpu_memory_utilization
    ),
    "actor_rollout_ref.rollout.multi_stage_wake_up": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.rollout_multi_stage_wake_up
    ),
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.dynamic_batching
    ),
    "actor_rollout_ref.rollout.max_num_batched_tokens": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.rollout_max_num_batched_tokens
    ),
    "actor_rollout_ref.rollout.max_model_len": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.rollout_max_model_len
    ),
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": _OwnedVerlField(
        "runtime",
        lambda runtime, _plan: runtime.rollout_logprob_micro_batch_size_per_gpu,
    ),
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.rollout_logprob_max_token_len_per_gpu
    ),
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": _OwnedVerlField(
        "runtime",
        lambda runtime, _plan: runtime.reference_logprob_micro_batch_size_per_gpu,
    ),
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": _OwnedVerlField(
        "runtime",
        lambda runtime, _plan: runtime.reference_logprob_max_token_len_per_gpu,
    ),
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.dynamic_batching
    ),
    "actor_rollout_ref.ref.entropy_from_logits_with_chunking": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.entropy_from_logits_with_chunking
    ),
    "actor_rollout_ref.ref.fsdp_config.param_offload": _OwnedVerlField(
        "runtime", lambda runtime, _plan: runtime.reference_param_offload
    ),
}

VERL_FIELD_OWNERS: Mapping[str, str] = MappingProxyType(
    {
        destination: field.owner
        for destination, field in _AUTHORITATIVE_VERL_FIELDS.items()
    }
)


def _leaf_paths(mapping: Mapping[str, Any], prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            paths.extend(_leaf_paths(value, path))
        else:
            paths.append(path)
    return tuple(paths)


def _paths_overlap(left: str, right: str) -> bool:
    return (
        left == right
        or left.startswith(f"{right}.")
        or right.startswith(f"{left}.")
    )


def _authoritative_collision(path: str) -> str | None:
    return next(
        (
            destination
            for destination in VERL_FIELD_OWNERS
            if _paths_overlap(path, destination)
        ),
        None,
    )


def _validate_bindings(
    raw_verl: Mapping[str, Any],
    asset_bindings: Mapping[str, Any],
    output_bindings: Mapping[str, Any],
) -> None:
    experiment_paths = _leaf_paths(raw_verl)
    asset_destinations = tuple(str(item) for item in asset_bindings)
    output_destinations = tuple(str(item) for item in output_bindings)
    for label, destinations in (
        ("experiment.asset_bindings", asset_destinations),
        ("experiment.output_bindings", output_destinations),
    ):
        for destination in destinations:
            collision = _authoritative_collision(destination)
            if collision is not None:
                raise ConfigError(
                    f"{label} destination {destination} collides with "
                    f"{VERL_FIELD_OWNERS[collision]}-owned field {collision}"
                )
            if any(_paths_overlap(destination, path) for path in experiment_paths):
                raise ConfigError(
                    f"{label} destination {destination} collides with "
                    "experiment.verl ownership"
                )
    for asset_destination in asset_destinations:
        if any(
            _paths_overlap(asset_destination, output_destination)
            for output_destination in output_destinations
        ):
            raise ConfigError(
                f"asset/output binding ownership collision at {asset_destination}"
            )


def build_verl_config(
    experiment: Mapping[str, Any],
    assets: Mapping[str, AssetSpec],
    machine: MachineConfig,
    runtime: RuntimeConfig,
    plan: TopologyPlan,
) -> dict[str, Any]:
    """Build the effective veRL config without inspecting host hardware.

    Unmapped ``experiment.verl`` leaves are experiment-owned. Destinations in
    ``asset_bindings`` and ``output_bindings`` are asset- and output-owned,
    respectively. ``VERL_FIELD_OWNERS`` owns every adapter-injected topology or
    runtime leaf, and overlap across any of those categories fails closed.
    """

    raw_verl = experiment.get("verl")
    if not isinstance(raw_verl, Mapping):
        raise ConfigError("experiment must contain a verl mapping")
    for path in _leaf_paths(raw_verl):
        destination = _authoritative_collision(path)
        if destination is not None:
            raise ConfigError(
                "experiment layer illegally owns runtime/topology field "
                f"{destination} (authoritative owner: "
                f"{VERL_FIELD_OWNERS[destination]})"
            )
    config: dict[str, Any] = deepcopy(dict(raw_verl))

    bindings = experiment.get("asset_bindings", {})
    if not isinstance(bindings, Mapping):
        raise ConfigError("experiment.asset_bindings must be a mapping")
    output_bindings = experiment.get("output_bindings", {})
    if not isinstance(output_bindings, Mapping):
        raise ConfigError("experiment.output_bindings must be a mapping")
    _validate_bindings(raw_verl, bindings, output_bindings)

    for destination, asset_name in bindings.items():
        if asset_name not in assets:
            raise ConfigError(f"unknown logical asset {asset_name!r} for {destination}")
        dotted_set(config, str(destination), str(assets[str(asset_name)].path))

    expanded_outputs = expand_templates(output_bindings, machine.template_values())
    for destination, value in expanded_outputs.items():
        dotted_set(config, str(destination), value)

    for destination, field in _AUTHORITATIVE_VERL_FIELDS.items():
        dotted_set(config, destination, field.resolve(runtime, plan))
    return config
