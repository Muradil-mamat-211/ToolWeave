"""Typed configuration objects at the machine/runtime boundary.

Nothing in this module imports ToolWeave reward, advantage, matching, lifecycle,
or policy-objective code.  These objects describe resources and integration
inputs only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """A deterministic configuration or resource-plan error."""


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{label} must be positive")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise ConfigError(f"{label} must be nonnegative")
    return parsed


@dataclass(frozen=True)
class MachineConfig:
    """Paths and executables that vary when the checkout moves machines."""

    source_root: Path
    asset_root: Path
    models_root: Path
    data_root: Path
    shared_data_root: Path
    artifacts_root: Path
    outputs_root: Path
    logs_root: Path
    reports_root: Path
    evals_root: Path
    cache_root: Path
    temp_root: Path
    short_temp_root: Path
    python_executable: str
    synthesis_python_executable: str
    conda_environment: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MachineConfig":
        raw = _as_mapping(raw.get("machine", raw), "machine")
        required_paths = (
            "source_root",
            "asset_root",
            "models_root",
            "data_root",
            "shared_data_root",
            "artifacts_root",
            "outputs_root",
            "logs_root",
            "reports_root",
            "evals_root",
            "cache_root",
            "temp_root",
            "short_temp_root",
        )
        missing = [name for name in required_paths if not str(raw.get(name, "")).strip()]
        if missing:
            raise ConfigError(f"machine is missing paths: {', '.join(missing)}")
        python = str(raw.get("python_executable", "")).strip()
        if not python:
            raise ConfigError("machine.python_executable is required")
        return cls(
            **{name: Path(str(raw[name])).expanduser().resolve() for name in required_paths},
            python_executable=python,
            synthesis_python_executable=str(
                raw.get("synthesis_python_executable", python)
            ).strip()
            or python,
            conda_environment=str(raw.get("conda_environment", "")),
        )

    def template_values(self) -> dict[str, str]:
        return {
            f"machine.{name}": str(getattr(self, name))
            for name in (
                "source_root",
                "asset_root",
                "models_root",
                "data_root",
                "shared_data_root",
                "artifacts_root",
                "outputs_root",
                "logs_root",
                "reports_root",
                "evals_root",
                "cache_root",
                "temp_root",
                "short_temp_root",
                "python_executable",
                "synthesis_python_executable",
                "conda_environment",
            )
        }


@dataclass(frozen=True)
class NodeHardware:
    """Declarative resources for one node; GPU IDs are node-local physical IDs."""

    name: str
    gpu_ids: tuple[int, ...]
    gpu_model: str
    gpu_memory_gib: float
    cpu_cores: int
    ram_gib: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NodeHardware":
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ConfigError("hardware node name is required")
        gpu_ids_raw = raw.get("gpu_ids", [])
        if not isinstance(gpu_ids_raw, list):
            raise ConfigError(f"hardware node {name}.gpu_ids must be a list")
        try:
            gpu_ids = tuple(int(item) for item in gpu_ids_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"hardware node {name}.gpu_ids must contain integers") from exc
        if any(item < 0 for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
            raise ConfigError(f"hardware node {name}.gpu_ids must be unique nonnegative IDs")
        gpu_memory = float(raw.get("gpu_memory_gib", 0.0))
        ram = float(raw.get("ram_gib", 0.0))
        if gpu_ids and gpu_memory <= 0:
            raise ConfigError(f"hardware node {name}.gpu_memory_gib must be positive")
        if ram <= 0:
            raise ConfigError(f"hardware node {name}.ram_gib must be positive")
        return cls(
            name=name,
            gpu_ids=gpu_ids,
            gpu_model=str(raw.get("gpu_model", "unknown")),
            gpu_memory_gib=gpu_memory,
            cpu_cores=_positive_int(raw.get("cpu_cores"), f"hardware node {name}.cpu_cores"),
            ram_gib=ram,
        )


@dataclass(frozen=True)
class HardwareConfig:
    nodes: tuple[NodeHardware, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HardwareConfig":
        raw = _as_mapping(raw.get("hardware", raw), "hardware")
        nodes_raw = raw.get("nodes")
        if not isinstance(nodes_raw, list) or not nodes_raw:
            raise ConfigError("hardware.nodes must be a nonempty list")
        nodes = tuple(NodeHardware.from_mapping(_as_mapping(item, "hardware node")) for item in nodes_raw)
        names = [node.name for node in nodes]
        if len(set(names)) != len(names):
            raise ConfigError("hardware node names must be unique")
        return cls(nodes=nodes)

    @property
    def by_name(self) -> dict[str, NodeHardware]:
        return {node.name: node for node in self.nodes}

    @property
    def total_gpus(self) -> int:
        return sum(len(node.gpu_ids) for node in self.nodes)

    @property
    def total_cpu_cores(self) -> int:
        return sum(node.cpu_cores for node in self.nodes)

    @property
    def total_ram_gib(self) -> float:
        return sum(node.ram_gib for node in self.nodes)


@dataclass(frozen=True)
class RoleResources:
    """Resources assigned to one process role."""

    nodes: dict[str, tuple[int, ...]]
    cpu_cores: int
    ram_gib: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], label: str) -> "RoleResources":
        nodes_raw = raw.get("nodes", {})
        if not isinstance(nodes_raw, Mapping):
            raise ConfigError(f"runtime.roles.{label}.nodes must be a mapping")
        nodes: dict[str, tuple[int, ...]] = {}
        for node_name, ids_raw in nodes_raw.items():
            if not isinstance(ids_raw, list):
                raise ConfigError(f"runtime.roles.{label}.nodes.{node_name} must be a list")
            try:
                ids = tuple(int(item) for item in ids_raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"runtime role {label} GPU IDs must be integers") from exc
            if any(item < 0 for item in ids) or len(set(ids)) != len(ids):
                raise ConfigError(f"runtime role {label} GPU IDs must be unique nonnegative IDs")
            nodes[str(node_name)] = ids
        cpu = _nonnegative_int(raw.get("cpu_cores", 0), f"runtime.roles.{label}.cpu_cores")
        ram = float(raw.get("ram_gib", 0.0))
        if ram < 0:
            raise ConfigError(f"runtime.roles.{label}.ram_gib must be nonnegative")
        return cls(nodes=nodes, cpu_cores=cpu, ram_gib=ram)

    @property
    def gpu_count(self) -> int:
        return sum(len(ids) for ids in self.nodes.values())


@dataclass(frozen=True)
class RuntimeConfig:
    """Framework/runtime choices; none are reward or policy-objective values."""

    cluster_mode: str
    roles: dict[str, RoleResources]
    ray_num_cpus: int | None
    ray_cpus_per_worker: float
    ray_placement_strategy: str
    ray_address: str | None
    rollout_backend: str
    rollout_tp: int
    rollout_dp: int | None
    rollout_replicas: int | None
    rollout_gpu_memory_utilization: float
    fsdp_world_size: int | None
    sequence_parallel_size: int
    model_gradient_checkpointing: bool
    model_activation_offload: bool
    model_use_remove_padding: bool
    model_use_fused_kernels: bool
    actor_param_offload: bool
    actor_optimizer_offload: bool
    reference_param_offload: bool
    dynamic_batching: bool
    entropy_checkpointing: bool
    entropy_from_logits_with_chunking: bool
    dataloader_num_workers: int
    rollout_multi_stage_wake_up: bool
    actor_micro_batch_size_per_gpu: int
    actor_max_token_len_per_gpu: int
    rollout_logprob_micro_batch_size_per_gpu: int
    rollout_logprob_max_token_len_per_gpu: int
    reference_logprob_micro_batch_size_per_gpu: int
    reference_logprob_max_token_len_per_gpu: int
    rollout_max_num_batched_tokens: int
    rollout_max_model_len: int
    generator_enabled: bool
    generator_backend: str
    generator_host: str
    generator_port: int
    generator_tp: int
    generator_dtype: str
    generator_max_model_len: int
    generator_gpu_memory_utilization: float
    generator_backend_concurrency: int
    generator_seed_workers: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RuntimeConfig":
        raw = _as_mapping(raw.get("runtime", raw), "runtime")
        if "multi_node_supported" in raw:
            raise ConfigError(
                "runtime.multi_node_supported is not a feature switch; "
                "multi-node currently unsupported by the ToolWeave veRL adapter"
            )
        roles_raw = _as_mapping(raw.get("roles", {}), "runtime.roles")
        roles = {
            str(name): RoleResources.from_mapping(_as_mapping(value, f"runtime role {name}"), str(name))
            for name, value in roles_raw.items()
        }
        if "learner" not in roles:
            raise ConfigError("runtime.roles.learner is required")
        ray = _as_mapping(raw.get("ray", {}), "runtime.ray")
        rollout = _as_mapping(raw.get("rollout", {}), "runtime.rollout")
        fsdp = _as_mapping(raw.get("fsdp", {}), "runtime.fsdp")
        batching = _as_mapping(raw.get("batching", {}), "runtime.batching")
        data = _as_mapping(raw.get("data", {}), "runtime.data")
        generator = _as_mapping(raw.get("generator", {}), "runtime.generator")
        ray_num_cpus_raw = ray.get("num_cpus", "auto")
        ray_num_cpus = None if ray_num_cpus_raw == "auto" else _positive_int(ray_num_cpus_raw, "runtime.ray.num_cpus")
        rollout_dp_raw = rollout.get("data_parallel_size", "auto")
        rollout_replicas_raw = rollout.get("replicas", "auto")
        fsdp_world_raw = fsdp.get("world_size", "auto")
        placement = str(ray.get("placement_strategy", "STRICT_PACK")).upper()
        if placement not in {"PACK", "STRICT_PACK", "SPREAD", "STRICT_SPREAD"}:
            raise ConfigError(f"unsupported Ray placement strategy: {placement}")
        memory_utilization = float(rollout.get("gpu_memory_utilization", 0.5))
        if not 0.0 < memory_utilization <= 1.0:
            raise ConfigError("runtime.rollout.gpu_memory_utilization must be in (0, 1]")
        generator_memory = float(generator.get("gpu_memory_utilization", 0.9))
        if not 0.0 < generator_memory <= 1.0:
            raise ConfigError("runtime.generator.gpu_memory_utilization must be in (0, 1]")
        return cls(
            cluster_mode=str(raw.get("cluster_mode", "local")),
            roles=roles,
            ray_num_cpus=ray_num_cpus,
            ray_cpus_per_worker=float(ray.get("cpus_per_worker", 1.0)),
            ray_placement_strategy=placement,
            ray_address=(str(ray["address"]) if ray.get("address") else None),
            rollout_backend=str(rollout.get("backend", "sglang")),
            rollout_tp=_positive_int(rollout.get("tensor_parallel_size", 1), "runtime.rollout.tensor_parallel_size"),
            rollout_dp=(None if rollout_dp_raw == "auto" else _positive_int(rollout_dp_raw, "runtime.rollout.data_parallel_size")),
            rollout_replicas=(None if rollout_replicas_raw == "auto" else _positive_int(rollout_replicas_raw, "runtime.rollout.replicas")),
            rollout_gpu_memory_utilization=memory_utilization,
            fsdp_world_size=(None if fsdp_world_raw == "auto" else _positive_int(fsdp_world_raw, "runtime.fsdp.world_size")),
            sequence_parallel_size=_positive_int(fsdp.get("sequence_parallel_size", 1), "runtime.fsdp.sequence_parallel_size"),
            model_gradient_checkpointing=bool(fsdp.get("gradient_checkpointing", False)),
            model_activation_offload=bool(fsdp.get("activation_offload", False)),
            model_use_remove_padding=bool(fsdp.get("use_remove_padding", True)),
            model_use_fused_kernels=bool(fsdp.get("use_fused_kernels", True)),
            actor_param_offload=bool(fsdp.get("actor_param_offload", False)),
            actor_optimizer_offload=bool(fsdp.get("actor_optimizer_offload", False)),
            reference_param_offload=bool(fsdp.get("reference_param_offload", False)),
            dynamic_batching=bool(batching.get("dynamic", True)),
            entropy_checkpointing=bool(batching.get("entropy_checkpointing", True)),
            entropy_from_logits_with_chunking=bool(batching.get("entropy_from_logits_with_chunking", True)),
            dataloader_num_workers=_nonnegative_int(data.get("dataloader_num_workers", 0), "runtime.data.dataloader_num_workers"),
            rollout_multi_stage_wake_up=bool(rollout.get("multi_stage_wake_up", False)),
            actor_micro_batch_size_per_gpu=_positive_int(batching.get("actor_micro_batch_size_per_gpu", 1), "runtime.batching.actor_micro_batch_size_per_gpu"),
            actor_max_token_len_per_gpu=_positive_int(batching.get("actor_max_token_len_per_gpu", 20480), "runtime.batching.actor_max_token_len_per_gpu"),
            rollout_logprob_micro_batch_size_per_gpu=_positive_int(batching.get("rollout_logprob_micro_batch_size_per_gpu", 1), "runtime.batching.rollout_logprob_micro_batch_size_per_gpu"),
            rollout_logprob_max_token_len_per_gpu=_positive_int(batching.get("rollout_logprob_max_token_len_per_gpu", 65536), "runtime.batching.rollout_logprob_max_token_len_per_gpu"),
            reference_logprob_micro_batch_size_per_gpu=_positive_int(batching.get("reference_logprob_micro_batch_size_per_gpu", 1), "runtime.batching.reference_logprob_micro_batch_size_per_gpu"),
            reference_logprob_max_token_len_per_gpu=_positive_int(batching.get("reference_logprob_max_token_len_per_gpu", 65536), "runtime.batching.reference_logprob_max_token_len_per_gpu"),
            rollout_max_num_batched_tokens=_positive_int(rollout.get("max_num_batched_tokens", 131072), "runtime.rollout.max_num_batched_tokens"),
            rollout_max_model_len=_positive_int(rollout.get("max_model_len", 32768), "runtime.rollout.max_model_len"),
            generator_enabled=bool(generator.get("enabled", False)),
            generator_backend=str(generator.get("backend", "vllm_openai")),
            generator_host=str(generator.get("host", "127.0.0.1")),
            generator_port=_positive_int(generator.get("port", 8000), "runtime.generator.port"),
            generator_tp=_positive_int(generator.get("tensor_parallel_size", 1), "runtime.generator.tensor_parallel_size"),
            generator_dtype=str(generator.get("dtype", "bfloat16")),
            generator_max_model_len=_positive_int(generator.get("max_model_len", 24576), "runtime.generator.max_model_len"),
            generator_gpu_memory_utilization=generator_memory,
            generator_backend_concurrency=_positive_int(generator.get("backend_concurrency", 1), "runtime.generator.backend_concurrency"),
            generator_seed_workers=_positive_int(generator.get("seed_workers", 1), "runtime.generator.seed_workers"),
        )


@dataclass(frozen=True)
class AssetSpec:
    name: str
    kind: str
    path: Path
    required: bool = True
    sha256: str | None = None
    row_count: int | None = None
    required_files: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyPlan:
    """The single derived resource source consumed by framework adapters."""

    cluster_mode: str
    node_names: tuple[str, ...]
    role_assignments: dict[str, dict[str, tuple[int, ...]]]
    cuda_visible_devices: dict[str, str]
    learner_world_size: int
    fsdp_world_size: int
    nnodes: int
    learner_gpus_per_node: tuple[int, ...]
    ray_resource_pool: tuple[int, ...]
    ray_bundles: tuple[dict[str, float], ...]
    ray_num_cpus: int
    ray_placement_strategy: str
    rollout_tp: int
    rollout_dp: int
    rollout_replicas: int
    rollout_backend: str
    generator_world_size: int
    generator_tp: int
    generator_dp: int

    @property
    def uniform_learner_gpus_per_node(self) -> int:
        values = set(self.learner_gpus_per_node)
        if len(values) != 1:
            raise ConfigError(
                "the current veRL adapter requires a uniform learner GPU count per node"
            )
        return self.learner_gpus_per_node[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_mode": self.cluster_mode,
            "node_names": list(self.node_names),
            "role_assignments": {
                role: {node: list(ids) for node, ids in nodes.items()}
                for role, nodes in self.role_assignments.items()
            },
            "cuda_visible_devices": dict(self.cuda_visible_devices),
            "learner_world_size": self.learner_world_size,
            "fsdp_world_size": self.fsdp_world_size,
            "nnodes": self.nnodes,
            "learner_gpus_per_node": list(self.learner_gpus_per_node),
            "ray_resource_pool": list(self.ray_resource_pool),
            "ray_bundles": [dict(item) for item in self.ray_bundles],
            "ray_num_cpus": self.ray_num_cpus,
            "ray_placement_strategy": self.ray_placement_strategy,
            "rollout_tp": self.rollout_tp,
            "rollout_dp": self.rollout_dp,
            "rollout_replicas": self.rollout_replicas,
            "rollout_backend": self.rollout_backend,
            "generator_world_size": self.generator_world_size,
            "generator_tp": self.generator_tp,
            "generator_dp": self.generator_dp,
        }
