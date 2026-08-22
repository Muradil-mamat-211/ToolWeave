"""Generic topology derivation and resource validation."""

from __future__ import annotations

from collections import defaultdict

from .models import ConfigError, HardwareConfig, RuntimeConfig, TopologyPlan


def _validate_role_resources(hardware: HardwareConfig, runtime: RuntimeConfig) -> None:
    nodes = hardware.by_name
    assigned: dict[tuple[str, int], str] = {}
    cpu_by_node: dict[str, int] = defaultdict(int)
    ram_by_node: dict[str, float] = defaultdict(float)

    # Current role CPU/RAM requests describe a single-node process role.  The
    # architecture can represent multiple nodes, while unsupported multi-node
    # execution fails explicitly below instead of guessing a distribution.
    for role, resources in runtime.roles.items():
        for node_name, gpu_ids in resources.nodes.items():
            if node_name not in nodes:
                raise ConfigError(f"runtime role {role} references unknown node {node_name}")
            available = set(nodes[node_name].gpu_ids)
            for gpu_id in gpu_ids:
                if gpu_id not in available:
                    raise ConfigError(
                        f"runtime role {role} GPU ID {gpu_id} is out of range for {node_name}"
                    )
                key = (node_name, gpu_id)
                if key in assigned:
                    raise ConfigError(
                        f"GPU {node_name}:{gpu_id} overlaps roles {assigned[key]} and {role}"
                    )
                assigned[key] = role
        used_nodes = [name for name, ids in resources.nodes.items() if ids]
        if resources.cpu_cores or resources.ram_gib:
            if len(used_nodes) > 1:
                raise ConfigError(
                    f"runtime role {role} spans multiple nodes but CPU/RAM distribution is unspecified"
                )
            target = used_nodes[0] if used_nodes else hardware.nodes[0].name
            cpu_by_node[target] += resources.cpu_cores
            ram_by_node[target] += resources.ram_gib

    for node_name, requested in cpu_by_node.items():
        available = nodes[node_name].cpu_cores
        if requested > available:
            raise ConfigError(
                f"CPU resources insufficient on {node_name}: requested {requested}, available {available}"
            )
    for node_name, requested in ram_by_node.items():
        available = nodes[node_name].ram_gib
        if requested > available:
            raise ConfigError(
                f"RAM resources insufficient on {node_name}: requested {requested:g} GiB, available {available:g} GiB"
            )


def build_topology_plan(hardware: HardwareConfig, runtime: RuntimeConfig) -> TopologyPlan:
    """Derive every framework topology value from hardware plus role config.

    There are no GPU-count-specific branches.  Unsupported capabilities are
    rejected through generic invariants.
    """

    if runtime.cluster_mode not in {"local", "existing"}:
        raise ConfigError("runtime.cluster_mode must be 'local' or 'existing'")
    _validate_role_resources(hardware, runtime)

    learner = runtime.roles["learner"]
    learner_nodes = tuple(
        node.name for node in hardware.nodes if learner.nodes.get(node.name, ())
    )
    if not learner_nodes or learner.gpu_count == 0:
        raise ConfigError("runtime role learner must own at least one GPU")
    if len(learner_nodes) > 1:
        # The data model deliberately represents nodes, but this release has
        # no audited remote-node launcher.  A profile flag must never be able
        # to turn an unimplemented capability into an apparently valid plan.
        raise ConfigError("multi-node currently unsupported by the ToolWeave veRL adapter")
    if runtime.cluster_mode == "local" and len(learner_nodes) != 1:
        raise ConfigError("cluster_mode=local supports exactly one learner node")
    if runtime.cluster_mode == "existing" and not runtime.ray_address:
        raise ConfigError("cluster_mode=existing requires runtime.ray.address")

    world_size = learner.gpu_count
    fsdp_world_size = runtime.fsdp_world_size or world_size
    if fsdp_world_size != world_size:
        raise ConfigError(
            f"illegal FSDP world size relation: configured {fsdp_world_size}, derived {world_size}"
        )
    if world_size % runtime.sequence_parallel_size:
        raise ConfigError(
            "learner world size must be divisible by FSDP/Ulysses sequence_parallel_size"
        )
    if world_size % runtime.rollout_tp:
        raise ConfigError("learner world size must be divisible by rollout tensor parallel size")
    rollout_dp = world_size // runtime.rollout_tp
    if runtime.rollout_dp is not None and runtime.rollout_dp != rollout_dp:
        raise ConfigError(
            f"illegal rollout TP/DP relation: world_size={world_size}, "
            f"tp={runtime.rollout_tp}, configured_dp={runtime.rollout_dp}"
        )
    rollout_replicas = runtime.rollout_replicas or rollout_dp
    if rollout_replicas != rollout_dp:
        raise ConfigError(
            f"rollout replicas must equal derived data parallel size {rollout_dp} "
            f"for the current hybrid adapter"
        )
    generator = runtime.roles.get("generator")
    generator_world_size = generator.gpu_count if generator else 0
    if runtime.generator_enabled:
        if generator_world_size == 0:
            raise ConfigError("runtime.generator.enabled requires GPU resources for role generator")
        if generator_world_size % runtime.generator_tp:
            raise ConfigError(
                "generator GPU count must be divisible by generator tensor parallel size"
            )
        generator_dp = generator_world_size // runtime.generator_tp
    else:
        if generator_world_size:
            raise ConfigError(
                "generator GPUs are assigned while runtime.generator.enabled is false"
            )
        generator_dp = 0
    if runtime.ray_cpus_per_worker <= 0:
        raise ConfigError("runtime.ray.cpus_per_worker must be positive")

    learner_cpu = learner.cpu_cores
    minimum_ray_cpu = world_size * runtime.ray_cpus_per_worker
    ray_num_cpus = runtime.ray_num_cpus or learner_cpu
    if ray_num_cpus <= 0:
        raise ConfigError("Ray requires a positive CPU allocation")
    if ray_num_cpus > learner_cpu:
        raise ConfigError(
            f"Ray resources exceed learner CPU allocation: requested {ray_num_cpus}, available {learner_cpu}"
        )
    if ray_num_cpus < minimum_ray_cpu:
        raise ConfigError(
            f"Ray resources insufficient: {ray_num_cpus} CPUs cannot satisfy "
            f"{world_size} workers x {runtime.ray_cpus_per_worker:g} CPU"
        )

    assignments = {
        role: {node: tuple(ids) for node, ids in resources.nodes.items() if ids}
        for role, resources in runtime.roles.items()
    }
    cuda_visible = {
        role: ",".join(
            str(gpu_id)
            for node in hardware.nodes
            for gpu_id in resources.nodes.get(node.name, ())
        )
        for role, resources in runtime.roles.items()
        if resources.gpu_count
    }
    counts = tuple(len(learner.nodes.get(name, ())) for name in learner_nodes)
    bundles = tuple(
        {"CPU": float(runtime.ray_cpus_per_worker), "GPU": 1.0}
        for _ in range(world_size)
    )

    return TopologyPlan(
        cluster_mode=runtime.cluster_mode,
        node_names=learner_nodes,
        role_assignments=assignments,
        cuda_visible_devices=cuda_visible,
        learner_world_size=world_size,
        fsdp_world_size=fsdp_world_size,
        nnodes=len(learner_nodes),
        learner_gpus_per_node=counts,
        ray_resource_pool=counts,
        ray_bundles=bundles,
        ray_num_cpus=ray_num_cpus,
        ray_placement_strategy=runtime.ray_placement_strategy,
        rollout_tp=runtime.rollout_tp,
        rollout_dp=rollout_dp,
        rollout_replicas=rollout_replicas,
        rollout_backend=runtime.rollout_backend,
        generator_world_size=generator_world_size,
        generator_tp=(runtime.generator_tp if runtime.generator_enabled else 0),
        generator_dp=generator_dp,
    )
