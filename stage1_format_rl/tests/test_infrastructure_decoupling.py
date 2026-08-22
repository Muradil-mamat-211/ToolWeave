from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from stage1_format_rl.infrastructure.assets import validate_assets
from stage1_format_rl.infrastructure.cli import (
    build_role_environment,
    build_training_command,
)
from stage1_format_rl.infrastructure.integrations.verl import build_verl_config
from stage1_format_rl.infrastructure.inventory import validate_local_inventory
from stage1_format_rl.infrastructure.loader import load_yaml
from stage1_format_rl.infrastructure.models import (
    ConfigError,
    HardwareConfig,
    RuntimeConfig,
)
from stage1_format_rl.infrastructure.qualification import (
    qualify_observed_reference,
    qualify_reference,
)
from stage1_format_rl.infrastructure.resolver import resolve_profile
from stage1_format_rl.infrastructure.topology import build_topology_plan
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer import main_ppo as main_ppo_module


ROOT = Path(__file__).resolve().parents[2]
LAYERS = ROOT / "stage1_format_rl/configs/layers"
REFERENCE_PROFILE = LAYERS / "profiles/stage3_reference.yaml"
ONLINE_PROFILE = LAYERS / "profiles/stage3_online_2gpu.yaml"
ALTERNATE_PROFILE = LAYERS / "profiles/stage3_portable_8gpu.yaml"
SINGLE_GPU_PROFILE = LAYERS / "profiles/single_gpu_eval.yaml"
GOLDEN = ROOT / "stage1_format_rl/tests/fixtures/stage3_algorithm_golden.json"
DEFAULT_ASSET_ROOT = Path(os.environ.get("TOOLWEAVE_ASSET_ROOT", ROOT)).resolve()


def _environment(tmp_path: Path | None = None, **overrides: str) -> dict[str, str]:
    asset_root = DEFAULT_ASSET_ROOT
    runtime_root = tmp_path or ROOT / ".runtime"
    values = {
        "TOOLWEAVE_SOURCE_ROOT": str(ROOT),
        "TOOLWEAVE_ASSET_ROOT": str(asset_root),
        "TOOLWEAVE_MODELS_ROOT": str(asset_root / "models"),
        "TOOLWEAVE_DATA_ROOT": str(asset_root / "stage1_format_rl/data"),
        "TOOLWEAVE_SHARED_DATA_ROOT": str(asset_root / "data"),
        "TOOLWEAVE_ARTIFACTS_ROOT": str(asset_root / "stage1_format_rl/artifacts"),
        "TOOLWEAVE_OUTPUTS_ROOT": str(runtime_root / "outputs"),
        "TOOLWEAVE_LOGS_ROOT": str(runtime_root / "logs"),
        "TOOLWEAVE_REPORTS_ROOT": str(runtime_root / "reports"),
        "TOOLWEAVE_EVALS_ROOT": str(runtime_root / "evals"),
        "TOOLWEAVE_CACHE_ROOT": str(runtime_root / "cache"),
        "TOOLWEAVE_TEMP_ROOT": str(runtime_root / "temp"),
        "TOOLWEAVE_SHORT_TEMP_ROOT": str(runtime_root / "short-temp"),
        "TOOLWEAVE_PYTHON": sys.executable,
        "TOOLWEAVE_SYNTH_PYTHON": sys.executable,
        "TOOLWEAVE_CONDA_ENV": "",
    }
    values.update(overrides)
    return values


def _resolve(profile: Path, tmp_path: Path | None = None):
    return resolve_profile(profile, environ=_environment(tmp_path))


def _raw_configs() -> tuple[dict, dict]:
    hardware = load_yaml(LAYERS / "hardware/reference_2x_rtx_pro_6000.yaml")
    runtime = load_yaml(LAYERS / "runtime/stage3_reference_training.yaml")
    return hardware, runtime


def _plan(hardware_raw: dict, runtime_raw: dict):
    hardware = HardwareConfig.from_mapping(hardware_raw)
    runtime = RuntimeConfig.from_mapping(runtime_raw)
    return hardware, runtime, build_topology_plan(hardware, runtime)


def test_reference_plan_is_single_source_for_verl_ray_fsdp_and_shell(tmp_path: Path) -> None:
    resolved = _resolve(REFERENCE_PROFILE, tmp_path)
    plan = resolved.topology
    config = resolved.effective_verl
    assert plan.learner_world_size == plan.fsdp_world_size == 2
    assert plan.nnodes == 1
    assert plan.learner_gpus_per_node == (2,)
    assert plan.rollout_tp == 1 and plan.rollout_dp == 2
    assert plan.ray_resource_pool == (2,)
    assert plan.ray_bundles == (
        {"CPU": 1.0, "GPU": 1.0},
        {"CPU": 1.0, "GPU": 1.0},
    )
    assert config["trainer"]["n_gpus_per_node"] == 2
    assert config["trainer"]["nnodes"] == 1
    assert config["trainer"]["ray_placement_strategy"] == "STRICT_PACK"
    assert config["ray_init"]["num_cpus"] == 48
    assert config["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 1
    assert config["data"]["dataloader_num_workers"] == 0
    assert config["actor_rollout_ref"]["model"]["enable_activation_offload"] is False
    assert config["actor_rollout_ref"]["model"]["enable_gradient_checkpointing"] is True
    assert config["actor_rollout_ref"]["model"]["use_remove_padding"] is True
    assert config["actor_rollout_ref"]["model"]["use_fused_kernels"] is True
    assert config["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is True
    assert config["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] == 20480
    assert config["actor_rollout_ref"]["actor"]["fsdp_config"] == {
        "param_offload": False,
        "optimizer_offload": True,
    }
    assert config["actor_rollout_ref"]["rollout"]["log_prob_use_dynamic_bsz"] is True
    assert config["actor_rollout_ref"]["rollout"]["max_model_len"] == 32768
    assert config["actor_rollout_ref"]["ref"]["fsdp_config"]["param_offload"] is True
    shell = build_role_environment(resolved, "learner")
    assert shell["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert shell["TOOLWEAVE_LEARNER_WORLD_SIZE"] == "2"
    assert shell["TOOLWEAVE_RAY_NUM_CPUS"] == "48"
    assert shell["TOOLWEAVE_SEQUENCE_PARALLEL_SIZE"] == "2"
    assert shell["TOOLWEAVE_ROLLOUT_MAX_MODEL_LEN"] == "32768"
    assert shell["TOOLWEAVE_ROLLOUT_MAX_NUM_BATCHED_TOKENS"] == "131072"


def test_alternate_8x80_topology_requires_no_python_branch(tmp_path: Path) -> None:
    resolved = _resolve(ALTERNATE_PROFILE, tmp_path)
    plan = resolved.topology
    assert resolved.mode == "portable"
    assert plan.learner_world_size == 8
    assert plan.fsdp_world_size == 8
    assert plan.rollout_tp == 2
    assert plan.rollout_dp == plan.rollout_replicas == 4
    assert plan.ray_resource_pool == (8,)
    assert len(plan.ray_bundles) == 8
    assert all(bundle == {"CPU": 1.0, "GPU": 1.0} for bundle in plan.ray_bundles)
    assert resolved.effective_verl["trainer"]["n_gpus_per_node"] == 8
    assert resolved.effective_verl["trainer"]["ray_placement_strategy"] == "PACK"
    command = build_training_command(
        resolved, tmp_path / "resolved-portable-8gpu.yaml"
    )
    assert command[:3] == [sys.executable, "-m", "verl.trainer.main_ppo"]


def test_alternate_topology_launcher_dry_run(tmp_path: Path) -> None:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT),
        **_environment(tmp_path),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stage1_format_rl.infrastructure.cli",
            "--profile",
            str(ALTERNATE_PROFILE),
            "launch",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["topology"]["learner_world_size"] == 8
    assert payload["topology"]["rollout_dp"] == 4


def test_single_gpu_profile_uses_a_matching_hardware_inventory(tmp_path: Path) -> None:
    resolved = _resolve(SINGLE_GPU_PROFILE, tmp_path)
    assert resolved.hardware.total_gpus == 1
    assert resolved.topology.learner_world_size == 1
    assert resolved.topology.cuda_visible_devices["learner"] == "0"


def test_online_roles_are_process_isolated_and_generator_is_derived(tmp_path: Path) -> None:
    resolved = _resolve(ONLINE_PROFILE, tmp_path)
    assert resolved.topology.cuda_visible_devices == {
        "learner": "1",
        "generator": "0",
    }
    assert resolved.topology.generator_world_size == 1
    assert resolved.topology.generator_tp == resolved.topology.generator_dp == 1
    assert resolved.effective_generator is not None
    deployment = resolved.effective_generator["deployment"]
    assert deployment["cuda_visible_devices"] == "0"
    assert deployment["tensor_parallel_size"] == 1
    assert resolved.effective_generator["llm"]["concurrency"] == 4
    assert resolved.effective_generator["seed_worker_count"] == 4
    assert resolved.effective_verl["actor_rollout_ref"]["rollout"]["multi_stage_wake_up"] is True


def test_machine_paths_change_without_experiment_or_python_changes(tmp_path: Path) -> None:
    custom = tmp_path / "machine"
    env = _environment(
        tmp_path,
        TOOLWEAVE_ASSET_ROOT=str(custom / "assets"),
        TOOLWEAVE_MODELS_ROOT=str(custom / "models"),
        TOOLWEAVE_DATA_ROOT=str(custom / "datasets"),
        TOOLWEAVE_ARTIFACTS_ROOT=str(custom / "artifacts"),
        TOOLWEAVE_OUTPUTS_ROOT=str(custom / "outputs"),
    )
    resolved = resolve_profile(REFERENCE_PROFILE, environ=env)
    assert resolved.assets["stage3_train_400"].path == custom / "datasets/bfcl_stage3_train_all_400_shuffled_seed42.parquet"
    assert resolved.assets["stage2_step25_model"].path == custom / "assets/stage1_format_rl/artifacts/stage2_eval/merged/global_step_25"
    assert resolved.effective_verl["trainer"]["rods_stage3_lifecycle"]["seed_output_path"] == str(
        custom / "artifacts/stage3_queues/boundary_seeds.jsonl"
    )
    assert resolved.effective_verl["trainer"]["default_local_dir"] == str(
        custom / "outputs/stage3_rods_matchtir_v1_training_branch/checkpoints"
    )
    assert resolved.effective_verl["algorithm"]["matchtir_local"]["gamma"] == 0.9


def test_machine_cleanup_helper_is_confined_to_runtime_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "outputs" / "job"
    forbidden = tmp_path / "outside"
    allowed.mkdir(parents=True)
    forbidden.mkdir()
    helper = ROOT / "stage1_format_rl/scripts/_machine.sh"
    env = {"PATH": os.environ.get("PATH", ""), **_environment(tmp_path)}

    removed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; toolweave_safe_rm_rf "$2"',
            "bash",
            str(helper),
            str(allowed),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert removed.returncode == 0, removed.stderr
    assert not allowed.exists()

    rejected_env = dict(env)
    rejected_env["TOOLWEAVE_OUTPUTS_ROOT"] = "/"
    rejected = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; toolweave_safe_rm_rf "$2"',
            "bash",
            str(helper),
            str(forbidden),
        ],
        env=rejected_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert forbidden.is_dir()


def test_asset_manifest_checks_hashes_and_parquet_rows_without_loading_rows(tmp_path: Path) -> None:
    resolved = _resolve(REFERENCE_PROFILE, tmp_path)
    validate_assets(resolved.assets)


def test_algorithm_sources_and_fields_equal_approved_formal_golden(tmp_path: Path) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for relative, expected in golden["source_sha256"].items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected, relative

    cfg = _resolve(REFERENCE_PROFILE, tmp_path).effective_verl
    actor = cfg["actor_rollout_ref"]["actor"]
    algorithm = cfg["algorithm"]
    actual = {
        "actor": {
            "clip_ratio_c": actor["clip_ratio_c"],
            "clip_ratio_high": actor["clip_ratio_high"],
            "clip_ratio_low": actor["clip_ratio_low"],
            "entropy_coeff": actor["entropy_coeff"],
            "grad_clip": actor["grad_clip"],
            "kl_loss_coef": actor["kl_loss_coef"],
            "kl_loss_type": actor["kl_loss_type"],
            "loss_agg_mode": actor["loss_agg_mode"],
            "lr": actor["optim"]["lr"],
            "ppo_epochs": actor["ppo_epochs"],
            "ppo_mini_batch_size": actor["ppo_mini_batch_size"],
            "use_kl_loss": actor["use_kl_loss"],
        },
        "algorithm": copy.deepcopy(algorithm),
        "data": {
            key: cfg["data"][key]
            for key in ("max_prompt_length", "max_response_length", "train_batch_size")
        },
        "lifecycle": {
            key: copy.deepcopy(cfg["trainer"]["rods_stage3_lifecycle"][key])
            for key in golden["algorithm_fields"]["lifecycle"]
        },
        "rollout": {
            key: cfg["actor_rollout_ref"]["rollout"][key]
            for key in ("n", "temperature", "top_k", "top_p")
        },
        "trainer": {"total_epochs": cfg["trainer"]["total_epochs"]},
    }
    assert actual == golden["algorithm_fields"]


def test_ray_override_reaches_framework_resource_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hardware_raw, runtime_raw = _raw_configs()
    runtime_raw["runtime"]["ray"]["placement_strategy"] = "SPREAD"
    runtime_raw["runtime"]["ray"]["cpus_per_worker"] = 2
    hardware, runtime, plan = _plan(hardware_raw, runtime_raw)
    assert hardware.total_gpus == 2
    resolved = _resolve(REFERENCE_PROFILE, tmp_path)
    config = build_verl_config(
        resolved.experiment, resolved.assets, resolved.machine, runtime, plan
    )
    assert config["trainer"]["ray_placement_strategy"] == "SPREAD"
    assert config["trainer"]["ray_cpus_per_worker"] == 2.0
    manager = ResourcePoolManager(
        resource_pool_spec={"global_pool": list(plan.ray_resource_pool)},
        mapping={Role.ActorRollout: "global_pool"},
        placement_strategy=config["trainer"]["ray_placement_strategy"],
        cpus_per_worker=config["trainer"]["ray_cpus_per_worker"],
    )
    monkeypatch.setattr(manager, "_check_resource_available", lambda: None)
    manager.create_resource_pool()
    pool = manager.resource_pool_dict["global_pool"]
    assert pool.placement_strategy == "SPREAD"
    assert pool.cpus_per_worker == 2.0


def test_experiment_cannot_redeclare_topology(tmp_path: Path) -> None:
    resolved = _resolve(REFERENCE_PROFILE, tmp_path)
    experiment = copy.deepcopy(resolved.experiment)
    experiment["verl"].setdefault("trainer", {})["nnodes"] = 99
    with pytest.raises(ConfigError, match="illegally owns runtime/topology"):
        build_verl_config(
            experiment,
            resolved.assets,
            resolved.machine,
            resolved.runtime,
            resolved.topology,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda _h, r: r["runtime"]["roles"]["learner"]["nodes"].update({"node-0": [0, 2]}), "out of range"),
        (lambda _h, r: r["runtime"]["roles"]["generator"].update({"nodes": {"node-0": [1]}}), "overlaps roles"),
        (lambda _h, r: r["runtime"]["roles"]["learner"].update({"cpu_cores": 51}), "CPU resources insufficient"),
        (lambda _h, r: r["runtime"]["roles"]["learner"].update({"ram_gib": 241}), "RAM resources insufficient"),
        (lambda _h, r: r["runtime"]["fsdp"].update({"world_size": 1}), "illegal FSDP world size"),
        (lambda _h, r: r["runtime"]["rollout"].update({"tensor_parallel_size": 3}), "divisible by rollout tensor parallel"),
        (lambda _h, r: r["runtime"]["ray"].update({"num_cpus": 1, "cpus_per_worker": 1}), "Ray resources insufficient"),
    ],
)
def test_generic_resource_negative_cases(mutation, message: str) -> None:
    hardware_raw, runtime_raw = _raw_configs()
    mutation(hardware_raw, runtime_raw)
    with pytest.raises(ConfigError, match=message):
        _plan(hardware_raw, runtime_raw)


def test_illegal_rollout_dp_is_rejected() -> None:
    hardware_raw, runtime_raw = _raw_configs()
    runtime_raw["runtime"]["rollout"]["data_parallel_size"] = 1
    with pytest.raises(ConfigError, match="illegal rollout TP/DP relation"):
        _plan(hardware_raw, runtime_raw)


def test_multinode_fails_closed_when_runtime_does_not_support_it() -> None:
    hardware_raw, runtime_raw = _raw_configs()
    node = copy.deepcopy(hardware_raw["hardware"]["nodes"][0])
    node.update({"name": "node-1", "gpu_ids": [0, 1]})
    hardware_raw["hardware"]["nodes"].append(node)
    learner = runtime_raw["runtime"]["roles"]["learner"]
    learner.update({"nodes": {"node-0": [0], "node-1": [0]}, "cpu_cores": 0, "ram_gib": 0})
    with pytest.raises(ConfigError, match="multi-node currently unsupported"):
        _plan(hardware_raw, runtime_raw)


def test_multinode_cannot_be_enabled_by_a_profile_flag() -> None:
    _hardware_raw, runtime_raw = _raw_configs()
    runtime_raw["runtime"]["multi_node_supported"] = True
    with pytest.raises(ConfigError, match="not a feature switch"):
        RuntimeConfig.from_mapping(runtime_raw)


def test_reference_qualification_is_strict_but_portable_mode_is_not(tmp_path: Path) -> None:
    portable = _resolve(ALTERNATE_PROFILE, tmp_path)
    assert portable.mode == "portable" and portable.topology.learner_world_size == 8
    reference = _resolve(REFERENCE_PROFILE, tmp_path)
    with pytest.raises(ConfigError, match="reference qualification mismatch"):
        qualify_reference(
            {"qualification": {"total_gpus": 7}},
            reference.hardware,
            reference.runtime,
            reference.topology,
        )


def test_portable_capacity_validation_ignores_gpu_brand_but_reference_does_not(
    tmp_path: Path,
) -> None:
    resolved = _resolve(REFERENCE_PROFILE, tmp_path)
    observed = {
        "cpu_cores": 50,
        "ram_gib": 240.0,
        "gpus": [
            {"id": 0, "model": "Equivalent Accelerator", "memory_gib": 95.6},
            {"id": 1, "model": "Equivalent Accelerator", "memory_gib": 95.6},
        ],
    }
    validate_local_inventory(resolved.hardware, observed)
    with pytest.raises(ConfigError, match="observed reference qualification mismatch for gpu_model"):
        qualify_observed_reference(resolved.qualification, observed)


def test_observed_reference_qualification_accepts_nominal_capacity_reporting(
    tmp_path: Path,
) -> None:
    resolved = _resolve(REFERENCE_PROFILE, tmp_path)
    observed = {
        "cpu_cores": 50,
        "ram_gib": 240.0,
        "gpus": [
            {
                "id": gpu_id,
                "model": "NVIDIA RTX PRO 6000 Blackwell",
                "memory_gib": 95.6,
            }
            for gpu_id in (0, 1)
        ],
    }
    validate_local_inventory(resolved.hardware, observed)
    qualify_observed_reference(resolved.qualification, observed)


def test_existing_ray_address_is_runtime_owned(tmp_path: Path) -> None:
    hardware_raw, runtime_raw = _raw_configs()
    runtime_raw["runtime"]["cluster_mode"] = "existing"
    runtime_raw["runtime"]["ray"]["address"] = "ray://scheduler.example:10001"
    _hardware, runtime, plan = _plan(hardware_raw, runtime_raw)
    resolved = _resolve(REFERENCE_PROFILE, tmp_path)
    config = build_verl_config(
        resolved.experiment, resolved.assets, resolved.machine, runtime, plan
    )
    assert plan.cluster_mode == "existing"
    assert config["ray_init"]["address"] == "ray://scheduler.example:10001"

    runtime_raw["runtime"]["ray"]["address"] = None
    with pytest.raises(ConfigError, match="requires runtime.ray.address"):
        _plan(hardware_raw, runtime_raw)


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("ray://scheduler.example:10001", {"address": "ray://scheduler.example:10001"}),
        (None, {"num_cpus": 17}),
    ],
)
def test_ray_init_consumes_local_or_existing_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
    address: str | None,
    expected: dict[str, object],
) -> None:
    calls: list[dict[str, object]] = []

    class RemoteMethod:
        def remote(self, *_args, **_kwargs):
            return "task-ref"

    class Runner:
        run = RemoteMethod()

    class DummyTaskRunner:
        @staticmethod
        def remote():
            return Runner()

        @staticmethod
        def options(**_kwargs):
            return DummyTaskRunner

    monkeypatch.setattr(main_ppo_module.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(
        main_ppo_module.ray, "init", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(main_ppo_module.ray, "get", lambda _value: None)
    monkeypatch.setattr(main_ppo_module, "TaskRunner", DummyTaskRunner)
    config = OmegaConf.create(
        {
            "ray_init": {
                "address": address,
                "num_cpus": 17,
                "timeline_json_file": None,
            },
            "trainer": {"profile_steps": []},
        }
    )
    main_ppo_module.run_ppo(config)
    assert len(calls) == 1
    for key, value in expected.items():
        assert calls[0][key] == value
    if address:
        assert "num_cpus" not in calls[0]
    else:
        assert "address" not in calls[0]
