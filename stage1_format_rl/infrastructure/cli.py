"""Resolve, preflight, dry-run, or launch a layered ToolWeave profile."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from .assets import validate_assets
from .inventory import discover_local_inventory, validate_local_inventory
from .models import ConfigError
from .qualification import qualify_observed_reference
from .resolver import ResolvedProjectConfig, resolve_profile


TRAINING_GUARD = "ALLOW_RODS_MATCHTIR_STAGE3_TRAINING"
GENERATOR_SERVER_GUARD = "RODS_ALLOW_VLLM_SERVER"
EVALUATION_SERVER_GUARD = "RODS_ALLOW_QWEN_EVAL_AFTER_GENERATOR"


class EvaluationLifecycleError(ConfigError):
    """The existing Generator lifecycle guard has not been satisfied."""


class EvaluationGpuPreflightError(ConfigError):
    """The topology-selected learner GPU is not ready for evaluation."""


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _runtime_config_path(resolved: ResolvedProjectConfig) -> Path:
    return (
        resolved.machine.temp_root
        / "resolved_configs"
        / f"{resolved.profile_path.stem}.yaml"
    )


def build_training_command(resolved: ResolvedProjectConfig, config_path: Path) -> list[str]:
    return [
        resolved.machine.python_executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config_path.parent}",
        f"--config-name={config_path.stem}",
    ]


def build_generator_server_command(resolved: ResolvedProjectConfig) -> list[str]:
    if resolved.effective_generator is None:
        raise ConfigError("profile does not define a Generator experiment")
    generator = resolved.effective_generator
    deployment = generator.get("deployment", {})
    model = generator.get("llm", {}).get("model")
    if not model:
        raise ConfigError("resolved Generator model is missing")
    return [
        resolved.machine.synthesis_python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--host",
        resolved.runtime.generator_host,
        "--port",
        str(resolved.runtime.generator_port),
        "--tensor-parallel-size",
        str(deployment["tensor_parallel_size"]),
        "--dtype",
        str(deployment["dtype"]),
        "--max-model-len",
        str(deployment["max_model_len"]),
        "--gpu-memory-utilization",
        str(deployment["gpu_memory_utilization"]),
        "--served-model-name",
        str(model),
    ]


def selected_role_physical_gpu(
    resolved: ResolvedProjectConfig, role: str = "learner"
) -> int:
    """Return the sole physical GPU for a single-GPU role, or fail closed."""

    assignments = resolved.topology.role_assignments.get(role, {})
    selected = [gpu_id for gpu_ids in assignments.values() for gpu_id in gpu_ids]
    if len(selected) != 1:
        raise ConfigError(
            "single-GPU evaluation requires exactly one "
            f"{role} physical GPU; resolved {len(selected)}"
        )
    return selected[0]


def build_evaluation_server_command(
    resolved: ResolvedProjectConfig, backend: str
) -> list[str]:
    """Build a specialized eval command from runtime plus TopologyPlan."""

    runtime = resolved.runtime.evaluation_server
    model = resolved.machine.models_root / "Qwen3-4B-RODS"
    tensor_parallel_size = str(resolved.topology.rollout_tp)
    if backend == "sglang":
        command = [
            resolved.machine.python_executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model),
            "--served-model-name",
            "Qwen3-4B-RODS",
            "--host",
            runtime.host,
            "--port",
            str(runtime.port),
            "--tp-size",
            tensor_parallel_size,
            "--dtype",
            runtime.dtype,
            "--context-length",
            str(runtime.sglang_context_length),
            "--mem-fraction-static",
            str(runtime.sglang_memory_fraction_static),
            "--max-running-requests",
            str(runtime.sglang_max_running_requests),
            "--cuda-graph-max-bs",
            str(runtime.sglang_cuda_graph_max_batch_size),
            "--attention-backend",
            runtime.sglang_attention_backend,
            "--sampling-backend",
            runtime.sglang_sampling_backend,
        ]
        if runtime.sglang_enable_tokenizer_batch_encode:
            command.append("--enable-tokenizer-batch-encode")
        return command
    if backend == "vllm":
        command = [
            str(Path(resolved.machine.synthesis_python_executable).parent / "vllm"),
            "serve",
            str(model),
            "--host",
            runtime.host,
            "--port",
            str(runtime.port),
            "--api-key",
            "EMPTY",
            "--served-model-name",
            "Qwen3-4B-RODS",
            "--tensor-parallel-size",
            tensor_parallel_size,
            "--dtype",
            runtime.dtype,
            "--max-model-len",
            str(runtime.vllm_max_model_len),
            "--gpu-memory-utilization",
            str(runtime.vllm_gpu_memory_utilization),
            "--max-num-seqs",
            str(runtime.vllm_max_num_seqs),
            "--max-num-batched-tokens",
            str(runtime.vllm_max_num_batched_tokens),
        ]
        if runtime.vllm_enable_prefix_caching:
            command.append("--enable-prefix-caching")
        if runtime.vllm_language_model_only:
            command.append("--language-model-only")
        if runtime.vllm_skip_mm_profiling:
            command.append("--skip-mm-profiling")
        command.extend(["--generation-config", runtime.vllm_generation_config])
        return command
    raise ConfigError(f"unsupported evaluation server backend: {backend}")


def _check_evaluation_lifecycle(backend: str) -> None:
    pattern = (
        "gemma-4-31B|vllm.entrypoints.openai.api_server"
        if backend == "sglang"
        else "gemma-4-31B|vllm.entrypoints.openai.api_server.*gemma"
    )
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        detail = result.stdout.strip()
        message = (
            "Refusing to start Qwen eval while Gemma/vLLM is still present."
            if backend == "sglang"
            else "Refusing to start Qwen eval while the Gemma generator is present."
        )
        raise EvaluationLifecycleError(
            f"{message}{os.linesep + detail if detail else ''}"
        )
    if result.returncode not in {0, 1}:
        raise EvaluationLifecycleError(
            f"cannot inspect evaluation lifecycle with pgrep: {result.stderr.strip()}"
        )


def check_selected_learner_gpu(
    selected_physical_gpu: int, max_used_memory_mib: int
) -> int:
    """Query only the topology-selected physical GPU and enforce release."""

    command = [
        "nvidia-smi",
        f"--id={selected_physical_gpu}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise EvaluationGpuPreflightError(
            "selected learner GPU "
            f"{selected_physical_gpu} could not be inspected: {exc}"
        ) from exc
    observed = result.stdout.strip()
    if result.returncode != 0 or not observed.isdigit():
        detail = result.stderr.strip() or observed or "unknown"
        raise EvaluationGpuPreflightError(
            "selected learner GPU "
            f"{selected_physical_gpu} returned invalid nvidia-smi data: {detail}"
        )
    used_mib = int(observed)
    if used_mib > max_used_memory_mib:
        raise EvaluationGpuPreflightError(
            "selected learner GPU "
            f"{selected_physical_gpu} is not released (used={used_mib} MiB)"
        )
    return used_mib


def build_role_environment(
    resolved: ResolvedProjectConfig, role: str
) -> dict[str, str]:
    """Derive process-local resource variables from the single TopologyPlan."""

    resources = resolved.runtime.roles.get(role)
    visible = resolved.topology.cuda_visible_devices.get(role)
    if resources is None or not visible:
        raise ConfigError(f"runtime role {role!r} has no assigned GPU resources")
    threads = max(1, resources.cpu_cores // max(1, resources.gpu_count))
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "NUMEXPR_MAX_THREADS": str(max(1, resources.cpu_cores)),
        "RAYON_NUM_THREADS": str(max(1, resources.cpu_cores)),
        "TOOLWEAVE_ROLE_GPU_COUNT": str(resources.gpu_count),
        "TOOLWEAVE_LEARNER_WORLD_SIZE": str(resolved.topology.learner_world_size),
        "TOOLWEAVE_LEARNER_GPUS_PER_NODE": str(
            resolved.topology.uniform_learner_gpus_per_node
        ),
        "TOOLWEAVE_NNODES": str(resolved.topology.nnodes),
        "TOOLWEAVE_RAY_NUM_CPUS": str(resolved.topology.ray_num_cpus),
        "TOOLWEAVE_ROLLOUT_TP": str(resolved.topology.rollout_tp),
        "TOOLWEAVE_ROLLOUT_DP": str(resolved.topology.rollout_dp),
        "TOOLWEAVE_SEQUENCE_PARALLEL_SIZE": str(
            resolved.runtime.sequence_parallel_size
        ),
        "TOOLWEAVE_ROLLOUT_GPU_MEMORY_UTILIZATION": str(
            resolved.runtime.rollout_gpu_memory_utilization
        ),
        "TOOLWEAVE_ROLLOUT_MAX_MODEL_LEN": str(
            resolved.runtime.rollout_max_model_len
        ),
        "TOOLWEAVE_ROLLOUT_MAX_NUM_BATCHED_TOKENS": str(
            resolved.runtime.rollout_max_num_batched_tokens
        ),
        "TOOLWEAVE_GENERATOR_TP": str(resolved.topology.generator_tp),
        "TOOLWEAVE_GENERATOR_DP": str(resolved.topology.generator_dp),
    }


def _resolve(args: argparse.Namespace) -> ResolvedProjectConfig:
    return resolve_profile(
        args.profile,
        mode=args.mode,
        machine_path=args.machine,
    )


def command_resolve(args: argparse.Namespace) -> int:
    resolved = _resolve(args)
    payload = resolved.summary()
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.suffix in {".yaml", ".yml"}:
            _write_yaml(output, payload)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    resolved = _resolve(args)
    if args.check_assets:
        validate_assets(resolved.assets)
    observed = None
    if args.observe_hardware:
        observed = discover_local_inventory()
        validate_local_inventory(resolved.hardware, observed)
        if resolved.mode == "reference":
            qualify_observed_reference(resolved.qualification, observed)
    output_path = _runtime_config_path(resolved)
    _write_yaml(output_path, resolved.effective_verl)
    result = {
        "status": "PASS",
        "mode": resolved.mode,
        "topology": resolved.topology.to_dict(),
        "assets_checked": bool(args.check_assets),
        "hardware_observed": observed,
        "resolved_verl_config": str(output_path),
        "command": build_training_command(resolved, output_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_launch(args: argparse.Namespace) -> int:
    resolved = _resolve(args)
    if args.execute and os.environ.get(TRAINING_GUARD) != "1":
        raise ConfigError(
            f"formal training is disabled; set {TRAINING_GUARD}=1 only after explicit approval"
        )
    # Command/topology dry-runs must remain usable without heavyweight model
    # and dataset assets. Execution remains fail-closed on the full manifest.
    if args.execute:
        validate_assets(resolved.assets)
    if args.observe_hardware:
        observed = discover_local_inventory()
        validate_local_inventory(resolved.hardware, observed)
        if resolved.mode == "reference":
            qualify_observed_reference(resolved.qualification, observed)
    config_path = _runtime_config_path(resolved)
    _write_yaml(config_path, resolved.effective_verl)
    command = build_training_command(resolved, config_path)
    if not args.execute:
        print(json.dumps({"dry_run": True, "command": command, "topology": resolved.topology.to_dict()}, indent=2, sort_keys=True))
        return 0
    env = os.environ.copy()
    env.update(build_role_environment(resolved, "learner"))
    envtuning = resolved.machine.source_root / "code/AWorld-RL-stage1-worktree/EnvTuning"
    python_paths = [str(envtuning), str(envtuning / "verl")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return subprocess.call(command, cwd=envtuning, env=env)


def command_generator_server(args: argparse.Namespace) -> int:
    resolved = _resolve(args)
    if resolved.effective_generator is None:
        raise ConfigError("profile does not define a Generator experiment")
    command = build_generator_server_command(resolved)
    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "command": command,
                    "cuda_visible_devices": resolved.topology.cuda_visible_devices.get(
                        "generator", ""
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if os.environ.get(GENERATOR_SERVER_GUARD) != "1":
        raise ConfigError(
            f"Generator server launch is disabled; set {GENERATOR_SERVER_GUARD}=1 explicitly"
        )
    validate_assets(resolved.assets, check_rows=False)
    env = os.environ.copy()
    env.update(build_role_environment(resolved, "generator"))
    return subprocess.call(command, env=env)


def command_generator_daemon(args: argparse.Namespace) -> int:
    resolved = _resolve(args)
    if resolved.effective_generator is None:
        raise ConfigError("profile does not define a Generator experiment")
    output = (
        resolved.machine.temp_root
        / "resolved_configs"
        / f"{resolved.profile_path.stem}_generator.yaml"
    )
    _write_yaml(output, resolved.effective_generator)
    command = [
        resolved.machine.synthesis_python_executable,
        "-m",
        "env_tuning.rods_data_generation_v1.launcher",
        "--config",
        str(output),
    ]
    if args.once:
        command.append("--once")
    if args.allow_generation:
        command.append("--allow-generation")
    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "command": command,
                    "config": str(output),
                    "cuda_visible_devices": resolved.topology.cuda_visible_devices.get(
                        "generator", ""
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    env = os.environ.copy()
    env.update(build_role_environment(resolved, "generator"))
    envtuning = resolved.machine.source_root / "code/AWorld-RL-stage1-worktree/EnvTuning"
    paths = [str(envtuning), str(envtuning / "verl")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return subprocess.call(command, cwd=envtuning, env=env)


def command_evaluation_server(args: argparse.Namespace) -> int:
    resolved = _resolve(args)
    selected_gpu = selected_role_physical_gpu(resolved)
    role_environment = build_role_environment(resolved, "learner")
    command = build_evaluation_server_command(resolved, args.backend)
    payload = {
        "dry_run": not args.execute,
        "backend": args.backend,
        "selected_learner_gpu": selected_gpu,
        "cuda_visible_devices": role_environment["CUDA_VISIBLE_DEVICES"],
        "tensor_parallel_size": resolved.topology.rollout_tp,
        "command": command,
    }
    if not args.execute:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if os.environ.get(EVALUATION_SERVER_GUARD) != "1":
        raise ConfigError(
            "Generator terminal reconciliation has not been acknowledged; set "
            f"{EVALUATION_SERVER_GUARD}=1 only after reconciliation"
        )
    _check_evaluation_lifecycle(args.backend)
    check_selected_learner_gpu(
        selected_gpu, resolved.runtime.evaluation_server.max_used_memory_mib
    )
    env = os.environ.copy()
    env.update(role_environment)
    if args.backend == "sglang":
        env["TOKENIZERS_PARALLELISM"] = "true"
    else:
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.execvpe(command[0], command, env)
    return 0


def command_shell_env(args: argparse.Namespace) -> int:
    resolved = _resolve(args)
    values = build_role_environment(resolved, args.role)
    for name, value in values.items():
        print(f"export {name}={shlex.quote(value)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--mode", choices=("portable", "reference"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--output", type=Path)
    resolve.set_defaults(handler=command_resolve)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--check-assets", action="store_true")
    preflight.add_argument("--observe-hardware", action="store_true")
    preflight.set_defaults(handler=command_preflight)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--observe-hardware", action="store_true")
    launch.set_defaults(handler=command_launch)

    generator_server = subparsers.add_parser("generator-server")
    generator_server.add_argument("--execute", action="store_true")
    generator_server.set_defaults(handler=command_generator_server)

    generator_daemon = subparsers.add_parser("generator-daemon")
    generator_daemon.add_argument("--execute", action="store_true")
    generator_daemon.add_argument("--once", action="store_true")
    generator_daemon.add_argument("--allow-generation", action="store_true")
    generator_daemon.set_defaults(handler=command_generator_daemon)

    evaluation_server = subparsers.add_parser(
        "evaluation-server",
        help="dry-run or execute a topology-owned specialized eval server",
    )
    evaluation_server.add_argument("--backend", choices=("sglang", "vllm"), required=True)
    evaluation_server.add_argument("--execute", action="store_true")
    evaluation_server.set_defaults(handler=command_evaluation_server)

    shell_env = subparsers.add_parser(
        "shell-env", help="emit role-specific process environment as shell exports"
    )
    shell_env.add_argument("--role", required=True)
    shell_env.set_defaults(handler=command_shell_env)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(args.handler(args))
    except EvaluationGpuPreflightError as exc:
        print(f"GPU PREFLIGHT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(4) from exc
    except EvaluationLifecycleError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(3) from exc
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
