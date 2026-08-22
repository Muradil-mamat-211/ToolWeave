"""Durable, smoke-only serialization of complete Stage-3 update evidence.

This module is imported only when ``RODS_SMOKE_ARTIFACT_DIR`` is set. It does
not participate in reward, advantage, or loss construction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _builtin(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return [_builtin(item) for item in value.tolist()]
    if hasattr(value, "model_dump"):
        return _builtin(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_builtin(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_builtin(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_torch(path: Path, value: Any) -> None:
    """Atomically persist a torch payload and fsync it before publication."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _row_value(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, np.ndarray) and value.shape and value.shape[0] == batch_size:
        return value[index]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def dump_raw_rollout_batch(
    *,
    artifact_root: str | os.PathLike[str],
    batch: Any,
    tokenizer: Any,
    epoch: int,
    global_step: int,
    batch_index: int,
) -> None:
    """Durably save all generated trajectories before log-prob/PPO work.

    This smoke-only function is deliberately called before reward, advantage,
    and optimizer computation.  It never mutates ``batch`` and therefore has
    no effect on training semantics.  Tensor dimensions retain the full
    trajectory batch and token axes exactly as received from rollout.
    """

    update_dir = Path(artifact_root) / "training" / f"update_{global_step}"
    update_dir.mkdir(parents=True, exist_ok=True)

    tensor_payload: dict[str, torch.Tensor] = {}
    tensor_manifest: dict[str, dict[str, Any]] = {}
    for key, value in batch.batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        cpu_value = value.detach().cpu()
        tensor_payload[str(key)] = cpu_value
        tensor_manifest[str(key)] = {
            "shape": list(cpu_value.shape),
            "dtype": str(cpu_value.dtype),
            "finite": (
                bool(torch.isfinite(cpu_value).all().item())
                if cpu_value.is_floating_point()
                else True
            ),
        }
    _write_torch(update_dir / "raw_rollout_tensors.pt", tensor_payload)
    _write_json(update_dir / "raw_rollout_tensor_manifest.json", tensor_manifest)

    batch_size = int(batch.batch.batch_size[0])
    prompts = tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
    responses = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    response_mask = batch.batch["response_mask"].detach().cpu()
    attention_mask = batch.batch["attention_mask"].detach().cpu()
    trajectory_records: list[dict[str, Any]] = []
    for index in range(batch_size):
        non_tensor = {
            str(key): _builtin(_row_value(value, index, batch_size))
            for key, value in batch.non_tensor_batch.items()
        }
        trajectory_records.append(
            {
                "trajectory_index": index,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "batch_index": int(batch_index),
                "prompt_text": prompts[index],
                "response_text": responses[index],
                "effective_total_tokens": int(attention_mask[index].sum().item()),
                "trainable_response_tokens": int(response_mask[index].sum().item()),
                "non_tensor": non_tensor,
            }
        )
    _write_jsonl(update_dir / "raw_trajectories.jsonl", trajectory_records)
    _write_json(
        update_dir / "raw_rollout_manifest.json",
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "batch_index": int(batch_index),
            "trajectory_count": batch_size,
            "prompt_group_count": len(
                {str(item) for item in batch.non_tensor_batch["uid"]}
            ),
            "stage": "post_rollout_pre_reward_logprob_optimizer",
            "all_floating_tensors_finite": all(
                item["finite"] for item in tensor_manifest.values()
            ),
        },
    )


def dump_training_update(
    *,
    artifact_root: str | os.PathLike[str],
    batch: Any,
    tokenizer: Any,
    metrics: Mapping[str, Any],
    timing_raw: Mapping[str, Any],
    epoch: int,
    global_step: int,
    batch_index: int,
    matchtir_step_records: Sequence[Mapping[str, Any]],
) -> None:
    """Save tensor-level and structured rollout evidence for one counted update."""

    update_dir = Path(artifact_root) / "training" / f"update_{global_step}"
    update_dir.mkdir(parents=True, exist_ok=True)

    tensor_payload: dict[str, torch.Tensor] = {}
    tensor_manifest: dict[str, dict[str, Any]] = {}
    for key, value in batch.batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        cpu_value = value.detach().cpu()
        tensor_payload[str(key)] = cpu_value
        tensor_manifest[str(key)] = {
            "shape": list(cpu_value.shape),
            "dtype": str(cpu_value.dtype),
            "finite": (
                bool(torch.isfinite(cpu_value).all().item())
                if cpu_value.is_floating_point()
                else True
            ),
        }
    tensor_target = update_dir / "batch_tensors.pt"
    _write_torch(tensor_target, tensor_payload)
    _write_json(update_dir / "tensor_manifest.json", tensor_manifest)

    batch_size = int(batch.batch.batch_size[0])
    prompts = tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
    responses = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    progress_rewards = (
        batch.batch["token_level_scores"].detach().sum(dim=-1).cpu().tolist()
    )
    trajectory_records: list[dict[str, Any]] = []
    for index in range(batch_size):
        non_tensor = {
            str(key): _builtin(_row_value(value, index, batch_size))
            for key, value in batch.non_tensor_batch.items()
        }
        trajectory_records.append(
            {
                "trajectory_index": index,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "batch_index": int(batch_index),
                "prompt_text": prompts[index],
                "response_text": responses[index],
                "progress_reward": float(progress_rewards[index]),
                "non_tensor": non_tensor,
            }
        )
    _write_jsonl(update_dir / "trajectories.jsonl", trajectory_records)
    _write_jsonl(update_dir / "matchtir_policy_steps.jsonl", list(matchtir_step_records))
    _write_json(update_dir / "metrics.json", dict(metrics))
    _write_json(update_dir / "timing_raw.json", dict(timing_raw))
    _write_json(
        update_dir / "update_manifest.json",
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "batch_index": int(batch_index),
            "trajectory_count": batch_size,
            "prompt_group_count": len({str(item) for item in batch.non_tensor_batch["uid"]}),
            "matchtir_policy_step_count": len(matchtir_step_records),
            "tensor_file": str(tensor_target),
            "all_floating_tensors_finite": all(
                item["finite"] for item in tensor_manifest.values()
            ),
        },
    )
