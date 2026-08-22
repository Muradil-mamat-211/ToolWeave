from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.workers.fsdp_workers import ActorRolloutRefWorker


class _FakeData:
    def __init__(self) -> None:
        self.meta_info: dict[str, object] = {}

    def to(self, _device: object) -> "_FakeData":
        return self


class _IdentityShardingManager:
    def __enter__(self) -> "_IdentityShardingManager":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def preprocess_data(self, data: _FakeData) -> _FakeData:
        return data

    def postprocess_data(self, data: object) -> object:
        return data


def _worker(compute_log_prob: object) -> ActorRolloutRefWorker:
    worker = object.__new__(ActorRolloutRefWorker)
    worker._is_lora = False
    worker._is_ref = True
    worker._is_ref_phase_offload = True
    worker.ref_module_fsdp = object()
    worker.ref_policy = SimpleNamespace(
        compute_log_prob=compute_log_prob,
        actor_module=object(),
    )
    worker.config = OmegaConf.create(
        {
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 131072,
                "log_prob_use_dynamic_bsz": True,
            },
            "rollout": {"temperature": 1.0},
        }
    )
    worker.ulysses_sharding_manager = _IdentityShardingManager()
    worker._world_size = 1
    return worker


def test_reference_phase_offload_orders_load_forward_offload(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def compute_log_prob(*, data: _FakeData, calculate_entropy: bool):
        assert calculate_entropy is False
        assert data.meta_info["max_token_len"] == 131072
        events.append("forward")
        return torch.zeros((1, 2), dtype=torch.float32), None

    worker = _worker(compute_log_prob)
    monkeypatch.setattr(
        "verl.workers.fsdp_workers.load_fsdp_model_to_gpu",
        lambda _model: events.append("load"),
    )
    monkeypatch.setattr(
        "verl.workers.fsdp_workers.offload_fsdp_model_to_cpu",
        lambda _model: events.append("offload"),
    )
    monkeypatch.setattr(
        "verl.workers.fsdp_workers.log_gpu_memory_usage",
        lambda *_args, **_kwargs: None,
    )

    method = inspect.unwrap(ActorRolloutRefWorker.compute_ref_log_prob)
    output = method(worker, _FakeData())

    assert events == ["load", "forward", "offload"]
    assert output.batch["ref_log_prob"].device.type == "cpu"


def test_reference_phase_offload_runs_on_forward_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def compute_log_prob(**_kwargs: object):
        events.append("forward")
        raise RuntimeError("synthetic reference failure")

    worker = _worker(compute_log_prob)
    monkeypatch.setattr(
        "verl.workers.fsdp_workers.load_fsdp_model_to_gpu",
        lambda _model: events.append("load"),
    )
    monkeypatch.setattr(
        "verl.workers.fsdp_workers.offload_fsdp_model_to_cpu",
        lambda _model: events.append("offload"),
    )
    monkeypatch.setattr(
        "verl.workers.fsdp_workers.log_gpu_memory_usage",
        lambda *_args, **_kwargs: None,
    )

    method = inspect.unwrap(ActorRolloutRefWorker.compute_ref_log_prob)
    with pytest.raises(RuntimeError, match="synthetic reference failure"):
        method(worker, _FakeData())

    assert events == ["load", "forward", "offload"]
