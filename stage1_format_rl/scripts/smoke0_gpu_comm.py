#!/usr/bin/env python3
"""Smoke 0 only: validate CUDA peer visibility, NCCL, and FlashAttention."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    if value.item() != 3.0:
        raise RuntimeError(f"NCCL all-reduce mismatch on rank {rank}: {value.item()}")

    # A small real BF16 kernel invocation verifies more than package metadata.
    q = torch.randn(1, 128, 8, 64, device=f"cuda:{local_rank}", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    if out.shape != q.shape or not torch.isfinite(out).all():
        raise RuntimeError(f"FlashAttention output failed validation on rank {rank}")

    result = {
        "rank": rank,
        "device": local_rank,
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "nccl_all_reduce": "PASS",
        "all_reduce_value": value.item(),
        "flash_attention_bf16": "PASS",
        "flash_attention_shape": list(out.shape),
        "peer_0_to_1": torch.cuda.can_device_access_peer(0, 1),
        "peer_1_to_0": torch.cuda.can_device_access_peer(1, 0),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
