#!/usr/bin/env python3
"""Verify Stage 2 merged-HF model is logits-consistent with the raw FSDP checkpoint.

Reconstructs the full state dict from the FSDP shards (same logic as
verl.model_merger.FSDPModelMerger), then:
  1. compares every parameter against the merged safetensors (weight equality), and
  2. runs a fixed input through the same model loaded from each weight source and
     compares logits (must be allclose).

Usage: verify_stage2_merge_logits.py ACTOR_DIR MERGED_HF_DIR
Exit 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root/autodl-tmp/rods-workspace/code/AWorld-RL-stage1-worktree/EnvTuning/verl")
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.model_merger.base_model_merger import ModelMergerConfig


def load_shard_state_dict(actor_dir: Path) -> dict[str, torch.Tensor]:
    """Rebuild the full model state dict from FSDP per-rank shards."""
    cfg = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(actor_dir),
        target_dir="/tmp/placeholder_merge_target",
        hf_upload_path=None,
        private=False,
        test_hf_dir=None,
        tie_word_embedding=False,
        is_value_model=False,
        hf_model_config_path=str(actor_dir / "huggingface"),
        use_cpu_initialization=True,
    )
    merger = FSDPModelMerger(cfg)
    world_size = merger._get_world_size()
    rank0 = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(rank0, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_dim_names)
    sd = merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor_dir")
    ap.add_argument("merged_hf_dir")
    args = ap.parse_args()

    actor_dir = Path(args.actor_dir)
    hf_dir = Path(args.merged_hf_dir)

    # 1) rebuild state dict from FSDP shards
    print(f"Rebuilding state dict from FSDP shards: {actor_dir}")
    sd_fsdp = load_shard_state_dict(actor_dir)

    # 2) load merged safetensors
    print(f"Loading merged HF safetensors: {hf_dir}")
    from safetensors.torch import load_file
    sd_hf = {}
    index = __import__("json").loads((hf_dir / "model.safetensors.index.json").read_text())
    for fn in sorted(set(index["weight_map"].values())):
        sd_hf.update(load_file(hf_dir / fn))

    common = sorted(set(sd_fsdp) & set(sd_hf))
    only_fsdp = sorted(set(sd_fsdp) - set(sd_hf))
    only_hf = sorted(set(sd_hf) - set(sd_fsdp))
    print(f"params common={len(common)} only_fsdp={len(only_fsdp)} only_hf={len(only_hf)}")
    assert not only_fsdp and not only_hf, f"key mismatch: {only_fsdp[:5]} / {only_hf[:5]}"

    # 3) weight equality (exact for bf16)
    max_wdiff = 0.0
    for k in common:
        a, b = sd_fsdp[k].float(), sd_hf[k].float()
        d = (a - b).abs().max().item()
        max_wdiff = max(max_wdiff, d)
    print(f"max |weight diff| (float) = {max_wdiff:.3e}")

    # 4) fixed-input logits comparison on GPU (or CPU fallback)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
    texts = [
        "Hello, what is the weather in Beijing today?",
        "Translate this to French: good morning, how are you?",
        "Write a python function to compute fibonacci numbers.",
    ]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in enc.items()}

    model = AutoModelForCausalLM.from_pretrained(
        str(hf_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    sd_fsdp_bf = {k: v.to(device).to(torch.bfloat16) for k, v in sd_fsdp.items()}

    with torch.no_grad():
        model.load_state_dict(sd_fsdp_bf, strict=True)
        logits_fsdp = model(**inputs).logits
        model.load_state_dict(sd_hf, strict=True)
        logits_hf = model(**inputs).logits

    diff = (logits_fsdp - logits_hf).abs()
    print(f"logits shape={tuple(logits_hf.shape)} dtype={logits_hf.dtype}")
    print(f"logits max abs diff = {diff.max().item():.3e}  mean abs diff = {diff.mean().item():.3e}")
    rel = diff.max().item() / (logits_hf.abs().max().item() + 1e-12)
    print(f"logits max relative diff = {rel:.3e}")

    ok = diff.max().item() <= 1e-5
    print("RESULT:", "PASS (logits identical)" if ok else "FAIL (logits differ)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
