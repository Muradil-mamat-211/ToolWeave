#!/usr/bin/env python3
"""Prepare the four-type public BFCL train split for the executable protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from machine_paths import project_roots
from stage1_contract import json_dump, normalize_nested, protocol_aligned_system_prompt


EXPECTED_COUNTS = {
    "multi_turn_base": 100,
    "multi_turn_miss_func": 100,
    "multi_turn_miss_param": 100,
    "multi_turn_long_context": 100,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_row(raw_row: dict) -> dict:
    row = normalize_nested(raw_row)
    prompt = row["prompt"]
    if len(prompt) != 2 or prompt[0].get("role") != "system":
        raise ValueError("unexpected public BFCL prompt structure")
    prompt[0]["content"] = protocol_aligned_system_prompt(prompt[0]["content"])
    row["extra_info"]["protocol_alignment"] = {
        "source": "public EnvTuning train split",
        "runtime_contract": "<think> plus exactly one <tool_call>|<answer>",
        "tools_environment_questions_ground_truth_unchanged": True,
    }
    return row


def main() -> None:
    roots = project_roots()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=roots.asset_root
        / "code/AWorld-RL-stage1-worktree/EnvTuning/data/bfcl_train.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=roots.stage_data_root
        / "bfcl_stage3_train_all_400_shuffled_seed42.parquet",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [patch_row(row) for row in pd.read_parquet(args.source).to_dict(orient="records")]
    counts = Counter(row["data_source"] for row in rows)
    if len(rows) != 400 or dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"unexpected public BFCL train composition: {counts}")

    ids: list[str] = []
    for row in rows:
        if set(row) != {"data_source", "prompt", "ability", "reward_model", "extra_info"}:
            raise ValueError("unexpected BFCL columns")
        kwargs = row["extra_info"]["interaction_kwargs"]
        sample_id = str(kwargs["id"])
        ids.append(sample_id)
        if len(kwargs["question"]) != len(kwargs["ground_truth"]):
            raise ValueError(f"question/GT turn mismatch for {sample_id}")
        json.loads(kwargs["initial_config"])
        if row["reward_model"]["style"] != "interaction":
            raise ValueError(f"non-interaction reward row: {sample_id}")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate BFCL train IDs")

    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    manifest_path = args.output.with_suffix(".manifest.json")
    json_dump(
        manifest_path,
        {
            "source": str(args.source),
            "source_sha256": sha256(args.source),
            "output": str(args.output),
            "output_sha256": sha256(args.output),
            "shuffle_seed": args.seed,
            "count": len(frame),
            "counts_by_type": dict(counts),
            "unique_ids": len(set(ids)),
            "protocol_only_patch": True,
        },
    )
    print(json.dumps(json.loads(manifest_path.read_text()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
