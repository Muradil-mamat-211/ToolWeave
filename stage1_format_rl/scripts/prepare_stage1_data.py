#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from stage1_contract import json_dump, normalize_nested, protocol_aligned_system_prompt


EXPECTED_TYPES = {
    "multi_turn_base": "BFCL_v3_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v3_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v3_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v3_multi_turn_long_context.json",
}


def read_jsonl_ids(path: Path) -> list[str]:
    ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                raise ValueError(f"Missing id in {path}:{line_number}")
            ids.append(str(row["id"]))
    return ids


def row_id(row: dict) -> str:
    return str(row["extra_info"]["original_id"])


def patch_row(raw_row: dict) -> dict:
    row = normalize_nested(raw_row)
    prompt = row["prompt"]
    if len(prompt) != 2 or prompt[0]["role"] != "system" or prompt[1]["role"] != "user":
        raise ValueError(f"Unexpected official prompt structure for {row_id(row)}")
    legacy_system = prompt[0]["content"]
    prompt[0]["content"] = protocol_aligned_system_prompt(legacy_system)

    extra = row["extra_info"]
    extra["protocol_alignment"] = {
        "source": "public EnvTuning parquet plus current executable parser contract",
        "legacy_thinking_tag": "<thinking>",
        "runtime_thinking_tag": "<think>",
        "runtime_answer_tag": "<answer>",
        "reason": "AWorld-RL issue #12 and parser implementation require a consistent rule",
    }
    return row


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-data-dir", type=Path, required=True)
    parser.add_argument("--raw-bfcl-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    source_train_base = args.official_data_dir / "bfcl_train_base.parquet"
    source_train_all = args.official_data_dir / "bfcl_train.parquet"
    source_val = args.official_data_dir / "bfcl_val.parquet"
    source_test = args.official_data_dir / "bfcl_test.parquet"
    for path in (source_train_base, source_train_all, source_val, source_test):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_base_df = pd.read_parquet(source_train_base)
    train_all_df = pd.read_parquet(source_train_all)
    held_in_df = pd.concat(
        [pd.read_parquet(source_val), pd.read_parquet(source_test)],
        ignore_index=True,
    )

    train_rows = [patch_row(row) for row in train_base_df.to_dict(orient="records")]
    held_rows = [patch_row(row) for row in held_in_df.to_dict(orient="records")]
    train_all_rows = [normalize_nested(row) for row in train_all_df.to_dict(orient="records")]

    train_ids = [row_id(row) for row in train_rows]
    validation_ids = [row_id(row) for row in held_rows]
    if len(train_rows) != 100 or Counter(row["data_source"] for row in train_rows) != {"multi_turn_base": 100}:
        raise ValueError("Official Stage 1 Base train split is not exactly 100 Base rows")
    expected_validation_counts = {name: 100 for name in EXPECTED_TYPES}
    actual_validation_counts = Counter(row["data_source"] for row in held_rows)
    if len(held_rows) != 400 or dict(actual_validation_counts) != expected_validation_counts:
        raise ValueError(f"Unexpected held-in validation counts: {actual_validation_counts}")
    if len(set(train_ids)) != 100 or len(set(validation_ids)) != 400:
        raise ValueError("Duplicate IDs in official train or held-in validation split")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Train/validation overlap detected")

    raw_counts = {}
    raw_ids_by_type = {}
    for data_type, filename in EXPECTED_TYPES.items():
        raw_path = args.raw_bfcl_dir / filename
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        ids = read_jsonl_ids(raw_path)
        if len(ids) != 200 or len(set(ids)) != 200:
            raise ValueError(f"Expected 200 unique raw records in {raw_path}")
        raw_counts[data_type] = len(ids)
        raw_ids_by_type[data_type] = set(ids)

    official_train_ids_by_type = {
        data_type: {
            row_id(row) for row in train_all_rows if row["data_source"] == data_type
        }
        for data_type in EXPECTED_TYPES
    }
    validation_ids_by_type = {
        data_type: {
            row_id(row) for row in held_rows if row["data_source"] == data_type
        }
        for data_type in EXPECTED_TYPES
    }
    for data_type in EXPECTED_TYPES:
        if len(official_train_ids_by_type[data_type]) != 100:
            raise ValueError(f"Official train seed count is not 100 for {data_type}")
        if official_train_ids_by_type[data_type] & validation_ids_by_type[data_type]:
            raise ValueError(f"Official train/held-in overlap for {data_type}")
        if official_train_ids_by_type[data_type] | validation_ids_by_type[data_type] != raw_ids_by_type[data_type]:
            raise ValueError(f"Official split does not partition all raw IDs for {data_type}")

    required_columns = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    for row in train_rows + held_rows:
        if set(row) != required_columns:
            raise ValueError(f"Unexpected columns for {row_id(row)}: {set(row)}")
        kwargs = row["extra_info"]["interaction_kwargs"]
        required_kwargs = {
            "name",
            "id",
            "initial_config",
            "involved_classes",
            "ground_truth",
            "processed_question",
            "question",
        }
        if not required_kwargs.issubset(kwargs):
            raise ValueError(f"Missing interaction kwargs for {row_id(row)}")
        json.loads(kwargs["initial_config"])
        if row["reward_model"]["style"] != "interaction":
            raise ValueError(f"Invalid reward style for {row_id(row)}")
        visible = json.dumps(row["prompt"], ensure_ascii=False)
        ground_truth = json.dumps(kwargs["ground_truth"], ensure_ascii=False)
        if ground_truth in visible:
            raise ValueError(f"Ground truth leaked into policy prompt for {row_id(row)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.out_dir / "bfcl_stage1_train_base_100.parquet"
    validation_out = args.out_dir / "bfcl_val_400.parquet"
    pd.DataFrame(train_rows).to_parquet(train_out, index=False)
    pd.DataFrame(held_rows).to_parquet(validation_out, index=False)

    train_ids_by_type = {
        data_type: sorted(ids) for data_type, ids in official_train_ids_by_type.items()
    }
    validation_ids_json = {
        data_type: sorted(ids) for data_type, ids in validation_ids_by_type.items()
    }
    manifest = {
        "selection_method": "exact public EnvTuning parquet split",
        "random_seed": None,
        "exact_public_envtuning_split": True,
        "exact_rods_experiment_id_manifest_independently_published": False,
        "rods_split_note": (
            "RODS points to the EnvTuning pipeline, and AWorld-RL issue #9 confirms "
            "the published EnvTuning data matches internal data. RODS does not publish "
            "a separate Stage-1 ID manifest, so independent identity is not claimed."
        ),
        "protocol_patch": (
            "Replaced stale <thinking>/plain-answer instructions in public parquet "
            "with the current executable <think> plus <tool_call>|<answer> contract; "
            "IDs, tools, environment state, questions, and ground truth are unchanged."
        ),
        "train_ids": sorted(train_ids),
        "validation_ids": validation_ids_json,
        "official_train_seed_ids_all_four_types": train_ids_by_type,
        "source_files": {
            str(path): {"sha256": hash_file(path)}
            for path in (source_train_base, source_train_all, source_val, source_test)
        },
    }
    stats = {
        "raw_counts": raw_counts,
        "train_count": len(train_rows),
        "train_by_type": dict(Counter(row["data_source"] for row in train_rows)),
        "validation_count": len(held_rows),
        "validation_by_type": dict(actual_validation_counts),
        "train_unique_ids": len(set(train_ids)),
        "validation_unique_ids": len(set(validation_ids)),
        "train_validation_overlap": len(set(train_ids) & set(validation_ids)),
        "train_output": str(train_out),
        "validation_output": str(validation_out),
        "train_output_sha256": hash_file(train_out),
        "validation_output_sha256": hash_file(validation_out),
    }
    samples = {
        "note": "Three complete converted records; ground truth remains metadata only.",
        "records": train_rows[:3],
    }
    json_dump(args.out_dir / "stage1_split_manifest.json", manifest)
    json_dump(args.out_dir / "stage1_dataset_stats.json", stats)
    json_dump(args.out_dir / "stage1_sample_records.json", samples)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

