#!/usr/bin/env python3
"""Validate and package the existing balanced BFCL-100 evaluation subset.

This script does not sample or rewrite examples.  It verifies that the existing
100-row subset is an exact-content subset of the canonical 400-row evaluation
dataset, then emits an auditable Parquet/JSONL copy and a provenance manifest.
It is intentionally CPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
EXPECTED_COUNTS = {
    "multi_turn_base": 25,
    "multi_turn_miss_func": 25,
    "multi_turn_miss_param": 25,
    "multi_turn_long_context": 25,
}
DISPLAY_NAMES = {
    "multi_turn_base": "Base",
    "multi_turn_miss_func": "Miss Func",
    "multi_turn_miss_param": "Miss Param",
    "multi_turn_long_context": "Long Context",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id(extra_info: Any) -> str:
    if not isinstance(extra_info, dict):
        raise TypeError(f"extra_info must be dict, got {type(extra_info)!r}")
    value = extra_info.get("original_id", extra_info.get("index"))
    if value is None:
        interaction = extra_info.get("interaction_kwargs")
        if isinstance(interaction, dict):
            value = interaction.get("id")
    if value is None:
        raise KeyError("extra_info has no canonical sample identifier")
    return str(value)


def _builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    return value


def _canonical_record(record: dict[str, Any]) -> str:
    return json.dumps(
        _builtin(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _atomic_text(destination: Path, text: str) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-eval-100", type=Path, required=True)
    parser.add_argument("--canonical-eval-400", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    subset = pd.read_parquet(args.existing_eval_100)
    canonical = pd.read_parquet(args.canonical_eval_400)
    if list(subset.columns) != EXPECTED_COLUMNS:
        raise RuntimeError(f"unexpected eval-100 columns: {list(subset.columns)!r}")
    if list(canonical.columns) != EXPECTED_COLUMNS:
        raise RuntimeError(f"unexpected eval-400 columns: {list(canonical.columns)!r}")
    if len(subset) != 100 or len(canonical) != 400:
        raise RuntimeError(f"unexpected row counts: subset={len(subset)} canonical={len(canonical)}")

    counts = subset["data_source"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"unexpected eval-100 class counts: {counts!r}")
    canonical_counts = canonical["data_source"].value_counts().to_dict()
    expected_400 = {name: 100 for name in EXPECTED_COUNTS}
    if canonical_counts != expected_400:
        raise RuntimeError(f"unexpected eval-400 class counts: {canonical_counts!r}")

    subset_records = [_builtin(row) for row in subset.to_dict(orient="records")]
    canonical_records = [_builtin(row) for row in canonical.to_dict(orient="records")]
    subset_by_id = {_sample_id(row["extra_info"]): row for row in subset_records}
    canonical_by_id = {_sample_id(row["extra_info"]): row for row in canonical_records}
    duplicate_count = len(subset_records) - len(subset_by_id)
    if duplicate_count:
        raise RuntimeError(f"eval-100 contains {duplicate_count} duplicate sample IDs")
    if not set(subset_by_id).issubset(canonical_by_id):
        missing = sorted(set(subset_by_id) - set(canonical_by_id))
        raise RuntimeError(f"eval-100 IDs missing from canonical eval-400: {missing}")
    mismatched = [
        sample_id
        for sample_id, row in subset_by_id.items()
        if _canonical_record(row) != _canonical_record(canonical_by_id[sample_id])
    ]
    if mismatched:
        raise RuntimeError(f"eval-100 rows differ from canonical eval-400: {mismatched}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = args.output_dir / "eval_rods_bfcl_multiturn_100.parquet"
    jsonl_out = args.output_dir / "eval_rods_bfcl_multiturn_100.jsonl"
    manifest_out = args.output_dir / "eval_rods_bfcl_multiturn_100_manifest.json"
    _atomic_copy(args.existing_eval_100, parquet_out)
    _atomic_text(jsonl_out, "".join(_canonical_record(row) + "\n" for row in subset_records))

    ids_by_class = {
        DISPLAY_NAMES[source]: sorted(
            _sample_id(row["extra_info"])
            for row in subset_records
            if row["data_source"] == source
        )
        for source in EXPECTED_COUNTS
    }
    manifest = {
        "dataset_contract": "RODS_BFCL_MULTITURN_EVAL100_EXISTING_V1",
        "dataset_source": "EXISTING",
        "source_eval_100_path": str(args.existing_eval_100.resolve()),
        "source_eval_100_sha256": _sha256(args.existing_eval_100),
        "source_eval_100_manifest_path": str(args.existing_manifest.resolve()),
        "source_eval_100_manifest_sha256": _sha256(args.existing_manifest),
        "source_eval_400_path": str(args.canonical_eval_400.resolve()),
        "source_eval_400_sha256": _sha256(args.canonical_eval_400),
        "selection_method": (
            "REUSED existing per-category sample(25, random_state=42); no resampling; "
            "all rows verified byte-content-equivalent after canonical JSON normalization "
            "to rows in canonical eval-400"
        ),
        "selection_version": "EXISTING_VAL100_STRATIFIED_SEED42",
        "total": len(subset_records),
        "class_counts": {DISPLAY_NAMES[key]: value for key, value in EXPECTED_COUNTS.items()},
        "internal_class_counts": counts,
        "selected_sample_ids": ids_by_class,
        "duplicate_count": duplicate_count,
        "schema": {
            "columns": EXPECTED_COLUMNS,
            "pandas_dtypes": {name: str(dtype) for name, dtype in subset.dtypes.items()},
        },
        "canonical_membership": {
            "all_ids_present": True,
            "all_row_contents_equal": True,
        },
        "outputs": {
            "jsonl": {"path": str(jsonl_out.resolve()), "sha256": _sha256(jsonl_out)},
            "parquet": {"path": str(parquet_out.resolve()), "sha256": _sha256(parquet_out)},
        },
    }
    _atomic_text(manifest_out, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": str(parquet_out), "manifest": str(manifest_out), "counts": counts}))


if __name__ == "__main__":
    main()
