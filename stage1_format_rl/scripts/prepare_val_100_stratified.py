#!/usr/bin/env python3
"""Stratified 100-sample validation subset for cheap per-epoch validation.

Randomly draws 25 samples per category from bfcl_val_400 (seed 42,
reproducible), so each epoch's training-time validation (test_freq=5) costs
~25 minutes instead of ~2 hours, while preserving the 25/25/25/25 proportion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED_SOURCES = {
    "multi_turn_base": 100,
    "multi_turn_long_context": 100,
    "multi_turn_miss_func": 100,
    "multi_turn_miss_param": 100,
}
PER_CATEGORY = 25
SEED = 42


def sample_id(extra_info: object) -> str:
    if not isinstance(extra_info, dict):
        raise TypeError(f"extra_info must be dict, got {type(extra_info)!r}")
    value = extra_info.get("original_id", extra_info.get("index"))
    if value is None:
        raise KeyError("extra_info has neither original_id nor index")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.source)
    if len(frame) != 400:
        raise RuntimeError(f"expected 400 validation rows, found {len(frame)}")
    counts = frame["data_source"].value_counts().to_dict()
    if counts != EXPECTED_SOURCES:
        raise RuntimeError(f"unexpected category counts: {counts!r}")

    work = frame.copy()
    work["_sample_id"] = work["extra_info"].map(sample_id)
    if work["_sample_id"].duplicated().any():
        dup = work.loc[work["_sample_id"].duplicated(), "_sample_id"].tolist()
        raise RuntimeError(f"duplicate validation IDs: {dup}")

    # Stratified random draw: 25 per category, fixed seed, deterministic.
    picked: list[pd.DataFrame] = []
    for source in sorted(EXPECTED_SOURCES):
        category = work.loc[work["data_source"] == source]
        subset = category.sample(PER_CATEGORY, random_state=SEED)
        picked.append(subset)
    selected = pd.concat(picked, ignore_index=True)

    ids = set(selected["_sample_id"])
    if len(selected) != 100 or len(ids) != 100:
        raise RuntimeError(
            f"invalid subset: rows={len(selected)} ids={len(ids)}"
        )
    got = selected["data_source"].value_counts().to_dict()
    if any(got.get(s, 0) != PER_CATEGORY for s in EXPECTED_SOURCES):
        raise RuntimeError(f"unexpected subset counts: {got}")

    # No overlap with the 100-prompt training pool (Base split must be disjoint).
    train = pd.read_parquet(args.train_base)
    train_ids = {sample_id(r["extra_info"]) for _, r in train.iterrows()}
    overlap = sorted(ids & train_ids)
    if overlap:
        raise RuntimeError(f"val_100 overlaps training Base IDs: {overlap[:10]}")

    out = selected.sort_values("_sample_id").drop(columns=["_sample_id"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    manifest = {
        "source": str(args.source.resolve()),
        "rows": len(out),
        "selection": f"per-category random sample({PER_CATEGORY}, random_state={SEED}), sorted by sample_id",
        "category_counts": out["data_source"].value_counts().to_dict(),
        "overlap_with_train_base": 0,
        "sample_ids": sorted(ids),
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(out), "counts": manifest["category_counts"],
                      "train_overlap": 0, "out": str(args.out),
                      "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
