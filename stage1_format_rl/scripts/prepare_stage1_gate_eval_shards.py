#!/usr/bin/env python3
"""Create two deterministic, category-balanced shards from Stage 1 val_400."""

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
    parser.add_argument("--out-dir", type=Path, required=True)
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
        duplicates = work.loc[work["_sample_id"].duplicated(), "_sample_id"].tolist()
        raise RuntimeError(f"duplicate validation IDs: {duplicates}")

    shard_rows: list[list[pd.DataFrame]] = [[], []]
    manifests: list[list[dict[str, str]]] = [[], []]
    for source in sorted(EXPECTED_SOURCES):
        category = work.loc[work["data_source"] == source].sort_values(
            "_sample_id", kind="stable"
        )
        for shard_index in (0, 1):
            selected = category.iloc[shard_index::2]
            shard_rows[shard_index].append(selected)
            manifests[shard_index].extend(
                {"sample_id": row["_sample_id"], "data_source": source}
                for _, row in selected.iterrows()
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_ids: list[set[str]] = []
    for shard_index in (0, 1):
        shard = pd.concat(shard_rows[shard_index], ignore_index=True)
        ids = set(shard["_sample_id"])
        all_ids.append(ids)
        if len(shard) != 200 or len(ids) != 200:
            raise RuntimeError(f"invalid shard {shard_index}: rows={len(shard)} ids={len(ids)}")
        shard = shard.drop(columns=["_sample_id"])
        parquet_path = args.out_dir / f"val_400_part{shard_index}.parquet"
        manifest_path = args.out_dir / f"val_400_part{shard_index}.manifest.json"
        shard.to_parquet(parquet_path, index=False)
        manifest_path.write_text(
            json.dumps(
                {
                    "source": str(args.source.resolve()),
                    "shard_index": shard_index,
                    "rows": len(shard),
                    "selection": "per-category ID sort followed by even/odd interleave",
                    "category_counts": shard["data_source"].value_counts().to_dict(),
                    "records": manifests[shard_index],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if all_ids[0] & all_ids[1]:
        raise RuntimeError("gate-eval shards overlap")
    if len(all_ids[0] | all_ids[1]) != 400:
        raise RuntimeError("gate-eval shards do not cover all validation IDs")

    # The dual-GPU evaluator uses one distributed veRL process. Keep the same
    # deterministic order as part0 followed by part1 and persist its manifest.
    combined_frames = [pd.read_parquet(args.out_dir / f"val_400_part{i}.parquet") for i in (0, 1)]
    combined = pd.concat(combined_frames, ignore_index=True)
    combined_records = manifests[0] + manifests[1]
    combined.to_parquet(args.out_dir / "val_400_combined.parquet", index=False)
    (args.out_dir / "val_400_combined.manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "rows": len(combined),
                "selection": "part0 followed by part1; each part is category-balanced",
                "category_counts": combined["data_source"].value_counts().to_dict(),
                "records": combined_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": 400, "part0": 200, "part1": 200, "counts": counts}))


if __name__ == "__main__":
    main()
