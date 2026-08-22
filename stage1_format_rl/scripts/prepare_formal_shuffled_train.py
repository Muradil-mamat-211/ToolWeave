#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sample_id(row: pd.Series) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        return str(extra.get("original_id", extra.get("index")))
    return str(extra)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_parquet(args.source)
    if len(frame) != 100:
        raise AssertionError(f"expected 100 rows, got {len(frame)}")
    source_ids = [sample_id(row) for _, row in frame.iterrows()]
    if len(set(source_ids)) != 100:
        raise AssertionError("source IDs are not unique")

    order = np.random.default_rng(args.seed).permutation(len(frame)).tolist()
    shuffled = frame.iloc[order].reset_index(drop=True)
    shuffled_ids = [sample_id(row) for _, row in shuffled.iterrows()]
    if set(shuffled_ids) != set(source_ids):
        raise AssertionError("physical shuffle changed dataset membership")
    if shuffled_ids == source_ids:
        raise AssertionError("physical shuffle did not change row order")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shuffled.to_parquet(args.output, index=False)
    reloaded = pd.read_parquet(args.output)
    reloaded_ids = [sample_id(row) for _, row in reloaded.iterrows()]
    if reloaded_ids != shuffled_ids:
        raise AssertionError("parquet round-trip changed shuffled order")

    manifest = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "rows": len(frame),
        "source_sha256": digest(args.source),
        "output_sha256": digest(args.output),
        "source_row_order": order,
        "sample_ids_in_training_order": shuffled_ids,
        "membership_unchanged": True,
        "order_changed": True,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in manifest.items() if k not in {"sample_ids_in_training_order", "source_row_order"}}
    summary["first_20_source_indices"] = order[:20]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
