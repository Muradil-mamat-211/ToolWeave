#!/usr/bin/env python3
"""Build a fixed held-out Base subset for read-only checkpoint selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def sample_id(extra_info: object) -> str:
    if not isinstance(extra_info, dict):
        raise TypeError(f"extra_info must be a dict, got {type(extra_info)!r}")
    value = extra_info.get("original_id", extra_info.get("index"))
    if value is None:
        raise KeyError("extra_info has neither original_id nor index")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_parquet(args.source)
    base = frame.loc[frame["data_source"] == "multi_turn_base"].copy()
    base["_sample_id"] = base["extra_info"].map(sample_id)
    base = base.sort_values("_sample_id", kind="stable")

    if len(base) < args.count:
        raise RuntimeError(f"requested {args.count} Base rows, found {len(base)}")
    if base["_sample_id"].duplicated().any():
        duplicates = base.loc[base["_sample_id"].duplicated(), "_sample_id"].tolist()
        raise RuntimeError(f"duplicate Base IDs: {duplicates}")

    selected = base.head(args.count).drop(columns=["_sample_id"])
    ids = [sample_id(value) for value in selected["extra_info"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)
    args.manifest.write_text(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "output": str(args.output.resolve()),
                "selection": "lexicographically first held-out multi_turn_base IDs",
                "count": len(selected),
                "sample_ids": ids,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    reread = pd.read_parquet(args.output)
    if len(reread) != args.count:
        raise RuntimeError(f"round-trip row mismatch: {len(reread)} != {args.count}")
    print(f"wrote {len(reread)} rows to {args.output}")
    print("sample_ids:", ", ".join(ids))


if __name__ == "__main__":
    main()
