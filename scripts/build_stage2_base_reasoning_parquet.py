#!/usr/bin/env python3
"""Convert normalized Stage 2 JSONL into the minimal veRL parquet schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            gold_calls = record.get("gold_calls")
            if not isinstance(gold_calls, list) or not gold_calls:
                raise ValueError(f"Missing gold_calls at {args.input}:{line_number}")
            rows.append(
                {
                    "data_source": "stage2_base_reasoning",
                    "prompt": [{"role": "user", "content": str(record["prompt"])}],
                    "ability": "stage2_base_reasoning",
                    "reward_model": {
                        # Keep the variable per-tool argument dictionaries intact.
                        # Arrow otherwise promotes them to one union struct and adds
                        # null-valued keys from unrelated tools.
                        "ground_truth": json.dumps(gold_calls, ensure_ascii=True),
                        "style": "reference_progress",
                    },
                    "extra_info": {
                        "dataset_type": "bfcl_v3_multiturn_base_stage2_reference",
                        "index": str(record["id"]),
                        "split": "base",
                        "tools": record.get("tools", []),
                    },
                }
            )
    if len(rows) < 80:
        raise ValueError(f"Only {len(rows)} rows found; Stage 2 requires at least 80.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"input={args.input}")
    print(f"output={args.out}")
    print(f"samples={len(rows)}")
    print(f"first_gold_calls={json.loads(rows[0]['reward_model']['ground_truth'])[:2]}")


if __name__ == "__main__":
    main()
