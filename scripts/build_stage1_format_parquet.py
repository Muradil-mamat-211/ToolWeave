#!/usr/bin/env python3
"""Convert the explicit Stage 1 JSONL artifact into the minimal veRL parquet schema."""

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
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {args.input}:{line_number}")
            rows.append(
                {
                    "data_source": "stage1_format",
                    "prompt": [{"role": "user", "content": str(record["prompt"])}],
                    "ability": "stage1_format",
                    "reward_model": {
                        "ground_truth": record.get("gold", {}).get("ground_truth"),
                        "style": "format",
                    },
                    "extra_info": {
                        "dataset_type": "bfcl_v3_multiturn_base_stage1_format",
                        "index": str(record.get("id", line_number)),
                        "split": "base",
                        "tools": record.get("tools", []),
                    },
                }
            )

    if not rows:
        raise ValueError(f"No JSONL records found in {args.input}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    first = rows[0]
    print(f"input={args.input}")
    print(f"output={args.out}")
    print(f"samples={len(rows)}")
    print(f"first_prompt_chars={len(first['prompt'][0]['content'])}")
    print(f"first_tool_names={[t.get('name') for t in first['extra_info']['tools'][:5]]}")


if __name__ == "__main__":
    main()
