#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import REPORTS_ROOT, STAGE_DATA_ROOT


SOURCE = STAGE_DATA_ROOT / "bfcl_val_400.parquet"
OUT_JSON = REPORTS_ROOT / "bfcl_four_class_eval_samples_full.json"
OUT_MD = REPORTS_ROOT / "bfcl_four_class_eval_samples_full.md"
CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]


def normalize(value):
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, np.ndarray):
        return [normalize(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    frame = pd.read_parquet(SOURCE)
    rows = [normalize(row) for row in frame.to_dict(orient="records")]
    selected = []
    source_counts = {}
    for category in CATEGORIES:
        category_rows = [row for row in rows if row["data_source"] == category]
        source_counts[category] = len(category_rows)
        if len(category_rows) < 2:
            raise ValueError(f"{category}: expected at least 2 rows, got {len(category_rows)}")
        selected.extend(category_rows[:2])

    payload = {
        "source_file": str(SOURCE),
        "source_description": (
            "Local four-class eval pool converted from official EnvTuning "
            "val+test parquet by prepare_stage1_data.py."
        ),
        "selection": "First two records in stable parquet order per data_source; no field omitted.",
        "categories": CATEGORIES,
        "counts_in_source": source_counts,
        "selected_counts": {category: 2 for category in CATEGORIES},
        "records": selected,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# BFCL Four-Class Eval Samples (Complete Records)",
        "",
        f"- Source: `{SOURCE}`",
        "- Selection: stable parquet order, first 2 records per `data_source`.",
        "- Scope: complete converted records; no top-level or nested field omitted.",
        f"- Total exported records: {len(selected)}",
        "",
        "## Source Counts",
        "",
        "| data_source | source rows | exported rows |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| `{category}` | {source_counts[category]} | 2 |" for category in CATEGORIES
    )
    lines.append("")
    for index, row in enumerate(selected, 1):
        extra = row.get("extra_info", {})
        sample_id = extra.get("original_id", extra.get("index", f"record_{index}"))
        lines.extend(
            [
                f"## Record {index}: `{sample_id}` ({row['data_source']})",
                "",
                "```json",
                json.dumps(row, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "source": str(SOURCE),
        "source_rows": len(rows),
        "source_counts": source_counts,
        "selected_ids": [
            row.get("extra_info", {}).get("original_id") for row in selected
        ],
        "json": str(OUT_JSON),
        "json_bytes": OUT_JSON.stat().st_size,
        "md": str(OUT_MD),
        "md_bytes": OUT_MD.stat().st_size,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
