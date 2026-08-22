#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/rods-workspace/data/Berkeley-Function-Calling-Leaderboard")
ENV_DATA = Path("/root/autodl-tmp/rods-workspace/code/AWorld-RL-stage1-worktree/EnvTuning/data")


def normalize(value):
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, np.ndarray):
        return [normalize(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    return value


def text(value) -> str:
    if isinstance(value, list):
        return " || ".join(text(item) for item in value)
    if isinstance(value, dict):
        return str(value.get("content", value))
    return str(value)


def load_jsonl(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {str(row["id"]): row for row in (json.loads(line) for line in handle if line.strip())}


def main() -> None:
    parquet_rows = {}
    for name in ["bfcl_train.parquet", "bfcl_val.parquet", "bfcl_test.parquet"]:
        for row in pd.read_parquet(ENV_DATA / name).to_dict(orient="records"):
            row = normalize(row)
            extra = row["extra_info"]
            parquet_rows[str(extra.get("original_id", extra.get("index")))] = row

    results = []
    for category in ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]:
        raw = load_jsonl(ROOT / f"BFCL_v3_{category}.json")
        for sample_id, record in raw.items():
            processed = parquet_rows[sample_id]["extra_info"]["interaction_kwargs"]["processed_question"]
            processed = [str(item) for item in processed]
            raw_followups = [text(item) for item in record["question"][1:]]
            raw_nonempty_followups = [item for item in raw_followups if item.strip()]
            matches_in_order = []
            cursor = 0
            for item in raw_nonempty_followups:
                found = next((i for i in range(cursor, len(processed)) if processed[i] == item), None)
                matches_in_order.append(found is not None)
                if found is not None:
                    cursor = found + 1
            results.append({
                "id": sample_id,
                "category": category,
                "raw_followup_count": len(raw_followups),
                "raw_nonempty_followup_count": len(raw_nonempty_followups),
                "processed_question_count": len(processed),
                "all_nonempty_raw_followups_preserved_in_order": all(matches_in_order),
                "exact_raw_followups_equal_processed": raw_followups == processed,
                "raw_followups": raw_followups,
                "processed_question": processed,
            })

    for category in ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]:
        subset = [row for row in results if row["category"] == category]
        print("\n", category)
        print("records", len(subset))
        print("exact_equal", sum(row["exact_raw_followups_equal_processed"] for row in subset))
        print("all_nonempty_preserved_in_order", sum(row["all_nonempty_raw_followups_preserved_in_order"] for row in subset))
        mismatches = [row["id"] for row in subset if not row["exact_raw_followups_equal_processed"]]
        print("mismatch_ids", mismatches)
        for row in subset:
            if not row["all_nonempty_raw_followups_preserved_in_order"]:
                print("missing_raw_followup_ids", row["id"])

    output = Path("/root/autodl-tmp/rods-workspace/reports/processed_question_raw_comparison.json")
    output.write_text(json.dumps({"records": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", output)


if __name__ == "__main__":
    main()
