#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

from project_paths import REPORTS_ROOT, SHARED_DATA_ROOT, SOURCE_ROOT


RAW_ROOT = SHARED_DATA_ROOT / "Berkeley-Function-Calling-Leaderboard"
ENV_DATA = SOURCE_ROOT / "code/AWorld-RL-stage1-worktree/EnvTuning/data"
REPORT_JSON = REPORTS_ROOT / "missing_turn_cardinality_audit.json"
REPORT_MD = REPORTS_ROOT / "missing_turn_cardinality_audit.md"
CATEGORIES = ["multi_turn_miss_func", "multi_turn_miss_param"]


def load_jsonl(path: Path) -> dict[str, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["id"])] = row
    return result


def q_is_empty(turn) -> bool:
    return turn == [] or turn == "" or turn is None


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


def make_record(category: str, raw: dict, gold: dict, parquet_row: dict | None) -> dict:
    questions = raw.get("question", [])
    ground_truth = gold.get("ground_truth", [])
    missed = raw.get("missed_function", {}) or {}
    missed_positions = sorted(int(k) for k in missed.keys())
    empty_question_positions = [i for i, value in enumerate(questions) if q_is_empty(value)]
    empty_ground_truth_positions = [i for i, value in enumerate(ground_truth) if not value]

    processed_len = None
    processed = None
    if parquet_row is not None:
        kwargs = parquet_row.get("extra_info", {}).get("interaction_kwargs", {})
        processed = normalize(kwargs.get("processed_question", []))
        processed_len = len(processed)

    return {
        "id": str(raw["id"]),
        "category": category,
        "question_count": len(questions),
        "empty_question_positions": empty_question_positions,
        "missed_function_positions": missed_positions,
        "missed_function_count": len(missed_positions),
        "missed_function_names_by_position": {
            str(k): list(v) for k, v in missed.items()
        },
        "ground_truth_count": len(ground_truth),
        "empty_ground_truth_positions": empty_ground_truth_positions,
        "missing_parameter_turn_count": len(empty_ground_truth_positions)
        if category == "multi_turn_miss_param"
        else 0,
        "processed_question_count": processed_len,
        "processed_question": processed,
    }


def distribution(records: list[dict], field: str) -> dict[str, int]:
    counts = Counter(record[field] for record in records)
    return {str(key): counts[key] for key in sorted(counts)}


def main() -> None:
    parquet_rows: dict[str, dict] = {}
    for filename in ["bfcl_train.parquet", "bfcl_val.parquet", "bfcl_test.parquet"]:
        frame = pd.read_parquet(ENV_DATA / filename)
        for row in frame.to_dict(orient="records"):
            extra = row.get("extra_info", {})
            sample_id = str(extra.get("original_id", extra.get("index")))
            parquet_rows[sample_id] = row

    all_records = []
    category_summary = {}
    for category in CATEGORIES:
        raw = load_jsonl(RAW_ROOT / f"BFCL_v3_{category}.json")
        gold = load_jsonl(RAW_ROOT / "possible_answer" / f"BFCL_v3_{category}.json")
        records = [make_record(category, raw[sample_id], gold[sample_id], parquet_rows.get(sample_id)) for sample_id in sorted(raw)]
        all_records.extend(records)

        category_summary[category] = {
            "raw_records": len(records),
            "question_count_distribution": distribution(records, "question_count"),
            "empty_question_count_distribution": distribution(
                [
                    {"count": len(record["empty_question_positions"])}
                    for record in records
                ],
                "count",
            ),
            "missed_function_turn_count_distribution": distribution(
                records, "missed_function_count"
            ),
            "empty_ground_truth_count_distribution": distribution(
                [
                    {"count": len(record["empty_ground_truth_positions"])}
                    for record in records
                ],
                "count",
            ),
            "processed_question_count_distribution": distribution(
                [
                    record
                    for record in records
                    if record["processed_question_count"] is not None
                ],
                "processed_question_count",
            ),
            "records_with_more_than_one_missing_function_turn": [
                record["id"] for record in records if record["missed_function_count"] > 1
            ],
            "records_with_more_than_one_missing_parameter_turn": [
                record["id"]
                for record in records
                if record["missing_parameter_turn_count"] > 1
            ],
            "records_with_question_ground_truth_length_mismatch": [
                record["id"]
                for record in records
                if record["question_count"] != record["ground_truth_count"]
            ],
        }

    payload = {
        "raw_source": str(RAW_ROOT),
        "processed_source": str(ENV_DATA),
        "scope": "All 200 records in each BFCL v3 missing-function/missing-parameter raw split.",
        "category_summary": category_summary,
        "records": all_records,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# BFCL Missing-Turn Cardinality Audit",
        "",
        "This audit covers all 200 raw records in each BFCL v3 missing-function and missing-parameter split.",
        "",
        "## Summary",
        "",
        "| category | raw records | question-count distribution | missing-turn distribution | >1 missing turns |",
        "|---|---:|---|---|---|",
    ]
    for category in CATEGORIES:
        summary = category_summary[category]
        if category == "multi_turn_miss_func":
            missing = summary["missed_function_turn_count_distribution"]
            more = summary["records_with_more_than_one_missing_function_turn"]
        else:
            missing = summary["empty_ground_truth_count_distribution"]
            more = summary["records_with_more_than_one_missing_parameter_turn"]
        lines.append(
            f"| `{category}` | {summary['raw_records']} | "
            f"`{summary['question_count_distribution']}` | `{missing}` | "
            f"{len(more)} records |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `question_count` is the number of user-turn slots in one BFCL record.",
        "- Missing Function turns are identified from the raw `missed_function` map.",
        "- Missing Parameter turns are identified from empty `ground_truth` turn lists in the possible-answer file.",
        "- `processed_question` is the runtime queue stored by EnvTuning in the converted parquet.",
        "",
        "## Records With More Than One Missing Turn",
        "",
    ])
    for category in CATEGORIES:
        summary = category_summary[category]
        key = (
            "records_with_more_than_one_missing_function_turn"
            if category == "multi_turn_miss_func"
            else "records_with_more_than_one_missing_parameter_turn"
        )
        lines.append(f"- `{category}`: `{summary[key]}`")

    lines.extend(["", "## Concrete Records", ""])
    for record in all_records:
        if record["id"] in {
            "multi_turn_miss_func_4",
            "multi_turn_miss_param_9",
        }:
            lines.append(f"### {record['id']}")
            lines.append("```json")
            lines.append(json.dumps(record, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["category_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
