#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("/root/autodl-tmp/rods-workspace/data/Berkeley-Function-Calling-Leaderboard")
ENV_DATA = Path("/root/autodl-tmp/rods-workspace/code/AWorld-RL-stage1-worktree/EnvTuning/data")
TARGETS = {
    "multi_turn_miss_func": ["multi_turn_miss_func_4", "multi_turn_miss_func_9"],
    "multi_turn_miss_param": ["multi_turn_miss_param_167", "multi_turn_miss_param_180"],
}


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {str(row["id"]): row for row in (json.loads(line) for line in handle if line.strip())}


def text(value) -> str:
    if isinstance(value, list):
        return " || ".join(text(item) for item in value)
    if isinstance(value, dict):
        return str(value.get("content", value))
    return str(value)


def main() -> None:
    parquet_rows = {}
    for name in ["bfcl_train.parquet", "bfcl_val.parquet", "bfcl_test.parquet"]:
        for row in pd.read_parquet(ENV_DATA / name).to_dict(orient="records"):
            extra = row["extra_info"]
            parquet_rows[str(extra.get("original_id", extra.get("index")))] = row

    for category, ids in TARGETS.items():
        raw = load(ROOT / f"BFCL_v3_{category}.json")
        gold = load(ROOT / "possible_answer" / f"BFCL_v3_{category}.json")
        print(f"\n### {category}")
        for sample_id in ids:
            row = raw[sample_id]
            gt = gold[sample_id]["ground_truth"]
            kwargs = parquet_rows[sample_id]["extra_info"]["interaction_kwargs"]
            print(f"\nID: {sample_id}")
            print("missed_function:", repr(row.get("missed_function")))
            print("question_count:", len(row["question"]))
            for index, question in enumerate(row["question"]):
                print(f"question[{index}]: {text(question)}")
            print("ground_truth:")
            for index, calls in enumerate(gt):
                print(f"gt[{index}]: {calls!r}")
            print("processed_question:")
            for index, question in enumerate(kwargs["processed_question"]):
                print(f"processed[{index}]: {str(question)[:3000]}")


if __name__ == "__main__":
    main()
