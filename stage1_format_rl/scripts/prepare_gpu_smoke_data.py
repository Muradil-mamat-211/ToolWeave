#!/usr/bin/env python3
"""Select deterministic BFCL Base rows for GPU smoke phases."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


WORKSPACE = Path("/root/autodl-tmp/rods-workspace")
SOURCE = WORKSPACE / "stage1_format_rl/data/bfcl_stage1_train_base_100.parquet"
OUTPUT_DIR = WORKSPACE / "stage1_format_rl/data/smoke"
MODEL = WORKSPACE / "models/Qwen3-4B"


def to_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def count_gold_calls(reward_model: dict) -> int:
    groups = to_list(reward_model["ground_truth"])
    return sum(len(to_list(group)) for group in groups)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(SOURCE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    records = []
    for row_idx, row in frame.iterrows():
        prompt = to_list(row["prompt"])
        rendered = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        interaction = row["extra_info"]["interaction_kwargs"]
        user_turns = len(to_list(interaction["question"]))
        gold_calls = count_gold_calls(row["reward_model"])
        system_chars = len(prompt[0]["content"])
        records.append(
            {
                "row_idx": int(row_idx),
                "sample_id": row["extra_info"]["index"],
                "prompt_tokens": prompt_tokens,
                "system_schema_chars": system_chars,
                "user_turns": user_turns,
                "gold_calls": gold_calls,
            }
        )

    # Pick one unique row for each requested stress dimension. If maxima overlap,
    # take the next-ranked unused row for that dimension.
    chosen = []
    dimensions = [
        ("longest_prompt_tokens", "prompt_tokens"),
        ("longest_tool_schema", "system_schema_chars"),
        ("most_user_turns", "user_turns"),
        ("longest_gold_tool_chain", "gold_calls"),
    ]
    used = set()
    for label, key in dimensions:
        ranked = sorted(records, key=lambda item: (-item[key], item["sample_id"]))
        selected = next(item for item in ranked if item["row_idx"] not in used)
        used.add(selected["row_idx"])
        chosen.append({"selection_reason": label, **selected})

    ordered = sorted(records, key=lambda item: (item["prompt_tokens"], item["sample_id"]))
    functional = ordered[len(ordered) // 2]
    extreme_indices = [item["row_idx"] for item in chosen]

    frame.iloc[[functional["row_idx"]]].reset_index(drop=True).to_parquet(
        OUTPUT_DIR / "smoke1_functional.parquet", index=False
    )
    frame.iloc[extreme_indices].reset_index(drop=True).to_parquet(
        OUTPUT_DIR / "smoke2_extremes_4.parquet", index=False
    )
    for case_index, item in enumerate(chosen):
        frame.iloc[[item["row_idx"]]].reset_index(drop=True).to_parquet(
            OUTPUT_DIR / f"smoke2_extreme_{case_index}.parquet", index=False
        )
    frame.iloc[extreme_indices].reset_index(drop=True).to_parquet(
        OUTPUT_DIR / "smoke3_stress_4.parquet", index=False
    )
    frame.iloc[extreme_indices].reset_index(drop=True).to_parquet(
        OUTPUT_DIR / "smoke4_train_4.parquet", index=False
    )

    output = {
        "source": str(SOURCE),
        "source_rows": len(frame),
        "functional": functional,
        "extremes": chosen,
        "smoke2_case_files": [
            str(OUTPUT_DIR / f"smoke2_extreme_{case_index}.parquet")
            for case_index in range(len(chosen))
        ],
        "selection_is_deterministic": True,
    }
    (OUTPUT_DIR / "gpu_smoke_selection.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
