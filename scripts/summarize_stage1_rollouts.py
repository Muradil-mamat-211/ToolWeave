#!/usr/bin/env python3
"""Summarize per-step Stage 1 format metrics from veRL rollout JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


METRICS = (
    "format_reward",
    "valid_tool_call_block",
    "valid_json_parse",
    "valid_function_name",
    "valid_arguments_object",
    "required_arguments_present",
    "direct_answer_without_tool",
    "malformed_output",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected_steps", type=int, default=100)
    args = parser.parse_args()

    values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(args.rollouts.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                step = int(row["step"])
                if step < 1 or step > args.expected_steps:
                    continue
                for metric in METRICS:
                    values[step][metric].append(float(row.get(metric, 0.0)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for step in sorted(values):
            row = {"step": step, "rollout_count": len(values[step]["format_reward"])}
            row["mean_format_reward"] = round(sum(values[step]["format_reward"]) / row["rollout_count"], 6)
            for metric in METRICS[1:]:
                row[f"{metric}_rate"] = round(sum(values[step][metric]) / row["rollout_count"], 6)
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    missing = [step for step in range(1, args.expected_steps + 1) if step not in values]
    print(f"output={args.out}")
    print(f"steps_found={len(values)}")
    print(f"missing_steps={missing}")


if __name__ == "__main__":
    main()
