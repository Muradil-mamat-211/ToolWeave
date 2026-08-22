#!/usr/bin/env python3
"""Summarize Stage 2 custom-reward metrics from veRL rollout JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


METRICS = (
    "stage2_reward",
    "format_reward",
    "progress_reward",
    "first_tool_name_accuracy",
    "first_required_arg_key_score",
    "first_arg_value_score",
    "sequence_prefix_progress",
    "extra_call_rate",
    "malformed_rate",
)


def step_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**9, path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected_steps", type=int, required=True)
    args = parser.parse_args()

    values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(args.rollouts.glob("*.jsonl"), key=step_key):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                step = int(row["step"])
                if not 1 <= step <= args.expected_steps:
                    continue
                for metric in METRICS:
                    raw = row.get(metric, row.get("score", 0.0) if metric == "stage2_reward" else 0.0)
                    values[step][metric].append(float(raw))

    if not values:
        raise RuntimeError(f"No Stage 2 rollout rows found under {args.rollouts}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for step in sorted(values):
            count = len(values[step]["stage2_reward"])
            result = {"step": step, "rollout_count": count}
            for metric in METRICS:
                result[f"mean_{metric}"] = round(sum(values[step][metric]) / count, 6)
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    missing = [step for step in range(1, args.expected_steps + 1) if step not in values]
    print(f"output={args.out}")
    print(f"steps_found={len(values)}")
    print(f"missing_steps={missing}")


if __name__ == "__main__":
    main()
