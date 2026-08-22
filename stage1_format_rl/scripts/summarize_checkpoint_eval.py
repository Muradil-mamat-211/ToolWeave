#!/usr/bin/env python3
"""Summarize read-only Stage 1 checkpoint-selection rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


METRICS = (
    "score",
    "progress",
    "format_reward",
    "tool_call_reward",
    "is_tool_call",
    "total_interaction_rounds",
)


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def aggregate(label: str, rows: list[dict], scope: str) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {
        "checkpoint": label,
        "scope": scope,
        "samples": len(rows),
    }
    for key in METRICS:
        result[key] = fmean(float(row[key]) for row in rows)
    lengths = [len(str(row.get("output", ""))) for row in rows]
    result["mean_output_chars"] = fmean(lengths)
    result["max_output_chars"] = max(lengths)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step25", type=Path, required=True)
    parser.add_argument("--step50", type=Path, required=True)
    parser.add_argument("--step75", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "global_step_25": load(args.step25),
        "global_step_50": load(args.step50),
        "global_step_75": load(args.step75),
    }
    if len(sources["global_step_25"]) != 20 or len(sources["global_step_50"]) != 20:
        raise RuntimeError("step 25 and step 50 must each contain 20 rows")
    if len(sources["global_step_75"]) != 4:
        raise RuntimeError("step 75 confirmation set must contain 4 rows")

    summaries = [
        aggregate(label, rows[:4], "same_first_4")
        for label, rows in sources.items()
    ]
    summaries.extend(
        aggregate(label, rows, "heldout_base_20")
        for label, rows in sources.items()
        if len(rows) == 20
    )

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Stage 1 Checkpoint Selection",
        "",
        "All rows use the same official BFCL multi-turn interaction, official Stage 1",
        "format reward, deterministic decoding, and held-out Base IDs disjoint from training.",
        "",
        "## Same Four Held-out Samples",
        "",
        "| checkpoint | score | format | tool | progress | is tool | rounds | output chars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries[:3]:
        lines.append(
            "| {checkpoint} | {score:.4f} | {format_reward:.4f} | "
            "{tool_call_reward:.4f} | {progress:.4f} | {is_tool_call:.4f} | "
            "{total_interaction_rounds:.2f} | {mean_output_chars:.0f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Full 20-sample Results Available",
            "",
            "| checkpoint | score | format | tool | progress | is tool | rounds |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries[3:]:
        lines.append(
            "| {checkpoint} | {score:.4f} | {format_reward:.4f} | "
            "{tool_call_reward:.4f} | {progress:.4f} | {is_tool_call:.4f} | "
            "{total_interaction_rounds:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "`global_step_25` is the best preserved checkpoint. Step 50 has severe format",
            "regression, while step 75 has collapsed to zero parser/tool/reward signal.",
            "The original run must not resume from step 75.",
            "",
        ]
    )
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
