#!/usr/bin/env python3
"""Extract formal Stage 1 trainer metrics and flag reward-collapse regions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import fmean


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+(.*)$")

KEY_METRICS = (
    "critic/score/mean",
    "critic/score/max",
    "critic/advantages/mean",
    "critic/advantages/max",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/entropy",
    "actor/pg_clipfrac",
    "actor/grad_norm",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "timing_s/generate_sequences",
    "timing_s/update_actor",
    "timing_s/step",
)


def parse_metrics(path: Path) -> list[dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        row: dict[str, float] = {"step": float(step)}
        for field in match.group(2).split(" - "):
            if ":" not in field:
                continue
            key, value = field.rsplit(":", 1)
            try:
                row[key.strip()] = float(value.strip())
            except ValueError:
                continue
        by_step[step] = row
    return [by_step[step] for step in sorted(by_step)]


def mean(rows: list[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in rows if key in row and math.isfinite(row[key])]
    return fmean(values) if values else None


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = parse_metrics(args.log)
    if not rows:
        raise RuntimeError(f"no trainer metric lines found in {args.log}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["step"] + sorted({key for row in rows for key in row if key != "step"})
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    ranges = ((1, 25), (26, 50), (51, 75), (76, max(int(r["step"]) for r in rows)))
    summaries = []
    for start, end in ranges:
        region = [row for row in rows if start <= row["step"] <= end]
        if not region:
            continue
        summaries.append(
            {
                "start_step": start,
                "end_step": end,
                "num_updates": len(region),
                **{key: mean(region, key) for key in KEY_METRICS},
            }
        )

    zero_reward_steps = [
        int(row["step"])
        for row in rows
        if row.get("critic/score/max") == 0.0 and row.get("critic/score/mean") == 0.0
    ]
    clipped_majority_steps = [
        int(row["step"])
        for row in rows
        if row.get("response_length/clip_ratio", 0.0) >= 0.5
    ]
    payload = {
        "source_log": str(args.log.resolve()),
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
        "num_updates": len(rows),
        "zero_reward_steps": zero_reward_steps,
        "majority_response_clipping_steps": clipped_majority_steps,
        "regions": summaries,
    }
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    header = (
        "| steps | updates | score | format proxy* | KL loss | entropy | PG loss | "
        "response tokens | clip ratio | step seconds |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    table = []
    for item in summaries:
        table.append(
            "| {start_step}-{end_step} | {num_updates} | {score} | n/a | {kl} | "
            "{entropy} | {pg} | {length} | {clip} | {seconds} |".format(
                **item,
                score=fmt(item["critic/score/mean"]),
                kl=fmt(item["actor/kl_loss"]),
                entropy=fmt(item["actor/entropy"]),
                pg=fmt(item["actor/pg_loss"]),
                length=fmt(item["response_length/mean"]),
                clip=fmt(item["response_length/clip_ratio"]),
                seconds=fmt(item["timing_s/step"]),
            )
        )

    report = f"""# Stage 1 Formal Training Collapse Audit

## Scope

- Source log: `{args.log.resolve()}`
- Parsed updates: {len(rows)} (steps {int(rows[0]['step'])}-{int(rows[-1]['step'])})
- This is a read-only log analysis. It did not run training or alter weights.

## Regional Metrics

{header}{chr(10).join(table)}

\* The trainer line exposes the official aggregate `critic/score/mean`; individual
format/tool components are retained in rollout logs rather than this metric line.

## Critical Signals

- All-zero reward steps: {zero_reward_steps or 'none'}
- Response clip ratio >= 0.5: {clipped_majority_steps or 'none'}
- At an all-zero group, GRPO advantages and policy-gradient loss are zero. Any
remaining update pressure comes from auxiliary loss terms, including loss-side KL.

## Interpretation

The run should not be resumed from the latest checkpoint until fixed held-out
evaluation identifies the best preserved checkpoint and a lower-risk continuation
profile is selected.
"""
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
