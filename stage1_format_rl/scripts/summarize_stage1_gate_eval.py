#!/usr/bin/env python3
"""Combine and summarize the three-checkpoint Stage 1 gate evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


MODEL_LABELS = (
    "old_run_step25",
    "recovery_run_step25_logical50",
    "recovery_run_step50_logical75",
)
METRICS = (
    "score",
    "progress",
    "format_reward",
    "tool_call_reward",
    "is_tool_call",
    "total_interaction_rounds",
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_run(root: Path, model: str, manifest_dir: Path) -> list[dict]:
    rollout_path = root / "runs" / model / "rollouts" / "0.jsonl"
    manifest_path = manifest_dir / "val_400_combined.manifest.json"
    rows = read_jsonl(rollout_path)
    records = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
    if len(rows) != len(records):
        raise RuntimeError(
            f"{model}: rollout rows {len(rows)} != manifest rows {len(records)}"
        )
    enriched = []
    for row, record in zip(rows, records, strict=True):
        item = dict(row)
        item.update(
            {
                "model_label": model,
                "sample_id": record["sample_id"],
                "data_source": record["data_source"],
                "eval_layout": "dual_gpu_distributed",
            }
        )
        enriched.append(item)
    return enriched


def aggregate(model: str, scope: str, rows: list[dict]) -> dict:
    result: dict[str, object] = {
        "model": model,
        "scope": scope,
        "samples": len(rows),
    }
    for metric in METRICS:
        result[metric] = fmean(float(row[metric]) for row in rows)
    result["full_progress_rate"] = fmean(float(row["progress"] == 1.0) for row in rows)
    result["nonzero_progress_rate"] = fmean(float(row["progress"] > 0.0) for row in rows)
    result["zero_score_rate"] = fmean(float(row["score"] == 0.0) for row in rows)
    output_lengths = [len(str(row.get("output", ""))) for row in rows]
    result["mean_output_chars"] = fmean(output_lengths)
    result["max_output_chars"] = max(output_lengths)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    model_rows: dict[str, list[dict]] = {}
    outputs_dir = args.eval_root / "outputs"
    comparison_dir = args.eval_root / "comparison"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for model in MODEL_LABELS:
        rows = load_run(args.eval_root, model, args.manifest_dir)
        if len(rows) != 400:
            raise RuntimeError(f"{model}: expected 400 rows, found {len(rows)}")
        ids = [row["sample_id"] for row in rows]
        if len(set(ids)) != 400:
            raise RuntimeError(f"{model}: duplicate sample IDs")
        model_rows[model] = rows
        with (outputs_dir / f"{model}_outputs.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries.append(aggregate(model, "all_400", rows))
        by_source: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_source[row["data_source"]].append(row)
        for source in sorted(by_source):
            summaries.append(aggregate(model, source, by_source[source]))

    json_path = comparison_dir / "three_checkpoint_gate_summary.json"
    csv_path = comparison_dir / "three_checkpoint_gate_summary.csv"
    md_path = comparison_dir / "three_checkpoint_gate_comparison.md"
    json_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Stage 1 Three-Checkpoint Gate Evaluation",
        "",
        "All models used the same official multi-turn interaction, format reward,",
        "deterministic decoding, validation records, ordering, and length limits.",
        "",
        "| model | scope | n | score | format | tool | progress | full progress | nonzero progress | zero score | rounds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {model} | {scope} | {samples} | {score:.4f} | {format_reward:.4f} | "
            "{tool_call_reward:.4f} | {progress:.4f} | {full_progress_rate:.4f} | "
            "{nonzero_progress_rate:.4f} | {zero_score_rate:.4f} | "
            "{total_interaction_rounds:.2f} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps([row for row in summaries if row["scope"] == "all_400"], indent=2))


if __name__ == "__main__":
    main()
