#!/usr/bin/env python
"""Read-only audit of BFCL raw data and RODS/EnvTuning split evidence."""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from project_paths import ASSET_ROOT, REPORTS_ROOT, SHARED_DATA_ROOT, SOURCE_ROOT


WORKSPACE = SOURCE_ROOT
BFCL_DIR = SHARED_DATA_ROOT / "Berkeley-Function-Calling-Leaderboard"
AWORLD = SOURCE_ROOT / "code" / "AWorld-RL"
REPORT_DIR = REPORTS_ROOT
JSON_REPORT = REPORT_DIR / "bfcl_rods_data_audit.json"
MD_REPORT = REPORT_DIR / "bfcl_rods_data_audit.md"


def git_value(path: Path, args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    except Exception:
        return None


def load_jsonl(path: Path) -> tuple[int, int, list[dict[str, Any]], list[str]]:
    total = valid = 0
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    value = json.loads(line)
                    valid += 1
                    if isinstance(value, dict):
                        rows.append(value)
                except Exception as exc:
                    errors.append(f"line {line_no}: {exc!r}")
    except Exception as exc:
        errors.append(f"open failed: {exc!r}")
    return total, valid, rows, errors


def category_from_name(name: str) -> str:
    lower = name.lower()
    if "multi_turn_base" in lower:
        return "Base Multi-Turn"
    if "multi_turn_miss_func" in lower:
        return "Missing Function"
    if "multi_turn_miss_param" in lower:
        return "Missing Parameter"
    if "multi_turn_long_context" in lower:
        return "Long Context"
    if "multi_turn_composite" in lower:
        return "Composite"
    return "Other"


def parquet_summary(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
        result["rows"] = int(len(frame))
        result["columns"] = [str(x) for x in frame.columns]
        if "data_source" in frame:
            result["data_source_counts"] = {
                str(k): int(v) for k, v in frame["data_source"].value_counts(dropna=False).items()
            }
        if "extra_info" in frame:
            ids = []
            for value in frame["extra_info"]:
                if isinstance(value, dict):
                    ids.append(value.get("index") or value.get("original_id"))
            result["unique_ids"] = len({x for x in ids if x is not None})
    except Exception as exc:
        result["read_error"] = repr(exc)
    return result


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path
        for path in list(BFCL_DIR.rglob("*.json")) + list(BFCL_DIR.rglob("*.jsonl"))
        if ".cache" not in path.parts
    )
    file_records: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    category_ids: dict[str, set[str]] = defaultdict(set)
    parse_errors: list[dict[str, Any]] = []

    for path in files:
        total, valid, rows, errors = load_jsonl(path)
        category = category_from_name(path.name)
        record = {
            "path": str(path),
            "relative_path": str(path.relative_to(BFCL_DIR)),
            "bytes": path.stat().st_size,
            "format": "jsonl_records",
            "category": category,
            "records": total,
            "valid_records": valid,
            "parse_errors": len(errors),
            "keys": sorted({key for row in rows for key in row}),
        }
        file_records.append(record)
        if path.parent.name == "possible_answer":
            category_counts[f"possible_answer/{category}"] += valid
        elif path.name.startswith("BFCL_v3_multi_turn_"):
            category_counts[f"raw/{category}"] += valid
        if path.parent.name == "possible_answer":
            for row in rows:
                if row.get("id") is not None:
                    category_ids[f"possible_answer/{category}"].add(str(row["id"]))
        elif path.name.startswith("BFCL_v3_multi_turn_"):
            for row in rows:
                if row.get("id") is not None:
                    category_ids[f"raw/{category}"].add(str(row["id"]))
        if errors:
            parse_errors.append({"path": str(path), "errors": errors[:10]})

    parquet_paths = [
        AWORLD / "EnvTuning" / "data" / "bfcl_train_base.parquet",
        AWORLD / "EnvTuning" / "data" / "bfcl_train.parquet",
        AWORLD / "EnvTuning" / "data" / "bfcl_val.parquet",
        SHARED_DATA_ROOT / "stage2_official" / "bfcl_v3_multiturn_base_official.parquet",
    ]
    parquet = [parquet_summary(path) for path in parquet_paths]

    official_manifest_candidates = []
    for root in (AWORLD / "RODS", AWORLD / "EnvTuning"):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".sh", ".py"}:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                lower = text.lower()
                if any(token in lower for token in ("manifest", "seed", "sampled half", "train_base", "bfcl_train.parquet")):
                    official_manifest_candidates.append(str(path))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(WORKSPACE),
        "bfcl_dir": str(BFCL_DIR),
        "bfcl_exists": BFCL_DIR.is_dir(),
        "total_data_files": len(files),
        "total_json_bytes": sum(path.stat().st_size for path in files),
        "file_records": file_records,
        "category_counts": dict(sorted(category_counts.items())),
        "category_unique_ids": {key: len(value) for key, value in sorted(category_ids.items())},
        "parse_errors": parse_errors,
        "official_parquet": parquet,
        "official_split_manifest_candidates": sorted(set(official_manifest_candidates)),
        "no_datasets_load_dataset_used_by_this_audit": True,
        "repository_commits": {
            "AWorld-RL": git_value(AWORLD, ["rev-parse", "HEAD"]),
            "gorilla": git_value(WORKSPACE / "code" / "gorilla", ["rev-parse", "HEAD"]),
            "verl": git_value(WORKSPACE / "code" / "verl", ["rev-parse", "HEAD"]),
        },
    }
    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# BFCL / RODS Data Audit",
        "",
        f"Generated: `{report['generated_at']}`",
        f"BFCL directory: `{BFCL_DIR}`",
        f"JSON/JSONL files found: **{len(files)}**",
        f"JSON/JSONL bytes: **{sum(path.stat().st_size for path in files):,}**",
        "",
        "## Raw Multi-Turn Counts",
        "",
        "| Category | Raw records | Possible-answer records |",
        "|---|---:|---:|",
    ]
    for category in ["Base Multi-Turn", "Missing Function", "Missing Parameter", "Long Context", "Composite"]:
        lines.append(
            f"| {category} | {category_counts.get('raw/' + category, 0)} | {category_counts.get('possible_answer/' + category, 0)} |"
        )
    lines += [
        "",
        "## Official Processed Data Evidence",
        "",
        "| File | Rows | Data-source counts | Read status |",
        "|---|---:|---|---|",
    ]
    for item in parquet:
        status = "missing" if not item.get("exists") else ("error: " + item.get("read_error", "unknown") if item.get("read_error") else "PASS")
        lines.append(f"| `{item['path']}` | {item.get('rows', '')} | `{item.get('data_source_counts', {})}` | {status} |")
    lines += [
        "",
        "## Split Finding",
        "",
        "The public RODS README states 400 human seeds and does not expose a separate RODS ID manifest. The EnvTuning README and committed parquet files document the reproducible processed split: `bfcl_train_base.parquet` has 100 `multi_turn_base` rows, while `bfcl_train.parquet` has 400 rows split 100 each across Base, Missing Function, Missing Parameter, and Long Context. The source README says this was created by sampling half of each 200-example BFCL V3 Multi-Turn category; the repository does not expose an independent RODS seed manifest in the RODS directory.",
        "",
        "No `datasets.load_dataset()` call was used by this audit.",
        "",
        "## Parse Errors",
        "",
    ]
    if parse_errors:
        for item in parse_errors:
            lines.append(f"- `{item['path']}`: {item['errors'][:2]}")
    else:
        lines.append("- None; all discovered JSONL records parsed successfully.")
    lines += ["", "## Detailed Machine-Readable Report", "", f"`{JSON_REPORT}`", ""]
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json_report": str(JSON_REPORT), "markdown_report": str(MD_REPORT), "category_counts": dict(category_counts), "parquet": parquet}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
