#!/usr/bin/env python3
"""Prepare a small, explicit BFCL V3 multi-turn Base JSONL artifact.

The official EnvTuning parquet remains the training input because it preserves
veRL's multi-turn interaction metadata. This JSONL is a transparent inspection
and evaluation artifact and is never loaded as a whole into memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def read_json_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(record, dict):
                yield record


def load_first_candidate(bfcl_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    exact = bfcl_dir / "BFCL_v3_multi_turn_base.json"
    candidates = [exact] if exact.is_file() else []
    if not candidates:
        candidates = sorted(
            path
            for path in bfcl_dir.rglob("*")
            if path.is_file()
            and ".cache" not in path.parts
            and "v3" in path.name.lower()
            and "multi" in path.name.lower()
            and "base" in path.name.lower()
            and path.suffix.lower() in {".json", ".jsonl"}
        )
    if not candidates:
        raise FileNotFoundError(
            "No unambiguous BFCL V3 multi-turn Base JSON/JSONL file was found. "
            "Inspect the candidate files before training."
        )
    path = candidates[0]
    records = list(read_json_records(path))
    if not records:
        raise ValueError(f"Candidate file is empty: {path}")
    return path, records


def load_ground_truth(bfcl_dir: Path) -> dict[str, Any]:
    path = bfcl_dir / "possible_answer" / "BFCL_v3_multi_turn_base.json"
    if not path.is_file():
        return {}
    return {record.get("id"): record.get("ground_truth") for record in read_json_records(path)}


def load_tool_docs(bfcl_dir: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    doc_dir = bfcl_dir / "multi_turn_func_doc"
    for path in sorted(doc_dir.glob("*.json")):
        for record in read_json_records(path):
            name = record.get("name")
            if name:
                docs[str(name)] = record
    return docs


def method_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1]


def render_conversation(question: list[list[dict[str, Any]]]) -> str:
    turns = []
    for index, turn in enumerate(question, start=1):
        turns.append(
            f"Turn {index}:\n"
            + "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in turn)
        )
    return "\n\n".join(turns)


def build_record(raw: dict[str, Any], ground_truth: dict[str, Any], docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = []
    tools = []
    for qualified in raw.get("path", []):
        name = method_name(str(qualified))
        if name in names:
            continue
        names.append(name)
        tools.append(
            docs.get(
                name,
                {
                    "name": name,
                    "description": "Schema was not present in the downloaded BFCL function documentation.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            )
        )
    return {
        "id": raw.get("id"),
        "stage": "stage1_format",
        "split": "base",
        "prompt": render_conversation(raw.get("question", [])),
        "conversation": raw.get("question", []),
        "tools": tools,
        "gold": {
            "path": raw.get("path", []),
            "ground_truth": ground_truth.get(raw.get("id")),
            "involved_classes": raw.get("involved_classes", []),
        },
        "raw": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    source, records = load_first_candidate(args.bfcl_dir)
    ground_truth = load_ground_truth(args.bfcl_dir)
    docs = load_tool_docs(args.bfcl_dir)
    if len(records) < args.num_samples:
        raise ValueError(f"Only {len(records)} records found in {source}; cannot produce {args.num_samples}.")
    selected = records[: args.num_samples]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parents[1] / "code" / "rods_stage1_local"))
    from prompt_template import render_prompt

    with args.out.open("w", encoding="utf-8") as handle:
        for raw in selected:
            record = build_record(raw, ground_truth, docs)
            record["prompt"] = render_prompt(record["conversation"], record["tools"])
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"source={source}")
    print(f"output={args.out}")
    print(f"samples={len(selected)}")
    for record in selected[:3]:
        print(
            json.dumps(
                {
                    "id": record.get("id"),
                    "turns": len(record.get("question", [])),
                    "tools": [method_name(str(name)) for name in record.get("path", [])],
                    "gold_turns": len(ground_truth.get(record.get("id"), []) or []),
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
