#!/usr/bin/env python3
"""Convert Stage 1 Base samples into explicit Stage 2 gold-call trajectories."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def read_json_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def load_tool_docs(bfcl_dir: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for path in sorted((bfcl_dir / "multi_turn_func_doc").glob("*.json")):
        for record in read_json_records(path):
            name = record.get("name")
            if isinstance(name, str) and name:
                docs[name] = record
    return docs


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


def parse_call(call_text: str, docs: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Parse BFCL's Python-like reference call into JSON-compatible arguments."""
    try:
        node = ast.parse(call_text.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call):
        return None
    name = call_name(node.func)
    if not name:
        return None
    properties = docs.get(name, {}).get("parameters", {}).get("properties", {})
    property_names = list(properties) if isinstance(properties, dict) else []
    arguments: dict[str, Any] = {}
    for index, value in enumerate(node.args):
        if index >= len(property_names):
            return None
        arguments[property_names[index]] = literal_value(value)
    for keyword in node.keywords:
        if keyword.arg is None:
            return None
        arguments[keyword.arg] = literal_value(keyword.value)
    return {"name": name.strip(), "arguments": arguments}


def iter_call_texts(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_call_texts(item)
    elif hasattr(value, "tolist"):
        yield from iter_call_texts(value.tolist())


def parse_gold_calls(ground_truth: Any, docs: dict[str, dict[str, Any]]) -> list[dict[str, Any]] | None:
    texts = list(iter_call_texts(ground_truth))
    if not texts:
        return None
    calls: list[dict[str, Any]] = []
    for text in texts:
        parsed = parse_call(text, docs)
        if parsed is None:
            return None
        calls.append(parsed)
    return calls or None


def augment_tools(
    existing_tools: list[dict[str, Any]], gold_calls: list[dict[str, Any]], docs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Ensure every referenced gold function has its downloaded BFCL schema."""
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool in existing_tools:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            tools.append(tool)
            seen.add(tool["name"])
    for call in gold_calls:
        name = call["name"]
        if name in seen:
            continue
        tool = docs.get(name)
        if tool is None:
            raise ValueError(f"Missing BFCL schema for gold function {name!r}")
        tools.append(tool)
        seen.add(name)
    return tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_file", type=Path, required=True)
    parser.add_argument("--bfcl_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.stage1_file.is_file():
        raise FileNotFoundError(args.stage1_file)
    docs = load_tool_docs(args.bfcl_dir)
    if not docs:
        raise FileNotFoundError(f"No tool schemas found below {args.bfcl_dir / 'multi_turn_func_doc'}")

    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for record in read_json_records(args.stage1_file):
        gold = record.get("gold", {}).get("ground_truth")
        calls = parse_gold_calls(gold, docs)
        if not calls:
            skipped.append({"id": str(record.get("id")), "reason": "gold_calls_unparseable_or_empty"})
            continue
        conversation = record.get("conversation") or record.get("raw", {}).get("question")
        if not isinstance(conversation, list):
            skipped.append({"id": str(record.get("id")), "reason": "conversation_missing"})
            continue
        try:
            tools = augment_tools(record.get("tools", []), calls, docs)
        except ValueError as exc:
            skipped.append({"id": str(record.get("id")), "reason": str(exc)})
            continue
        accepted.append(
            {
                "id": record.get("id"),
                "stage": "stage2_base_reasoning",
                "split": "base",
                "prompt": "",
                "conversation": conversation,
                "tools": tools,
                "gold_calls": calls,
                "raw": record.get("raw", record),
            }
        )

    if len(accepted) < 80:
        raise RuntimeError(
            f"Only {len(accepted)} samples have parseable gold_calls; at least 80 are required. "
            f"Skipped={len(skipped)}"
        )

    sys.path.insert(0, str(Path(__file__).parents[1] / "code" / "rods_stage2_local"))
    from prompt_template import render_prompt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in accepted:
            record["prompt"] = render_prompt(record["conversation"], record["tools"])
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    skipped_path = args.out.with_suffix(".skipped.jsonl")
    with skipped_path.open("w", encoding="utf-8") as handle:
        for row in skipped:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"stage1_file={args.stage1_file}")
    print(f"output={args.out}")
    print(f"samples={len(accepted)}")
    print(f"skipped={len(skipped)}")
    print(f"skipped_file={skipped_path}")
    for record in accepted[:3]:
        print(
            json.dumps(
                {
                    "id": record["id"],
                    "prompt_preview": record["prompt"][:180],
                    "tools_count": len(record["tools"]),
                    "gold_calls_length": len(record["gold_calls"]),
                    "gold_calls_first_two": record["gold_calls"][:2],
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
