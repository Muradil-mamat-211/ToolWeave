#!/usr/bin/env python3
"""Static audit of every active BFCL function's public Python return contract."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path

import bfcl_env.multi_turn_utils as multi_turn

from env_tuning.rods_data_generation_v1.function_catalog import FunctionCatalog
from env_tuning.rods_data_generation_v1.queue import atomic_write_json
from env_tuning.rods_data_generation_v1.result_semantics import (
    DOMAIN_NEGATIVE_CONTRACTS,
    HARD_FAILURE_FLAG_CONTRACTS,
)


def _literal_strings(node: ast.AST) -> list[str]:
    return sorted(
        {
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
    )


def _false_dict_keys(node: ast.AST) -> list[str]:
    output: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Dict):
            continue
        for key, value in zip(child.keys, child.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Constant)
                and value.value is False
            ):
                output.add(key.value)
    return sorted(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = FunctionCatalog.from_training_parquet(args.catalog)
    class_for_function = catalog.class_for_function()
    modules: dict[str, ast.Module] = {}
    module_paths: dict[str, str] = {}
    class_nodes: dict[str, ast.ClassDef] = {}
    for class_name, module_name in multi_turn.CLASS_FILE_PATH_MAPPING_WO_AUG.items():
        module = __import__(module_name, fromlist=[class_name])
        path = Path(inspect.getsourcefile(module) or "")
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        modules[class_name] = parsed
        module_paths[class_name] = str(path.resolve())
        class_nodes[class_name] = next(
            node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )

    functions = []
    for name in catalog.names():
        class_name = class_for_function[name]
        method = next(
            (
                node
                for node in class_nodes[class_name].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
            ),
            None,
        )
        if method is None:
            raise RuntimeError(f"active function has no public VM method: {class_name}.{name}")
        strings = _literal_strings(method)
        functions.append(
            {
                "class_name": class_name,
                "function_name": name,
                "source_file": module_paths[class_name],
                "line": method.lineno,
                "has_explicit_error_key": any(
                    isinstance(child, ast.Constant) and child.value == "error"
                    for child in ast.walk(method)
                ),
                "false_result_keys": _false_dict_keys(method),
                "negative_or_unknown_literals": [
                    value
                    for value in strings
                    if any(marker in value.casefold() for marker in ("not found", "unknown", "not following", "already following"))
                ],
                "returns_none_literal": any(
                    isinstance(child, ast.Return)
                    and isinstance(child.value, ast.Constant)
                    and child.value.value is None
                    for child in ast.walk(method)
                ),
            }
        )

    by_name = {item["function_name"]: item for item in functions}
    contract_checks = []
    for contract in DOMAIN_NEGATIVE_CONTRACTS:
        item = by_name.get(contract.function_name)
        if item is None:
            raise RuntimeError(f"domain-negative contract is absent from active catalog: {contract.function_name}")
        source = Path(item["source_file"]).read_text(encoding="utf-8")
        if isinstance(contract.exact_value, str) and contract.exact_value not in source:
            raise RuntimeError(f"domain-negative sentinel missing from source: {contract}")
        contract_checks.append(
            {
                "function_name": contract.function_name,
                "path": list(contract.source_path),
                "exact_value": contract.exact_value,
                "source_verified": True,
            }
        )
    for function_name, path in HARD_FAILURE_FLAG_CONTRACTS.items():
        item = by_name.get(function_name)
        if item is None or path[-1] not in item["false_result_keys"]:
            raise RuntimeError(f"hard false-flag contract not verified: {function_name}.{path[-1]}")

    report = {
        "schema_version": "bfcl_result_contract_audit.v1",
        "catalog": str(args.catalog.resolve()),
        "active_function_count": len(functions),
        "all_active_functions_resolved_to_public_vm_source": len(functions) == len(catalog.names()),
        "domain_negative_contracts": contract_checks,
        "hard_failure_flag_contracts": HARD_FAILURE_FLAG_CONTRACTS,
        "generic_not_found_substring_rule_used": False,
        "functions": functions,
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: report[key] for key in report if key != "functions"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
