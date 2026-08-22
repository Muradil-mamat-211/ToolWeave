"""Deterministic tests for the source-locked RODS/BFCL evaluation path."""

from __future__ import annotations

import ast
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("TOOLWEAVE_ASSET_ROOT", SOURCE_ROOT)).expanduser().resolve()
SCRIPTS = SOURCE_ROOT / "stage1_format_rl/scripts"
ENVTUNING = SOURCE_ROOT / "code/AWorld-RL-stage1-worktree/EnvTuning"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ENVTUNING))

from env_tuning.rods_matchtir_v1.provenance import extract_available_functions
from rods_official_bfcl_protocol import (
    QWEN_HANDLER_SOURCE,
    EnvTuningDiagnosticCode,
    assert_source_contract,
    build_bfcl_test_entry,
    extract_qwen_tool_calls,
    format_qwen_prompt,
    parse_qwen_response,
)
from run_qwen_rods_bfcl100_official_eval import (
    strict_diagnostic_for_step,
    to_builtin,
    validate_full_run_integrity,
)


DATASET = (
    WORKSPACE
    / "stage1_format_rl/artifacts/stage3_generator_full_historical_revalidation_20260813T083630Z"
    / "11_official_rods_eval/dataset/eval_rods_bfcl_multiturn_100.parquet"
)


def _upstream_method(name: str):
    """Compile one dependency-free method directly from audited BFCL source."""

    tree = ast.parse(QWEN_HANDLER_SOURCE.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QwenFCHandler"
    )
    source_node = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    function_node = copy.deepcopy(source_node)
    function_node.decorator_list = []
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"json": json, "re": __import__("re")}
    exec(compile(module, str(QWEN_HANDLER_SOURCE), "exec"), namespace)
    return namespace[name]


def test_audited_source_contract_is_present() -> None:
    assert_source_contract()


def test_qwen_parser_matches_public_source_exactly() -> None:
    upstream = _upstream_method("_extract_tool_calls")
    cases = [
        '<tool_call>\n{"name":"f","arguments":{"x":1}}\n</tool_call>',
        '<tool_call>{"name":"f","arguments":{"x":1}}</tool_call>',
        '<think>r</think>\n<tool_call>\n{"name":"f","arguments":{}}\n</tool_call>',
        '<tool_call>\nnot json\n</tool_call>',
    ]
    for raw in cases:
        assert extract_qwen_tool_calls(raw) == upstream(raw)


def test_qwen_prompt_formatter_matches_public_source_exactly() -> None:
    upstream = _upstream_method("_format_prompt")
    functions = [
        {
            "name": "lookup",
            "description": "Lookup a value.",
            "parameters": {
                "type": "dict",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Find Tesla."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should look it up.",
            "tool_calls": [{"name": "lookup", "arguments": {"name": "Tesla"}}],
        },
        {"role": "tool", "name": "lookup(name='Tesla')", "content": "TSLA"},
    ]
    assert format_qwen_prompt(messages, functions) == upstream(
        object(), copy.deepcopy(messages), copy.deepcopy(functions)
    )


def test_optional_think_is_separated_but_not_required() -> None:
    with_think = parse_qwen_response(
        '<think>lookup first</think>\n<tool_call>\n'
        '{"name":"lookup","arguments":{"name":"Tesla"}}\n</tool_call>'
    )
    assert with_think.reasoning_content == "lookup first"
    assert with_think.decoded_calls == ["lookup(name='Tesla')"]

    without_think = parse_qwen_response(
        '<tool_call>\n{"name":"lookup","arguments":{}}\n</tool_call>'
    )
    assert without_think.reasoning_content == ""
    assert without_think.decoded_calls == ["lookup()"]


def test_missing_function_row_reconstructs_official_update() -> None:
    frame = pd.read_parquet(DATASET)
    record = to_builtin(
        next(
            row
            for row in frame.to_dict(orient="records")
            if row["data_source"] == "multi_turn_miss_func"
        )
    )
    entry = build_bfcl_test_entry(
        record,
        extract_available_functions=extract_available_functions,
        to_builtin=to_builtin,
    )
    empty_turns = [index for index, turn in enumerate(entry["question"]) if not turn]
    assert empty_turns
    assert set(entry["missed_function"]) == {str(index) for index in empty_turns}
    assert all(entry["missed_function"][str(index)] for index in empty_turns)


def test_envtuning_diagnostic_codes_follow_public_meanings() -> None:
    parse_error = strict_diagnostic_for_step(
        raw_response="plain BFCL answer",
        official_decoded_calls=[],
        execution_results=None,
        terminal_success=False,
    )
    assert parse_error["code"] == EnvTuningDiagnosticCode.FORMAT_OR_PARSE_ERROR

    tool = '<think>x</think><tool_call>\n{"name":"f","arguments":{}}\n</tool_call>'
    success = strict_diagnostic_for_step(
        raw_response=tool,
        official_decoded_calls=["f()"],
        execution_results=["ok"],
        terminal_success=None,
    )
    assert success["code"] == EnvTuningDiagnosticCode.TOOL_EXECUTION_SUCCESS

    failure = strict_diagnostic_for_step(
        raw_response=tool,
        official_decoded_calls=["f()"],
        execution_results=["Error during execution: boom"],
        terminal_success=None,
    )
    assert failure["code"] == EnvTuningDiagnosticCode.TOOL_EXECUTION_ERROR

    answer = "<think>x</think><answer>done</answer>"
    terminal_ok = strict_diagnostic_for_step(
        raw_response=answer,
        official_decoded_calls=[],
        execution_results=None,
        terminal_success=True,
    )
    terminal_bad = strict_diagnostic_for_step(
        raw_response=answer,
        official_decoded_calls=[],
        execution_results=None,
        terminal_success=False,
    )
    assert terminal_ok["code"] == EnvTuningDiagnosticCode.TERMINAL_TURN_SUCCESS
    assert terminal_bad["code"] == EnvTuningDiagnosticCode.TERMINAL_TURN_FAILURE


def test_envtuning_execution_code_is_not_guessed_on_parser_divergence() -> None:
    raw = '<think>x</think><tool_call>{"name":"f","arguments":{}}</tool_call>'
    diagnostic = strict_diagnostic_for_step(
        raw_response=raw,
        official_decoded_calls=[],
        execution_results=None,
        terminal_success=None,
    )
    assert diagnostic["response_type"] == "tool_call"
    assert diagnostic["code"] is None
    assert diagnostic["code_reliable"] is False


def test_bfcl_force_quit_is_scored_failure_not_invalid_run() -> None:
    validate_full_run_integrity(
        100,
        {"runtime_failures": 0, "force_quit_samples": 1},
    )


def test_bfcl_runtime_failure_still_invalidates_full_run() -> None:
    with pytest.raises(RuntimeError, match="clean full eval violated"):
        validate_full_run_integrity(
            100,
            {"runtime_failures": 1, "force_quit_samples": 0},
        )
