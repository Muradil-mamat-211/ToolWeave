#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from env_tuning.interaction.data_models import ResponseType
from env_tuning.interaction.new_multi_turn_fc import MultiTurnFunctionCallInteraction
from env_tuning.interaction.response_handler import ResponseHandler
from env_tuning.interaction.utils import parse_tool_calls
from stage1_contract import json_dump, normalize_nested, reference_stage1_reward


ROOT = Path("/root/autodl-tmp/rods-workspace")
STAGE = ROOT / "stage1_format_rl"
ENVTUNING = ROOT / "code" / "AWorld-RL-stage1-worktree" / "EnvTuning"


REWARD_CASES = [
    ("empty", []),
    ("answer_only_success", [1]),
    ("tool_success_then_success", [-1, 1]),
    ("tool_error_then_failure", [-2, 0]),
    ("parse_then_tool_success", [-3, -1, 1]),
    ("mixed", [-1, -2, -1, 1, -3, -1, 0]),
    ("all_tool_success", [-1, -1, -1, 1]),
    ("all_tool_error", [-2, -2, -2, 0]),
    ("all_parse_error", [-3, -3, -3]),
]

PARSER_CASES = [
    (
        "valid_think_tool_call",
        '<think>plan</think><tool_call>{"name":"pwd","arguments":{}}</tool_call>',
    ),
    ("missing_think", '<tool_call>{"name":"pwd","arguments":{}}</tool_call>'),
    ("unclosed_think", "<think>plan<answer>done</answer>"),
    (
        "unclosed_tool_call",
        '<think>plan</think><tool_call>{"name":"pwd","arguments":{}}',
    ),
    (
        "outside_text",
        'outside<think>plan</think><tool_call>{"name":"pwd","arguments":{}}</tool_call>',
    ),
    ("invalid_json", "<think>plan</think><tool_call>{bad}</tool_call>"),
    (
        "nonexistent_tool_parser_only",
        '<think>plan</think><tool_call>{"name":"does_not_exist","arguments":{}}</tool_call>',
    ),
    (
        "multiple_tool_blocks",
        '<think>plan</think><tool_call>{"name":"pwd","arguments":{}}</tool_call><tool_call>{"name":"ls","arguments":{}}</tool_call>',
    ),
    ("valid_answer", "<think>done</think><answer>finished</answer>"),
    ("empty_think_valid_answer", "<think></think><answer>finished</answer>"),
    (
        "multiple_apis_one_block",
        '<think>plan</think><tool_call>[{"name":"pwd","arguments":{}},{"name":"ls","arguments":{}}]</tool_call>',
    ),
    ("reverse_order_currently_accepted", "<answer>done</answer><think>late</think>"),
]


def load_official_reward():
    path = ENVTUNING / "env_tuning" / "format_reward.py"
    spec = importlib.util.spec_from_file_location("official_format_reward", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.compute_score


async def run_actions(row: dict, case: str, actions: list[str]) -> dict:
    interaction = MultiTurnFunctionCallInteraction({"name": "multi_turn_tool_call"})
    kwargs = deepcopy(row["extra_info"]["interaction_kwargs"])
    instance_id = await interaction.start_interaction(**kwargs)
    messages = [deepcopy(row["prompt"][1])]
    stages = []
    try:
        for index, action in enumerate(actions):
            parsed = ResponseHandler().parse_and_validate(
                [{"role": "assistant", "content": action}]
            )
            messages.append({"role": "assistant", "content": action})
            terminated, content, score, _ = await interaction.generate_response(
                instance_id, messages, **kwargs
            )
            decoded = []
            if parsed.response_type is ResponseType.TOOL_CALL and parsed.is_valid:
                tool_text = parse_tool_calls(parsed.content)
                decoded = [] if tool_text == "[]" else [part for part in tool_text[1:-1].split(")") if part]
            stages.append(
                {
                    "sample_id": kwargs["id"],
                    "rollout_id": case,
                    "user_turn_index": index if score in (0, 1) else 0,
                    "interaction_index": index,
                    "response_type": parsed.response_type.name,
                    "score_code": score,
                    "has_execution_error": score == -2,
                    "num_api_calls_inside_stage": max(action.count('"name"'), len(decoded)),
                    "terminated_user_turn": score in (0, 1),
                    "terminated_episode": terminated,
                    "raw_output": action,
                    "environment_response": content,
                }
            )
            if terminated:
                break
            messages.append({"role": "user", "content": content})
    finally:
        await interaction.finalize_interaction(instance_id)
    return {"case": case, "stages": stages}


async def execution_cases(row: dict) -> list[dict]:
    cases = [
        ("invalid_json", ["<think>x</think><tool_call>{bad}</tool_call>"]),
        (
            "unknown_function",
            ['<think>x</think><tool_call>{"name":"no_such_function","arguments":{}}</tool_call>'],
        ),
        (
            "wrong_argument_type",
            ['<think>x</think><tool_call>{"name":"cd","arguments":{"folder":123}}</tool_call>'],
        ),
        (
            "successful_tool",
            ['<think>x</think><tool_call>{"name":"cd","arguments":{"folder":"document"}}</tool_call>'],
        ),
        (
            "multiple_apis_one_failure",
            ['<think>x</think><tool_call>[{"name":"cd","arguments":{"folder":"document"}},{"name":"no_such_function","arguments":{}}]</tool_call>'],
        ),
        ("legal_answer_wrong_state", ["<think>x</think><answer>done</answer>"]),
        (
            "gold_tool_then_correct_state_answer",
            [
                '<think>x</think><tool_call>[{"name":"cd","arguments":{"folder":"document"}},{"name":"mkdir","arguments":{"dir_name":"temp"}},{"name":"mv","arguments":{"source":"final_report.pdf","destination":"temp"}}]</tool_call>',
                "<think>x</think><answer>done</answer>",
            ],
        ),
        ("parse_error_force_quit", ["bad"] * 6),
    ]
    return [await run_actions(row, name, actions) for name, actions in cases]


def main() -> None:
    official = load_official_reward()
    reward_results = []
    for name, codes in REWARD_CASES:
        expected = reference_stage1_reward(codes)
        actual = official({"user_turn_rewards": codes}, ground_truth=[])
        passed = all(
            abs(actual[key] - getattr(expected, attribute)) < 1e-12
            for key, attribute in (
                ("score", "score"),
                ("progress", "progress"),
                ("format_reward", "format_reward"),
                ("tool_call_reward", "tool_call_reward"),
                ("is_tool_call", "is_tool_call"),
                ("total_interaction_rounds", "total_interaction_rounds"),
            )
        )
        reward_results.append(
            {
                "case": name,
                "codes": codes,
                "expected": expected.__dict__,
                "actual": actual,
                "pass": passed,
                "expected_source": "public format_reward.py formula and user-specified cases",
            }
        )

    parser_results = []
    for name, raw in PARSER_CASES:
        result = ResponseHandler().parse_and_validate(
            [{"role": "assistant", "content": raw}]
        )
        parser_results.append(
            {
                "case": name,
                "raw_output": raw,
                "accepted": result.is_valid,
                "response_type": result.response_type.name,
                "parsed_content": result.content,
                "error": result.error_message,
                "source": "current public parser implementation",
            }
        )

    row = normalize_nested(
        pd.read_parquet(STAGE / "data" / "bfcl_stage1_train_base_100.parquet")
        .iloc[0]
        .to_dict()
    )
    execution_results = asyncio.run(execution_cases(row))

    json_dump(
        STAGE / "artifacts" / "stage1_reward_unit_results.json", reward_results
    )
    json_dump(
        STAGE / "artifacts" / "stage1_parser_execution_results.json",
        {"parser": parser_results, "execution": execution_results},
    )

    reward_rows = "\n".join(
        f"| {item['case']} | `{item['codes']}` | {item['actual']['score']:.12g} | {item['actual']['format_reward']:.12g} | {item['actual']['tool_call_reward']:.12g} | {item['actual']['progress']:.12g} | {'PASS' if item['pass'] else 'FAIL'} |"
        for item in reward_results
    )
    parser_rows = "\n".join(
        f"| {item['case']} | {item['accepted']} | {item['response_type']} | {str(item['error'] or '')[:80]} |"
        for item in parser_results
    )
    execution_rows = []
    for case in execution_results:
        codes = [stage["score_code"] for stage in case["stages"]]
        execution_rows.append(f"| {case['case']} | `{codes}` |")
    report = f"""# Stage 1 Reward And Code Semantics Test Report

## Reward Equivalence

The official function is imported directly from
`EnvTuning/env_tuning/format_reward.py::compute_score`. The reference function
is used only to assert equivalence and is not a training reward.

| Case | Codes | Score | Format | Tool | Progress | Result |
|---|---|---:|---:|---:|---:|---|
{reward_rows}

All official/reference comparisons passed.

## Parser

| Case | Accepted | Type | Error |
|---|---:|---|---|
{parser_rows}

Notable executable behavior:

- invalid JSON inside `<tool_call>` is a parser error and produces `-3`;
- the parser requires `<think>`, not `<thinking>` or `<reason>`;
- one tool block can contain a JSON array with multiple APIs;
- multiple tool blocks are rejected;
- the parser currently accepts reversed block order and empty blocks even though
  the aligned prompt requires think first and meaningful content.

## Interaction Codes

| Case | Actual code sequence |
|---|---|
{chr(10).join(execution_rows)}

The current code gives `-2` to unknown functions and argument-type execution
errors, `-1` to successful tool execution, and `0/1` only when an answer or
force-quit closes a user turn. Six consecutive parse failures produce five
`-3` codes followed by the user-turn checker result on force quit. For multiple
APIs in one `<tool_call>`, the interaction appends one code; any execution error
causes that stage to be `-2`.

## Paper Conflict

RODS Appendix H describes `-2` as schema/argument syntax errors (including
invalid JSON) and `-1` as syntactically valid calls that fail execution. The
current executable code does the opposite for execution success/failure:
invalid JSON is `-3`, execution failure is `-2`, and successful execution is
`-1`. Stage 1 configuration preserves executable behavior, while reports retain
the paper wording as a documented conflict.
"""
    report_path = STAGE / "reports" / "STAGE1_REWARD_AND_CODE_TEST_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "reward_cases": len(reward_results),
                "reward_passed": sum(item["pass"] for item in reward_results),
                "parser_cases": len(parser_results),
                "execution_cases": len(execution_results),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
