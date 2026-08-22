from __future__ import annotations

import asyncio
from copy import deepcopy

from env_tuning.interaction.new_multi_turn_fc import MultiTurnFunctionCallInteraction


def run_actions(row, actions):
    async def scenario():
        interaction = MultiTurnFunctionCallInteraction({"name": "multi_turn_tool_call"})
        kwargs = deepcopy(row["extra_info"]["interaction_kwargs"])
        instance_id = await interaction.start_interaction(**kwargs)
        messages = [deepcopy(row["prompt"][1])]
        scores = []
        contents = []
        try:
            for action in actions:
                messages.append({"role": "assistant", "content": action})
                terminated, content, score, _ = await interaction.generate_response(
                    instance_id, messages, **kwargs
                )
                scores.append(score)
                contents.append(content)
                if terminated:
                    break
                messages.append({"role": "user", "content": content})
        finally:
            await interaction.finalize_interaction(instance_id)
        return scores, contents

    return asyncio.run(scenario())


def test_invalid_json_is_parse_error_minus_three(train_rows):
    scores, _ = run_actions(
        train_rows[0], ["<think>x</think><tool_call>{bad}</tool_call>"]
    )
    assert scores == [-3.0]


def test_unknown_function_is_execution_error_minus_two(train_rows):
    scores, _ = run_actions(
        train_rows[0],
        [
            '<think>x</think><tool_call>{"name":"no_such_function","arguments":{}}</tool_call>'
        ],
    )
    assert scores == [-2.0]


def test_wrong_argument_type_is_execution_error_minus_two(train_rows):
    scores, _ = run_actions(
        train_rows[0],
        ['<think>x</think><tool_call>{"name":"cd","arguments":{"folder":123}}</tool_call>'],
    )
    assert scores == [-2.0]


def test_successful_tool_execution_is_category_minus_one(train_rows):
    scores, _ = run_actions(
        train_rows[0],
        ['<think>x</think><tool_call>{"name":"cd","arguments":{"folder":"document"}}</tool_call>'],
    )
    assert scores == [-1.0]


def test_multiple_apis_append_one_stage_code_and_any_failure_makes_minus_two(train_rows):
    scores, _ = run_actions(
        train_rows[0],
        [
            '<think>x</think><tool_call>[{"name":"cd","arguments":{"folder":"document"}},{"name":"no_such_function","arguments":{}}]</tool_call>'
        ],
    )
    assert scores == [-2.0]


def test_legal_answer_without_required_state_is_zero(train_rows):
    scores, _ = run_actions(
        train_rows[0], ["<think>x</think><answer>done</answer>"]
    )
    assert scores == [0.0]


def test_gold_execution_then_legal_answer_is_one(train_rows):
    scores, _ = run_actions(
        train_rows[0],
        [
            '<think>x</think><tool_call>[{"name":"cd","arguments":{"folder":"document"}},{"name":"mkdir","arguments":{"dir_name":"temp"}},{"name":"mv","arguments":{"source":"final_report.pdf","destination":"temp"}}]</tool_call>',
            "<think>x</think><answer>done</answer>",
        ],
    )
    assert scores == [-1.0, 1.0]


def test_parse_error_retry_and_force_quit_codes(train_rows):
    scores, _ = run_actions(train_rows[0], ["bad"] * 6)
    assert scores[:5] == [-3.0] * 5
    assert scores[5] == 0.0


def test_empty_ground_truth_special_branch(train_rows):
    row = deepcopy(train_rows[0])
    row["extra_info"]["interaction_kwargs"]["ground_truth"][0] = []
    answer_scores, _ = run_actions(
        row, ["<think>x</think><answer>tool unavailable</answer>"]
    )
    tool_scores, _ = run_actions(
        row,
        ['<think>x</think><tool_call>{"name":"pwd","arguments":{}}</tool_call>'],
    )
    assert answer_scores == [1.0]
    assert tool_scores == [0.0]

