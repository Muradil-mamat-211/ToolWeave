from __future__ import annotations

import pytest

from env_tuning.interaction.data_models import ResponseType
from env_tuning.interaction.response_handler import ResponseHandler
from env_tuning.interaction.utils import parse_model_response


@pytest.mark.parametrize(
    "raw,expected_type",
    [
        (
            '<think>plan</think><tool_call>{"name":"pwd","arguments":{}}</tool_call>',
            ResponseType.TOOL_CALL,
        ),
        ("<think>done</think><answer>finished</answer>", ResponseType.ANSWER),
        (
            '<think>plan</think><tool_call>[{"name":"pwd","arguments":{}},{"name":"ls","arguments":{}}]</tool_call>',
            ResponseType.TOOL_CALL,
        ),
    ],
)
def test_valid_actions(raw, expected_type):
    parsed = ResponseHandler().parse_and_validate(
        [{"role": "assistant", "content": raw}]
    )
    assert parsed.is_valid
    assert parsed.response_type is expected_type


@pytest.mark.parametrize(
    "raw,error_fragment",
    [
        ('<tool_call>{"name":"pwd","arguments":{}}</tool_call>', "Missing <think>"),
        (
            '<thinking>plan</thinking><tool_call>{"name":"pwd","arguments":{}}</tool_call>',
            "Missing <think>",
        ),
        (
            '<reason>plan</reason><tool_call>{"name":"pwd","arguments":{}}</tool_call>',
            "Missing <think>",
        ),
        ("<think>plan<answer>done</answer>", "Missing <think>"),
        ("<think>plan</think><tool_call>{bad}</tool_call>", "Invalid JSON"),
        (
            '<think>plan</think><tool_call>{"name":"pwd","arguments":{}}</tool_call>outside',
            "must not contain text outside",
        ),
        (
            '<think>plan</think><tool_call>{"name":"pwd","arguments":{}}</tool_call><answer>x</answer>',
            "cannot contain both",
        ),
        (
            '<think>plan</think><tool_call>{"name":"pwd","arguments":{}}</tool_call><tool_call>{"name":"ls","arguments":{}}</tool_call>',
            "Multiple <tool_call>",
        ),
    ],
)
def test_invalid_actions_are_parse_errors(raw, error_fragment):
    parsed = ResponseHandler().parse_and_validate(
        [{"role": "assistant", "content": raw}]
    )
    assert not parsed.is_valid
    assert parsed.response_type is ResponseType.PARSE_ERROR
    assert error_fragment in parsed.error_message


def test_parser_does_not_enforce_block_order():
    # Current executable behavior, not a recommendation: regex validation accepts
    # the action block before the think block even though prompts require think first.
    raw = '<answer>done</answer><think>late reasoning</think>'
    content, flag = parse_model_response(raw)
    assert (content, flag) == ("done", "answer")


def test_empty_think_and_empty_answer_are_currently_accepted():
    content, flag = parse_model_response("<think></think><answer></answer>")
    assert (content, flag) == ("", "answer")

