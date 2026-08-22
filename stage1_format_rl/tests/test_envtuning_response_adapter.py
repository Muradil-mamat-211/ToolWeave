"""Regression tests for lossless EnvTuning serving compatibility."""

from __future__ import annotations

from envtuning_response_adapter import (
    build_envtuning_parser_input,
    parse_envtuning_input,
)


TOOL = '<tool_call>{"name":"lookup","arguments":{"x":1}}</tool_call>'


def test_true_raw_strict_serialization_needs_no_reconstruction() -> None:
    raw = f"<think>A</think>{TOOL}"
    result = build_envtuning_parser_input(
        true_raw_decoded_text=raw,
        reasoning_content=None,
        content=raw,
        tool_calls=None,
        reasoning_parser_name=None,
    )
    assert result.envtuning_parser_input == raw
    assert result.serialization_source == "true_raw"
    assert not result.reconstructed_think_from_reasoning_content
    assert parse_envtuning_input(result.envtuning_parser_input)["parse_success"]


def test_explicit_reasoning_channel_can_restore_split_serialization() -> None:
    result = build_envtuning_parser_input(
        true_raw_decoded_text=None,
        reasoning_content="A",
        content=TOOL,
        tool_calls=None,
        reasoning_parser_name="qwen3",
    )
    assert result.envtuning_parser_input == f"<think>A</think>{TOOL}"
    assert result.serialization_source == "reconstructed_from_reasoning_content"
    assert result.reconstructed_think_from_reasoning_content
    assert parse_envtuning_input(result.envtuning_parser_input)["parse_success"]


def test_plain_prose_is_never_promoted_to_thinking() -> None:
    content = f"A\n{TOOL}"
    result = build_envtuning_parser_input(
        true_raw_decoded_text=content,
        reasoning_content=None,
        content=content,
        tool_calls=None,
        reasoning_parser_name=None,
    )
    assert result.envtuning_parser_input == content
    assert result.serialization_source == "no_valid_thinking_channel"
    assert not result.reconstructed_think_from_reasoning_content
    assert not parse_envtuning_input(result.envtuning_parser_input)["parse_success"]


def test_existing_think_block_is_not_nested() -> None:
    raw = f"<think>A</think>{TOOL}"
    result = build_envtuning_parser_input(
        true_raw_decoded_text=raw,
        reasoning_content="A",
        content=TOOL,
        tool_calls=None,
        reasoning_parser_name="qwen3",
    )
    assert result.envtuning_parser_input.count("<think>") == 1
    assert result.envtuning_parser_input.count("</think>") == 1
    assert result.serialization_source == "true_raw"


def test_structured_tool_call_restoration_preserves_payload() -> None:
    result = build_envtuning_parser_input(
        true_raw_decoded_text=None,
        reasoning_content="A",
        content=None,
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"x":1,"flag":false}',
                },
            }
        ],
        reasoning_parser_name="qwen3",
    )
    assert result.reconstructed_action_from_tool_calls
    assert (
        result.envtuning_parser_input
        == '<think>A</think><tool_call>{"name":"lookup","arguments":{"x":1,"flag":false}}</tool_call>'
    )
    assert parse_envtuning_input(result.envtuning_parser_input)["parse_success"]


def test_true_raw_without_think_overrides_claimed_reasoning_channel() -> None:
    result = build_envtuning_parser_input(
        true_raw_decoded_text=TOOL,
        reasoning_content="A",
        content=TOOL,
        tool_calls=None,
        reasoning_parser_name="qwen3",
    )
    assert result.serialization_source == "no_valid_thinking_channel"
    assert not result.reconstructed_think_from_reasoning_content
    assert not parse_envtuning_input(result.envtuning_parser_input)["parse_success"]


def test_true_raw_terminal_special_token_is_not_sent_to_strict_parser() -> None:
    raw = f"<think>A</think>{TOOL}<|im_end|>"
    result = build_envtuning_parser_input(
        true_raw_decoded_text=raw,
        reasoning_content=None,
        content=f"<think>A</think>{TOOL}",
        tool_calls=None,
        reasoning_parser_name=None,
    )
    assert result.serialization_source == "true_raw"
    assert result.envtuning_parser_input == f"<think>A</think>{TOOL}"
    assert parse_envtuning_input(result.envtuning_parser_input)["parse_success"]
