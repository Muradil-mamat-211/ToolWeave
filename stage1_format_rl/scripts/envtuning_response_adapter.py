"""Lossless serving-to-EnvTuning response serialization diagnostics.

This module does not relax or modify EnvTuning's strict response parser.  It
keeps four separate evidence layers and only restores XML serialization when
the serving API exposes an explicit reasoning or structured-tool channel.
Ordinary prose is never promoted to a ``<think>`` block.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


_COMPLETE_THINK = re.compile(r"<think>([\s\S]*?)</think>")
_TOOL_ACTION = re.compile(r"\s*(<tool_call>[\s\S]*?</tool_call>)\s*\Z")
_ANSWER_ACTION = re.compile(r"\s*(<answer>[\s\S]*?</answer>)\s*\Z")
_COMPLETE_STRICT = re.compile(
    r"\s*<think>[\s\S]*?</think>\s*"
    r"(?:<tool_call>[\s\S]*?</tool_call>|<answer>[\s\S]*?</answer>)\s*\Z"
)
_KNOWN_TERMINAL_SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>")


@dataclass(frozen=True)
class EnvTuningParserInput:
    """Auditable result of restoring an EnvTuning parser input string."""

    envtuning_parser_input: str
    serialization_source: str
    reconstructed_think_from_reasoning_content: bool
    reconstructed_action_from_tool_calls: bool
    true_raw_available: bool
    raw_has_open_think: bool
    raw_has_close_think: bool
    raw_has_complete_think_block: bool
    api_reasoning_content_nonempty: bool
    api_content_has_complete_think_block: bool
    api_tool_calls_present: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value.strip() else None


def api_reasoning_content(message: Mapping[str, Any]) -> str | None:
    """Read both current vLLM ``reasoning`` and legacy API field names."""

    return _nonempty_text(message.get("reasoning_content")) or _nonempty_text(
        message.get("reasoning")
    )


def _literal_action(content: str | None) -> str | None:
    """Return one already-serialized action only when no outside text exists."""

    if not isinstance(content, str):
        return None
    for pattern in (_TOOL_ACTION, _ANSWER_ACTION):
        match = pattern.fullmatch(content)
        if match is not None:
            return match.group(1)
    return None


def _strip_known_terminal_special_token(text: str) -> str:
    """Remove only a tokenizer terminal marker from the parser serialization.

    TRUE RAW evidence remains untouched in its own artifact.  EnvTuning parses
    assistant content rather than the generated EOS marker, matching the
    OpenAI serving response's content boundary.
    """

    stripped = text
    for token in _KNOWN_TERMINAL_SPECIAL_TOKENS:
        if stripped.endswith(token):
            return stripped[: -len(token)]
    return stripped


def _normalize_structured_tool_call(raw_call: Mapping[str, Any]) -> dict[str, Any]:
    function = raw_call.get("function", raw_call)
    if not isinstance(function, Mapping):
        raise ValueError("structured tool call has no function object")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError("structured tool call requires a non-empty function name")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, Mapping):
        raise ValueError("structured tool-call arguments must be a JSON object")
    # Preserve names, values, and JSON scalar types.  Transport-only fields
    # such as call IDs are intentionally not part of EnvTuning's action body.
    return {"name": name, "arguments": dict(arguments)}


def serialize_structured_tool_calls(
    tool_calls: Sequence[Mapping[str, Any]] | None,
) -> str | None:
    """Losslessly restore OpenAI structured calls to one EnvTuning block."""

    if not tool_calls:
        return None
    calls = [_normalize_structured_tool_call(item) for item in tool_calls]
    body: Any = calls[0] if len(calls) == 1 else calls
    return (
        "<tool_call>"
        + json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        + "</tool_call>"
    )


def build_envtuning_parser_input(
    *,
    true_raw_decoded_text: str | None,
    reasoning_content: str | None,
    content: str | None,
    tool_calls: Sequence[Mapping[str, Any]] | None,
    reasoning_parser_name: str | None,
) -> EnvTuningParserInput:
    """Build the exact string passed to the frozen EnvTuning parser.

    Priority and safety rules:

    * A valid strict serialization present in generated token evidence is used
      verbatim.
    * When generated-token evidence is unavailable, an explicit serving
      reasoning channel may be combined with an already-serialized action or
      structured tool call.  A configured reasoning parser is required.
    * If true generated-token evidence is available but does not contain a
      complete thinking block, API prose is never relabelled as reasoning.
    * Without an explicit reasoning channel, content is passed through and the
      strict parser remains fail-closed.
    """

    raw_available = isinstance(true_raw_decoded_text, str)
    raw = true_raw_decoded_text or ""
    raw_parser_serialization = _strip_known_terminal_special_token(raw)
    raw_open = "<think>" in raw
    raw_close = "</think>" in raw
    raw_complete = bool(_COMPLETE_THINK.search(raw))
    content_text = content if isinstance(content, str) else ""
    content_complete = bool(_COMPLETE_THINK.search(content_text))
    reason = _nonempty_text(reasoning_content)
    structured_action = serialize_structured_tool_calls(tool_calls)

    if raw_complete and _COMPLETE_STRICT.fullmatch(raw_parser_serialization):
        return EnvTuningParserInput(
            envtuning_parser_input=raw_parser_serialization,
            serialization_source="true_raw",
            reconstructed_think_from_reasoning_content=False,
            reconstructed_action_from_tool_calls=False,
            true_raw_available=True,
            raw_has_open_think=raw_open,
            raw_has_close_think=raw_close,
            raw_has_complete_think_block=True,
            api_reasoning_content_nonempty=reason is not None,
            api_content_has_complete_think_block=content_complete,
            api_tool_calls_present=bool(tool_calls),
        )

    # Some APIs preserve literal strict XML in message.content while not
    # exposing output token IDs.  Keep this distinct from TRUE RAW evidence.
    if not raw_available and _COMPLETE_STRICT.fullmatch(content_text):
        return EnvTuningParserInput(
            envtuning_parser_input=content_text,
            serialization_source="api_content_literal_serialization",
            reconstructed_think_from_reasoning_content=False,
            reconstructed_action_from_tool_calls=False,
            true_raw_available=False,
            raw_has_open_think=False,
            raw_has_close_think=False,
            raw_has_complete_think_block=False,
            api_reasoning_content_nonempty=reason is not None,
            api_content_has_complete_think_block=True,
            api_tool_calls_present=bool(tool_calls),
        )

    # Compatibility restoration is only justified when the true token stream
    # was not available and a configured serving parser exposed an explicit
    # reasoning channel.  If token IDs prove there was no <think> block, this
    # branch deliberately does not manufacture one.
    if not raw_available and reason is not None and reasoning_parser_name:
        literal_action = _literal_action(content)
        action = literal_action or structured_action
        if action is not None:
            return EnvTuningParserInput(
                envtuning_parser_input=f"<think>{reason}</think>{action}",
                serialization_source="reconstructed_from_reasoning_content",
                reconstructed_think_from_reasoning_content=True,
                reconstructed_action_from_tool_calls=(
                    literal_action is None and structured_action is not None
                ),
                true_raw_available=False,
                raw_has_open_think=False,
                raw_has_close_think=False,
                raw_has_complete_think_block=False,
                api_reasoning_content_nonempty=True,
                api_content_has_complete_think_block=content_complete,
                api_tool_calls_present=bool(tool_calls),
            )

    return EnvTuningParserInput(
        envtuning_parser_input=content_text,
        serialization_source="no_valid_thinking_channel",
        reconstructed_think_from_reasoning_content=False,
        reconstructed_action_from_tool_calls=False,
        true_raw_available=raw_available,
        raw_has_open_think=raw_open,
        raw_has_close_think=raw_close,
        raw_has_complete_think_block=raw_complete,
        api_reasoning_content_nonempty=reason is not None,
        api_content_has_complete_think_block=content_complete,
        api_tool_calls_present=bool(tool_calls),
    )


def parse_envtuning_input(parser_input: str) -> dict[str, Any]:
    """Run the frozen parser and return explicit Level-4 diagnostics."""

    from env_tuning.interaction.utils import parse_model_response

    content, message = parse_model_response(parser_input)
    thinking_matches = _COMPLETE_THINK.findall(parser_input)
    success = message in {"tool_call", "answer"}
    return {
        "parse_success": success,
        "parse_error": None if success else message,
        "parsed_thinking": thinking_matches[0] if len(thinking_matches) == 1 else None,
        "parsed_tool_call": content if message == "tool_call" else None,
        "parsed_answer": content if message == "answer" else None,
        "parsed_response_type": message if success else "parse_error",
    }
