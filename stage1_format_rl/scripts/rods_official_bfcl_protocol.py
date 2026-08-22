#!/usr/bin/env python3
"""Source-locked adapters for the released RODS model's BFCL evaluation.

This module deliberately keeps two public protocols separate:

* The primary evaluation protocol is the public BFCL Qwen prompting handler.
  Its prompt formatting and tool-call extraction below are a narrow port of
  ``bfcl_eval/model_handler/local_inference/qwen_fc.py``.  Importing that class
  pulls in unrelated Java/JavaScript parser dependencies, so the adapter keeps
  only the Python/Qwen methods required for BFCL multi-turn evaluation and
  records the upstream source hash at runtime.
* EnvTuning's ``-3/-2/-1/0/1`` values are interaction diagnostics.  They are
  retained as diagnostics and never used to replace the BFCL sample-accuracy
  score reported by RODS.

No Training Branch code imports this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from machine_paths import project_roots
from typing import Any, Mapping, Sequence


WORKSPACE = project_roots().source_root
GORILLA_ROOT = WORKSPACE / "code/gorilla/berkeley-function-call-leaderboard"
QWEN_HANDLER_SOURCE = (
    GORILLA_ROOT / "bfcl_eval/model_handler/local_inference/qwen_fc.py"
)
BFCL_BASE_HANDLER_SOURCE = GORILLA_ROOT / "bfcl_eval/model_handler/base_handler.py"
BFCL_DEFAULT_PROMPTS_SOURCE = (
    GORILLA_ROOT / "bfcl_eval/constants/default_prompts.py"
)
AWORLD_ENVTUNING_ROOT = WORKSPACE / "code/AWorld-RL-stage1-worktree/EnvTuning"
ENVTUNING_DIAGNOSTIC_SOURCES = (
    AWORLD_ENVTUNING_ROOT / "env_tuning/interaction/response_handler.py",
    AWORLD_ENVTUNING_ROOT / "env_tuning/interaction/new_multi_turn_fc.py",
    AWORLD_ENVTUNING_ROOT / "env_tuning/interaction/execution_manager.py",
    AWORLD_ENVTUNING_ROOT / "env_tuning/interaction/score_calculator.py",
)

# Public BFCL constants at the audited Gorilla commit.  They are asserted
# against the source file by ``assert_source_contract`` before evaluation.
BFCL_MAXIMUM_STEP_LIMIT = 20
BFCL_ADDITIONAL_FUNCTION_MESSAGE = (
    "I have updated some more functions you can choose from. What about now?"
)


class EnvTuningDiagnosticCode(IntEnum):
    """Exact meanings used by AWorld-RL/EnvTuning's interaction state machine."""

    FORMAT_OR_PARSE_ERROR = -3
    TOOL_EXECUTION_ERROR = -2
    TOOL_EXECUTION_SUCCESS = -1
    TERMINAL_TURN_FAILURE = 0
    TERMINAL_TURN_SUCCESS = 1


@dataclass(frozen=True)
class QwenProtocolResponse:
    """Official-Qwen-handler interpretation of one raw completion."""

    raw_response: str
    cleaned_response: str
    reasoning_content: str
    tool_calls: list[dict[str, Any]]
    decoded_calls: list[str]
    decode_error: str | None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.decoded_calls)

    def assistant_history_message(self) -> dict[str, Any]:
        if self.tool_calls:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": self.tool_calls,
            }
        else:
            message = {"role": "assistant", "content": self.cleaned_response}
        message["reasoning_content"] = self.reasoning_content
        return message


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_source_hashes() -> dict[str, str]:
    return {
        str(QWEN_HANDLER_SOURCE): sha256(QWEN_HANDLER_SOURCE),
        str(BFCL_BASE_HANDLER_SOURCE): sha256(BFCL_BASE_HANDLER_SOURCE),
        str(BFCL_DEFAULT_PROMPTS_SOURCE): sha256(BFCL_DEFAULT_PROMPTS_SOURCE),
    }


def envtuning_diagnostic_source_hashes() -> dict[str, str]:
    """Hashes of the exact public files defining ``-3/-2/-1/0/1``."""

    return {str(path): sha256(path) for path in ENVTUNING_DIAGNOSTIC_SOURCES}


def assert_source_contract() -> None:
    """Fail closed if the locally audited BFCL source no longer matches the port."""

    qwen_source = QWEN_HANDLER_SOURCE.read_text(encoding="utf-8")
    defaults_source = BFCL_DEFAULT_PROMPTS_SOURCE.read_text(encoding="utf-8")
    required_qwen_fragments = (
        'pattern = r"<tool_call>\\n(.*?)\\n</tool_call>"',
        "formatted_prompt += \"<|im_start|>system\\n\"",
        'formatted_prompt += "<|im_start|>assistant\\n"',
    )
    missing = [item for item in required_qwen_fragments if item not in qwen_source]
    if missing:
        raise RuntimeError(f"audited BFCL Qwen source contract changed: {missing}")
    if f"MAXIMUM_STEP_LIMIT = {BFCL_MAXIMUM_STEP_LIMIT}" not in defaults_source:
        raise RuntimeError("audited BFCL MAXIMUM_STEP_LIMIT changed")
    if BFCL_ADDITIONAL_FUNCTION_MESSAGE not in defaults_source:
        raise RuntimeError("audited BFCL missing-function update message changed")

    interaction_source = ENVTUNING_DIAGNOSTIC_SOURCES[1].read_text(encoding="utf-8")
    execution_source = ENVTUNING_DIAGNOSTIC_SOURCES[2].read_text(encoding="utf-8")
    required_diagnostic_fragments = (
        "-3.0",
        "return should_term, content, 1.0, extra",
        "return should_term, warning_hint + content, 0.0, extra",
    )
    missing_diagnostics = [
        item for item in required_diagnostic_fragments if item not in interaction_source
    ]
    if missing_diagnostics:
        raise RuntimeError(
            "audited EnvTuning terminal/parse diagnostic contract changed: "
            f"{missing_diagnostics}"
        )
    if "score = -2.0 if has_error else -1.0" not in execution_source:
        raise RuntimeError("audited EnvTuning execution diagnostic contract changed")


def _json_call_to_execution(call: Mapping[str, Any]) -> str:
    """Port of BFCL ``convert_to_function_call`` for one Qwen JSON call."""

    name = call["name"]
    arguments = call["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("Qwen tool call requires string name and object arguments")
    return f"{name}({','.join(f'{key}={value!r}' for key, value in arguments.items())})"


def extract_qwen_tool_calls(raw_response: str) -> list[dict[str, Any]]:
    """Exact extraction boundary from the public BFCL ``QwenFCHandler``.

    The official handler requires newlines immediately inside both XML tags and
    ignores malformed JSON blocks.  This intentionally does not canonicalize
    inline tags or repair JSON.
    """

    matches = re.findall(
        r"<tool_call>\n(.*?)\n</tool_call>", raw_response, flags=re.DOTALL
    )
    result: list[dict[str, Any]] = []
    for match in matches:
        try:
            decoded = json.loads(match)
        except Exception:
            continue
        result.append(decoded)
    return result


def parse_qwen_response(raw_response: str) -> QwenProtocolResponse:
    """Apply the public BFCL Qwen parser and execution-call conversion."""

    tool_calls = extract_qwen_tool_calls(raw_response)
    reasoning_content = ""
    cleaned_response = raw_response
    if "</think>" in raw_response:
        parts = raw_response.split("</think>")
        reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        cleaned_response = parts[-1].lstrip("\n")

    decode_error: str | None = None
    decoded_calls: list[str] = []
    try:
        if type(tool_calls) is not list or any(type(item) is not dict for item in tool_calls):
            raise ValueError(
                f"Model did not return a list of function calls: {cleaned_response}"
            )
        decoded_calls = [_json_call_to_execution(item) for item in tool_calls]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        decode_error = f"{type(exc).__name__}: {exc}"
        decoded_calls = []

    return QwenProtocolResponse(
        raw_response=raw_response,
        cleaned_response=cleaned_response,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        decoded_calls=decoded_calls,
        decode_error=decode_error,
    )


def format_qwen_prompt(
    messages: Sequence[Mapping[str, Any]], functions: Sequence[Mapping[str, Any]]
) -> str:
    """Narrow source-locked port of BFCL ``QwenFCHandler._format_prompt``."""

    if not messages:
        raise ValueError("official Qwen prompt requires at least one message")
    formatted_prompt = ""

    if functions:
        formatted_prompt += "<|im_start|>system\n"
        if messages[0]["role"] == "system":
            formatted_prompt += str(messages[0]["content"]) + "\n\n"
        formatted_prompt += (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>"
        )
        for tool in functions:
            formatted_prompt += f"\n{json.dumps(tool)}"
        formatted_prompt += (
            "\n</tools>\n\nFor each function call, return a json object with function name "
            "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call><|im_end|>\n"
        )
    elif messages[0]["role"] == "system":
        formatted_prompt += (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
        )

    last_query_index = len(messages) - 1
    for offset, message in enumerate(reversed(messages)):
        index = len(messages) - 1 - offset
        content = message.get("content")
        if (
            message.get("role") == "user"
            and type(content) is str
            and not (
                content.startswith("<tool_response>")
                and content.endswith("</tool_response>")
            )
        ):
            last_query_index = index
            break

    for index, raw_message in enumerate(messages):
        message = dict(raw_message)
        role = message["role"]
        content = message.get("content")
        content = content if isinstance(content, str) else ""

        if role == "user" or (role == "system" and index != 0):
            formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        elif role == "assistant":
            reasoning_content = ""
            if isinstance(message.get("reasoning_content"), str) and message.get(
                "reasoning_content"
            ):
                reasoning_content = str(message["reasoning_content"])
            elif "</think>" in content:
                parts = content.split("</think>")
                reasoning_content = (
                    parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                )
                content = parts[-1].lstrip("\n")

            if index > last_query_index:
                if index == len(messages) - 1 or reasoning_content:
                    formatted_prompt += (
                        f"<|im_start|>{role}\n<think>\n"
                        + reasoning_content.strip("\n")
                        + "\n</think>\n\n"
                        + content.lstrip("\n")
                    )
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"
            else:
                formatted_prompt += f"<|im_start|>{role}\n{content}"

            for call_index, raw_call in enumerate(message.get("tool_calls", [])):
                if (call_index == 0 and content) or call_index != 0:
                    formatted_prompt += "\n"
                tool_call = raw_call.get("function", raw_call)
                formatted_prompt += '<tool_call>\n{"name": "'
                formatted_prompt += str(tool_call["name"])
                formatted_prompt += '", "arguments": '
                arguments = tool_call["arguments"]
                formatted_prompt += (
                    arguments if isinstance(arguments, str) else json.dumps(arguments)
                )
                formatted_prompt += "}\n</tool_call>"
            formatted_prompt += "<|im_end|>\n"
        elif role == "tool":
            previous_role = messages[index - 1]["role"] if index > 0 else None
            next_role = messages[index + 1]["role"] if index + 1 < len(messages) else None
            if index == 0 or previous_role != "tool":
                formatted_prompt += "<|im_start|>user"
            formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"
            if index == len(messages) - 1 or next_role != "tool":
                formatted_prompt += "<|im_end|>\n"

    formatted_prompt += "<|im_start|>assistant\n"
    return formatted_prompt


def add_tool_results_to_history(
    messages: list[dict[str, Any]],
    execution_results: Sequence[str],
    decoded_calls: Sequence[str],
) -> None:
    """Port of BFCL OSS handler's tool-result history construction."""

    if len(execution_results) != len(decoded_calls):
        raise ValueError("execution results and decoded calls have different lengths")
    for execution_result, decoded_call in zip(execution_results, decoded_calls):
        messages.append(
            {
                "role": "tool",
                "name": decoded_call,
                "content": execution_result,
            }
        )


def parse_missed_function_update(processed_turn: str) -> list[dict[str, Any]]:
    """Recover the held-out BFCL function docs from EnvTuning's lossless row."""

    stripped = processed_turn.lstrip()
    decoded, end = json.JSONDecoder().raw_decode(stripped)
    remainder = stripped[end:].strip()
    if remainder != BFCL_ADDITIONAL_FUNCTION_MESSAGE:
        raise ValueError(
            "missing-function processed turn does not contain the official BFCL update message"
        )
    if not isinstance(decoded, list) or not all(isinstance(item, dict) for item in decoded):
        raise ValueError("held-out function payload is not a list of function objects")
    return decoded


def build_bfcl_test_entry(
    record: Mapping[str, Any],
    *,
    extract_available_functions: Any,
    to_builtin: Any,
) -> dict[str, Any]:
    """Reconstruct the official BFCL V3 entry from the lossless EnvTuning row."""

    normalized = to_builtin(record)
    kwargs = normalized["extra_info"]["interaction_kwargs"]
    questions = to_builtin(kwargs["question"])
    processed = to_builtin(kwargs["processed_question"])
    if len(processed) != max(0, len(questions) - 1):
        raise ValueError("processed_question is not aligned with BFCL user turns")

    missed_function: dict[str, list[dict[str, Any]]] = {}
    for turn_index, turn_messages in enumerate(questions):
        if turn_messages:
            continue
        if turn_index == 0:
            raise ValueError("first BFCL turn cannot be an empty missing-function update")
        missed_function[str(turn_index)] = parse_missed_function_update(
            processed[turn_index - 1]
        )

    initial_config = kwargs["initial_config"]
    if isinstance(initial_config, str):
        initial_config = json.loads(initial_config)
    return {
        "id": str(kwargs["id"]),
        "question": questions,
        "function": extract_available_functions(normalized["prompt"]),
        "initial_config": to_builtin(initial_config),
        "involved_classes": to_builtin(kwargs["involved_classes"]),
        "missed_function": missed_function,
        "ground_truth": to_builtin(kwargs["ground_truth"]),
    }
