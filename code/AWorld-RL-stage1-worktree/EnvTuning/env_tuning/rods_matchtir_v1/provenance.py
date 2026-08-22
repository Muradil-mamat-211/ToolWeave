"""Serializable BFCL rollout provenance helpers.

The training path uses structured parser output and token spans captured while the
conversation is assembled.  Nothing in this module attempts to reconstruct calls
or actor spans from a decoded full trajectory.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


FUNCTION_MARKER = "Here is a list of functions in JSON format that you can invoke.\n"
SCHEMA_VERSION = "rods_matchtir_rollout.v3"


def to_builtin(value: Any) -> Any:
    """Convert Arrow/numpy/Pydantic containers into JSON-stable Python values."""

    if hasattr(value, "model_dump"):
        return to_builtin(value.model_dump())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return to_builtin(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def extract_available_functions(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Decode the official BFCL function JSON from the system message.

    The function marker is part of the existing dataset contract.  Splitting on
    that exact marker is deterministic and does not inspect model-generated text.
    """

    system_content: str | None = None
    for raw_message in messages:
        message = to_builtin(raw_message)
        if isinstance(message, Mapping) and message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                system_content = content
                break
    if system_content is None:
        raise ValueError("BFCL prompt has no string-valued system message")
    if system_content.count(FUNCTION_MARKER) != 1:
        raise ValueError("BFCL function-list marker must occur exactly once")
    functions = json.loads(system_content.split(FUNCTION_MARKER, 1)[1])
    if not isinstance(functions, list) or not all(isinstance(item, dict) for item in functions):
        raise ValueError("BFCL available_functions must decode to a list of objects")
    return to_builtin(functions)


def build_rollout_context(
    *,
    messages: Sequence[Any],
    interaction_kwargs: Mapping[str, Any],
    data_source: str | None,
) -> dict[str, Any]:
    """Build prompt-level context copied into each rollout's provenance."""

    kwargs = to_builtin(interaction_kwargs)
    available_functions: list[dict[str, Any]] = []
    context_reliable = True
    context_error: str | None = None
    try:
        available_functions = extract_available_functions(messages)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        context_reliable = False
        context_error = str(exc)

    initial_config = kwargs.get("initial_config", {})
    if isinstance(initial_config, str):
        try:
            initial_config = json.loads(initial_config)
        except json.JSONDecodeError as exc:
            context_reliable = False
            context_error = f"invalid initial_config JSON: {exc}"

    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_id": str(kwargs.get("id", "")),
        "data_type": str(data_source or ""),
        "questions": to_builtin(kwargs.get("question", [])),
        "ground_truth": to_builtin(kwargs.get("ground_truth", [])),
        "available_functions": available_functions,
        "initial_config": to_builtin(initial_config),
        "context_reliable": context_reliable,
        "context_error": context_error,
    }


def response_relative_step(
    step: Mapping[str, Any],
    *,
    prompt_length: int,
    response_length: int,
) -> dict[str, Any]:
    """Convert one absolute actor span to the padded response coordinate system."""

    output = to_builtin(step)
    absolute_start = int(output.pop("actor_token_start_absolute", -1))
    absolute_end = int(output.pop("actor_token_end_absolute", -1))
    start = max(0, absolute_start - prompt_length)
    end = min(response_length, absolute_end - prompt_length)
    span_reliable = (
        absolute_start >= prompt_length
        and absolute_end > absolute_start
        and absolute_end <= prompt_length + response_length
        and end > start
    )
    output["actor_span"] = {"start": start, "end": end}
    # Actor-span reliability is independent of temporal ownership and call
    # parsing. Keep the legacy aggregate for old consumers; formal local credit
    # reads the split fields directly and uses runtime interaction depth.
    bound_span_reliable = bool(output.get("actor_span_reliable", True))
    output["actor_span_reliable"] = bool(bound_span_reliable and span_reliable)
    output["provenance_reliable"] = bool(output.get("provenance_reliable", False) and span_reliable)
    if not span_reliable:
        output["provenance_error"] = "actor span was absent or truncated"
    return output
