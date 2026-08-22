"""Prompt loading helpers.

Prompt files carry an auditable metadata header followed by ``---PROMPT---``.
Only the body after that delimiter is sent to the backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


PROMPT_ROOT = Path(__file__).parent
PROMPT_DELIMITER = "---PROMPT---\n"


class _StrictFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        raise KeyError(f"prompt placeholder is missing: {key}")


def load_prompt(relative_path: str, values: Mapping[str, Any] | None = None) -> str:
    path = PROMPT_ROOT / relative_path
    raw = path.read_text(encoding="utf-8")
    if raw.count(PROMPT_DELIMITER) != 1:
        raise ValueError(f"prompt metadata delimiter is invalid: {path}")
    body = raw.split(PROMPT_DELIMITER, 1)[1].strip()
    if "<think>" in body or "</think>" in body:
        raise ValueError(f"Generator prompt must use <reason>, not <think>: {path}")
    return body.format_map(_StrictFormatDict(values or {}))
