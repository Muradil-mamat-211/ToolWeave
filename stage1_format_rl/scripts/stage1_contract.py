from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from machine_paths import project_roots
from typing import Any, Iterable

import numpy as np


ROOTS = project_roots()
WORKSPACE = ROOTS.source_root
STAGE_ROOT = ROOTS.source_root / "stage1_format_rl"
AWORLD_WORKTREE = ROOTS.source_root / "code" / "AWorld-RL-stage1-worktree"
ENVTUNING_ROOT = AWORLD_WORKTREE / "EnvTuning"
ADAPTED_VERL_ROOT = ENVTUNING_ROOT / "verl"
MODEL_PATH = ROOTS.models_root / "Qwen3-4B"
RAW_BFCL_ROOT = ROOTS.shared_data_root / "Berkeley-Function-Calling-Leaderboard"

# The user's authoritative experiment setting. Several copied sections in the
# request still say K=8; those stale values are intentionally not used.
ROLLOUT_K = 16
GPU_COUNT_TARGET = 2


PROTOCOL_HEADER = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, make the function/tool calls needed to complete the user's request.

Every assistant action must contain exactly one reasoning block followed by exactly one action block. Do not emit any text outside these XML blocks.

Use exactly one of these two forms:

<think>your step-by-step reasoning</think><tool_call>{\"name\": \"function_name\", \"arguments\": {\"argument_name\": \"value\"}}</tool_call>

<think>your step-by-step reasoning</think><answer>a concise user-facing answer</answer>

The content of <tool_call> must be valid JSON. To invoke multiple APIs in one interaction stage, put a JSON array of call objects inside one and only one <tool_call> block. Never emit multiple <tool_call> blocks in the same assistant action. Use <answer> only when no further tool call is needed or possible.

At each turn, try to complete the current user request. After a tool call, use the environment result to decide whether another tool call is required or whether to finish with <answer>.

"""

FUNCTION_MARKER = "Here is a list of functions in JSON format that you can invoke.\n"


def normalize_nested(value: Any) -> Any:
    """Convert Arrow/numpy containers into JSON/Parquet-stable Python values."""
    if isinstance(value, np.ndarray):
        return [normalize_nested(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize_nested(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_nested(item) for item in value]
    return value


def protocol_aligned_system_prompt(legacy_prompt: str) -> str:
    """Keep official tool schemas while aligning instructions to the parser."""
    if FUNCTION_MARKER not in legacy_prompt:
        raise ValueError("Official function-list marker is missing from system prompt")
    function_section = legacy_prompt.split(FUNCTION_MARKER, 1)[1]
    return PROTOCOL_HEADER + FUNCTION_MARKER + function_section


@dataclass(frozen=True)
class RewardBreakdown:
    score: float
    progress: float
    format_reward: float
    tool_call_reward: float
    is_tool_call: float
    total_interaction_rounds: int
    n_parse_error: int
    n_tool_execution_error: int
    n_tool_execution_success: int


def reference_stage1_reward(codes: Iterable[float]) -> RewardBreakdown:
    """Pure reference implementation of the public format_reward.py formula."""
    values = list(codes)
    total = len(values)
    n_success = values.count(-1)
    n_execution_error = values.count(-2)
    n_parse_error = values.count(-3)
    tool_gate = 1.0 if n_success + n_execution_error > 0 else 0.0
    format_reward = (total - n_parse_error) / total if total else 0.0
    tool_reward = n_success / (n_success + n_execution_error) if tool_gate else 0.0
    relevant = [score for score in values if score in (0, 1)]
    progress = sum(relevant) / len(relevant) if relevant else 0.0
    return RewardBreakdown(
        score=tool_gate * (format_reward + tool_reward),
        progress=progress,
        format_reward=format_reward,
        tool_call_reward=tool_reward,
        is_tool_call=tool_gate,
        total_interaction_rounds=total,
        n_parse_error=n_parse_error,
        n_tool_execution_error=n_execution_error,
        n_tool_execution_success=n_success,
    )


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalize_nested(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
