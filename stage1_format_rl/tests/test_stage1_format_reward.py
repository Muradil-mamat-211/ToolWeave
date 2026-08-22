from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from stage1_contract import reference_stage1_reward


CASES = [
    ("empty", [], 0.0, 0.0, 0.0, 0.0, 0.0),
    ("answer_only_success", [1], 0.0, 1.0, 0.0, 0.0, 1.0),
    ("tool_success_then_success", [-1, 1], 2.0, 1.0, 1.0, 1.0, 1.0),
    ("tool_error_then_failure", [-2, 0], 1.0, 1.0, 0.0, 1.0, 0.0),
    ("parse_then_tool_success", [-3, -1, 1], 5 / 3, 2 / 3, 1.0, 1.0, 1.0),
    (
        "mixed",
        [-1, -2, -1, 1, -3, -1, 0],
        6 / 7 + 3 / 4,
        6 / 7,
        3 / 4,
        1.0,
        1 / 2,
    ),
    ("all_tool_success", [-1, -1, -1, 1], 2.0, 1.0, 1.0, 1.0, 1.0),
    ("all_tool_error", [-2, -2, -2, 0], 1.0, 1.0, 0.0, 1.0, 0.0),
    ("all_parse_error", [-3, -3, -3], 0.0, 0.0, 0.0, 0.0, 0.0),
]


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("official_stage1_format_reward", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.compute_score


@pytest.mark.parametrize(
    "name,codes,score,format_reward,tool_reward,gate,progress", CASES
)
def test_reference_and_official_are_exactly_equivalent(
    envtuning_root,
    name,
    codes,
    score,
    format_reward,
    tool_reward,
    gate,
    progress,
):
    official = load_official(envtuning_root / "env_tuning" / "format_reward.py")
    actual = official({"user_turn_rewards": codes}, ground_truth=[])
    reference = reference_stage1_reward(codes)

    assert math.isclose(actual["score"], score)
    assert math.isclose(actual["format_reward"], format_reward)
    assert math.isclose(actual["tool_call_reward"], tool_reward)
    assert math.isclose(actual["is_tool_call"], gate)
    assert math.isclose(actual["progress"], progress)
    assert math.isclose(reference.score, actual["score"])
    assert math.isclose(reference.format_reward, actual["format_reward"])
    assert math.isclose(reference.tool_call_reward, actual["tool_call_reward"])
    assert math.isclose(reference.is_tool_call, actual["is_tool_call"])
    assert math.isclose(reference.progress, actual["progress"])


def test_progress_is_diagnostic_not_stage1_score(envtuning_root):
    official = load_official(envtuning_root / "env_tuning" / "format_reward.py")
    result = official({"user_turn_rewards": [1]}, ground_truth=[])
    assert result["progress"] == 1.0
    assert result["score"] == 0.0

