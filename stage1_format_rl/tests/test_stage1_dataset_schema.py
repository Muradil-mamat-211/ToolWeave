from __future__ import annotations

import json
from collections import Counter


REQUIRED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
REQUIRED_INTERACTION = {
    "name",
    "id",
    "initial_config",
    "involved_classes",
    "ground_truth",
    "processed_question",
    "question",
}


def assert_schema(row):
    assert set(row) == REQUIRED_COLUMNS
    assert row["ability"] == "multi_turn_function_calling"
    assert row["reward_model"]["style"] == "interaction"
    assert len(row["prompt"]) == 2
    assert [message["role"] for message in row["prompt"]] == ["system", "user"]
    system = row["prompt"][0]["content"]
    assert "<think>" in system
    assert "<answer>" in system
    assert "<reason>" not in system
    assert "<thinking>" not in system
    kwargs = row["extra_info"]["interaction_kwargs"]
    assert REQUIRED_INTERACTION.issubset(kwargs)
    assert kwargs["name"] == "multi_turn_tool_call"
    assert isinstance(json.loads(kwargs["initial_config"]), dict)
    assert kwargs["ground_truth"] == row["reward_model"]["ground_truth"]
    assert json.dumps(kwargs["ground_truth"], ensure_ascii=False) not in json.dumps(
        row["prompt"], ensure_ascii=False
    )


def test_train_schema(train_rows):
    assert len(train_rows) == 100
    assert Counter(row["data_source"] for row in train_rows) == {
        "multi_turn_base": 100
    }
    for row in train_rows:
        assert_schema(row)


def test_validation_schema(validation_rows):
    assert len(validation_rows) == 400
    assert Counter(row["data_source"] for row in validation_rows) == {
        "multi_turn_base": 100,
        "multi_turn_miss_func": 100,
        "multi_turn_miss_param": 100,
        "multi_turn_long_context": 100,
    }
    for row in validation_rows:
        assert_schema(row)

