from __future__ import annotations

from check_qwen3_4b import inspect_model


def test_qwen3_4b_is_post_trained_not_base(workspace):
    report = inspect_model(workspace / "models" / "Qwen3-4B")
    assert report["identity_check"] == "PASS"
    assert report["model_card_identity"] == "Qwen/Qwen3-4B"
    assert report["is_base_checkpoint"] is False
    assert report["model_type"] == "qwen3"
    assert report["architecture"] == ["Qwen3ForCausalLM"]
    assert 3.8e9 < report["parameter_estimate"] < 4.2e9
    assert report["chat_template_present"] is True
    assert report["default_thinking_mode"] is True
    assert report["missing_shards"] == []

