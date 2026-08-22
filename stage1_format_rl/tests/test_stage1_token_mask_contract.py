from __future__ import annotations

from transformers import AutoTokenizer

from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    TokenizationSanityCheckModeEnum,
)


def make_request(tokenizer):
    return AsyncRolloutRequest(
        processing_class=tokenizer,
        request_id="mask-contract",
        state=AsyncRolloutRequestStateEnum.PENDING,
        messages=[
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ],
        input_ids=[],
        prompt_ids=[],
        response_ids=[],
        attention_mask=[],
        prompt_attention_mask=[],
        response_attention_mask=[],
        position_ids=[],
        prompt_position_ids=[],
        response_position_ids=[],
        loss_mask=[],
        prompt_loss_mask=[],
        response_loss_mask=[],
        reward_scores={},
        max_prompt_len=8192,
        max_response_len=1000,
        max_model_len=32768,
        use_inference_chat_template=False,
        tokenization_sanity_check_mode=TokenizationSanityCheckModeEnum.OFF,
        generation_prompt_ids=[],
        base_conv_wo_gen_prompt_end_pos=0,
        base_conv_with_gen_prompt_end_pos=0,
    )


def test_policy_and_environment_token_masks(workspace):
    tokenizer = AutoTokenizer.from_pretrained(
        workspace / "models" / "Qwen3-4B", local_files_only=True
    )
    request = make_request(tokenizer)
    assert set(request.prompt_loss_mask) == {0}

    before = len(request.loss_mask)
    request.add_assistant_message(
        tokenizer,
        '<think>reason</think><tool_call>{"name":"pwd","arguments":{}}</tool_call>',
    )
    assistant_mask = request.loss_mask[before:]
    assert assistant_mask and set(assistant_mask) == {1}

    before = len(request.loss_mask)
    # EnvTuning interaction observations and next questions are appended as user.
    request.add_user_message(tokenizer, "environment execution result")
    environment_mask = request.loss_mask[before:]
    assert environment_mask and set(environment_mask) == {0}


def test_padding_uses_zero_loss_mask(envtuning_root):
    source = (
        envtuning_root
        / "verl"
        / "verl"
        / "workers"
        / "rollout"
        / "sglang_rollout"
        / "sglang_rollout.py"
    ).read_text()
    assert "response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)" in source

