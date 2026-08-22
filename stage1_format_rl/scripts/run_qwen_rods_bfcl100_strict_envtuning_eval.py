#!/usr/bin/env python3
"""Strict EnvTuning evaluation companion for Qwen3-4B-RODS BFCL100.

Unlike the source-locked BFCL Qwen-FC run, this companion uses the exact
training/validation interaction contract checked into AWorld-RL/EnvTuning:
``<think>`` plus exactly one ``<tool_call>`` or ``<answer>`` block, the public
``-3/-2/-1/0/1`` transition codes, and the real stateful BFCL environment.

The reported strict score is mean per-sample Progress Reward with the frozen
fixed expected-user-turn denominator.  Complete-episode accuracy and the
public observed-terminal calculation are retained as separate diagnostics.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import aiohttp
import pandas as pd
import yaml
from transformers import AutoConfig, AutoTokenizer


WORKSPACE = Path("/root/autodl-tmp/rods-workspace")
ENVTUNING = WORKSPACE / "code/AWorld-RL-stage1-worktree/EnvTuning"
SCRIPTS = WORKSPACE / "stage1_format_rl/scripts"
sys.path.insert(0, str(ENVTUNING))
sys.path.insert(0, str(SCRIPTS))

from env_tuning.bfcl_reward import compute_score as public_bfcl_reward  # noqa: E402
from env_tuning.interaction.new_multi_turn_fc import (  # noqa: E402
    MultiTurnFunctionCallInteraction,
)
from env_tuning.rods_matchtir_v1.provenance import (  # noqa: E402
    extract_available_functions,
)
from rods_official_bfcl_protocol import (  # noqa: E402
    EnvTuningDiagnosticCode,
    assert_source_contract,
    envtuning_diagnostic_source_hashes,
)
from envtuning_response_adapter import (  # noqa: E402
    api_reasoning_content,
    build_envtuning_parser_input,
    parse_envtuning_input,
)
from run_qwen_rods_bfcl100_official_eval import (  # noqa: E402
    CLASS_LABELS,
    REFERENCE,
    atomic_text,
    sample_id,
    sha256,
    to_builtin,
)


VALID_CODES = {int(item) for item in EnvTuningDiagnosticCode}


class OpenAIChatBackend:
    """OpenAI chat transport retaining prompt, token, API, and parser evidence."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        tokenizer: Any,
        max_context_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        max_output_tokens: int,
        chat_template: str,
        chat_template_source: str,
        enable_thinking: bool,
        parser_input_mode: str,
        reasoning_parser_name: str,
        tool_call_parser_name: str,
        timeout_seconds: float,
        transport_retries: int,
    ) -> None:
        self.endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.chat_template = chat_template
        self.chat_template_source = chat_template_source
        self.enable_thinking = enable_thinking
        self.parser_input_mode = parser_input_mode
        self.reasoning_parser_name = reasoning_parser_name
        self.tool_call_parser_name = tool_call_parser_name
        self.timeout_seconds = timeout_seconds
        self.transport_retries = transport_retries
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "OpenAIChatBackend":
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
            connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=300),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self.session is not None:
            await self.session.close()

    def render_prompt(self, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
        kwargs = {
            "chat_template": self.chat_template,
            "add_generation_prompt": True,
            "enable_thinking": self.enable_thinking,
        }
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            **kwargs,
        )
        token_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
        if not isinstance(text, str) or not isinstance(token_ids, list):
            raise TypeError("chat template did not return text and token IDs")
        return text, [int(item) for item in token_ids]

    async def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("backend session is not open")
        rendered_prompt_text, rendered_prompt_token_ids = self.render_prompt(messages)
        available_functions = extract_available_functions(messages)
        prompt_tokens_local = len(rendered_prompt_token_ids)
        room = self.max_context_length - prompt_tokens_local - 2
        if room <= 0:
            raise ValueError(
                f"strict EnvTuning prompt has {prompt_tokens_local} tokens and "
                f"exceeds native context {self.max_context_length}"
            )
        max_tokens = min(self.max_output_tokens, room)
        request_id = f"envtuning100-strict-{uuid.uuid4()}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template": self.chat_template,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "return_token_ids": True,
            "return_prompt_text": True,
        }
        last_error: Exception | None = None
        for attempt in range(1, self.transport_retries + 2):
            started = time.perf_counter()
            try:
                async with self.session.post(
                    self.endpoint,
                    json=payload,
                    headers={
                        "Authorization": "Bearer EMPTY",
                        "X-Request-ID": request_id,
                    },
                ) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {body[:4000]}")
                    decoded = json.loads(body)
                    choice = decoded["choices"][0]
                    message = choice["message"]
                    if not isinstance(message, Mapping):
                        raise RuntimeError("Chat endpoint returned no assistant message")
                    content = message.get("content")
                    if content is not None and not isinstance(content, str):
                        raise RuntimeError("Chat endpoint returned non-text content")
                    generated_token_ids = choice.get("token_ids")
                    if not isinstance(generated_token_ids, list) or not all(
                        isinstance(item, int) for item in generated_token_ids
                    ):
                        raise RuntimeError(
                            "vLLM did not return generated token IDs; TRUE RAW is unavailable"
                        )
                    true_raw = self.tokenizer.decode(
                        generated_token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    server_prompt_ids = decoded.get("prompt_token_ids")
                    server_prompt_text = decoded.get("prompt_text")
                    if server_prompt_ids != rendered_prompt_token_ids:
                        raise RuntimeError(
                            "server prompt token IDs differ from local rendered prompt"
                        )
                    if server_prompt_text != rendered_prompt_text:
                        raise RuntimeError(
                            "server prompt text differs from local rendered prompt"
                        )
                    reasoning = api_reasoning_content(message)
                    tool_calls = message.get("tool_calls") or []
                    if not isinstance(tool_calls, list):
                        raise RuntimeError("Chat endpoint returned malformed tool_calls")
                    compatibility = build_envtuning_parser_input(
                        true_raw_decoded_text=true_raw,
                        reasoning_content=reasoning,
                        content=content,
                        tool_calls=tool_calls,
                        reasoning_parser_name=self.reasoning_parser_name,
                    )
                    original_input = content or ""
                    compatible_input = compatibility.envtuning_parser_input
                    usage = decoded.get("usage") or {}
                    return {
                        "request_id": request_id,
                        "server_request_id": decoded.get("id"),
                        # Backward-compatible name: this remains API content,
                        # not TRUE decoder raw text.
                        "content": original_input,
                        "api_content": content,
                        "api_reasoning_content": reasoning,
                        "api_tool_calls": copy.deepcopy(tool_calls),
                        "finish_reason": choice.get("finish_reason"),
                        "prompt_tokens_local": prompt_tokens_local,
                        "prompt_tokens": int(usage.get("prompt_tokens", prompt_tokens_local)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "max_tokens": max_tokens,
                        "latency_seconds": time.perf_counter() - started,
                        "transport_attempt": attempt,
                        "input_messages": copy.deepcopy(messages),
                        "request_generation_parameters": {
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "top_k": self.top_k,
                            "seed": self.seed,
                            "max_tokens": max_tokens,
                            "stop": None,
                        },
                        "level_1_rendered_prompt": {
                            "messages": copy.deepcopy(messages),
                            "tools": available_functions,
                            "tools_transport_note": (
                                "EnvTuning row embeds the ordered function schemas in the "
                                "system message; no separate OpenAI tools field was sent"
                            ),
                            "rendered_prompt_text": rendered_prompt_text,
                            "rendered_prompt_token_ids": rendered_prompt_token_ids,
                            "server_prompt_text": server_prompt_text,
                            "server_prompt_token_ids": server_prompt_ids,
                            "prompt_text_identical": True,
                            "prompt_token_ids_identical": True,
                            "enable_thinking": self.enable_thinking,
                            "chat_template_kwargs": {
                                "enable_thinking": self.enable_thinking
                            },
                            "chat_template_source": self.chat_template_source,
                            "chat_template_sha256": hashlib.sha256(
                                self.chat_template.encode("utf-8")
                            ).hexdigest(),
                        },
                        "level_2_true_decoder_raw_output": {
                            "true_raw_generated_token_ids": generated_token_ids,
                            "true_raw_decoded_text": true_raw,
                            "raw_has_open_think": compatibility.raw_has_open_think,
                            "raw_has_close_think": compatibility.raw_has_close_think,
                            "raw_has_complete_think_block": (
                                compatibility.raw_has_complete_think_block
                            ),
                            "decode_skip_special_tokens": False,
                        },
                        "level_3_serving_api_response": {
                            "api_reasoning_content": reasoning,
                            "api_content": content,
                            "api_tool_calls": copy.deepcopy(tool_calls),
                            "full_api_response_json": copy.deepcopy(decoded),
                            "reasoning_parser_name": self.reasoning_parser_name or None,
                            "tool_call_parser_name": self.tool_call_parser_name or None,
                            "backend_name": "vLLM OpenAI-compatible chat completions",
                            "backend_version": decoded.get("system_fingerprint"),
                        },
                        "level_4_envtuning_parser_input": {
                            "original": {
                                "envtuning_parser_input": original_input,
                                **parse_envtuning_input(original_input),
                            },
                            "compatible": {
                                **compatibility.to_dict(),
                                **parse_envtuning_input(compatible_input),
                            },
                            "selected_mode": self.parser_input_mode,
                        },
                        "envtuning_compatibility": compatibility.to_dict(),
                        "raw_response": decoded,
                    }
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                json.JSONDecodeError,
                KeyError,
                RuntimeError,
            ) as exc:
                last_error = exc
                if attempt > self.transport_retries:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"strict inference request failed after retries: {last_error}")


async def evaluate_one(
    record: Mapping[str, Any],
    *,
    backend: OpenAIChatBackend,
    semaphore: asyncio.Semaphore,
    output_root: Path,
    max_assistant_actions: int,
) -> dict[str, Any]:
    sid = sample_id(record)
    started = time.perf_counter()
    data_type = str(record["data_source"])
    kwargs = copy.deepcopy(to_builtin(record["extra_info"]["interaction_kwargs"]))
    messages = copy.deepcopy(to_builtin(record["prompt"]))
    expected_user_turns = len(kwargs["ground_truth"])
    interaction = MultiTurnFunctionCallInteraction(
        {
            "name": "multi_turn_tool_call",
            "is_augmented": False,
            "environment_feedback_mode": "standard",
        }
    )
    instance_id = f"envtuning100-strict-{sid}-{uuid.uuid4()}"
    rewards: list[float] = []
    actions: list[dict[str, Any]] = []
    terminal_by_turn: dict[int, float] = {}
    raw_backend_responses: list[dict[str, Any]] = []
    runtime_error: str | None = None
    should_terminate = False
    await interaction.start_interaction(instance_id, **kwargs)
    try:
        for action_index in range(max_assistant_actions):
            try:
                async with semaphore:
                    generated = await backend.generate(messages)
            except Exception as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"
                break
            raw_backend_responses.append(generated)
            original_parser_input = generated["level_4_envtuning_parser_input"][
                "original"
            ]["envtuning_parser_input"]
            compatible_parser_input = generated["level_4_envtuning_parser_input"][
                "compatible"
            ]["envtuning_parser_input"]
            assistant_text = (
                compatible_parser_input
                if backend.parser_input_mode == "compatible"
                else original_parser_input
            )
            messages.append({"role": "assistant", "content": assistant_text})
            try:
                should_terminate, feedback, reward, metrics = await interaction.generate_response(
                    instance_id,
                    messages,
                    **kwargs,
                )
            except Exception as exc:
                runtime_error = f"interaction error: {type(exc).__name__}: {exc}"
                break

            reward = float(reward)
            if reward not in VALID_CODES:
                raise AssertionError(f"unexpected EnvTuning diagnostic code: {reward}")
            rewards.append(reward)
            provenance = to_builtin((metrics or {}).get("rods_matchtir_v1_step", {}))
            turn_index = int(provenance.get("user_turn_id", len(terminal_by_turn)))
            if reward >= 0:
                if turn_index in terminal_by_turn:
                    raise AssertionError(f"duplicate terminal code for user turn {turn_index}")
                terminal_by_turn[turn_index] = reward
            actions.append(
                {
                    "action_index": action_index,
                    "user_turn_id": turn_index,
                    "policy_step_id": provenance.get("policy_step_id"),
                    "raw_response": assistant_text,
                    "api_content_original_parser_input": original_parser_input,
                    "compatible_parser_input": compatible_parser_input,
                    "serialization_source": generated["envtuning_compatibility"][
                        "serialization_source"
                    ],
                    "reconstructed_think_from_reasoning_content": generated[
                        "envtuning_compatibility"
                    ]["reconstructed_think_from_reasoning_content"],
                    "reconstructed_action_from_tool_calls": generated[
                        "envtuning_compatibility"
                    ]["reconstructed_action_from_tool_calls"],
                    "original_parse": generated["level_4_envtuning_parser_input"][
                        "original"
                    ],
                    "compatible_parse": generated["level_4_envtuning_parser_input"][
                        "compatible"
                    ],
                    "response_type": provenance.get("response_type"),
                    "parsed_calls": provenance.get("calls", []),
                    "diagnostic_code": int(reward),
                    "diagnostic_name": EnvTuningDiagnosticCode(int(reward)).name,
                    "feedback": feedback,
                    "should_terminate_sequence": bool(should_terminate),
                    "prompt_tokens": generated.get("prompt_tokens"),
                    "completion_tokens": generated.get("completion_tokens"),
                    "latency_seconds": generated.get("latency_seconds"),
                }
            )
            if should_terminate:
                break
            messages.append({"role": "user", "content": feedback})
        else:
            runtime_error = f"max_assistant_actions={max_assistant_actions} exhausted"

        terminal_rewards = [
            terminal_by_turn[index]
            for index in range(expected_user_turns)
            if index in terminal_by_turn
        ]
        fixed_denominator_progress = (
            sum(terminal_rewards) / expected_user_turns if expected_user_turns else 0.0
        )
        public_metrics = public_bfcl_reward(
            reward_scores={"user_turn_rewards": rewards},
            ground_truth=kwargs["ground_truth"],
        )
        terminal_complete = len(terminal_by_turn) == expected_user_turns
        complete_episode_success = (
            terminal_complete
            and all(terminal_by_turn[index] == 1.0 for index in range(expected_user_turns))
        )
        result = to_builtin(
            {
                "sample_id": sid,
                "data_type": data_type,
                "model": backend.model,
                "parser_input_mode": backend.parser_input_mode,
                "question": kwargs["question"],
                "ground_truth": kwargs["ground_truth"],
                "model_trajectory": messages,
                "actions": actions,
                "diagnostic_codes": [int(value) for value in rewards],
                "terminal_by_turn": terminal_by_turn,
                "expected_user_turns": expected_user_turns,
                "terminal_complete": terminal_complete,
                "fixed_denominator_progress": fixed_denominator_progress,
                "strict_score_percent": 100.0 * fixed_denominator_progress,
                "complete_episode_success": complete_episode_success,
                "public_envtuning_bfcl_reward_metrics": public_metrics,
                "runtime_error": runtime_error,
                "response_lengths": {
                    "assistant_actions": len(actions),
                    "prompt_tokens_total": sum(
                        int(item.get("prompt_tokens", 0)) for item in raw_backend_responses
                    ),
                    "completion_tokens_total": sum(
                        int(item.get("completion_tokens", 0))
                        for item in raw_backend_responses
                    ),
                },
                "wall_time_seconds": time.perf_counter() - started,
                "raw_backend_responses": raw_backend_responses,
            }
        )
        atomic_text(
            output_root / "per_sample" / f"{sid}.json",
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
        )
        return result
    finally:
        await interaction.finalize_interaction(instance_id=instance_id)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_class[result["data_type"]].append(result)

    scores: dict[str, Any] = {}
    for source, label in CLASS_LABELS.items():
        rows = by_class[source]
        scores[label] = {
            "samples": len(rows),
            "progress_score_percent": (
                100.0 * sum(row["fixed_denominator_progress"] for row in rows) / len(rows)
                if rows
                else math.nan
            ),
            "complete_episode_correct": sum(
                bool(row["complete_episode_success"]) for row in rows
            ),
            "complete_episode_accuracy_percent": (
                100.0
                * sum(bool(row["complete_episode_success"]) for row in rows)
                / len(rows)
                if rows
                else math.nan
            ),
        }
    scores["Overall"] = {
        "samples": len(results),
        "progress_score_percent": (
            100.0
            * sum(row["fixed_denominator_progress"] for row in results)
            / len(results)
            if results
            else math.nan
        ),
        "complete_episode_correct": sum(
            bool(row["complete_episode_success"]) for row in results
        ),
        "complete_episode_accuracy_percent": (
            100.0
            * sum(bool(row["complete_episode_success"]) for row in results)
            / len(results)
            if results
            else math.nan
        ),
    }
    codes = Counter(
        str(code) for row in results for code in row["diagnostic_codes"]
    )
    all_actions = [action for row in results for action in row["actions"]]
    scores["Diagnostics"] = {
        "code_counts": dict(codes),
        "code_legend": {
            "-3": "FORMAT_OR_PARSE_ERROR",
            "-2": "TOOL_EXECUTION_ERROR",
            "-1": "TOOL_EXECUTION_SUCCESS_INTERMEDIATE",
            "0": "TERMINAL_TURN_FAILURE",
            "1": "TERMINAL_TURN_SUCCESS",
        },
        "runtime_failures": sum(row["runtime_error"] is not None for row in results),
        "incomplete_terminal_samples": sum(not row["terminal_complete"] for row in results),
        "format_reward_mean": (
            sum(
                row["public_envtuning_bfcl_reward_metrics"]["format_reward"]
                for row in results
            )
            / len(results)
            if results
            else math.nan
        ),
        "tool_call_reward_mean": (
            sum(
                row["public_envtuning_bfcl_reward_metrics"]["tool_call_reward"]
                for row in results
            )
            / len(results)
            if results
            else math.nan
        ),
        "assistant_actions": sum(
            row["response_lengths"]["assistant_actions"] for row in results
        ),
        "parser_input_mode": (
            results[0]["parser_input_mode"] if results else "unknown"
        ),
        "original_strict_parser_pass_actions": sum(
            bool(action["original_parse"]["parse_success"]) for action in all_actions
        ),
        "compatible_strict_parser_pass_actions": sum(
            bool(action["compatible_parse"]["parse_success"]) for action in all_actions
        ),
        "reconstructed_think_actions": sum(
            bool(action["reconstructed_think_from_reasoning_content"])
            for action in all_actions
        ),
        "reconstructed_tool_call_actions": sum(
            bool(action["reconstructed_action_from_tool_calls"])
            for action in all_actions
        ),
        "true_raw_complete_think_actions": sum(
            bool(action["compatible_parse"]["raw_has_complete_think_block"])
            for action in all_actions
        ),
        "prompt_tokens": sum(
            row["response_lengths"]["prompt_tokens_total"] for row in results
        ),
        "completion_tokens": sum(
            row["response_lengths"]["completion_tokens_total"] for row in results
        ),
    }
    return scores


async def async_main(args: argparse.Namespace) -> None:
    assert_source_contract()
    frame = pd.read_parquet(args.dataset)
    records = [to_builtin(row) for row in frame.to_dict(orient="records")]
    if args.sample_id:
        wanted = set(args.sample_id)
        records = [row for row in records if sample_id(row) in wanted]
        missing = wanted - {sample_id(row) for row in records}
        if missing:
            raise RuntimeError(f"requested sample IDs not found: {sorted(missing)}")
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("no strict evaluation records selected")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=False, local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=False, local_files_only=True
    )
    native_context = int(model_config.max_position_embeddings)
    if args.max_context_length > native_context:
        raise ValueError(
            f"configured context {args.max_context_length} exceeds native {native_context}"
        )
    template_tokenizer = AutoTokenizer.from_pretrained(
        args.chat_template_model_path or args.model_path,
        trust_remote_code=False,
        local_files_only=True,
    )
    chat_template = template_tokenizer.chat_template
    if not isinstance(chat_template, str) or not chat_template:
        raise RuntimeError("selected chat-template source has no chat template")
    chat_template_source = str((args.chat_template_model_path or args.model_path).resolve())
    config = {
        "mode": args.mode,
        "protocol": "AWORLD_ENVTUNING_STRICT",
        "model_path": str(args.model_path.resolve()),
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "endpoint": args.endpoint,
        "served_model_name": args.served_model_name,
        "num_samples": len(records),
        "client_concurrency": args.concurrency,
        "native_max_context_length": native_context,
        "configured_max_context_length": args.max_context_length,
        "max_output_tokens_per_action": args.max_output_tokens,
        "max_assistant_actions": args.max_assistant_actions,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "chat_template_source": chat_template_source,
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "parser_input_mode": args.parser_input_mode,
        "reasoning_parser_name": args.reasoning_parser_name or None,
        "tool_call_parser_name": args.tool_call_parser_name or None,
        "strict_score_definition": (
            "mean per-sample terminal 0/1 sum divided by fixed expected user turns, "
            "multiplied by 100"
        ),
        "diagnostic_source_hashes": envtuning_diagnostic_source_hashes(),
        "rods_reference_bfcl_score_percent_not_envtuning_threshold": REFERENCE,
    }
    atomic_text(
        args.output_root / "config" / "strict_envtuning_resolved_config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    async with OpenAIChatBackend(
        endpoint=args.endpoint,
        model=args.served_model_name,
        tokenizer=tokenizer,
        max_context_length=args.max_context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        chat_template=chat_template,
        chat_template_source=chat_template_source,
        enable_thinking=args.enable_thinking,
        parser_input_mode=args.parser_input_mode,
        reasoning_parser_name=args.reasoning_parser_name,
        tool_call_parser_name=args.tool_call_parser_name,
        timeout_seconds=args.timeout_seconds,
        transport_retries=args.transport_retries,
    ) as backend:
        tasks = [
            asyncio.create_task(
                evaluate_one(
                    record,
                    backend=backend,
                    semaphore=semaphore,
                    output_root=args.output_root,
                    max_assistant_actions=args.max_assistant_actions,
                )
            )
            for record in records
        ]
        results: list[dict[str, Any]] = []
        for completed in asyncio.as_completed(tasks):
            result = await completed
            results.append(result)
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "total": len(tasks),
                        "sample_id": result["sample_id"],
                        "strict_progress": result["fixed_denominator_progress"],
                        "complete_episode": result["complete_episode_success"],
                    }
                ),
                flush=True,
            )
    run_wall = time.perf_counter() - started

    results.sort(key=lambda row: row["sample_id"])
    public_results: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    for result in results:
        raw_results.append(
            {
                "sample_id": result["sample_id"],
                "responses": result.pop("raw_backend_responses"),
            }
        )
        public_results.append(result)
    atomic_text(
        args.output_root / "per_sample" / "per_sample_results.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in public_results
        ),
    )
    atomic_text(
        args.output_root / "raw_outputs" / "raw_model_responses.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in raw_results
        ),
    )
    metrics = aggregate(public_results)
    metrics["Diagnostics"]["run_wall_seconds"] = run_wall
    metrics["Diagnostics"]["completion_tokens_per_second"] = (
        metrics["Diagnostics"]["completion_tokens"] / run_wall if run_wall else 0.0
    )
    atomic_text(
        args.output_root / "metrics" / "strict_envtuning_metrics.json",
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
    )
    if args.mode == "full":
        if len(public_results) != 100:
            raise RuntimeError(f"full strict mode requires 100 samples, got {len(public_results)}")
        if metrics["Diagnostics"]["runtime_failures"]:
            raise RuntimeError(f"strict full eval has runtime failures: {metrics['Diagnostics']}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--max-context-length", type=int, default=262144)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--chat-template-model-path", type=Path)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--parser-input-mode", choices=("original", "compatible"), default="compatible"
    )
    parser.add_argument("--reasoning-parser-name", default="")
    parser.add_argument("--tool-call-parser-name", default="")
    parser.add_argument("--max-assistant-actions", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=72000.0)
    parser.add_argument("--transport-retries", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
