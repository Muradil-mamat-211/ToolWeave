#!/usr/bin/env python3
"""Stateful BFCL-100 evaluation for the released Qwen3-4B-RODS model.

Inference is served by an OpenAI-compatible local endpoint, while environment
interaction, parsing, and final correctness use the existing EnvTuning/BFCL
implementation.  Independent samples run concurrently; every individual
sample remains strictly sequential across assistant/tool/user turns.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import datetime as dt
import decimal
import enum
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer


WORKSPACE = Path("/root/autodl-tmp/rods-workspace")
ENVTUNING = WORKSPACE / "code/AWorld-RL-stage1-worktree/EnvTuning"
sys.path.insert(0, str(ENVTUNING))

from bfcl_env.multi_turn_checker import (  # noqa: E402
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from env_tuning.interaction.data_models import ResponseData, ResponseType  # noqa: E402
from env_tuning.interaction.new_multi_turn_fc import (  # noqa: E402
    MultiTurnFunctionCallInteraction,
)


CLASS_LABELS = {
    "multi_turn_base": "Base",
    "multi_turn_miss_func": "Miss Func",
    "multi_turn_miss_param": "Miss Param",
    "multi_turn_long_context": "Long Context",
}
REFERENCE = {
    "Overall": 56.00,
    "Base": 68.00,
    "Miss Func": 59.00,
    "Miss Param": 44.00,
    "Long Context": 53.00,
}


class QwenFCEvalResponseHandler:
    """BFCL Qwen-FC response adapter used only by this read-only evaluation.

    This follows the public BFCL Qwen handler's semantic boundary: function
    calls are JSON objects inside ``<tool_call>`` tags, while an output with no
    tool-call tag is an assistant answer.  Unlike the Training Branch parser,
    BFCL scoring does not require a ``<think>`` wrapper.  JSON and tag errors
    remain fail-closed.  The Training parser is neither imported nor modified.
    """

    _TOOL_BLOCK = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")

    def parse_and_validate(self, messages: list[dict[str, Any]]) -> ResponseData:
        if not messages or messages[-1].get("role") != "assistant":
            return self._error("Invalid message format", "")
        raw = messages[-1].get("content")
        if not isinstance(raw, str):
            return self._error("Model raw response must be text", "")

        open_count = raw.count("<tool_call>")
        close_count = raw.count("</tool_call>")
        blocks = self._TOOL_BLOCK.findall(raw)
        if open_count != close_count or len(blocks) != open_count:
            return self._error("Malformed <tool_call> tag structure", raw)
        if not blocks:
            return ResponseData(
                content=raw,
                response_type=ResponseType.ANSWER,
                is_valid=True,
            )

        calls: list[dict[str, Any]] = []
        try:
            for block in blocks:
                decoded = json.loads(block.strip())
                candidates = decoded if isinstance(decoded, list) else [decoded]
                for call in candidates:
                    if not isinstance(call, dict):
                        raise ValueError("tool call must be a JSON object")
                    name = call.get("name")
                    arguments = call.get("arguments")
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("tool call name must be a non-empty string")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool call arguments must be a JSON object")
                    calls.append({"name": name.strip(), "arguments": arguments})
        except (json.JSONDecodeError, ValueError) as exc:
            return self._error(f"Invalid Qwen tool call: {exc}", raw)
        if not calls:
            return self._error("Tool-call tag contains no calls", raw)

        content = json.dumps(calls if len(calls) > 1 else calls[0], ensure_ascii=False)
        return ResponseData(
            content=content,
            response_type=ResponseType.TOOL_CALL,
            is_valid=True,
            tool_calls=[
                {
                    "call_idx": index,
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "valid": True,
                }
                for index, call in enumerate(calls)
            ],
        )

    @staticmethod
    def _error(message: str, raw: str) -> ResponseData:
        return ResponseData(
            content=raw,
            response_type=ResponseType.PARSE_ERROR,
            is_valid=False,
            error_message=message,
            has_error=True,
        )


def to_builtin(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """Convert evaluation evidence to deterministic, cycle-safe JSON values."""

    if _seen is None:
        _seen = set()
    if _depth > 64:
        return {"__truncated_type__": f"{type(value).__module__}.{type(value).__name__}"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, enum.Enum):
        return {
            "__enum__": f"{type(value).__module__}.{type(value).__name__}",
            "name": value.name,
            "value": to_builtin(value.value, _seen=_seen, _depth=_depth + 1),
        }
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)

    identity = id(value)
    track_identity = isinstance(value, (dict, list, tuple, set, frozenset)) or hasattr(
        value, "__dict__"
    )
    if track_identity:
        if identity in _seen:
            return {"__cycle_ref__": f"{type(value).__module__}.{type(value).__name__}"}
        _seen.add(identity)
    child = {"_seen": _seen, "_depth": _depth + 1}
    try:
        if isinstance(value, dict):
            return {str(key): to_builtin(item, **child) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_builtin(item, **child) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [to_builtin(item, **child) for item in value]
            return sorted(
                normalized,
                key=lambda item: json.dumps(
                    item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            )
        if isinstance(value, np.ndarray):
            return to_builtin(value.tolist(), **child)
        if isinstance(value, np.generic):
            return value.item()
        if value is pd.NA:
            return None
        if hasattr(value, "model_dump"):
            return to_builtin(value.model_dump(), **child)
        if hasattr(value, "__dict__"):
            return {
                "__class__": f"{type(value).__module__}.{type(value).__name__}",
                "attributes": to_builtin(vars(value), **child),
            }
        return {
            "__class__": f"{type(value).__module__}.{type(value).__name__}",
            "__repr__": repr(value),
        }
    finally:
        if track_identity:
            _seen.remove(identity)


def sample_id(record: dict[str, Any]) -> str:
    extra = record["extra_info"]
    value = extra.get("original_id", extra.get("index"))
    if value is None:
        value = extra.get("interaction_kwargs", {}).get("id")
    if value is None:
        raise KeyError("evaluation row has no canonical sample ID")
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class OpenAIChatBackend:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        tokenizer: Any,
        max_model_len: int,
        max_output_tokens: int,
        timeout_seconds: float,
        transport_retries: int,
    ) -> None:
        self.endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.transport_retries = transport_retries
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "OpenAIChatBackend":
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self.session is not None:
            await self.session.close()

    def prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(ids)

    async def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("backend session is not open")
        prompt_tokens_local = self.prompt_tokens(messages)
        room = self.max_model_len - prompt_tokens_local - 1
        if room <= 0:
            raise ValueError(
                f"prompt length {prompt_tokens_local} leaves no generation room "
                f"under max_model_len={self.max_model_len}"
            )
        max_tokens = min(self.max_output_tokens, room)
        request_id = f"bfcl100-{uuid.uuid4()}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(1, self.transport_retries + 2):
            started = time.perf_counter()
            try:
                async with self.session.post(
                    self.endpoint,
                    json=payload,
                    headers={
                        "X-Request-ID": request_id,
                        "Authorization": "Bearer EMPTY",
                    },
                ) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {body[:4000]}")
                    decoded = json.loads(body)
                    content = decoded["choices"][0]["message"].get("content")
                    if not isinstance(content, str):
                        raise RuntimeError("endpoint returned no textual assistant content")
                    usage = decoded.get("usage") or {}
                    return {
                        "request_id": request_id,
                        "server_request_id": decoded.get("id"),
                        "content": content,
                        "finish_reason": decoded["choices"][0].get("finish_reason"),
                        "prompt_tokens_local": prompt_tokens_local,
                        "prompt_tokens": int(usage.get("prompt_tokens", prompt_tokens_local)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "total_tokens": int(
                            usage.get(
                                "total_tokens",
                                int(usage.get("prompt_tokens", prompt_tokens_local))
                                + int(usage.get("completion_tokens", 0)),
                            )
                        ),
                        "max_tokens": max_tokens,
                        "latency_seconds": time.perf_counter() - started,
                        "transport_attempt": attempt,
                        "raw_response": decoded,
                    }
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt > self.transport_retries:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"inference request failed after retries: {last_error}")


def decoded_calls_from_step(interaction: MultiTurnFunctionCallInteraction, raw: str) -> list[str]:
    response = interaction.response_handler.parse_and_validate(
        [{"role": "assistant", "content": raw}]
    )
    if response.response_type != ResponseType.TOOL_CALL or response.has_error:
        return []
    return interaction.execution_manager.decode_tool_calls(response.content)


def score_official_local(
    *,
    record: dict[str, Any],
    decoded_by_turn: list[list[list[str]]],
    completed_turns: int,
    terminal_scores: list[float],
    checker_model_name: str,
) -> dict[str, Any]:
    kwargs = record["extra_info"]["interaction_kwargs"]
    ground_truth = to_builtin(kwargs["ground_truth"])
    expected_turns = len(ground_truth)
    terminal_complete = completed_turns == expected_turns and len(terminal_scores) == expected_turns
    if not terminal_complete:
        return {
            "valid": False,
            "error": {
                "error_type": "multi_turn:force_terminated",
                "error_message": (
                    f"completed terminal turns={completed_turns}, terminal scores={len(terminal_scores)}, "
                    f"expected={expected_turns}"
                ),
            },
            "terminal_complete": False,
            "all_terminal_scores_one": False,
        }

    irrelevance = multi_turn_irrelevance_checker(decoded_by_turn, ground_truth)
    if not irrelevance.get("valid", False):
        return {
            "valid": False,
            "error": irrelevance,
            "terminal_complete": True,
            "all_terminal_scores_one": all(score == 1.0 for score in terminal_scores),
        }

    test_entry = {
        "id": kwargs["id"],
        "initial_config": json.loads(kwargs["initial_config"]),
        "involved_classes": to_builtin(kwargs["involved_classes"]),
    }
    try:
        checker = multi_turn_checker(
            decoded_by_turn,
            ground_truth,
            test_entry,
            record["data_source"],
            checker_model_name,
            is_augmented=False,
        )
    except Exception as exc:  # Official checker exceptions are fail-closed evidence.
        checker = {
            "valid": False,
            "error_type": "multi_turn:checker_exception",
            "error_message": f"{type(exc).__name__}: {exc}",
        }
    all_one = all(score == 1.0 for score in terminal_scores)
    valid = bool(checker.get("valid", False)) and all_one
    return {
        "valid": valid,
        "checker": checker,
        "irrelevance_checker": irrelevance,
        "terminal_complete": True,
        "all_terminal_scores_one": all_one,
        "terminal_scores": terminal_scores,
    }


async def evaluate_one(
    record: dict[str, Any],
    *,
    backend: OpenAIChatBackend,
    semaphore: asyncio.Semaphore,
    output_root: Path,
    max_assistant_turns: int,
) -> dict[str, Any]:
    sid = sample_id(record)
    started = time.perf_counter()
    interaction = MultiTurnFunctionCallInteraction(
        {
            "name": "multi_turn_tool_call",
            "is_augmented": False,
            "environment_feedback_mode": "standard",
        }
    )
    # The released model is evaluated with the public BFCL Qwen-FC output
    # contract.  The strict RODS Training parser remains untouched.
    interaction.response_handler = QwenFCEvalResponseHandler()
    instance_id = f"bfcl100-live-{sid}-{uuid.uuid4()}"
    kwargs = copy.deepcopy(to_builtin(record["extra_info"]["interaction_kwargs"]))
    messages = copy.deepcopy(to_builtin(record["prompt"]))
    expected_turns = len(kwargs["ground_truth"])
    decoded_by_turn: list[list[list[str]]] = [[] for _ in range(expected_turns)]
    policy_steps: list[dict[str, Any]] = []
    terminal_scores: list[float] = []
    raw_calls: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    runtime_error: str | None = None
    truncated = False
    await interaction.start_interaction(instance_id, **kwargs)
    try:
        for assistant_step in range(max_assistant_turns):
            try:
                async with semaphore:
                    generated = await backend.generate(messages)
            except ValueError as exc:
                truncated = True
                runtime_error = str(exc)
                break
            raw_calls.append(generated)
            assistant_text = generated["content"]
            messages.append({"role": "assistant", "content": assistant_text})
            try:
                should_terminate, content, reward, metrics = await interaction.generate_response(
                    instance_id,
                    messages,
                    **kwargs,
                )
            except Exception as exc:
                runtime_error = f"interaction error: {type(exc).__name__}: {exc}"
                break

            provenance = to_builtin((metrics or {}).get("rods_matchtir_v1_step", {}))
            turn_id = int(provenance.get("user_turn_id", len(terminal_scores)))
            decoded = decoded_calls_from_step(interaction, assistant_text)
            if 0 <= turn_id < expected_turns and decoded:
                decoded_by_turn[turn_id].append(decoded)
            policy_steps.append(
                {
                    "assistant_step": assistant_step,
                    "user_turn_id": turn_id,
                    "policy_step_id": provenance.get("policy_step_id"),
                    "raw_response": assistant_text,
                    "response_type": provenance.get("response_type"),
                    "parser_reliable": provenance.get("provenance_reliable", False),
                    "parsed_calls": provenance.get("calls", []),
                    "decoded_calls": decoded,
                    "reward": float(reward),
                    "should_terminate_sequence": bool(should_terminate),
                    "finish_reason": generated.get("finish_reason"),
                    "prompt_tokens": generated.get("prompt_tokens"),
                    "completion_tokens": generated.get("completion_tokens"),
                    "latency_seconds": generated.get("latency_seconds"),
                }
            )
            if reward >= 0:
                terminal_scores.append(float(reward))
            if should_terminate:
                break
            observation = {
                "after_assistant_step": assistant_step,
                "user_turn_id": turn_id,
                "content": content,
                "kind": (
                    "tool_observation"
                    if content.startswith("Here are the function's execution results")
                    else "next_user_turn_or_system_feedback"
                ),
            }
            observations.append(observation)
            messages.append({"role": "user", "content": content})
        else:
            runtime_error = f"max_assistant_turns={max_assistant_turns} exhausted"

        score = score_official_local(
            record=record,
            decoded_by_turn=decoded_by_turn,
            completed_turns=len(terminal_scores),
            terminal_scores=terminal_scores,
            checker_model_name=f"bfcl100-official-{sid}-{uuid.uuid4()}",
        )
        if runtime_error is not None:
            score["valid"] = False
            score["runtime_error"] = runtime_error
        parser_failures = sum(step["response_type"] == "parse_error" for step in policy_steps)
        result = to_builtin({
            "sample_id": sid,
            "data_type": record["data_source"],
            "model": backend.model,
            "question": to_builtin(kwargs["question"]),
            "ground_truth": to_builtin(kwargs["ground_truth"]),
            "model_trajectory": messages,
            "policy_steps": policy_steps,
            "tool_calls_by_turn": decoded_by_turn,
            "environment_observations": observations,
            "parser_status": {
                "parse_failures": parser_failures,
                "all_steps_reliable": parser_failures == 0,
            },
            "final_correctness": bool(score["valid"]),
            "failure_category": (
                None
                if score["valid"]
                else (
                    score.get("runtime_error")
                    or score.get("error", {}).get("error_type")
                    or score.get("checker", {}).get("error_type")
                    or score.get("checker", {}).get("error", {}).get("error_type")
                    or "official_checker_reject"
                )
            ),
            "official_local_checker": score,
            "response_lengths": {
                "assistant_steps": len(policy_steps),
                "completion_tokens_total": sum(int(call.get("completion_tokens", 0)) for call in raw_calls),
                "prompt_tokens_total": sum(int(call.get("prompt_tokens", 0)) for call in raw_calls),
                "completion_tokens_per_step": [int(call.get("completion_tokens", 0)) for call in raw_calls],
            },
            "turn_count": {
                "expected": expected_turns,
                "completed": len(terminal_scores),
            },
            "truncated": truncated,
            "runtime_error": runtime_error,
            "wall_time_seconds": time.perf_counter() - started,
            "raw_backend_responses": raw_calls,
        })
        atomic_text(
            output_root / "per_sample" / f"{sid}.json",
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
        )
        return result
    finally:
        await interaction.finalize_interaction(instance_id=instance_id)


def aggregate(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_class[result["data_type"]].append(result)
    primary: dict[str, Any] = {}
    for source, label in CLASS_LABELS.items():
        rows = by_class[source]
        correct = sum(bool(row["final_correctness"]) for row in rows)
        primary[label] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows) if rows else math.nan,
            "score_percent": 100.0 * correct / len(rows) if rows else math.nan,
        }
    correct_total = sum(bool(row["final_correctness"]) for row in results)
    primary["Overall"] = {
        "correct": correct_total,
        "total": len(results),
        "accuracy": correct_total / len(results) if results else math.nan,
        "score_percent": 100.0 * correct_total / len(results) if results else math.nan,
    }
    if len(results) == 100 and all(primary[label]["total"] == 25 for label in CLASS_LABELS.values()):
        macro = sum(primary[label]["score_percent"] for label in CLASS_LABELS.values()) / 4
        if not math.isclose(macro, primary["Overall"]["score_percent"], abs_tol=1e-12):
            raise AssertionError(f"balanced macro={macro} differs from overall={primary['Overall']}")

    completion_tokens = sum(row["response_lengths"]["completion_tokens_total"] for row in results)
    prompt_tokens = sum(row["response_lengths"]["prompt_tokens_total"] for row in results)
    total_wall = sum(row["wall_time_seconds"] for row in results)
    diagnostics = {
        "parser_failures": sum(row["parser_status"]["parse_failures"] for row in results),
        "runtime_failures": sum(row["runtime_error"] is not None for row in results),
        "truncated_samples": sum(bool(row["truncated"]) for row in results),
        "failure_categories": dict(Counter(row["failure_category"] for row in results if row["failure_category"])),
        "assistant_steps": sum(row["response_lengths"]["assistant_steps"] for row in results),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "sum_sample_wall_seconds": total_wall,
    }
    return primary, diagnostics


def write_report(output_root: Path, primary: dict[str, Any], diagnostics: dict[str, Any], config: dict[str, Any]) -> None:
    def pct(label: str) -> str:
        return f"{primary[label]['score_percent']:.2f}"

    lines = [
        "# Qwen3-4B-RODS BFCL100 Evaluation Report",
        "",
        "## Primary RODS/BFCL Scores",
        "",
        "| Model | Overall | Base | Miss Func | Miss Param | Long Context |",
        "|---|---:|---:|---:|---:|---:|",
        "| RODS paper/reference | 56.00 | 68.00 | 59.00 | 44.00 | 53.00 |",
        f"| Local Qwen3-4B-RODS / BFCL100 | {pct('Overall')} | {pct('Base')} | {pct('Miss Func')} | {pct('Miss Param')} | {pct('Long Context')} |",
        "",
        "## Counts",
        "",
        f"- Overall: {primary['Overall']['correct']}/{primary['Overall']['total']}",
        f"- Base: {primary['Base']['correct']}/{primary['Base']['total']}",
        f"- Miss Func: {primary['Miss Func']['correct']}/{primary['Miss Func']['total']}",
        f"- Miss Param: {primary['Miss Param']['correct']}/{primary['Miss Param']['total']}",
        f"- Long Context: {primary['Long Context']['correct']}/{primary['Long Context']['total']}",
        "",
        "The local evaluation uses a deterministic 100-sample balanced subset (25/class), "
        "whereas the RODS reported reference corresponds to its full BFCL multi-turn "
        "evaluation. Therefore numerical equality is NOT expected.",
        "",
        "## Evaluation Contract",
        "",
        f"- Dataset: `{config['dataset_path']}`",
        f"- Model: `{config['model_path']}`",
        "- Stateful environment: EnvTuning `MultiTurnFunctionCallInteraction`.",
        "- Primary scorer: local copy of the official BFCL state/response and irrelevance checkers.",
        "- Accuracy unit: complete multi-turn sample; all expected turns and official checks must pass.",
        "- Reported score: exact-sample accuracy multiplied by 100 (percentage points), matching the RODS table convention.",
        "- Decoding: deterministic local canonical evaluation configuration (`temperature=0`, `top_p=1`).",
        "",
        "## Diagnostics",
        "",
        "```json",
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        "```",
        "",
        "Diagnostics are not mixed into the primary RODS accuracy table. Full per-sample "
        "trajectories, calls, observations, parser status, and checker evidence are stored "
        "under `per_sample/` and `per_sample_results.jsonl`.",
    ]
    atomic_text(output_root / "report" / "QWEN3_4B_RODS_BFCL100_EVAL_REPORT.md", "\n".join(lines) + "\n")


async def async_main(args: argparse.Namespace) -> None:
    dataset = pd.read_parquet(args.dataset)
    records = [to_builtin(row) for row in dataset.to_dict(orient="records")]
    if args.sample_id:
        wanted = set(args.sample_id)
        records = [row for row in records if sample_id(row) in wanted]
        missing = wanted - {sample_id(row) for row in records}
        if missing:
            raise RuntimeError(f"requested sample IDs not found: {sorted(missing)}")
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("no evaluation records selected")

    args.output_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    config = {
        "mode": args.mode,
        "model_path": str(args.model_path.resolve()),
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "endpoint": args.endpoint,
        "served_model_name": args.served_model_name,
        "num_samples": len(records),
        "client_concurrency": args.concurrency,
        "max_model_len": args.max_model_len,
        "max_output_tokens_per_assistant_step": args.max_output_tokens,
        "max_assistant_turns": args.max_assistant_turns,
        "decoding": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "source": "LOCAL_CANONICAL_EVAL_CONFIG; not claimed as a paper-published setting",
        },
        "rods_reference_score_percent": REFERENCE,
        "primary_score_definition": (
            "complete-sample BFCL multi-turn exact accuracy multiplied by 100; "
            "not Progress Reward"
        ),
        "bfcl_evaluator": str((ENVTUNING / "bfcl_env/multi_turn_checker.py").resolve()),
        "bfcl_evaluator_sha256": sha256(ENVTUNING / "bfcl_env/multi_turn_checker.py"),
        "parser": str((ENVTUNING / "env_tuning/interaction/response_handler.py").resolve()),
        "parser_sha256": sha256(ENVTUNING / "env_tuning/interaction/response_handler.py"),
        "environment": str((ENVTUNING / "env_tuning/interaction/new_multi_turn_fc.py").resolve()),
        "environment_sha256": sha256(ENVTUNING / "env_tuning/interaction/new_multi_turn_fc.py"),
    }
    atomic_text(args.output_root / "config" / "eval_resolved_config.yaml", yaml.safe_dump(config, sort_keys=False))

    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    async with OpenAIChatBackend(
        endpoint=args.endpoint,
        model=args.served_model_name,
        tokenizer=tokenizer,
        max_model_len=args.max_model_len,
        max_output_tokens=args.max_output_tokens,
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
                    max_assistant_turns=args.max_assistant_turns,
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
                        "correct": result["final_correctness"],
                        "wall_seconds": round(result["wall_time_seconds"], 2),
                    }
                ),
                flush=True,
            )
    run_wall = time.perf_counter() - started
    results.sort(key=lambda row: row["sample_id"])
    public_results = []
    raw_results = []
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
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in public_results),
    )
    atomic_text(
        args.output_root / "raw_outputs" / "raw_model_responses.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in raw_results),
    )
    primary, diagnostics = aggregate(public_results)
    diagnostics["run_wall_seconds"] = run_wall
    diagnostics["aggregate_completion_tokens_per_second"] = (
        diagnostics["completion_tokens"] / run_wall if run_wall else 0.0
    )
    atomic_text(args.output_root / "metrics" / "primary_metrics.json", json.dumps(primary, indent=2) + "\n")
    atomic_text(args.output_root / "metrics" / "diagnostics.json", json.dumps(diagnostics, indent=2) + "\n")
    if args.mode == "full":
        if len(public_results) != 100:
            raise RuntimeError(f"full mode requires 100 samples, got {len(public_results)}")
        if diagnostics["runtime_failures"] or diagnostics["truncated_samples"]:
            raise RuntimeError(f"clean full eval violated: {diagnostics}")
        write_report(args.output_root, primary, diagnostics, config)
    print(json.dumps({"primary": primary, "diagnostics": diagnostics}, indent=2))


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
    parser.add_argument("--max-model-len", type=int, default=32768)
    # These defaults reproduce the previously exercised local eval-400 path.
    # They are LOCAL_CANONICAL_EVAL_CONFIG values, not paper-published values.
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    parser.add_argument("--max-assistant-turns", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--transport-retries", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
