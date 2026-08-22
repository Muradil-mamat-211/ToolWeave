#!/usr/bin/env python3
"""Source-locked BFCL V3 multi-turn evaluation for Qwen3-4B-RODS.

Primary accuracy follows the public BFCL Qwen prompting handler and the
stateful BFCL multi-turn checker vendored by AWorld-RL/EnvTuning.  The public
EnvTuning ``-3/-2/-1/0/1`` interaction values are captured separately as
strict-format diagnostics; they never alter the BFCL sample-accuracy result.

This is a read-only model evaluation.  It does not import or mutate the
Training Branch optimizer, reward, advantage, or lifecycle code.
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
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from machine_paths import project_roots
from typing import Any, Mapping, Sequence

import aiohttp
import numpy as np
import pandas as pd
import yaml
from transformers import AutoConfig, AutoTokenizer


WORKSPACE = project_roots().source_root
ENVTUNING = WORKSPACE / "code/AWorld-RL-stage1-worktree/EnvTuning"
SCRIPTS = WORKSPACE / "stage1_format_rl/scripts"
sys.path.insert(0, str(ENVTUNING))
sys.path.insert(0, str(SCRIPTS))

from bfcl_env.multi_turn_checker import (  # noqa: E402
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from bfcl_env.multi_turn_utils import execute_multi_turn_func_call  # noqa: E402
from env_tuning.interaction.data_models import ResponseType  # noqa: E402
from env_tuning.interaction.execution_manager import ExecutionManager  # noqa: E402
from env_tuning.interaction.response_handler import ResponseHandler  # noqa: E402
from env_tuning.interaction.utils import has_execution_error  # noqa: E402
from env_tuning.rods_matchtir_v1.provenance import (  # noqa: E402
    extract_available_functions,
)
from rods_official_bfcl_protocol import (  # noqa: E402
    BFCL_ADDITIONAL_FUNCTION_MESSAGE,
    BFCL_MAXIMUM_STEP_LIMIT,
    EnvTuningDiagnosticCode,
    add_tool_results_to_history,
    assert_source_contract,
    build_bfcl_test_entry,
    envtuning_diagnostic_source_hashes,
    format_qwen_prompt,
    official_source_hashes,
    parse_qwen_response,
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


def to_builtin(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """Convert BFCL evidence to deterministic, cycle-safe JSON values."""

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
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist(), _seen=_seen, _depth=_depth + 1)
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None

    identity = id(value)
    tracked = isinstance(value, (dict, list, tuple, set, frozenset)) or hasattr(
        value, "__dict__"
    )
    if tracked:
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
        if hasattr(value, "model_dump"):
            return to_builtin(value.model_dump(), **child)
        if hasattr(value, "__dict__"):
            return {
                key: to_builtin(item, **child)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        return repr(value)
    finally:
        if tracked:
            _seen.remove(identity)


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


def sample_id(record: Mapping[str, Any]) -> str:
    extra = record["extra_info"]
    value = extra.get("original_id", extra.get("index"))
    if value is None:
        value = extra.get("interaction_kwargs", {}).get("id")
    if value is None:
        raise KeyError("evaluation row has no canonical sample ID")
    return str(value)


def snapshot_instances(instances: Mapping[str, Any]) -> dict[str, Any]:
    return {
        class_name: {
            key: to_builtin(value)
            for key, value in vars(instance).items()
            if not key.startswith("_")
        }
        for class_name, instance in instances.items()
    }


class OpenAICompletionsBackend:
    """Async transport matching BFCL OSSHandler's Completions API contract."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        tokenizer: Any,
        max_context_length: int,
        temperature: float,
        timeout_seconds: float,
        transport_retries: int,
    ) -> None:
        self.endpoint = endpoint.rstrip("/") + "/v1/completions"
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.transport_retries = transport_retries
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "OpenAICompletionsBackend":
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=300),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self.session is not None:
            await self.session.close()

    def prompt_tokens(self, prompt: str) -> int:
        # This is intentionally the same token-count operation used by BFCL's
        # public OSSHandler rather than a chat-template approximation.
        return len(self.tokenizer.tokenize(prompt))

    async def generate(self, formatted_prompt: str) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("backend session is not open")
        prompt_tokens_local = self.prompt_tokens(formatted_prompt)
        room = self.max_context_length - prompt_tokens_local - 2
        if room <= 0:
            raise ValueError(
                f"official BFCL prompt has {prompt_tokens_local} tokens and exceeds "
                f"native context {self.max_context_length}"
            )
        # Public BFCL OSSHandler caps every assistant step at 4096 tokens.
        max_tokens = min(4096, room)
        request_id = f"bfcl100-official-{uuid.uuid4()}"
        payload = {
            "model": self.model,
            "prompt": formatted_prompt,
            "temperature": self.temperature,
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
                        "Authorization": "Bearer EMPTY",
                        "X-Request-ID": request_id,
                    },
                ) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {body[:4000]}")
                    decoded = json.loads(body)
                    text = decoded["choices"][0].get("text")
                    if not isinstance(text, str):
                        raise RuntimeError("Completions endpoint returned no text")
                    usage = decoded.get("usage") or {}
                    return {
                        "request_id": request_id,
                        "server_request_id": decoded.get("id"),
                        "content": text,
                        "finish_reason": decoded["choices"][0].get("finish_reason"),
                        "prompt_tokens_local": prompt_tokens_local,
                        "prompt_tokens": int(usage.get("prompt_tokens", prompt_tokens_local)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "max_tokens": max_tokens,
                        "latency_seconds": time.perf_counter() - started,
                        "transport_attempt": attempt,
                        "formatted_prompt": formatted_prompt,
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
        raise RuntimeError(f"inference request failed after retries: {last_error}")


def score_primary(
    *,
    entry: Mapping[str, Any],
    data_type: str,
    decoded_by_turn: list[list[list[str]]],
    force_quit: bool,
    checker_model_name: str,
) -> dict[str, Any]:
    """Run the public AWorld/BFCL state, response, and irrelevance checks."""

    ground_truth = entry["ground_truth"]
    if len(decoded_by_turn) != len(ground_truth):
        return {
            "valid": False,
            "error_type": "multi_turn:turn_count_mismatch",
            "decoded_turns": len(decoded_by_turn),
            "expected_turns": len(ground_truth),
        }
    irrelevance = multi_turn_irrelevance_checker(decoded_by_turn, ground_truth)
    try:
        checker = multi_turn_checker(
            decoded_by_turn,
            ground_truth,
            dict(entry),
            data_type,
            checker_model_name,
            is_augmented=False,
        )
    except Exception as exc:
        checker = {
            "valid": False,
            "error_type": "multi_turn:checker_exception",
            "error_message": f"{type(exc).__name__}: {exc}",
        }
    valid = (
        not force_quit
        and bool(irrelevance.get("valid", False))
        and bool(checker.get("valid", False))
    )
    return {
        "valid": valid,
        "force_quit": force_quit,
        "irrelevance_checker": to_builtin(irrelevance),
        "multi_turn_checker": to_builtin(checker),
    }


def score_prefix_for_terminal_diagnostic(
    *,
    entry: Mapping[str, Any],
    data_type: str,
    decoded_by_turn: list[list[list[str]]],
    turn_index: int,
    checker_namespace: str,
) -> bool:
    """Recover the terminal 0/1 meaning for a strict-format terminal action.

    EnvTuning awards an empty-ground-truth turn directly from response type.
    For a normal turn its terminal score is state/response correctness up to
    that turn, so the same public checker is run on the trajectory prefix.
    """

    ground_truth = entry["ground_truth"]
    if not ground_truth[turn_index]:
        return not decoded_by_turn[turn_index]
    prefix_entry = dict(entry)
    prefix_entry["ground_truth"] = ground_truth[: turn_index + 1]
    prefix = score_primary(
        entry=prefix_entry,
        data_type=data_type,
        decoded_by_turn=decoded_by_turn[: turn_index + 1],
        force_quit=False,
        checker_model_name=f"{checker_namespace}-turn-{turn_index}-{uuid.uuid4()}",
    )
    return bool(prefix["valid"])


def strict_diagnostic_for_step(
    *,
    raw_response: str,
    official_decoded_calls: Sequence[str],
    execution_results: Sequence[str] | None,
    terminal_success: bool | None,
) -> dict[str, Any]:
    """Apply EnvTuning's strict parser and exact numeric code semantics.

    This is a diagnostic shadow over a BFCL-primary run.  A numeric execution
    code is emitted only when the strict and BFCL parsers agree on the calls
    that were actually executed.  Parser-protocol divergence is retained as
    evidence rather than guessed.
    """

    response = ResponseHandler().parse_and_validate(
        [{"role": "assistant", "content": raw_response}]
    )
    result: dict[str, Any] = {
        "protocol": "AWORLD_ENVTUNING_STRICT_SHADOW",
        "response_type": response.response_type.value,
        "is_valid": bool(response.is_valid),
        "error_message": response.error_message,
        "code": None,
        "code_name": None,
        "code_reliable": False,
    }
    if response.has_error or response.response_type == ResponseType.PARSE_ERROR:
        code = EnvTuningDiagnosticCode.FORMAT_OR_PARSE_ERROR
        result.update(code=int(code), code_name=code.name, code_reliable=True)
        return result

    if response.response_type == ResponseType.ANSWER:
        if terminal_success is None:
            result["unassigned_reason"] = "terminal correctness not available"
            return result
        code = (
            EnvTuningDiagnosticCode.TERMINAL_TURN_SUCCESS
            if terminal_success
            else EnvTuningDiagnosticCode.TERMINAL_TURN_FAILURE
        )
        result.update(code=int(code), code_name=code.name, code_reliable=True)
        return result

    strict_decoded = ExecutionManager(is_augmented=False).decode_tool_calls(
        response.content
    )
    result["strict_decoded_calls"] = strict_decoded
    if not strict_decoded:
        if terminal_success is None:
            result["unassigned_reason"] = "empty tool action terminal correctness unavailable"
            return result
        code = (
            EnvTuningDiagnosticCode.TERMINAL_TURN_SUCCESS
            if terminal_success
            else EnvTuningDiagnosticCode.TERMINAL_TURN_FAILURE
        )
        result.update(code=int(code), code_name=code.name, code_reliable=True)
        return result
    if list(strict_decoded) != list(official_decoded_calls) or execution_results is None:
        result["unassigned_reason"] = (
            "strict/BFCL parser divergence: no matching actual execution to classify"
        )
        return result
    code = (
        EnvTuningDiagnosticCode.TOOL_EXECUTION_ERROR
        if has_execution_error(list(execution_results))
        else EnvTuningDiagnosticCode.TOOL_EXECUTION_SUCCESS
    )
    result.update(code=int(code), code_name=code.name, code_reliable=True)
    return result


async def evaluate_one(
    record: Mapping[str, Any],
    *,
    backend: OpenAICompletionsBackend,
    semaphore: asyncio.Semaphore,
    output_root: Path,
) -> dict[str, Any]:
    sid = sample_id(record)
    started = time.perf_counter()
    entry = build_bfcl_test_entry(
        record,
        extract_available_functions=extract_available_functions,
        to_builtin=to_builtin,
    )
    data_type = str(record["data_source"])
    functions = copy.deepcopy(entry["function"])
    messages: list[dict[str, Any]] = []
    decoded_by_turn: list[list[list[str]]] = []
    model_responses_by_turn: list[list[str]] = []
    turn_traces: list[dict[str, Any]] = []
    raw_backend_responses: list[dict[str, Any]] = []
    runtime_error: str | None = None
    force_quit = False
    run_namespace = f"bfcl100_{sid}_{uuid.uuid4().hex}"

    initial_config = entry["initial_config"]
    involved_classes = entry["involved_classes"]
    _, instances = execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        run_namespace,
        entry["id"],
        long_context=("long_context" in entry["id"] or "composite" in entry["id"]),
        is_evaL_run=False,
        is_augmented=False,
    )
    initial_state = snapshot_instances(instances)

    try:
        for turn_index, original_turn_messages in enumerate(entry["question"]):
            turn_messages = copy.deepcopy(original_turn_messages)
            held_out = entry["missed_function"].get(str(turn_index))
            if held_out is not None:
                if turn_messages:
                    raise ValueError("BFCL held-out-function turn unexpectedly has a user query")
                functions.extend(copy.deepcopy(held_out))
                turn_messages = [
                    {"role": "user", "content": BFCL_ADDITIONAL_FUNCTION_MESSAGE}
                ]
            messages.extend(turn_messages)
            current_decoded: list[list[str]] = []
            current_responses: list[str] = []
            steps: list[dict[str, Any]] = []
            call_step_count = 0

            while True:
                formatted_prompt = format_qwen_prompt(messages, functions)
                try:
                    async with semaphore:
                        generated = await backend.generate(formatted_prompt)
                except Exception as exc:
                    runtime_error = f"{type(exc).__name__}: {exc}"
                    break

                raw_backend_responses.append(generated)
                parsed = parse_qwen_response(generated["content"])
                messages.append(parsed.assistant_history_message())
                current_responses.append(parsed.cleaned_response)
                pre_state = snapshot_instances(instances)
                execution_results: list[str] | None = None
                post_state = pre_state

                if parsed.decoded_calls:
                    execution_results, instances = execute_multi_turn_func_call(
                        parsed.decoded_calls,
                        initial_config,
                        involved_classes,
                        run_namespace,
                        entry["id"],
                        long_context=(
                            "long_context" in entry["id"]
                            or "composite" in entry["id"]
                        ),
                        is_evaL_run=False,
                        is_augmented=False,
                    )
                    current_decoded.append(list(parsed.decoded_calls))
                    add_tool_results_to_history(
                        messages, execution_results, parsed.decoded_calls
                    )
                    post_state = snapshot_instances(instances)

                steps.append(
                    {
                        "step_index": len(steps),
                        "raw_response": parsed.raw_response,
                        "cleaned_response": parsed.cleaned_response,
                        "reasoning_content": parsed.reasoning_content,
                        "official_tool_calls": parsed.tool_calls,
                        "official_decoded_calls": parsed.decoded_calls,
                        "official_decode_error": parsed.decode_error,
                        "execution_results": execution_results,
                        "execution_has_error": (
                            has_execution_error(execution_results)
                            if execution_results is not None
                            else None
                        ),
                        "pre_state": pre_state,
                        "post_state": post_state,
                        "finish_reason": generated.get("finish_reason"),
                        "prompt_tokens": generated.get("prompt_tokens"),
                        "completion_tokens": generated.get("completion_tokens"),
                        "latency_seconds": generated.get("latency_seconds"),
                        "official_turn_terminal": not bool(parsed.decoded_calls),
                    }
                )

                if not parsed.decoded_calls:
                    break
                call_step_count += 1
                if call_step_count > BFCL_MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    break

            decoded_by_turn.append(current_decoded)
            model_responses_by_turn.append(current_responses)
            turn_traces.append(
                {
                    "turn_index": turn_index,
                    "begin_of_turn_query": turn_messages,
                    "held_out_functions_added": held_out or [],
                    "steps": steps,
                }
            )
            if runtime_error is not None or force_quit:
                break

        # Preserve the expected BFCL turn shape after an infrastructure stop so
        # the checker and evidence remain deterministic and fail closed.
        while len(decoded_by_turn) < len(entry["ground_truth"]):
            decoded_by_turn.append([])
            model_responses_by_turn.append([])

        score = score_primary(
            entry=entry,
            data_type=data_type,
            decoded_by_turn=decoded_by_turn,
            force_quit=force_quit,
            checker_model_name=f"{run_namespace}_final_{uuid.uuid4().hex}",
        )
        if runtime_error is not None:
            score["valid"] = False
            score["runtime_error"] = runtime_error

        # Fill exact EnvTuning strict diagnostics only after terminal turn
        # correctness is available.  These codes never feed ``score``.
        for turn_trace in turn_traces:
            turn_index = int(turn_trace["turn_index"])
            terminal_success: bool | None = None
            if runtime_error is None and turn_trace["steps"]:
                terminal_success = score_prefix_for_terminal_diagnostic(
                    entry=entry,
                    data_type=data_type,
                    decoded_by_turn=decoded_by_turn,
                    turn_index=turn_index,
                    checker_namespace=run_namespace,
                )
            for step in turn_trace["steps"]:
                is_terminal = bool(step["official_turn_terminal"])
                step["envtuning_strict_shadow_diagnostic"] = strict_diagnostic_for_step(
                    raw_response=step["raw_response"],
                    official_decoded_calls=step["official_decoded_calls"],
                    execution_results=step["execution_results"],
                    terminal_success=terminal_success if is_terminal else None,
                )

        failure_category = None
        if not score["valid"]:
            failure_category = (
                runtime_error
                or (
                    "multi_turn:force_terminated"
                    if force_quit
                    else score.get("irrelevance_checker", {}).get("error_type")
                )
                or score.get("multi_turn_checker", {}).get("error_type")
                or "bfcl_checker_reject"
            )
        result = to_builtin(
            {
                "sample_id": sid,
                "data_type": data_type,
                "model": backend.model,
                "question": entry["question"],
                "ground_truth": entry["ground_truth"],
                "initial_functions": entry["function"],
                "missed_function": entry["missed_function"],
                "initial_config": entry["initial_config"],
                "involved_classes": entry["involved_classes"],
                "initial_environment_state": initial_state,
                "final_environment_state": snapshot_instances(instances),
                "model_trajectory": messages,
                "model_responses_by_turn": model_responses_by_turn,
                "tool_calls_by_turn": decoded_by_turn,
                "turn_traces": turn_traces,
                "final_correctness": bool(score["valid"]),
                "failure_category": failure_category,
                "primary_bfcl_checker": score,
                "force_quit": force_quit,
                "runtime_error": runtime_error,
                "response_lengths": {
                    "assistant_steps": sum(len(turn["steps"]) for turn in turn_traces),
                    "prompt_tokens_total": sum(
                        int(item.get("prompt_tokens", 0)) for item in raw_backend_responses
                    ),
                    "completion_tokens_total": sum(
                        int(item.get("completion_tokens", 0))
                        for item in raw_backend_responses
                    ),
                    "completion_tokens_per_step": [
                        int(item.get("completion_tokens", 0))
                        for item in raw_backend_responses
                    ],
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
    except Exception as exc:
        # Preserve a per-sample failure artifact before propagating.  Full mode
        # is clean-only and will not silently count infrastructure errors.
        failure = {
            "sample_id": sid,
            "data_type": data_type,
            "error": f"{type(exc).__name__}: {exc}",
            "turn_traces": to_builtin(turn_traces),
            "decoded_by_turn": to_builtin(decoded_by_turn),
        }
        atomic_text(
            output_root / "failures" / f"{sid}.json",
            json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n",
        )
        raise


def aggregate(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostic_codes: Counter[str] = Counter()
    diagnostic_unassigned = 0
    for result in results:
        by_class[result["data_type"]].append(result)
        for turn in result["turn_traces"]:
            for step in turn["steps"]:
                diagnostic = step["envtuning_strict_shadow_diagnostic"]
                if diagnostic["code"] is None:
                    diagnostic_unassigned += 1
                else:
                    diagnostic_codes[str(diagnostic["code"])] += 1

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
    if len(results) == 100 and all(
        primary[label]["total"] == 25 for label in CLASS_LABELS.values()
    ):
        macro = sum(primary[label]["score_percent"] for label in CLASS_LABELS.values()) / 4
        if not math.isclose(macro, primary["Overall"]["score_percent"], abs_tol=1e-12):
            raise AssertionError("balanced macro average differs from overall accuracy")

    diagnostics = {
        "runtime_failures": sum(row["runtime_error"] is not None for row in results),
        "force_quit_samples": sum(bool(row["force_quit"]) for row in results),
        "failure_categories": dict(
            Counter(row["failure_category"] for row in results if row["failure_category"])
        ),
        "assistant_steps": sum(row["response_lengths"]["assistant_steps"] for row in results),
        "prompt_tokens": sum(
            row["response_lengths"]["prompt_tokens_total"] for row in results
        ),
        "completion_tokens": sum(
            row["response_lengths"]["completion_tokens_total"] for row in results
        ),
        "envtuning_strict_shadow_diagnostic_counts": dict(diagnostic_codes),
        "envtuning_strict_shadow_unassigned_protocol_divergence": diagnostic_unassigned,
        "envtuning_code_legend": {
            "-3": "FORMAT_OR_PARSE_ERROR",
            "-2": "TOOL_EXECUTION_ERROR",
            "-1": "TOOL_EXECUTION_SUCCESS_INTERMEDIATE",
            "0": "TERMINAL_TURN_FAILURE",
            "1": "TERMINAL_TURN_SUCCESS",
        },
    }
    return primary, diagnostics


def write_report(
    output_root: Path,
    primary: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    def pct(label: str) -> str:
        return f"{primary[label]['score_percent']:.2f}"

    lines = [
        "# Qwen3-4B-RODS BFCL100 Evaluation Report",
        "",
        "## Primary RODS/BFCL Score",
        "",
        "| Model | Overall | Base | Miss Func | Miss Param | Long Context |",
        "|---|---:|---:|---:|---:|---:|",
        "| RODS paper/reference | 56.00 | 68.00 | 59.00 | 44.00 | 53.00 |",
        f"| Local Qwen3-4B-RODS / BFCL100 | {pct('Overall')} | {pct('Base')} | {pct('Miss Func')} | {pct('Miss Param')} | {pct('Long Context')} |",
        "",
        f"The single headline eval score is **{pct('Overall')}** "
        f"({primary['Overall']['correct']}/{primary['Overall']['total']}).",
        "",
        "## Integer Counts",
        "",
        f"- Overall: {primary['Overall']['correct']}/{primary['Overall']['total']}",
        f"- Base: {primary['Base']['correct']}/{primary['Base']['total']}",
        f"- Miss Func: {primary['Miss Func']['correct']}/{primary['Miss Func']['total']}",
        f"- Miss Param: {primary['Miss Param']['correct']}/{primary['Miss Param']['total']}",
        f"- Long Context: {primary['Long Context']['correct']}/{primary['Long Context']['total']}",
        "",
        "The local evaluation uses a deterministic balanced 100-sample subset "
        "(25 per class), whereas the RODS reference uses the full 400-sample "
        "held-in BFCL V3 multi-turn test set. Numerical equality is not expected.",
        "",
        "## Source-Locked Evaluation Contract",
        "",
        f"- Dataset: `{config['dataset_path']}`",
        f"- Model: `{config['model_path']}`",
        "- Prompt/parser: public BFCL `QwenFCHandler` prompting path, source-locked by hash.",
        "- Stateful execution and scoring: AWorld-RL/EnvTuning BFCL environment, "
        "state/response checker, plus its explicit missing-turn irrelevance checker.",
        "- Decoding: BFCL CLI default `temperature=0.001`; per-step output cap 4096; "
        "no project-invented `<answer>` requirement.",
        "- `<think>` is optional in the public Qwen handler and is separated from the "
        "tool-call payload when present.",
        "",
        "## EnvTuning Diagnostic Codes",
        "",
        "The exact EnvTuning values are retained as per-step shadow diagnostics: "
        "`-3` parse/format error, `-2` execution error, `-1` successful intermediate "
        "tool execution, `0` terminal turn failure, and `1` terminal turn success. "
        "They are not summed, averaged, or substituted for the BFCL primary score. "
        "If BFCL and strict EnvTuning parsers disagree, no guessed execution code is emitted.",
        "",
        "```json",
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        "```",
        "",
        "Full prompts, raw responses, stateful execution traces, parser evidence, "
        "diagnostic codes, and checker evidence are stored per sample.",
    ]
    atomic_text(
        output_root / "report" / "QWEN3_4B_RODS_BFCL100_EVAL_REPORT.md",
        "\n".join(lines) + "\n",
    )


def validate_full_run_integrity(
    sample_count: int, diagnostics: Mapping[str, Any]
) -> None:
    """Reject incomplete or infrastructure-failed full runs.

    BFCL's public handler explicitly marks a trajectory that exceeds
    ``MAXIMUM_STEP_LIMIT`` as a failed *entry*.  It is therefore valid scored
    model behavior, not an infrastructure failure that invalidates the other
    99 entries.  The force-quit count remains in diagnostics and the affected
    sample remains incorrect.
    """

    if sample_count != 100:
        raise RuntimeError(f"full mode requires 100 samples, got {sample_count}")
    if diagnostics["runtime_failures"]:
        raise RuntimeError(f"clean full eval violated: {diagnostics}")


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
        raise RuntimeError("no evaluation records selected")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=False, local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=False, local_files_only=True
    )
    native_context = int(model_config.max_position_embeddings)
    if args.max_context_length != native_context:
        raise ValueError(
            "source-locked BFCL eval must use the model's native context length: "
            f"configured={args.max_context_length}, native={native_context}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": args.mode,
        "model_path": str(args.model_path.resolve()),
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "endpoint": args.endpoint,
        "served_model_name": args.served_model_name,
        "num_samples": len(records),
        "client_concurrency": args.concurrency,
        "native_max_context_length": native_context,
        "bfcl_maximum_tool_steps_per_turn": BFCL_MAXIMUM_STEP_LIMIT,
        "max_output_tokens_per_step": 4096,
        "decoding": {
            "temperature": args.temperature,
            "top_p": "OMITTED_AS_IN_PUBLIC_BFCL_OSS_HANDLER",
            "source": "PUBLIC_BFCL_CLI_AND_OSS_HANDLER",
        },
        "primary_score_definition": (
            "complete-sample BFCL multi-turn state/response plus missing-turn "
            "irrelevance accuracy multiplied by 100"
        ),
        "rods_reference_score_percent": REFERENCE,
        "bfcl_source_hashes": official_source_hashes(),
        "envtuning_diagnostic_source_hashes": envtuning_diagnostic_source_hashes(),
        "diagnostic_code_policy": (
            "exact EnvTuning strict-format shadow diagnostics; excluded from primary score"
        ),
    }
    atomic_text(
        args.output_root / "config" / "eval_resolved_config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    async with OpenAICompletionsBackend(
        endpoint=args.endpoint,
        model=args.served_model_name,
        tokenizer=tokenizer,
        max_context_length=native_context,
        temperature=args.temperature,
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
    primary, diagnostics = aggregate(public_results)
    diagnostics["run_wall_seconds"] = run_wall
    diagnostics["completion_tokens_per_second"] = (
        diagnostics["completion_tokens"] / run_wall if run_wall else 0.0
    )
    atomic_text(
        args.output_root / "metrics" / "primary_metrics.json",
        json.dumps(primary, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_text(
        args.output_root / "metrics" / "diagnostics.json",
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
    )
    if args.mode == "full":
        validate_full_run_integrity(len(public_results), diagnostics)
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
    parser.add_argument("--max-context-length", type=int, default=262144)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--timeout-seconds", type=float, default=72000.0)
    parser.add_argument("--transport-retries", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
