"""Isolated adapter over the real EnvTuning BFCL execution environment."""

from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import bfcl_env.multi_turn_utils as bfcl_multi_turn

from .models import FunctionCall, to_builtin
from .result_semantics import (
    ExecutionSemanticOutcome,
    classify_execution_result,
    find_explicit_vm_error,
    normalize_execution_result,
)


def _snapshot_value(value: Any, *, seen: set[int] | None = None) -> Any:
    """Best-effort deterministic snapshot of public environment state."""

    seen = seen or set()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    identity = id(value)
    if identity in seen:
        return "<cycle>"
    seen.add(identity)
    if isinstance(value, dict):
        return {
            str(key): _snapshot_value(item, seen=seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_snapshot_value(item, seen=seen) for item in value]
    if isinstance(value, set):
        return sorted((_snapshot_value(item, seen=seen) for item in value), key=repr)
    if hasattr(value, "__dict__"):
        return {
            str(key): _snapshot_value(item, seen=seen)
            for key, item in sorted(vars(value).items())
            if not key.startswith("_")
        }
    return repr(value)


def _decode_execution_result(raw: Any) -> Any:
    # ``normalize_execution_result`` supports JSON and bounded, safe
    # Python-literal-like containers.  The prior JSON-only transport left
    # stringified nested ``{"error": ...}`` payloads classified as SUCCESS.
    return to_builtin(normalize_execution_result(raw))


def _find_vm_error(value: Any) -> str | None:
    """Backward-compatible explicit VM-error helper.

    Semantic false flags require a function contract and are handled by
    :func:`classify_execution_result`; this helper deliberately does not treat
    every false ``*status`` value as an execution failure.
    """

    found = find_explicit_vm_error(value)
    return found[0] if found is not None else None


@dataclass(frozen=True)
class VMCallResult:
    result: Any
    success: bool
    error_detail: str | None
    pre_state: dict[str, Any]
    post_state: dict[str, Any]
    semantic_outcome: str = ExecutionSemanticOutcome.SUCCESS.value
    semantic_detail: str = ""


class EnvironmentSession(Protocol):
    environment_id: str

    def snapshot(self) -> dict[str, Any]: ...

    def execute(self, call: FunctionCall) -> VMCallResult: ...

    def close(self) -> None: ...


class EnvironmentFactory(Protocol):
    created_environment_ids: list[str]

    def create(
        self,
        *,
        initial_config: dict[str, Any],
        involved_classes: list[str],
        seed_id: str,
        long_context: bool,
        purpose: str,
    ) -> EnvironmentSession: ...


class BFCLSession:
    """One stateful, isolated namespace in EnvTuning's real CPU VM."""

    def __init__(
        self,
        *,
        initial_config: dict[str, Any],
        involved_classes: list[str],
        long_context: bool,
        is_augmented: bool,
        purpose: str,
    ) -> None:
        token = uuid.uuid4().hex
        self.environment_id = f"bfcl-{purpose}-{token}"
        self._model_name = f"rods_generator_{purpose}_{token}"
        self._test_entry_id = f"s_{token}"
        self._initial_config = copy.deepcopy(initial_config)
        self._classes = list(involved_classes)
        self._long_context = bool(long_context)
        self._is_augmented = bool(is_augmented)
        self._closed = False
        _, self._instances = bfcl_multi_turn.execute_multi_turn_func_call(
            [],
            self._initial_config,
            self._classes,
            self._model_name,
            self._test_entry_id,
            long_context=self._long_context,
            is_evaL_run=False,
            is_augmented=self._is_augmented,
        )

    def snapshot(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("cannot snapshot a closed BFCL session")
        return {
            class_name: _snapshot_value(instance)
            for class_name, instance in sorted(self._instances.items())
        }

    def execute(self, call: FunctionCall) -> VMCallResult:
        if self._closed:
            raise RuntimeError("cannot execute in a closed BFCL session")
        if call.class_name not in self._classes:
            raise ValueError(f"call class is absent from this VM: {call.class_name}")
        pre_state = self.snapshot()
        raw_results, instances = bfcl_multi_turn.execute_multi_turn_func_call(
            [call.canonical()],
            self._initial_config,
            self._classes,
            self._model_name,
            self._test_entry_id,
            long_context=self._long_context,
            is_evaL_run=False,
            is_augmented=self._is_augmented,
        )
        self._instances = instances
        if len(raw_results) != 1:
            raise RuntimeError("BFCL VM returned a non-unit result for one call")
        result = _decode_execution_result(raw_results[0])
        semantic = classify_execution_result(call.name, result)
        error = (
            semantic.detail
            if semantic.outcome == ExecutionSemanticOutcome.HARD_ERROR
            else None
        )
        return VMCallResult(
            result=result,
            success=error is None,
            error_detail=error,
            pre_state=pre_state,
            post_state=self.snapshot(),
            semantic_outcome=semantic.outcome.value,
            semantic_detail=semantic.detail,
        )

    def _global_instance_name(self, class_name: str) -> str:
        safe_model_name = "uuid" + self._model_name.replace("-", "_").replace(".", "_").replace("/", "_")
        return f"_{safe_model_name}_{self._test_entry_id}_{class_name.lower()}_instance"

    def close(self) -> None:
        if self._closed:
            return
        for class_name in self._classes:
            bfcl_multi_turn.__dict__.pop(self._global_instance_name(class_name), None)
        self._instances.clear()
        self._closed = True

    def __enter__(self) -> "BFCLSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class SynthesisEnvironmentAdapter:
    """Factory recording instance identity so fresh-VM use is auditable."""

    def __init__(self, *, is_augmented: bool = False):
        self.is_augmented = bool(is_augmented)
        self.created_environment_ids: list[str] = []

    def create(
        self,
        *,
        initial_config: dict[str, Any],
        involved_classes: list[str],
        seed_id: str,
        long_context: bool,
        purpose: str,
    ) -> BFCLSession:
        del seed_id  # Identity is represented by an isolated random namespace.
        session = BFCLSession(
            initial_config=initial_config,
            involved_classes=involved_classes,
            long_context=long_context,
            is_augmented=self.is_augmented,
            purpose=re.sub(r"[^a-zA-Z0-9_]", "_", purpose),
        )
        self.created_environment_ids.append(session.environment_id)
        return session
