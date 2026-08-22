"""Appendix F dual feedback: config patches plus cumulative action pruning."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .config_patch import ConfigPatchAgent, UnsafePatchError, apply_patch_operations
from .error_taxonomy import ERROR_GUIDANCE, PATCHABLE_ERRORS
from .llm_backend import BackendError
from .models import ErrorRecord, to_builtin
from .parsing import StructuredParseError


@dataclass
class FeedbackState:
    current_config: dict[str, Any]
    failures: list[ErrorRecord] = field(default_factory=list)
    blocked_functions: set[str] = field(default_factory=set)
    patch_history: list[dict[str, Any]] = field(default_factory=list)
    blocklist_history: list[list[str]] = field(default_factory=list)

    @classmethod
    def from_initial_config(cls, initial_config: dict[str, Any]) -> "FeedbackState":
        return cls(current_config=copy.deepcopy(initial_config))

    @classmethod
    def from_resume(
        cls,
        initial_config: dict[str, Any],
        *,
        failures: list[ErrorRecord],
        blocked_functions: set[str],
        patch_history: list[dict[str, Any]],
        blocklist_history: list[list[str]],
        current_config: dict[str, Any] | None,
    ) -> "FeedbackState":
        """Restore only a checkpoint written by this Generator state machine."""

        return cls(
            current_config=copy.deepcopy(current_config or initial_config),
            failures=list(failures),
            blocked_functions=set(blocked_functions),
            patch_history=copy.deepcopy(patch_history),
            blocklist_history=copy.deepcopy(blocklist_history),
        )

    def checkpoint(
        self, *, completed_failed_attempts: int, planner_calls: int = 0
    ) -> dict[str, Any]:
        """Serialize retryable state after fully completed failed attempts.

        A successful attempt is deliberately not represented by this counter.
        Terminal success/drop durability is owned by the terminal-result journal.
        """

        return {
            "completed_failed_attempts": int(completed_failed_attempts),
            "planner_calls": int(planner_calls),
            "failures": [error.to_dict() for error in self.failures],
            "patches": to_builtin(self.patch_history),
            "blocklist": sorted(self.blocked_functions),
            "blocklist_history": to_builtin(self.blocklist_history),
            "current_config": to_builtin(self.current_config),
        }

    async def register_failure(
        self,
        error: ErrorRecord,
        *,
        patch_agent: ConfigPatchAgent,
    ) -> None:
        self.failures.append(error)
        self.blocked_functions.update(error.function_names)
        self.blocklist_history.append(sorted(self.blocked_functions))
        if error.error_type not in PATCHABLE_ERRORS:
            return
        try:
            reason, operations = await patch_agent.propose(
                error, current_config=self.current_config
            )
            merged, patch = apply_patch_operations(self.current_config, operations)
        except (StructuredParseError, UnsafePatchError, BackendError, ValueError) as exc:
            self.patch_history.append(
                {
                    "attempt_id": error.attempt_id,
                    "error_type": error.error_type.value,
                    "applied": False,
                    "failure": str(exc),
                }
            )
            return
        self.current_config = merged
        self.patch_history.append(
            {
                "attempt_id": error.attempt_id,
                "error_type": error.error_type.value,
                "reason": reason,
                "operations": [
                    {
                        "class_name": operation.class_name,
                        "field_path": operation.field_path,
                        "value": to_builtin(operation.value),
                    }
                    for operation in operations
                ],
                "patch": patch,
                "applied": True,
            }
        )
        patch_agent.metrics.increment("execution/patch_count")

    def guidance(self) -> list[str]:
        return [ERROR_GUIDANCE[error.error_type] for error in self.failures]
