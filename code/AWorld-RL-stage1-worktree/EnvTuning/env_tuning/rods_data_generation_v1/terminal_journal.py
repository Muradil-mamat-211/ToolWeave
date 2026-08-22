"""Durable terminal-result journal for exactly-once logical finalization.

SOURCE_STATUS = RECONSTRUCTED_FROM_RODS_SPEC.  RODS Appendix B.4 specifies
immutable append-only generated-data logs and replay, but does not publish the
transaction implementation used between generation, queue append, and tracker
finalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .models import PipelineResult, stable_id, to_builtin, utc_now
from .queue import LockedJsonlQueue


TERMINAL_RESULT_SCHEMA_VERSION = "rods_generator_terminal_result.v1"
TERMINAL_STATUSES = frozenset({"SUCCEEDED", "DROPPED"})


def validate_terminal_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    record = to_builtin(raw)
    if record.get("schema_version") != TERMINAL_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported Generator terminal-result schema")
    terminal_id = record.get("terminal_record_id")
    seed_id = record.get("seed_id")
    status = record.get("status")
    if not isinstance(terminal_id, str) or not terminal_id:
        raise ValueError("terminal record requires terminal_record_id")
    if not isinstance(seed_id, str) or not seed_id:
        raise ValueError("terminal record requires seed_id")
    if status not in TERMINAL_STATUSES:
        raise ValueError("terminal record status must be SUCCEEDED or DROPPED")
    if not isinstance(record.get("attempts"), int) or record["attempts"] < 0:
        raise ValueError("terminal record requires a non-negative attempt count")
    checkpoint = record.get("checkpoint_metadata")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("terminal record requires checkpoint_metadata")

    candidate = record.get("candidate")
    candidate_id = record.get("candidate_id")
    if status == "SUCCEEDED":
        if not isinstance(candidate, Mapping):
            raise ValueError("SUCCEEDED terminal record requires complete candidate payload")
        if not isinstance(candidate_id, str) or candidate.get("candidate_id") != candidate_id:
            raise ValueError("SUCCEEDED terminal record candidate identity is inconsistent")
    elif candidate is not None or candidate_id is not None:
        raise ValueError("DROPPED terminal record cannot contain a candidate")
    return dict(record)


class TerminalResultJournal:
    """Append/fetch one immutable terminal record per seed identity."""

    def __init__(self, path: str | Path):
        self.queue = LockedJsonlQueue(path, key_field="terminal_record_id")

    @staticmethod
    def _record(result: PipelineResult) -> dict[str, Any]:
        if result.status not in TERMINAL_STATUSES:
            raise ValueError(f"pipeline result is not terminal: {result.status}")
        candidate_id = (
            result.candidate.get("candidate_id") if result.candidate is not None else None
        )
        record = {
            "schema_version": TERMINAL_RESULT_SCHEMA_VERSION,
            # One terminal outcome is permitted for each stable seed identity.
            "terminal_record_id": stable_id("terminal_result", {"seed_id": result.seed_id}),
            "timestamp": utc_now(),
            "seed_id": result.seed_id,
            "status": result.status,
            "attempts": int(result.attempts),
            "planner_calls": int(result.planner_calls),
            "candidate_id": candidate_id,
            "candidate": to_builtin(result.candidate),
            "drop_reason": result.reason if result.status == "DROPPED" else None,
            "errors": [error.to_dict() for error in result.errors],
            "checkpoint_metadata": to_builtin(result.checkpoint),
        }
        return validate_terminal_record(record)

    def commit(self, result: PipelineResult) -> dict[str, Any]:
        """Fsync a terminal result, then return the canonical durable payload."""

        proposed = self._record(result)
        self.queue.append([proposed])
        terminal_id = proposed["terminal_record_id"]
        matches = [
            validate_terminal_record(record)
            for record in self.queue.read()
            if record.get("terminal_record_id") == terminal_id
        ]
        if len(matches) != 1:
            raise RuntimeError("terminal journal did not durably resolve exactly one record")
        durable = matches[0]
        if durable["seed_id"] != result.seed_id or durable["status"] != result.status:
            raise RuntimeError("conflicting terminal result already exists for seed")
        return durable

    def read(self) -> list[dict[str, Any]]:
        return [validate_terminal_record(record) for record in self.queue.read()]
