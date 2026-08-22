"""Persistent seed state, append-only events, and crash recovery."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .models import SeedStatus, to_builtin, utc_now
from .queue import LockedJsonlQueue, atomic_write_json


TRACKER_SCHEMA_VERSION = "rods_generator_tracker.v2"
LEGACY_TRACKER_SCHEMA_VERSION = "rods_generator_tracker.v1"


def _failed_attempt_count(checkpoint: Mapping[str, Any]) -> int:
    explicit = checkpoint.get("completed_failed_attempts")
    if explicit is not None:
        return int(explicit)
    failures = checkpoint.get("failures", [])
    failure_attempt_ids: list[int] = []
    if isinstance(failures, list):
        for failure in failures:
            if not isinstance(failure, Mapping):
                continue
            try:
                attempt_id = int(failure.get("attempt_id", 0))
            except (TypeError, ValueError):
                continue
            if attempt_id > 0:
                failure_attempt_ids.append(attempt_id)
    if failure_attempt_ids:
        return max(failure_attempt_ids)
    # Legacy tracker.v1 fallback.  New writers never emit this overloaded key.
    return int(checkpoint.get("attempts", 0))


def _process_start_token(pid: int) -> str | None:
    """Return Linux /proc start-time ticks, which disambiguate PID reuse."""

    try:
        # Field 22 is starttime. The executable name in field 2 can contain
        # spaces inside parentheses, so split only after its final ')'.
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        suffix = raw.rsplit(")", 1)[1].strip().split()
        return suffix[19]
    except (FileNotFoundError, IndexError, OSError):
        return None


def _worker_is_alive(item: Mapping[str, Any]) -> bool:
    try:
        pid = int(item["worker_pid"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_start = item.get("worker_process_start")
    actual_start = _process_start_token(pid)
    return actual_start is not None and (
        expected_start is None or str(expected_start) == actual_start
    )


class PromptTracker:
    def __init__(self, state_path: str | Path, event_log_path: str | Path):
        self.state_path = Path(state_path)
        self.lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        self.events = LockedJsonlQueue(event_log_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._recover_running_on_startup()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"schema_version": TRACKER_SCHEMA_VERSION, "seeds": {}}

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return self._empty()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        schema_version = state.get("schema_version")
        if schema_version == LEGACY_TRACKER_SCHEMA_VERSION:
            for item in state.get("seeds", {}).values():
                if isinstance(item, dict):
                    item["completed_failed_attempts"] = _failed_attempt_count(item)
                    item.pop("attempts", None)
            state["schema_version"] = TRACKER_SCHEMA_VERSION
        elif schema_version != TRACKER_SCHEMA_VERSION:
            raise ValueError("unsupported Generator tracker schema")
        if not isinstance(state.get("seeds"), dict):
            raise ValueError("Generator tracker seeds must be an object")
        return state

    def _mutate(self, callback) -> Any:
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                state = self._load_unlocked()
                result = callback(state)
                atomic_write_json(self.state_path, state)
                return result
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _event(self, seed_id: str, event: str, **payload: Any) -> None:
        self.events.append(
            [
                {
                    "timestamp": utc_now(),
                    "seed_id": seed_id,
                    "event": event,
                    "payload": to_builtin(payload),
                }
            ]
        )

    def _recover_running_on_startup(self) -> None:
        def recover(state: dict[str, Any]) -> list[str]:
            recovered: list[str] = []
            for seed_id, item in state["seeds"].items():
                if item.get("status") == SeedStatus.RUNNING.value:
                    # Do not steal work from another live daemon sharing the
                    # same filesystem tracker. A genuinely crashed worker has
                    # no matching /proc identity and is recoverable.
                    if _worker_is_alive(item):
                        continue
                    item["status"] = SeedStatus.PENDING.value
                    item["recovered_after_crash"] = True
                    item["updated_at"] = utc_now()
                    recovered.append(seed_id)
            return recovered

        recovered = self._mutate(recover)
        for seed_id in recovered:
            self._event(seed_id, "RECOVERED_RUNNING_TO_PENDING")

    def register(self, seed_id: str) -> SeedStatus:
        def mutate(state: dict[str, Any]) -> SeedStatus:
            item = state["seeds"].setdefault(
                seed_id,
                {
                    "status": SeedStatus.PENDING.value,
                    "completed_failed_attempts": 0,
                    "failures": [],
                    "patches": [],
                    "blocklist": [],
                    "candidate_id": None,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                },
            )
            return SeedStatus(item["status"])

        status = self._mutate(mutate)
        if status == SeedStatus.PENDING:
            self._event(seed_id, "REGISTERED")
        return status

    def try_claim(self, seed_id: str) -> bool:
        def mutate(state: dict[str, Any]) -> bool:
            item = state["seeds"].get(seed_id)
            if item is None or item.get("status") != SeedStatus.PENDING.value:
                return False
            item["status"] = SeedStatus.RUNNING.value
            item["worker_pid"] = os.getpid()
            item["worker_process_start"] = _process_start_token(os.getpid())
            item["updated_at"] = utc_now()
            return True

        claimed = self._mutate(mutate)
        if claimed:
            self._event(seed_id, "CLAIMED", worker_pid=os.getpid())
        return claimed

    def update_running(
        self,
        seed_id: str,
        *,
        completed_failed_attempts: int,
        planner_calls: int = 0,
        failures: list[dict[str, Any]],
        patches: list[dict[str, Any]],
        blocklist: list[str],
        blocklist_history: list[list[str]] | None = None,
        current_config: dict[str, Any] | None = None,
    ) -> None:
        def mutate(state: dict[str, Any]) -> None:
            item = state["seeds"][seed_id]
            if item.get("status") != SeedStatus.RUNNING.value:
                raise RuntimeError("cannot update a seed that is not RUNNING")
            item.update(
                {
                    "completed_failed_attempts": int(completed_failed_attempts),
                    "planner_calls": int(planner_calls),
                    "failures": to_builtin(failures),
                    "patches": to_builtin(patches),
                    "blocklist": sorted(blocklist),
                    "blocklist_history": to_builtin(blocklist_history or []),
                    "current_config": to_builtin(current_config or {}),
                    "updated_at": utc_now(),
                }
            )

        self._mutate(mutate)
        self._event(
            seed_id,
            "RUNNING_CHECKPOINT",
            completed_failed_attempts=completed_failed_attempts,
        )

    def update_from_checkpoint(self, seed_id: str, checkpoint: Mapping[str, Any]) -> None:
        self.update_running(
            seed_id,
            completed_failed_attempts=_failed_attempt_count(checkpoint),
            planner_calls=int(checkpoint.get("planner_calls", 0)),
            failures=[dict(value) for value in checkpoint.get("failures", [])],
            patches=[dict(value) for value in checkpoint.get("patches", [])],
            blocklist=[str(value) for value in checkpoint.get("blocklist", [])],
            blocklist_history=[
                [str(item) for item in values]
                for values in checkpoint.get("blocklist_history", [])
            ],
            current_config=(
                dict(checkpoint["current_config"])
                if isinstance(checkpoint.get("current_config"), Mapping)
                else {}
            ),
        )

    def resume_state(self, seed_id: str) -> dict[str, Any]:
        state = self.snapshot()
        item = state["seeds"].get(seed_id)
        if not isinstance(item, dict):
            return {}
        return {
            key: to_builtin(item.get(key, default))
            for key, default in {
                "completed_failed_attempts": 0,
                "planner_calls": 0,
                "failures": [],
                "patches": [],
                "blocklist": [],
                "blocklist_history": [],
                "current_config": {},
            }.items()
        }

    def mark_succeeded(self, seed_id: str, candidate_id: str) -> None:
        def mutate(state: dict[str, Any]) -> None:
            item = state["seeds"][seed_id]
            if item.get("status") != SeedStatus.RUNNING.value:
                raise RuntimeError("only a RUNNING seed can succeed")
            item["status"] = SeedStatus.SUCCEEDED.value
            item["candidate_id"] = candidate_id
            item["updated_at"] = utc_now()

        self._mutate(mutate)
        self._event(seed_id, "SUCCEEDED", candidate_id=candidate_id)

    def mark_dropped(self, seed_id: str, reason: str) -> None:
        def mutate(state: dict[str, Any]) -> None:
            item = state["seeds"][seed_id]
            if item.get("status") != SeedStatus.RUNNING.value:
                raise RuntimeError("only a RUNNING seed can be dropped")
            item["status"] = SeedStatus.DROPPED.value
            item["drop_reason"] = reason
            item["updated_at"] = utc_now()

        self._mutate(mutate)
        self._event(seed_id, "DROPPED", reason=reason)

    def reconcile_terminal_result(self, record: Mapping[str, Any]) -> None:
        """Atomically project one durable terminal journal record into tracker state."""

        seed_id = str(record["seed_id"])
        terminal_record_id = str(record["terminal_record_id"])
        terminal_status = str(record["status"])
        checkpoint = record.get("checkpoint_metadata", {})
        if not isinstance(checkpoint, Mapping):
            raise ValueError("terminal checkpoint_metadata must be an object")
        candidate_id = record.get("candidate_id")
        drop_reason = record.get("drop_reason")

        def mutate(state: dict[str, Any]) -> bool:
            item = state["seeds"].setdefault(
                seed_id,
                {
                    "completed_failed_attempts": 0,
                    "failures": [],
                    "patches": [],
                    "blocklist": [],
                    "created_at": utc_now(),
                },
            )
            prior_terminal_id = item.get("terminal_record_id")
            if prior_terminal_id not in (None, terminal_record_id):
                raise RuntimeError("tracker contains a conflicting terminal record")
            prior_status = item.get("status")
            prior_candidate_id = item.get("candidate_id")
            if prior_status == SeedStatus.SUCCEEDED.value and (
                terminal_status != SeedStatus.SUCCEEDED.value
                or prior_candidate_id != candidate_id
            ):
                raise RuntimeError("tracker terminal success conflicts with journal")
            if prior_status == SeedStatus.DROPPED.value and terminal_status != SeedStatus.DROPPED.value:
                raise RuntimeError("tracker terminal drop conflicts with journal")
            if (
                prior_terminal_id == terminal_record_id
                and prior_status == terminal_status
                and prior_candidate_id == candidate_id
            ):
                return False

            item.update(
                {
                    "completed_failed_attempts": _failed_attempt_count(checkpoint),
                    "planner_calls": int(checkpoint.get("planner_calls", 0)),
                    "failures": to_builtin(checkpoint.get("failures", [])),
                    "patches": to_builtin(checkpoint.get("patches", [])),
                    "blocklist": sorted(str(value) for value in checkpoint.get("blocklist", [])),
                    "blocklist_history": to_builtin(
                        checkpoint.get("blocklist_history", [])
                    ),
                    "current_config": to_builtin(checkpoint.get("current_config", {})),
                    "status": terminal_status,
                    "candidate_id": candidate_id,
                    "drop_reason": drop_reason,
                    "terminal_record_id": terminal_record_id,
                    "updated_at": utc_now(),
                }
            )
            item.pop("attempts", None)
            item.pop("worker_pid", None)
            item.pop("worker_process_start", None)
            return prior_terminal_id != terminal_record_id or prior_status != terminal_status

        changed = self._mutate(mutate)
        if changed:
            self._event(
                seed_id,
                "TERMINAL_RECONCILED",
                terminal_record_id=terminal_record_id,
                status=terminal_status,
                candidate_id=candidate_id,
            )

    def snapshot(self) -> dict[str, Any]:
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            try:
                return self._load_unlocked()
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def reconcile_succeeded_candidates(self, candidates: list[Mapping[str, Any]]) -> None:
        by_seed: dict[str, str] = {}
        for record in candidates:
            metadata = record.get("generation_metadata", {})
            seed_id = metadata.get("source_seed_id") if isinstance(metadata, Mapping) else None
            candidate_id = record.get("candidate_id")
            if isinstance(seed_id, str) and isinstance(candidate_id, str):
                by_seed[seed_id] = candidate_id

        def mutate(state: dict[str, Any]) -> None:
            for seed_id, candidate_id in by_seed.items():
                item = state["seeds"].setdefault(
                    seed_id,
                    {
                        "completed_failed_attempts": 0,
                        "failures": [],
                        "patches": [],
                        "blocklist": [],
                        "created_at": utc_now(),
                    },
                )
                # A durable terminal journal is authoritative.  This legacy
                # candidate-only reconciliation must never override it.
                if item.get("terminal_record_id") is not None:
                    continue
                item["status"] = SeedStatus.SUCCEEDED.value
                item["candidate_id"] = candidate_id
                item["updated_at"] = utc_now()

        self._mutate(mutate)
