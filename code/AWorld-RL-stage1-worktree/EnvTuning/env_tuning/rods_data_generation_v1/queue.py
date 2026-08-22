"""Concurrency-safe, durable filesystem IPC queues."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .models import to_builtin


class ProductionQueueGuardError(RuntimeError):
    pass


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LockedJsonlQueue:
    """Append-only JSONL with a sidecar advisory lock and idempotent keys."""

    def __init__(
        self,
        path: str | Path,
        *,
        key_field: str | None = None,
        test_mode: bool = False,
        production_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.key_field = key_field
        if test_mode and production_path is not None:
            if self.path == Path(production_path).resolve():
                raise ProductionQueueGuardError(
                    "test/dry-run queue cannot target the production candidate path"
                )

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {self.path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"non-object record at {self.path}:{line_number}")
                records.append(record)
        return records

    def read(self) -> list[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_unlocked()
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def append(self, records: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
        values = [to_builtin(record) for record in records]
        if not values:
            return 0, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                existing_keys: set[str] = set()
                if self.key_field is not None:
                    for record in self._read_unlocked():
                        key = record.get(self.key_field)
                        if isinstance(key, str):
                            existing_keys.add(key)
                accepted: list[dict[str, Any]] = []
                duplicate_count = 0
                batch_keys: set[str] = set()
                for record in values:
                    if not isinstance(record, dict):
                        raise ValueError("queue append requires object records")
                    if self.key_field is not None:
                        key = record.get(self.key_field)
                        if not isinstance(key, str) or not key:
                            raise ValueError(f"queue record requires {self.key_field}")
                        if key in existing_keys or key in batch_keys:
                            duplicate_count += 1
                            continue
                        batch_keys.add(key)
                    accepted.append(record)
                if accepted:
                    with self.path.open("a", encoding="utf-8") as handle:
                        for record in accepted:
                            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    _fsync_parent(self.path)
                return len(accepted), duplicate_count
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(to_builtin(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    _fsync_parent(target)
