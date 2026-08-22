"""Thread-safe diagnostic counters and latency summaries."""

from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from .error_taxonomy import ErrorType


class GeneratorMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += float(value)

    def record_error(self, error_type: ErrorType) -> None:
        self.increment(f"errors/{error_type.value}")

    @contextmanager
    def latency(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            self.increment(f"latency/{name}_seconds_sum", elapsed)
            self.increment(f"latency/{name}_count")

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(sorted(self._counters.items()))

    def ensure_error_keys(self) -> None:
        for error_type in ErrorType:
            with self._lock:
                self._counters.setdefault(f"errors/{error_type.value}", 0.0)
