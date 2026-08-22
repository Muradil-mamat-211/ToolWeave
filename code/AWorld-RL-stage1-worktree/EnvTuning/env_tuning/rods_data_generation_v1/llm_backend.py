"""One shared synthesis backend with production, fake, and replay adapters."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx

from .config import LLMConfig
from .models import BackendResponse, to_builtin, utc_now


class BackendError(RuntimeError):
    pass


_REQUEST_METADATA: ContextVar[dict[str, Any]] = ContextVar(
    "rods_generator_request_metadata", default={}
)


def push_request_metadata(**values: Any) -> Token[dict[str, Any]]:
    """Attach seed/attempt provenance to all agent calls in this async task."""

    merged = dict(_REQUEST_METADATA.get())
    merged.update({key: value for key, value in values.items() if value is not None})
    return _REQUEST_METADATA.set(merged)


def pop_request_metadata(token: Token[dict[str, Any]]) -> None:
    _REQUEST_METADATA.reset(token)


def _merged_request_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(_REQUEST_METADATA.get())
    merged.update(dict(metadata or {}))
    return merged


class LLMBackend(ABC):
    """Logical agents share one backend and differ only by prompts/role."""

    @abstractmethod
    async def complete(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> BackendResponse:
        raise NotImplementedError


class VLLMOpenAIBackend(LLMBackend):
    """OpenAI-compatible adapter for a separately launched vLLM service.

    Importing this class does not import vLLM, inspect model tensors, or start a
    server.  Transport retries are independent from RODS pipeline attempts.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds)

    @property
    def chat_url(self) -> str:
        endpoint = self.config.endpoint.rstrip("/")
        return endpoint + "/chat/completions"

    async def aclose(self) -> None:
        await self._client.aclose()

    def _append_raw_log(self, record: Mapping[str, Any]) -> None:
        if not self.config.raw_response_log_path:
            return
        path = Path(self.config.raw_response_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(to_builtin(record), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    async def complete(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> BackendResponse:
        request_id = f"rods-{role}-{uuid.uuid4()}"
        request_metadata = _merged_request_metadata(metadata)
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": to_builtin(messages),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.disable_native_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        }
        last_error: Exception | None = None
        started = time.perf_counter()
        async with self._semaphore:
            for transport_attempt in range(self.config.transport_retries + 1):
                try:
                    response = await self._client.post(self.chat_url, headers=headers, json=body)
                    response.raise_for_status()
                    raw = response.json()
                    choices = raw.get("choices")
                    if not isinstance(choices, list) or not choices:
                        raise BackendError("vLLM response has no choices")
                    message = choices[0].get("message", {})
                    text = message.get("content")
                    if not isinstance(text, str) or not text.strip():
                        raise BackendError("vLLM response content is empty")
                    latency = time.perf_counter() - started
                    result = BackendResponse(
                        role=role,
                        text=text,
                        request_id=str(raw.get("id") or request_id),
                        raw_response=to_builtin(raw),
                        latency_seconds=latency,
                    )
                    self._append_raw_log(
                        {
                            "timestamp": utc_now(),
                            "request_id": request_id,
                            "role": role,
                            "metadata": to_builtin(request_metadata),
                            "request": body,
                            "response": raw,
                            "latency_seconds": latency,
                        }
                    )
                    return result
                except (httpx.HTTPError, ValueError, BackendError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError):
                        response_body = exc.response.text
                        last_error = BackendError(
                            f"HTTP {exc.response.status_code}: {response_body}"
                        )
                    else:
                        last_error = exc
                    if transport_attempt >= self.config.transport_retries:
                        break
                    await asyncio.sleep(min(0.25 * (2**transport_attempt), 2.0))
        self._append_raw_log(
            {
                "timestamp": utc_now(),
                "request_id": request_id,
                "role": role,
                "metadata": to_builtin(request_metadata),
                "request": body,
                "status": "ERROR",
                "error": str(last_error),
                "latency_seconds": time.perf_counter() - started,
            }
        )
        raise BackendError(f"vLLM transport exhausted retries: {last_error}")


class FakeLLMBackend(LLMBackend):
    """Deterministic structured fixture backend.

    ``script`` may be a global sequence or a mapping from logical agent role to
    a role-local sequence. Values may be strings or exceptions.
    """

    def __init__(
        self,
        script: Sequence[str | Exception] | Mapping[str, Sequence[str | Exception]],
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._global: deque[str | Exception] | None = None
        self._by_role: dict[str, deque[str | Exception]] = defaultdict(deque)
        if isinstance(script, Mapping):
            for role, values in script.items():
                self._by_role[str(role)].extend(values)
        else:
            self._global = deque(script)

    async def complete(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> BackendResponse:
        request_id = f"fake-{role}-{len(self.calls)}"
        request_metadata = _merged_request_metadata(metadata)
        self.calls.append(
            {
                "role": role,
                "messages": to_builtin(messages),
                "metadata": to_builtin(request_metadata),
                "request_id": request_id,
            }
        )
        queue = self._global if self._global is not None else self._by_role[role]
        if not queue:
            raise BackendError(f"FakeLLMBackend has no response for role {role!r}")
        value = queue.popleft()
        if isinstance(value, Exception):
            raise value
        return BackendResponse(
            role=role,
            text=value,
            request_id=request_id,
            raw_response={"fixture": value},
            latency_seconds=0.0,
        )

    def remaining(self, role: str | None = None) -> int:
        if self._global is not None:
            return len(self._global)
        if role is not None:
            return len(self._by_role[role])
        return sum(len(queue) for queue in self._by_role.values())


class ReplayLLMBackend(FakeLLMBackend):
    """Replay responses captured as JSONL without contacting a model."""

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ReplayLLMBackend":
        by_role: dict[str, list[str]] = defaultdict(list)
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                role = record.get("role")
                text = record.get("text", record.get("response"))
                if not isinstance(role, str) or not isinstance(text, str):
                    raise ValueError(f"invalid replay record at {path}:{line_number}")
                by_role[role].append(text)
        return cls(by_role)


def build_backend(config: LLMConfig, *, replay_path: str | Path | None = None) -> LLMBackend:
    if config.backend == "vllm_openai":
        return VLLMOpenAIBackend(config)
    if config.backend == "replay":
        if replay_path is None:
            raise ValueError("Replay backend requires replay_path")
        return ReplayLLMBackend.from_jsonl(replay_path)
    raise ValueError(
        "FakeLLMBackend must be constructed with an explicit fixture; "
        f"unsupported configured backend: {config.backend!r}"
    )
