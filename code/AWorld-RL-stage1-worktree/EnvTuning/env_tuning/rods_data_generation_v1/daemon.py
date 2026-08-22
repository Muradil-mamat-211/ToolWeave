"""Concurrency-safe queue consumer and crash-recoverable Generator daemon."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from jsonschema import ValidationError

from .config import GeneratorConfig
from .contracts import validate_seed_record
from .llm_backend import (
    LLMBackend,
    build_backend,
    pop_request_metadata,
    push_request_metadata,
)
from .metrics import GeneratorMetrics
from .models import SeedRecord, SeedStatus, stable_id, utc_now
from .pipeline import RODSDataGenerationPipeline
from .queue import LockedJsonlQueue
from .terminal_journal import TerminalResultJournal
from .tracker import PromptTracker


GENERATION_GUARD_ENV = "RODS_ALLOW_DATA_GENERATION"


class GenerationGuardError(RuntimeError):
    pass


class GeneratorDaemon:
    def __init__(
        self,
        *,
        config: GeneratorConfig,
        backend: LLMBackend,
        pipeline: RODSDataGenerationPipeline | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.metrics = pipeline.metrics if pipeline is not None else GeneratorMetrics()
        self.pipeline = pipeline or RODSDataGenerationPipeline(
            config=config, backend=backend, metrics=self.metrics
        )
        self._fault_injector = fault_injector
        self.seed_queue = LockedJsonlQueue(config.queues.seed_path)
        self.candidate_queue = LockedJsonlQueue(
            config.queues.candidate_path,
            key_field="candidate_id",
            test_mode=config.test_mode or config.dry_run,
            production_path=config.queues.production_candidate_path,
        )
        self.tracker = PromptTracker(
            config.queues.tracker_path, config.queues.event_log_path
        )
        expanded_path = Path(config.queues.expanded_log_dir) / "pipeline_results.jsonl"
        self.expanded_results = LockedJsonlQueue(expanded_path, key_field="result_id")
        terminal_path = Path(config.queues.expanded_log_dir) / "terminal_results.jsonl"
        self.terminal_results = TerminalResultJournal(terminal_path)

    def _inject_fault(self, point: str) -> None:
        """Invoke a deterministic test hook at a durable commit boundary."""

        if self._fault_injector is not None:
            self._fault_injector(point)

    def _append_candidate_exact(self, candidate: Mapping[str, Any]) -> None:
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("terminal candidate requires candidate_id")
        accepted, duplicates = self.candidate_queue.append([candidate])
        if accepted == 1:
            self.metrics.increment("lifecycle/candidates_written")
            return
        if duplicates != 1:
            raise RuntimeError("candidate queue did not durably accept the result")
        existing = [
            value
            for value in self.candidate_queue.read()
            if value.get("candidate_id") == candidate_id
        ]
        if len(existing) != 1:
            raise RuntimeError("candidate identity did not resolve exactly once")
        if existing[0] != dict(candidate):
            prior_metadata = existing[0].get("generation_metadata", {})
            new_metadata = candidate.get("generation_metadata", {})
            prior_fingerprint = (
                prior_metadata.get("content_fingerprint")
                if isinstance(prior_metadata, Mapping)
                else None
            )
            new_fingerprint = (
                new_metadata.get("content_fingerprint")
                if isinstance(new_metadata, Mapping)
                else None
            )
            if (
                not isinstance(prior_fingerprint, str)
                or not prior_fingerprint
                or prior_fingerprint != new_fingerprint
            ):
                raise RuntimeError("candidate identity collision has different Training content")
            # The terminal journal retains the second source seed's complete
            # payload.  Only the exact duplicate Training row is suppressed.
            self.metrics.increment("novelty/exact_content_duplicate")
        self.metrics.increment("queue/duplicate_skipped")

    def _finalize_terminal(
        self, record: Mapping[str, Any], *, inject_faults: bool
    ) -> None:
        status = record.get("status")
        if status == "SUCCEEDED":
            candidate = record.get("candidate")
            if not isinstance(candidate, Mapping):
                raise RuntimeError("successful terminal record has no candidate payload")
            if inject_faults:
                self._inject_fault("after_terminal_journal_before_candidate_append")
            self._append_candidate_exact(candidate)
            if inject_faults:
                self._inject_fault("after_candidate_append_before_tracker_succeeded")
        elif status == "DROPPED":
            if inject_faults:
                self._inject_fault("after_dropped_journal_before_tracker_dropped")
        else:
            raise RuntimeError(f"unsupported terminal status: {status}")
        self.tracker.reconcile_terminal_result(record)

    def _reconcile_terminal_results(self) -> None:
        """Replay journal payloads without invoking Planner or the LLM backend."""

        for record in self.terminal_results.read():
            self._finalize_terminal(record, inject_faults=False)

    def _allow_generation(self, explicit: bool) -> None:
        if self.config.dry_run:
            return
        if self.config.test_mode and explicit:
            return
        if not explicit or os.environ.get(GENERATION_GUARD_ENV) != "1":
            raise GenerationGuardError(
                f"non-dry Generator requires explicit permission and {GENERATION_GUARD_ENV}=1"
            )

    async def _process_seed(self, seed: SeedRecord) -> None:
        status = self.tracker.register(seed.sample_id)
        if status in {SeedStatus.SUCCEEDED, SeedStatus.DROPPED}:
            self.metrics.increment("queue/duplicate_skipped")
            return
        if not self.tracker.try_claim(seed.sample_id):
            self.metrics.increment("queue/duplicate_skipped")
            return
        self.metrics.increment("queue/seeds_claimed")
        resume = self.tracker.resume_state(seed.sample_id)

        def checkpoint(value: dict[str, Any]) -> None:
            self.tracker.update_from_checkpoint(seed.sample_id, value)

        try:
            request_context = push_request_metadata(seed_id=seed.sample_id)
            try:
                result = await self.pipeline.generate(
                    seed,
                    resume_state=resume,
                    checkpoint_callback=checkpoint,
                )
            finally:
                pop_request_metadata(request_context)
            # A successful attempt is not a failed-attempt checkpoint.  The
            # terminal journal is the first durable commit of its full result.
            self._inject_fault("after_pipeline_before_terminal_journal")
            terminal_record = self.terminal_results.commit(result)
            self._finalize_terminal(terminal_record, inject_faults=True)
        except Exception as exc:
            # RUNNING remains durable. On restart PromptTracker returns it to
            # PENDING and resumes from the last attempt checkpoint.
            self.expanded_results.append(
                [
                    {
                        "result_id": stable_id(
                            "result_exception",
                            {"seed_id": seed.sample_id, "detail": repr(exc)},
                        ),
                        "timestamp": utc_now(),
                        "seed_id": seed.sample_id,
                        "status": "INTERRUPTED",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                ]
            )
            raise

    async def run_once(self, *, allow_generation: bool = False) -> dict[str, float]:
        self._allow_generation(allow_generation)
        raw_seeds = self.seed_queue.read()
        self.metrics.increment("queue/seeds_seen", len(raw_seeds))
        seeds: list[SeedRecord] = []
        for raw in raw_seeds:
            try:
                seeds.append(validate_seed_record(raw))
            except (TypeError, ValueError, ValidationError):
                self.metrics.increment("queue/invalid_seed_records")
                self.metrics.increment("queue/seeds_dropped")

        if self.config.dry_run:
            self.metrics.ensure_error_keys()
            return self.metrics.snapshot()

        # Terminal journal replay is authoritative and occurs before any seed
        # can be claimed.  Candidate-only replay remains for tracker.v1 data.
        self._reconcile_terminal_results()
        self.tracker.reconcile_succeeded_candidates(self.candidate_queue.read())
        semaphore = asyncio.Semaphore(self.config.seed_worker_count)

        async def worker(seed: SeedRecord) -> None:
            async with semaphore:
                await self._process_seed(seed)

        await asyncio.gather(*(worker(seed) for seed in seeds))
        self.metrics.ensure_error_keys()
        return self.metrics.snapshot()

    async def run_forever(self, *, allow_generation: bool = False) -> None:
        self._allow_generation(allow_generation)
        if self.config.dry_run:
            await self.run_once(allow_generation=False)
            return
        while True:
            await self.run_once(allow_generation=allow_generation)
            await asyncio.sleep(self.config.queue_poll_seconds)


def daemon_from_config(config: GeneratorConfig) -> GeneratorDaemon:
    backend = build_backend(
        config.llm,
        replay_path=config.llm.replay_path or None,
    )
    return GeneratorDaemon(config=config, backend=backend)
