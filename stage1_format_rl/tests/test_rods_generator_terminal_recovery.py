"""Fault-injection regressions for terminal-result crash consistency."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from env_tuning.rods_data_generation_v1.daemon import GeneratorDaemon
from env_tuning.rods_data_generation_v1.error_taxonomy import ErrorType
from env_tuning.rods_data_generation_v1.llm_backend import FakeLLMBackend
from env_tuning.rods_data_generation_v1.metrics import GeneratorMetrics
from env_tuning.rods_data_generation_v1.models import ErrorRecord, PipelineResult
from env_tuning.rods_data_generation_v1.queue import LockedJsonlQueue

from rods_data_generation_v1_fixtures import make_config, make_seed


class InjectedCrash(RuntimeError):
    pass


class _CrashAt:
    def __init__(self, point: str):
        self.point = point
        self.triggered = False

    def __call__(self, point: str) -> None:
        if point == self.point and not self.triggered:
            self.triggered = True
            raise InjectedCrash(point)


def _errors(seed_id: str, count: int) -> list[ErrorRecord]:
    return [
        ErrorRecord(
            error_type=ErrorType.NO_PATTERN,
            seed_id=seed_id,
            attempt_id=attempt,
            turn_id=None,
            function_names=(),
            detail=f"controlled failed attempt {attempt}",
            patchable=False,
        )
        for attempt in range(1, count + 1)
    ]


def _checkpoint(errors: list[ErrorRecord], planner_calls: int) -> dict:
    return {
        "completed_failed_attempts": len(errors),
        "planner_calls": planner_calls,
        "failures": [error.to_dict() for error in errors],
        "patches": [],
        "blocklist": [],
        "blocklist_history": [[] for _ in errors],
        "current_config": {},
    }


def _candidate(seed_id: str) -> dict:
    return {
        "candidate_id": "candidate_terminal_attempt3",
        "validated": True,
        "validation": {"passed": True},
        "generation_metadata": {"source_seed_id": seed_id, "generated_epoch": 7},
        "sample": {"complete_payload_marker": [1, 2, 3]},
    }


class Attempt3TerminalPipeline:
    """Attempt 1/2 fail durably; attempt 3 returns one terminal result."""

    def __init__(self, *, dropped: bool = False):
        self.metrics = GeneratorMetrics()
        self.dropped = dropped
        self.resume_counts: list[int] = []
        self.generate_calls = 0

    async def generate(self, seed, *, resume_state=None, checkpoint_callback=None):
        self.generate_calls += 1
        completed = int((resume_state or {}).get("completed_failed_attempts", 0))
        self.resume_counts.append(completed)
        terminal_failed_count = 3 if self.dropped else 2
        errors = _errors(seed.sample_id, completed)
        for attempt in range(completed + 1, terminal_failed_count + 1):
            errors = _errors(seed.sample_id, attempt)
            if checkpoint_callback is not None:
                checkpoint_callback(_checkpoint(errors, planner_calls=attempt * 3))
        if self.dropped:
            return PipelineResult(
                seed_id=seed.sample_id,
                status="DROPPED",
                candidate=None,
                errors=errors,
                attempts=3,
                planner_calls=9,
                blocklist_history=[[]] * 3,
                config_patch_history=[],
                metrics={},
                reason="three feedback-conditioned pipeline attempts failed",
                checkpoint=_checkpoint(errors, planner_calls=9),
            )
        errors = _errors(seed.sample_id, 2)
        return PipelineResult(
            seed_id=seed.sample_id,
            status="SUCCEEDED",
            candidate=_candidate(seed.sample_id),
            errors=errors,
            attempts=3,
            planner_calls=7,
            blocklist_history=[[]] * 2,
            config_patch_history=[],
            metrics={},
            reason="success on attempt 3",
            checkpoint=_checkpoint(errors, planner_calls=7),
        )


class NeverRunPipeline:
    def __init__(self):
        self.metrics = GeneratorMetrics()
        self.generate_calls = 0

    async def generate(self, *args, **kwargs):
        self.generate_calls += 1
        raise AssertionError("journal recovery must not invoke the pipeline or LLM")


def _seed_queue(config) -> None:
    LockedJsonlQueue(config.queues.seed_path).append([make_seed()])


def _simulate_worker_process_death(config) -> None:
    state_path = config.queues.tracker_path
    state = json.loads(open(state_path, encoding="utf-8").read())
    item = state["seeds"]["seed-multi_turn_base"]
    item["worker_pid"] = 99999999
    item["worker_process_start"] = "gone"
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)


def _run_crashing_daemon(tmp_path, point: str, *, dropped: bool = False):
    config = make_config(tmp_path=tmp_path, dry_run=False, test_mode=True)
    _seed_queue(config)
    pipeline = Attempt3TerminalPipeline(dropped=dropped)
    daemon = GeneratorDaemon(
        config=config,
        backend=FakeLLMBackend({}),
        pipeline=pipeline,
        fault_injector=_CrashAt(point),
    )
    with pytest.raises(InjectedCrash, match=point):
        asyncio.run(daemon.run_once(allow_generation=True))
    return config, daemon, pipeline


def test_crash_1_attempt3_success_before_journal_reexecutes_only_attempt3(tmp_path):
    config, daemon, _ = _run_crashing_daemon(
        tmp_path, "after_pipeline_before_terminal_journal"
    )
    assert daemon.terminal_results.read() == []
    assert daemon.candidate_queue.read() == []
    tracker_item = daemon.tracker.snapshot()["seeds"]["seed-multi_turn_base"]
    assert tracker_item["completed_failed_attempts"] == 2

    _simulate_worker_process_death(config)
    resumed_pipeline = Attempt3TerminalPipeline()
    restarted = GeneratorDaemon(
        config=config,
        backend=FakeLLMBackend({}),
        pipeline=resumed_pipeline,
    )
    asyncio.run(restarted.run_once(allow_generation=True))
    assert resumed_pipeline.resume_counts == [2]
    assert len(restarted.candidate_queue.read()) == 1
    assert restarted.tracker.snapshot()["seeds"]["seed-multi_turn_base"]["status"] == "SUCCEEDED"


def test_crash_2_success_journal_replays_exact_candidate_without_pipeline(tmp_path):
    config, daemon, _ = _run_crashing_daemon(
        tmp_path, "after_terminal_journal_before_candidate_append"
    )
    terminal = daemon.terminal_results.read()
    assert len(terminal) == 1 and terminal[0]["status"] == "SUCCEEDED"
    assert daemon.candidate_queue.read() == []

    never = NeverRunPipeline()
    restarted = GeneratorDaemon(config=config, backend=FakeLLMBackend({}), pipeline=never)
    asyncio.run(restarted.run_once(allow_generation=True))
    assert never.generate_calls == 0
    assert restarted.candidate_queue.read() == [terminal[0]["candidate"]]
    assert restarted.tracker.snapshot()["seeds"]["seed-multi_turn_base"]["status"] == "SUCCEEDED"


def test_crash_3_candidate_fsync_before_tracker_reconciles_without_duplicate(tmp_path):
    config, daemon, _ = _run_crashing_daemon(
        tmp_path, "after_candidate_append_before_tracker_succeeded"
    )
    assert len(daemon.candidate_queue.read()) == 1
    never = NeverRunPipeline()
    restarted = GeneratorDaemon(config=config, backend=FakeLLMBackend({}), pipeline=never)
    asyncio.run(restarted.run_once(allow_generation=True))
    assert never.generate_calls == 0
    assert len(restarted.candidate_queue.read()) == 1
    assert restarted.tracker.snapshot()["seeds"]["seed-multi_turn_base"]["status"] == "SUCCEEDED"


def test_crash_4_dropped_journal_recovers_drop_without_regeneration(tmp_path):
    config, daemon, _ = _run_crashing_daemon(
        tmp_path, "after_dropped_journal_before_tracker_dropped", dropped=True
    )
    assert daemon.terminal_results.read()[0]["status"] == "DROPPED"
    never = NeverRunPipeline()
    restarted = GeneratorDaemon(config=config, backend=FakeLLMBackend({}), pipeline=never)
    asyncio.run(restarted.run_once(allow_generation=True))
    assert never.generate_calls == 0
    assert restarted.candidate_queue.read() == []
    assert restarted.tracker.snapshot()["seeds"]["seed-multi_turn_base"]["status"] == "DROPPED"


def test_crash_5_recovery_is_idempotent_across_repeated_runs(tmp_path):
    config, _, _ = _run_crashing_daemon(
        tmp_path, "after_terminal_journal_before_candidate_append"
    )
    never = NeverRunPipeline()
    restarted = GeneratorDaemon(config=config, backend=FakeLLMBackend({}), pipeline=never)
    asyncio.run(restarted.run_once(allow_generation=True))
    first_tracker = copy.deepcopy(restarted.tracker.snapshot()["seeds"]["seed-multi_turn_base"])
    asyncio.run(restarted.run_once(allow_generation=True))
    second_tracker = restarted.tracker.snapshot()["seeds"]["seed-multi_turn_base"]
    assert never.generate_calls == 0
    assert len(restarted.terminal_results.read()) == 1
    assert len(restarted.candidate_queue.read()) == 1
    assert first_tracker == second_tracker
