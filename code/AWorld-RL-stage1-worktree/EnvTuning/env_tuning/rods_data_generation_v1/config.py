"""Configuration with paper-derived values separated from project defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


WORKSPACE = Path("/root/autodl-tmp/rods-workspace")


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "replay"
    model: str = str(WORKSPACE / "models" / "gemma-4-31B-it-manual")
    endpoint: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 1.0  # RODS Appendix J
    top_p: float = 0.7  # RODS Appendix J
    max_tokens: int = 4096  # PROJECT/RECONSTRUCTED default
    timeout_seconds: float = 120.0  # PROJECT/RECONSTRUCTED default
    transport_retries: int = 2  # Transport only; not an algorithm retry.
    concurrency: int = 1  # Must be benchmarked for the target 2x48GB host.
    disable_native_thinking: bool = True
    raw_response_log_path: str = ""
    replay_path: str = ""  # PROJECT fixture/config path; never a RODS default.

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "LLMConfig":
        raw = raw or {}
        defaults = cls()
        return cls(
            backend=str(raw.get("backend", "replay")),
            model=str(raw.get("model", defaults.model)),
            endpoint=str(raw.get("endpoint", defaults.endpoint)),
            api_key=str(raw.get("api_key", "EMPTY")),
            temperature=float(raw.get("temperature", 1.0)),
            top_p=float(raw.get("top_p", 0.7)),
            max_tokens=int(raw.get("max_tokens", 4096)),
            timeout_seconds=float(raw.get("timeout_seconds", 120.0)),
            transport_retries=int(raw.get("transport_retries", 2)),
            concurrency=int(raw.get("concurrency", 1)),
            disable_native_thinking=bool(raw.get("disable_native_thinking", True)),
            raw_response_log_path=str(raw.get("raw_response_log_path", "")),
            replay_path=str(raw.get("replay_path", "")),
        )


@dataclass(frozen=True)
class QueueConfig:
    seed_path: str = str(WORKSPACE / "stage1_format_rl/artifacts/stage3_queues/boundary_seeds.jsonl")
    # Safe default: production queue requires an explicit config override plus
    # the launch guard.  Dry runs never target the Training ingestion queue.
    candidate_path: str = str(
        WORKSPACE / "stage1_format_rl/artifacts/stage3_generator/dry_run/validated_candidates.jsonl"
    )
    tracker_path: str = str(WORKSPACE / "stage1_format_rl/artifacts/stage3_generator/tracker.json")
    event_log_path: str = str(WORKSPACE / "stage1_format_rl/artifacts/stage3_generator/events.jsonl")
    expanded_log_dir: str = str(WORKSPACE / "stage1_format_rl/artifacts/stage3_generator/expanded")
    production_candidate_path: str = str(WORKSPACE / "stage1_format_rl/artifacts/stage3_queues/validated_candidates.jsonl")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "QueueConfig":
        raw = raw or {}
        defaults = cls()
        return cls(**{name: str(raw.get(name, getattr(defaults, name))) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class GeneratorConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    queues: QueueConfig = field(default_factory=QueueConfig)
    function_catalog_dir: str = str(
        WORKSPACE / "data/Berkeley-Function-Calling-Leaderboard/multi_turn_func_doc"
    )
    # PROJECT integration source of truth.  The schemas embedded in this
    # active Training dataset match EnvTuning's bfcl_env implementation; the
    # separately downloaded public function-doc checkout is version-skewed.
    function_schema_parquet: str = str(
        WORKSPACE
        / "stage1_format_rl/data/bfcl_stage3_train_all_400_shuffled_seed42.parquet"
    )
    max_pipeline_attempts: int = 3  # RODS Appendix J
    planner_retries: int = 3  # RODS Appendix J
    agent_parse_retries: int = 1  # PROJECT/RECONSTRUCTED default
    max_refinement_cycles: int = 1  # RODS Appendix G
    dry_run: bool = True
    test_mode: bool = False
    use_augmented_environment: bool = False
    seed_worker_count: int = 1  # PROJECT default; production must benchmark.
    queue_poll_seconds: float = 5.0  # PROJECT default.

    def __post_init__(self) -> None:
        if self.max_pipeline_attempts != 3:
            raise ValueError("RODS V1 requires exactly 3 full pipeline attempts")
        if self.planner_retries != 3:
            raise ValueError("RODS V1 requires exactly 3 planner retries")
        if self.max_refinement_cycles != 1:
            raise ValueError("RODS V1 permits at most one refinement cycle")
        if self.llm.temperature != 1.0 or self.llm.top_p != 0.7:
            raise ValueError("RODS paper-derived synthesis decoding is temperature=1.0, top_p=0.7")
        if self.llm.concurrency < 1 or self.llm.max_tokens < 1:
            raise ValueError("LLM concurrency and max_tokens must be positive")
        if self.seed_worker_count < 1 or self.queue_poll_seconds <= 0:
            raise ValueError("seed_worker_count and queue_poll_seconds must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "GeneratorConfig":
        raw = raw or {}
        defaults = cls()
        return cls(
            llm=LLMConfig.from_mapping(raw.get("llm")),
            queues=QueueConfig.from_mapping(raw.get("queues")),
            function_catalog_dir=str(raw.get("function_catalog_dir", defaults.function_catalog_dir)),
            function_schema_parquet=str(
                raw.get("function_schema_parquet", defaults.function_schema_parquet)
            ),
            max_pipeline_attempts=int(raw.get("max_pipeline_attempts", 3)),
            planner_retries=int(raw.get("planner_retries", 3)),
            agent_parse_retries=int(raw.get("agent_parse_retries", 1)),
            max_refinement_cycles=int(raw.get("max_refinement_cycles", 1)),
            dry_run=bool(raw.get("dry_run", True)),
            test_mode=bool(raw.get("test_mode", False)),
            use_augmented_environment=bool(raw.get("use_augmented_environment", False)),
            seed_worker_count=int(raw.get("seed_worker_count", 1)),
            queue_poll_seconds=float(raw.get("queue_poll_seconds", 5.0)),
        )
