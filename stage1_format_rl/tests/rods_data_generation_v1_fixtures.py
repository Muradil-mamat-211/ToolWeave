"""Deterministic fixtures for the no-GPU Generator test suite."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from env_tuning.rods_data_generation_v1.config import GeneratorConfig, LLMConfig, QueueConfig
from env_tuning.rods_data_generation_v1.function_catalog import FunctionCatalog
from env_tuning.rods_data_generation_v1.llm_backend import FakeLLMBackend
from env_tuning.rods_data_generation_v1.models import SEED_SCHEMA_VERSION


SOURCE_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get("TOOLWEAVE_ASSET_ROOT", SOURCE_ROOT)).expanduser().resolve()
CATALOG_DIR = ASSET_ROOT / "data/Berkeley-Function-Calling-Leaderboard/multi_turn_func_doc"

PLANNER_ADD_MULTIPLY = (
    "<reason>Preserve two arithmetic turns.</reason>"
    "<narrative>A user checks two independent calculations.</narrative>"
    "<turn>MathAPI: add</turn>"
    "<turn>MathAPI: multiply</turn>"
)
PARAM_ADD = '<reason>Use the requested operands.</reason><arguments>{"a": 2.0, "b": 3.0}</arguments>'
PARAM_MULTIPLY = '<reason>Use the requested operands.</reason><arguments>{"a": 4.0, "b": 5.0}</arguments>'
QUERY_ADD = "<reason>Describe the executed sum.</reason><query>What is two plus three?</query>"
QUERY_MULTIPLY = "<reason>Describe the executed product.</reason><query>What is four times five?</query>"
VERIFY_ACCEPT = "<reason>The request and executed behavior agree.</reason><verdict>accept</verdict>"
REWRITE_TWO = "<query>Could you work out two plus three?</query><query>Now, what is four times five?</query>"
JUDGE_ACCEPT = "<reason>All five criteria pass.</reason><decision>accept</decision><fail_reason></fail_reason>"


def make_catalog() -> FunctionCatalog:
    return FunctionCatalog.from_bfcl_directory(CATALOG_DIR)


def make_seed(data_type: str = "multi_turn_base", *, epoch: int = 7) -> dict[str, Any]:
    catalog = make_catalog()
    tools = [copy.deepcopy(catalog.get(name).schema) for name in ("add", "multiply")]
    return {
        "schema_version": SEED_SCHEMA_VERSION,
        "sample_id": f"seed-{data_type}",
        "data_type": data_type,
        "Q_old": [
            [{"role": "user", "content": "Please do one calculation."}],
            [{"role": "user", "content": "Please do a second calculation."}],
        ],
        "GT_old": [["add(a=1.0, b=1.0)"], ["multiply(a=2.0, b=2.0)"]],
        "available_functions": tools,
        "initial_config": {},
        "mean_progress": 0.5,
        "boundary_score_phi": 1.0,
        "training_epoch_or_step": {"epoch": epoch, "global_step": 70},
        "generation_metadata": {
            "source": "deterministic test fixture",
            "progress_source": "R_P_only",
        },
    }


def success_script(data_type: str = "multi_turn_base") -> dict[str, list[str]]:
    script: dict[str, list[str]] = {
        "planner": [PLANNER_ADD_MULTIPLY],
        "parameter_generator": [PARAM_ADD, PARAM_MULTIPLY],
        "query_generator": [QUERY_ADD, QUERY_MULTIPLY],
        "query_verifier": [VERIFY_ACCEPT, VERIFY_ACCEPT],
        "coherence_rewrite": [REWRITE_TWO],
        "final_query_verifier": [VERIFY_ACCEPT],
        "quality_judge": [JUDGE_ACCEPT],
    }
    if data_type == "multi_turn_miss_func":
        script["missing_function"] = [
            "<reason>The first turn requires a tool that can be restored.</reason>"
            "<affected_turn>0</affected_turn><missing_function>add</missing_function>"
        ]
    elif data_type == "multi_turn_miss_param":
        script["missing_parameter"] = [
            "<reason>Omit the first operand, then provide it.</reason>"
            "<affected_turn>0</affected_turn><missing_parameter>a</missing_parameter>"
            "<affected_query>Please combine an unspecified first value with three.</affected_query>"
            "<recovery_query>Use 2.0 as the first value.</recovery_query>"
        ]
    return script


def make_config(
    *,
    tmp_path: Path | None = None,
    dry_run: bool = True,
    test_mode: bool = True,
) -> GeneratorConfig:
    queues = QueueConfig()
    if tmp_path is not None:
        queues = QueueConfig(
            seed_path=str(tmp_path / "boundary_seeds.jsonl"),
            candidate_path=str(tmp_path / "validated_candidates.jsonl"),
            tracker_path=str(tmp_path / "tracker.json"),
            event_log_path=str(tmp_path / "events.jsonl"),
            expanded_log_dir=str(tmp_path / "expanded"),
            production_candidate_path=str(
                SOURCE_ROOT / ".runtime/test-production-queue-must-not-be-written.jsonl"
            ),
        )
    return GeneratorConfig(
        llm=LLMConfig(backend="fake", model="fixture-gemma-4-31b"),
        queues=queues,
        function_catalog_dir=str(CATALOG_DIR),
        dry_run=dry_run,
        test_mode=test_mode,
        seed_worker_count=2,
    )


def make_backend(data_type: str = "multi_turn_base") -> FakeLLMBackend:
    return FakeLLMBackend(success_script(data_type))
