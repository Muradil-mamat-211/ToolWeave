#!/usr/bin/env python3
"""Stream historical Stage-3 artifacts into one auditable Generator replay.

This script never selects boundaries and never mutates source artifacts.  It
has three explicit phases:

* ``discover`` deduplicates artifact copies by a stable boundary-event key;
* ``offline`` rechecks immutable historical candidates with deterministic
  precision gates and the real CPU BFCL fresh-VM path;
* ``finalize`` reconciles a completed fresh Generator replay into the required
  audit views and verifies queue/journal/tracker consistency.

The LLM replay itself remains the production Generator daemon's responsibility
so crash recovery, filesystem locks, prompts, and state-machine semantics are
exercised without a second implementation here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from machine_paths import project_roots
from stage1_format_rl.infrastructure.inventory import discover_local_inventory
from typing import Any, Iterable, Iterator, Mapping

from env_tuning.rods_data_generation_v1.candidate_builder import FUNCTION_MARKER
from env_tuning.rods_data_generation_v1.contracts import validate_seed_record
from env_tuning.rods_data_generation_v1.environment_adapter import (
    SynthesisEnvironmentAdapter,
)
from env_tuning.rods_data_generation_v1.function_catalog import FunctionCatalog
from env_tuning.rods_data_generation_v1.models import stable_id, to_builtin
from env_tuning.rods_data_generation_v1.result_semantics import (
    ExecutionSemanticOutcome,
    classify_execution_result,
    find_unclassified_suspicious_results,
)
from env_tuning.rods_data_generation_v1.revalidation import candidate_to_draft
from env_tuning.rods_data_generation_v1.validation.missing_parameter_validity import (
    missing_parameter_validity_gate,
)
from env_tuning.rods_data_generation_v1.validation.parameter_complexity import (
    parameter_complexity_gate,
)
from env_tuning.rods_data_generation_v1.validation.relational_resolution import (
    relational_resolution_gate,
)
from env_tuning.rods_data_generation_v1.validation.semantic_grounding import (
    semantic_grounding_gate,
)
from env_tuning.rods_data_generation_v1.validation.tool_availability import (
    tool_availability_gate,
)
from env_tuning.rods_data_generation_v1.validation.unit_semantics import (
    unit_semantic_gate,
)
from env_tuning.rods_data_generation_v1.validation.vm_reverify import (
    fresh_vm_reverify_gate,
)
from env_tuning.rods_matchtir_v1.lifecycle import validate_candidate_record


ROOTS = project_roots()
WORKSPACE = ROOTS.source_root
ARTIFACTS = ROOTS.artifacts_root
CATALOG_PARQUET = ROOTS.stage_data_root / "bfcl_stage3_train_all_400_shuffled_seed42.parquet"
DATA_TYPES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)
TYPE_LABELS = {
    "multi_turn_base": "Base",
    "multi_turn_miss_func": "Missing Function",
    "multi_turn_miss_param": "Missing Parameter",
    "multi_turn_long_context": "Long Context",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(
        to_builtin(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL at {path}:{line_number}")
            yield value


class JsonlWriter(AbstractContextManager["JsonlWriter"]):
    """Streaming JSONL writer with a durable close."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def write(self, value: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(to_builtin(value), ensure_ascii=False, sort_keys=True))
        self._handle.write("\n")

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        descriptor = os.open(self.path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact_root(path: Path) -> Path:
    relative = path.resolve().relative_to(ARTIFACTS.resolve())
    return ARTIFACTS / relative.parts[0]


def _source_files(pattern: str, destination: Path) -> list[Path]:
    output: list[Path] = []
    destination = destination.resolve()
    for path in ARTIFACTS.rglob(pattern):
        if destination == path.resolve() or destination in path.resolve().parents:
            continue
        if path.is_file():
            output.append(path.resolve())
    return sorted(set(output))


def _event_payload(seed: Mapping[str, Any]) -> dict[str, Any]:
    step = seed["training_epoch_or_step"]
    return {
        "source_epoch": int(step["epoch"]),
        "source_global_step": int(step["global_step"]),
        "sample_id": str(seed["sample_id"]),
        "data_type": str(seed["data_type"]),
        "canonical_boundary_payload": to_builtin(seed),
    }


def discover(root: Path) -> dict[str, Any]:
    boundary_files = _source_files("boundary_seeds*.jsonl", root)
    manifest_path = root / "00_manifest/discovered_boundary_manifest.json"
    all_path = root / "01_boundary/boundary_seeds_all.jsonl"
    unique_path = root / "01_boundary/boundary_seeds_unique.jsonl"
    duplicate_path = root / "01_boundary/boundary_duplicates.jsonl"
    replay_path = root / "03_replay/boundary_seeds_replayed.jsonl"
    pilot_path = root / "03_replay/pilot/boundary_seeds_pilot.jsonl"

    seen: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    raw_count = 0
    duplicate_count = 0
    with (
        JsonlWriter(all_path) as all_writer,
        JsonlWriter(unique_path) as unique_writer,
        JsonlWriter(duplicate_path) as duplicate_writer,
    ):
        for source in boundary_files:
            for line_number, raw in enumerate(_read_jsonl(source), 1):
                seed = validate_seed_record(raw)
                normalized = to_builtin(raw)
                content_hash = _sha256(normalized)
                event_key = stable_id("boundary_event", _event_payload(normalized), length=32)
                entry = {
                    "source_file": str(source),
                    "source_artifact_root": str(_artifact_root(source)),
                    "source_line": line_number,
                    "sample_id": seed.sample_id,
                    "data_type": seed.data_type,
                    "source_epoch": seed.source_epoch,
                    "source_global_step": seed.source_global_step,
                    "boundary_event_key": event_key,
                    "content_hash": content_hash,
                    "duplicate_of": None,
                }
                raw_count += 1
                all_writer.write(normalized)
                if event_key in seen:
                    duplicate_count += 1
                    entry["duplicate_of"] = seen[event_key]["source_file"] + ":" + str(
                        seen[event_key]["source_line"]
                    )
                    duplicate_writer.write({**entry, "record": normalized})
                else:
                    seen[event_key] = {**entry, "record": normalized}
                    unique_writer.write(normalized)
                entries.append(entry)

    # Every replay tracker identity corresponds to one boundary event, not
    # merely one sample.  Original identity remains explicit in metadata.
    replay_records: list[dict[str, Any]] = []
    for event_key, item in sorted(
        seen.items(),
        key=lambda pair: (
            pair[1]["source_epoch"],
            pair[1]["source_global_step"],
            pair[1]["data_type"],
            pair[1]["sample_id"],
            pair[0],
        ),
    ):
        replay = copy.deepcopy(item["record"])
        original_sample_id = replay["sample_id"]
        replay_seed_id = f"{original_sample_id}__{event_key[-12:]}"
        replay["sample_id"] = replay_seed_id
        replay["generation_metadata"] = {
            **replay["generation_metadata"],
            "historical_replay": True,
            "original_sample_id": original_sample_id,
            "boundary_event_key": event_key,
            "boundary_source_file": item["source_file"],
            "boundary_source_line": item["source_line"],
        }
        validate_seed_record(replay)
        replay_records.append(replay)

    with JsonlWriter(replay_path) as writer:
        for record in replay_records:
            writer.write(record)

    preferred = ("multi_turn_miss_func_83", "multi_turn_miss_param_136")
    pilot: list[dict[str, Any]] = []
    for sample_id in preferred:
        match = next(
            (
                item
                for item in replay_records
                if item["generation_metadata"]["original_sample_id"] == sample_id
            ),
            None,
        )
        if match is not None:
            pilot.append(match)
    for item in replay_records:
        if len(pilot) >= 2:
            break
        if item not in pilot:
            pilot.append(item)
    with JsonlWriter(pilot_path) as writer:
        for record in pilot[:2]:
            writer.write(record)

    manifest = {
        "schema_version": "rods_boundary_discovery.v1",
        "created_at": _utc_now(),
        "source_files": [str(path) for path in boundary_files],
        "raw_discovered_records": raw_count,
        "artifact_duplicate_records": duplicate_count,
        "unique_real_boundary_events": len(seen),
        "boundary_event_identity": (
            "sha256(source_epoch, source_global_step, sample_id, data_type, "
            "canonical full boundary payload)"
        ),
        "entries": entries,
        "replay_seed_aliasing": {
            "reason": "tracker identity must preserve repeated real selections of one sample",
            "original_sample_id_field": "generation_metadata.original_sample_id",
            "event_field": "generation_metadata.boundary_event_key",
        },
    }
    _write_json(manifest_path, manifest)
    shutil.copy2(manifest_path, root / "DISCOVERED_BOUNDARY_MANIFEST.json")
    shutil.copy2(all_path, root / "boundary_seeds_all.jsonl")
    return manifest


def _candidate_training_identity(candidate: Mapping[str, Any]) -> str:
    sample = candidate["sample"]
    kwargs = sample["extra_info"]["interaction_kwargs"]
    system = sample["prompt"][0]["content"]
    visible_tools: Any = []
    if FUNCTION_MARKER in system:
        visible_tools = json.loads(system.split(FUNCTION_MARKER, 1)[1])
    initial_config = kwargs["initial_config"]
    if isinstance(initial_config, str):
        initial_config = json.loads(initial_config)
    content = {
        "data_type": sample["data_source"],
        "initial_config": initial_config,
        "visible_tools": visible_tools,
        "question": kwargs["question"],
        "processed_question": kwargs.get("processed_question", []),
        "ground_truth": kwargs["ground_truth"],
    }
    return stable_id("candidate_content_v2", content, length=32)


def _gate_record(candidate: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    metadata = candidate.get("generation_metadata", {})
    for raw in metadata.get("deterministic_gate_results", []):
        if isinstance(raw, Mapping) and raw.get("name") == name:
            return dict(raw)
    return None


def _boundary_key_for_candidate(
    candidate: Mapping[str, Any],
    source_path: Path,
    boundary_manifest: Mapping[str, Any],
) -> tuple[str | None, list[str]]:
    metadata = candidate["generation_metadata"]
    sample_id = metadata.get("source_seed_id")
    epoch = int(metadata.get("source_epoch", -1))
    step = int(metadata.get("source_global_step", -1))
    entries = [
        item
        for item in boundary_manifest["entries"]
        if item["duplicate_of"] is None
        and item["sample_id"] == sample_id
        and item["source_epoch"] == epoch
        and item["source_global_step"] == step
    ]
    local_root = str(_artifact_root(source_path))
    local = [item for item in entries if item["source_artifact_root"] == local_root]
    resolved = local if len(local) == 1 else entries
    keys = sorted({item["boundary_event_key"] for item in resolved})
    return (keys[0] if len(keys) == 1 else None), keys


def _safe_gate(callable_gate, name: str) -> dict[str, Any]:
    try:
        gate = callable_gate()
        return to_builtin(asdict(gate))
    except Exception as exc:  # Historical corruption must be quarantined, not abort audit.
        return {
            "name": name,
            "passed": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "metadata": {"audit_exception": True},
        }


def offline_revalidate(root: Path) -> dict[str, Any]:
    boundary_manifest = json.loads(
        (root / "00_manifest/discovered_boundary_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_files = _source_files("validated_candidates*.jsonl", root)
    all_path = root / "02_offline_revalidation/historical_candidates_all.jsonl"
    revalidation_path = (
        root / "02_offline_revalidation/historical_candidate_revalidation.jsonl"
    )
    pass_path = root / "02_offline_revalidation/historical_pass.jsonl"
    quarantine_path = root / "02_offline_revalidation/historical_quarantined.jsonl"
    unit_path = root / "08_semantic_audit/unit_semantic_audit.jsonl"
    mp_path = root / "08_semantic_audit/missing_parameter_validity_audit.jsonl"
    execution_path = (
        root / "08_semantic_audit/execution_result_semantics_audit.jsonl"
    )
    suspicious_path = (
        root / "08_semantic_audit/unclassified_suspicious_results_offline.jsonl"
    )

    catalog = FunctionCatalog.from_training_parquet(CATALOG_PARQUET)
    candidate_groups: dict[str, dict[str, Any]] = {}
    raw_count = 0
    with JsonlWriter(all_path) as writer:
        for source in candidate_files:
            for line_number, candidate in enumerate(_read_jsonl(source), 1):
                raw_count += 1
                writer.write(candidate)
                identity = _candidate_training_identity(candidate)
                item = candidate_groups.setdefault(
                    identity,
                    {
                        "candidate": candidate,
                        "source_path": source,
                        "source_line": line_number,
                        "source_mtime": source.stat().st_mtime_ns,
                        "sources": [],
                    },
                )
                item["sources"].append(
                    {
                        "path": str(source),
                        "line": line_number,
                        "candidate_id": candidate.get("candidate_id"),
                    }
                )
                if source.stat().st_mtime_ns >= item["source_mtime"]:
                    item.update(
                        {
                            "candidate": candidate,
                            "source_path": source,
                            "source_line": line_number,
                            "source_mtime": source.stat().st_mtime_ns,
                        }
                    )

    status_counts = Counter()
    known_results: dict[str, dict[str, Any]] = {}
    with (
        JsonlWriter(revalidation_path) as review_writer,
        JsonlWriter(pass_path) as pass_writer,
        JsonlWriter(quarantine_path) as quarantine_writer,
        JsonlWriter(unit_path) as unit_writer,
        JsonlWriter(mp_path) as mp_writer,
        JsonlWriter(execution_path) as execution_writer,
        JsonlWriter(suspicious_path) as suspicious_writer,
    ):
        for identity, item in sorted(candidate_groups.items()):
            candidate = item["candidate"]
            source = item["source_path"]
            candidate_id = str(candidate.get("candidate_id", ""))
            source_seed_id = str(
                candidate.get("generation_metadata", {}).get("source_seed_id", "")
            )
            event_key, event_candidates = _boundary_key_for_candidate(
                candidate, source, boundary_manifest
            )
            try:
                draft = candidate_to_draft(candidate)
                unit = _safe_gate(
                    lambda: unit_semantic_gate(draft, catalog=catalog),
                    "unit_semantic_gate",
                )
                grounding = _safe_gate(
                    lambda: semantic_grounding_gate(draft, catalog=catalog),
                    "semantic_grounding_gate",
                )
                missing_parameter = _safe_gate(
                    lambda: missing_parameter_validity_gate(draft, catalog=catalog),
                    "missing_parameter_validity_gate",
                )
                relational = _safe_gate(
                    lambda: relational_resolution_gate(draft),
                    "relational_resolution_gate",
                )
                fresh = _safe_gate(
                    lambda: fresh_vm_reverify_gate(
                        draft,
                        environment_factory=SynthesisEnvironmentAdapter(),
                        seed_id=f"historical-{candidate_id}",
                    ),
                    "fresh_vm_gate",
                )
                tool = _safe_gate(
                    lambda: tool_availability_gate(draft), "tool_availability_gate"
                )
                complexity = _safe_gate(
                    lambda: parameter_complexity_gate(draft),
                    "parameter_complexity_gate",
                )
                hard_errors = 0
                domain_negatives = 0
                suspicious_count = 0
                for turn in draft.turns:
                    for record in turn.execution_records:
                        semantic = classify_execution_result(
                            record.call.name, record.execution_result
                        )
                        hard_errors += int(
                            semantic.outcome == ExecutionSemanticOutcome.HARD_ERROR
                        )
                        domain_negatives += int(
                            semantic.outcome == ExecutionSemanticOutcome.DOMAIN_NEGATIVE
                        )
                        execution_writer.write(
                            {
                                "phase": "historical_offline",
                                "candidate_id": candidate_id,
                                "source_seed_id": source_seed_id,
                                "turn_id": turn.turn_id,
                                "call_id": record.call_id,
                                "function": record.call.name,
                                "outcome": semantic.outcome.value,
                                "detail": semantic.detail,
                                "source_path": semantic.source_path,
                                "result": record.execution_result,
                            }
                        )
                        for observation in find_unclassified_suspicious_results(
                            record.call.name, record.execution_result
                        ):
                            suspicious_count += 1
                            suspicious_writer.write(
                                {
                                    "phase": "historical_offline",
                                    "candidate_id": candidate_id,
                                    "source_seed_id": source_seed_id,
                                    "turn_id": turn.turn_id,
                                    "call_id": record.call_id,
                                    **observation,
                                }
                            )
                recorded_global = _gate_record(candidate, "final_query_semantic_gate")
                global_semantic = {
                    "status": (
                        "RECORDED_PASS"
                        if recorded_global and recorded_global.get("passed") is True
                        else "RECORDED_FAIL"
                        if recorded_global
                        else "NOT_RECORDED_FAIL_CLOSED"
                    ),
                    "record": recorded_global,
                }
                try:
                    validate_candidate_record(candidate)
                    training_validator = {"passed": True, "detail": "passed"}
                except Exception as exc:
                    training_validator = {
                        "passed": False,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                gate_sequence = (
                    unit,
                    grounding,
                    missing_parameter,
                    relational,
                    fresh,
                    tool,
                    complexity,
                )
                failures = [gate["detail"] for gate in gate_sequence if not gate["passed"]]
                if hard_errors:
                    failures.append(f"execution_result_gate found {hard_errors} HARD_ERROR call(s)")
                if global_semantic["status"] != "RECORDED_PASS":
                    failures.append(global_semantic["status"])
                if not training_validator["passed"]:
                    failures.append(training_validator["detail"])
                status = "PASS" if not failures else "QUARANTINED"
                review = {
                    "candidate_id": candidate_id,
                    "content_fingerprint": identity,
                    "source_seed_id": source_seed_id,
                    "source_boundary_event_key": event_key,
                    "candidate_boundary_event_keys": event_candidates,
                    "old_status": "validated",
                    "new_status": status,
                    "source_candidate_path": str(source),
                    "all_source_records": item["sources"],
                    "unit_gate": unit,
                    "mp_validity_gate": missing_parameter,
                    "execution_result_gate": {
                        "passed": hard_errors == 0,
                        "hard_error_count": hard_errors,
                        "domain_negative_count": domain_negatives,
                    },
                    "semantic_grounding": grounding,
                    "relational": relational,
                    "global_semantic": global_semantic,
                    "fresh_vm": fresh,
                    "tool_availability": tool,
                    "parameter_complexity": complexity,
                    "training_validator": training_validator,
                    "suspicious_result_count": suspicious_count,
                    "quarantine_reason": failures[0] if failures else None,
                    "all_failures": failures,
                    "reviewed_at": _utc_now(),
                }
            except Exception as exc:
                status = "QUARANTINED"
                review = {
                    "candidate_id": candidate_id,
                    "content_fingerprint": identity,
                    "source_seed_id": source_seed_id,
                    "source_boundary_event_key": event_key,
                    "candidate_boundary_event_keys": event_candidates,
                    "old_status": "validated",
                    "new_status": status,
                    "source_candidate_path": str(source),
                    "all_source_records": item["sources"],
                    "quarantine_reason": f"audit reconstruction failed: {type(exc).__name__}: {exc}",
                    "all_failures": [f"{type(exc).__name__}: {exc}"],
                    "reviewed_at": _utc_now(),
                }
                unit = {"passed": False, "detail": "not reached", "metadata": {}}
                missing_parameter = {
                    "passed": False,
                    "detail": "not reached",
                    "metadata": {},
                }
            status_counts[status] += 1
            review_writer.write(review)
            unit_writer.write(
                {
                    "phase": "historical_offline",
                    "candidate_id": candidate_id,
                    "source_seed_id": source_seed_id,
                    **unit,
                }
            )
            mp_writer.write(
                {
                    "phase": "historical_offline",
                    "candidate_id": candidate_id,
                    "source_seed_id": source_seed_id,
                    **missing_parameter,
                }
            )
            output = {"audit": review, "candidate": candidate}
            (pass_writer if status == "PASS" else quarantine_writer).write(output)
            if source_seed_id in {
                "multi_turn_miss_func_83",
                "multi_turn_miss_param_136",
                "multi_turn_miss_param_125",
                "multi_turn_long_context_162",
            }:
                known_results[source_seed_id] = {
                    "candidate_id": candidate_id,
                    "status": status,
                    "reason": review.get("quarantine_reason"),
                }

    summary = {
        "schema_version": "rods_historical_candidate_revalidation.v1",
        "created_at": _utc_now(),
        "source_candidate_files": [str(path) for path in candidate_files],
        "raw_historical_validated_records": raw_count,
        "unique_training_content_candidates": len(candidate_groups),
        "artifact_or_content_duplicates": raw_count - len(candidate_groups),
        "still_pass": status_counts["PASS"],
        "quarantined": status_counts["QUARANTINED"],
        "global_semantic_policy": (
            "recorded current final semantic gate required; absent historical gate fails closed"
        ),
        "known_precision_patterns": known_results,
    }
    _write_json(root / "02_offline_revalidation/revalidation_summary.json", summary)
    return summary


def _load_replay_seeds(root: Path) -> dict[str, dict[str, Any]]:
    return {
        item["sample_id"]: item
        for item in _read_jsonl(root / "03_replay/boundary_seeds_replayed.jsonl")
    }


def _failure_bucket(record: Mapping[str, Any]) -> str:
    reason = str(record.get("drop_reason") or "")
    lowered = reason.casefold()
    mappings = (
        ("unit_semantic", "unit_semantic_failed"),
        ("semantic_grounding", "semantic_grounding_failed"),
        ("missing_parameter", "missing_parameter_not_genuine"),
        ("relational", "relational_failed"),
        ("global coherence", "global_coherence_failed"),
        ("final semantic", "global_coherence_failed"),
        ("fresh_vm", "fresh_vm_failed"),
        ("parameter_complexity", "complexity_failed"),
        ("quality judge", "judge_rejected"),
        ("duplicate", "duplicate_content"),
    )
    for marker, bucket in mappings:
        if marker in lowered:
            return bucket
    errors = record.get("errors", [])
    if isinstance(errors, list) and errors:
        last = errors[-1]
        if isinstance(last, Mapping):
            error_type = str(last.get("error_type", ""))
            if error_type:
                return error_type
    return "other"


def finalize(root: Path) -> dict[str, Any]:
    runtime = root / "03_replay/runtime"
    terminal_source = runtime / "terminal_results.jsonl"
    candidate_source = root / "05_validated/validated_candidates_all.jsonl"
    tracker_source = root / "07_tracker/tracker_final.json"
    if not terminal_source.is_file() or not tracker_source.is_file():
        raise FileNotFoundError("fresh replay terminal journal/tracker is incomplete")
    seeds = _load_replay_seeds(root)
    terminals = list(_read_jsonl(terminal_source))
    candidates = list(_read_jsonl(candidate_source)) if candidate_source.is_file() else []
    tracker = json.loads(tracker_source.read_text(encoding="utf-8"))
    terminal_by_seed = {record["seed_id"]: record for record in terminals}
    if len(terminal_by_seed) != len(terminals):
        raise RuntimeError("terminal journal contains duplicate seed identities")
    if set(terminal_by_seed) != set(seeds):
        missing = sorted(set(seeds) - set(terminal_by_seed))
        extra = sorted(set(terminal_by_seed) - set(seeds))
        raise RuntimeError(f"terminal coverage mismatch: missing={missing}, extra={extra}")
    tracker_seeds = tracker.get("seeds", {})
    nonterminal_tracker = {
        seed_id: item.get("status")
        for seed_id, item in tracker_seeds.items()
        if item.get("status") not in {"SUCCEEDED", "DROPPED"}
    }
    if nonterminal_tracker:
        raise RuntimeError(f"tracker contains RUNNING/PENDING seeds: {nonterminal_tracker}")

    succeeded = [record for record in terminals if record["status"] == "SUCCEEDED"]
    dropped = [record for record in terminals if record["status"] == "DROPPED"]
    queue_candidate_ids = {candidate["candidate_id"] for candidate in candidates}
    terminal_candidate_ids = {record["candidate_id"] for record in succeeded}
    if queue_candidate_ids != terminal_candidate_ids:
        raise RuntimeError(
            "validated queue and terminal SUCCEEDED candidate-id sets differ"
        )
    for candidate in candidates:
        validate_candidate_record(candidate)

    terminal_all = root / "04_terminal/terminal_results_all.jsonl"
    terminal_success = root / "04_terminal/terminal_succeeded.jsonl"
    terminal_drop = root / "04_terminal/terminal_dropped.jsonl"
    quarantined_all = root / "06_quarantined/quarantined_candidates_all.jsonl"
    quarantine_reasons = root / "06_quarantined/quarantine_reasons.jsonl"
    matrix_path = root / "08_semantic_audit/validated_candidate_quality_matrix.jsonl"
    unit_path = root / "08_semantic_audit/unit_semantic_audit_replay.jsonl"
    mp_path = root / "08_semantic_audit/missing_parameter_validity_audit_replay.jsonl"
    execution_path = root / "08_semantic_audit/execution_result_semantics_audit_replay.jsonl"
    suspicious_path = root / "08_semantic_audit/unclassified_suspicious_results.jsonl"

    source_suspicious = runtime / "unclassified_suspicious_results.jsonl"
    if source_suspicious.is_file():
        shutil.copy2(source_suspicious, suspicious_path)
    else:
        suspicious_path.touch()

    with (
        JsonlWriter(terminal_all) as all_writer,
        JsonlWriter(terminal_success) as success_writer,
        JsonlWriter(terminal_drop) as drop_writer,
        JsonlWriter(quarantined_all) as quarantine_writer,
        JsonlWriter(quarantine_reasons) as reason_writer,
    ):
        for record in terminals:
            all_writer.write(record)
            if record["status"] == "SUCCEEDED":
                success_writer.write(record)
            else:
                drop_writer.write(record)
                quarantine_writer.write(record)
                reason_writer.write(
                    {
                        "boundary_event_key": seeds[record["seed_id"]][
                            "generation_metadata"
                        ]["boundary_event_key"],
                        "seed_id": record["seed_id"],
                        "original_sample_id": seeds[record["seed_id"]][
                            "generation_metadata"
                        ]["original_sample_id"],
                        "data_type": seeds[record["seed_id"]]["data_type"],
                        "reason_bucket": _failure_bucket(record),
                        "drop_reason": record.get("drop_reason"),
                    }
                )

    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    suspicious_rows = list(_read_jsonl(suspicious_path)) if suspicious_path.stat().st_size else []
    suspicious_by_seed = Counter(str(row.get("seed_id", "")) for row in suspicious_rows)
    unresolved_by_seed = Counter(
        str(row.get("seed_id", ""))
        for row in suspicious_rows
        if row.get("changes_execution_semantics") is True
    )
    with (
        JsonlWriter(matrix_path) as matrix_writer,
        JsonlWriter(unit_path) as unit_writer,
        JsonlWriter(mp_path) as mp_writer,
        JsonlWriter(execution_path) as execution_writer,
    ):
        for terminal in succeeded:
            seed = seeds[terminal["seed_id"]]
            candidate = terminal["candidate"]
            validate_candidate_record(candidate)
            gates = {
                gate["name"]: gate
                for gate in candidate["generation_metadata"][
                    "deterministic_gate_results"
                ]
            }
            hard_errors = 0
            domain_negatives = 0
            trace = candidate["generation_metadata"]["execution_trace"]
            for turn in trace:
                for record in turn.get("records", []):
                    call = record["call"]
                    semantic = classify_execution_result(
                        call["name"], record.get("execution_result")
                    )
                    hard_errors += int(
                        semantic.outcome == ExecutionSemanticOutcome.HARD_ERROR
                    )
                    domain_negatives += int(
                        semantic.outcome == ExecutionSemanticOutcome.DOMAIN_NEGATIVE
                    )
                    execution_writer.write(
                        {
                            "phase": "fresh_replay",
                            "boundary_event_key": seed["generation_metadata"][
                                "boundary_event_key"
                            ],
                            "seed_id": terminal["seed_id"],
                            "candidate_id": terminal["candidate_id"],
                            "turn_id": turn["turn_id"],
                            "call_id": record["call_id"],
                            "function": call["name"],
                            "outcome": semantic.outcome.value,
                            "detail": semantic.detail,
                            "source_path": semantic.source_path,
                        }
                    )
            kwargs = candidate["sample"]["extra_info"]["interaction_kwargs"]
            judge = candidate["generation_metadata"]["judge_result"]
            matrix = {
                "boundary_event_key": seed["generation_metadata"]["boundary_event_key"],
                "seed_id": terminal["seed_id"],
                "original_sample_id": seed["generation_metadata"]["original_sample_id"],
                "candidate_id": terminal["candidate_id"],
                "data_type": seed["data_type"],
                "attempts": terminal["attempts"],
                "num_turns": len(kwargs["ground_truth"]),
                "num_gt_calls": sum(len(turn) for turn in kwargs["ground_truth"]),
                "unit_guard": gates.get("unit_semantic_gate", {}).get("passed", False),
                "argument_grounding": gates.get("semantic_grounding_gate", {}).get("passed", False),
                "missing_parameter_validity": gates.get("missing_parameter_validity_gate", {}).get("passed", False),
                "execution_semantics": hard_errors == 0,
                "domain_negative_count": domain_negatives,
                "hard_error_count": hard_errors,
                "relational_guard": gates.get("relational_resolution_gate", {}).get("passed", False),
                "global_semantic": gates.get("final_query_semantic_gate", {}).get("passed", False),
                "fresh_vm": gates.get("fresh_vm_gate", {}).get("passed", False),
                "tool_availability": gates.get("tool_availability_gate", {}).get("passed", False),
                "complexity": gates.get("parameter_complexity_gate", {}).get("passed", False),
                "judge": judge.get("decision") == "accept",
                "training_validator": True,
                "suspicious_result_count": suspicious_by_seed[terminal["seed_id"]],
                "unresolved_suspicious_count": unresolved_by_seed[terminal["seed_id"]],
                "final_verdict": "PASS",
            }
            matrix_writer.write(matrix)
            unit_writer.write(
                {
                    "phase": "fresh_replay",
                    "seed_id": terminal["seed_id"],
                    "candidate_id": terminal["candidate_id"],
                    **gates["unit_semantic_gate"],
                }
            )
            mp_writer.write(
                {
                    "phase": "fresh_replay",
                    "seed_id": terminal["seed_id"],
                    "candidate_id": terminal["candidate_id"],
                    **gates["missing_parameter_validity_gate"],
                }
            )

    by_type: dict[str, dict[str, int | float]] = {}
    for data_type in DATA_TYPES:
        type_terminals = [
            item for item in terminals if seeds[item["seed_id"]]["data_type"] == data_type
        ]
        valid = sum(item["status"] == "SUCCEEDED" for item in type_terminals)
        total = len(type_terminals)
        by_type[TYPE_LABELS[data_type]] = {
            "dispatched": total,
            "validated": valid,
            "dropped": total - valid,
            "p_valid": valid / total if total else 0.0,
        }
    failure_reasons = Counter(_failure_bucket(record) for record in dropped)
    summary = {
        "schema_version": "rods_generator_historical_replay_summary.v1",
        "created_at": _utc_now(),
        "dispatched": len(terminals),
        "validated_terminal_events": len(succeeded),
        "validated_unique_training_candidates": len(candidates),
        "exact_content_duplicates_suppressed": len(succeeded) - len(candidates),
        "dropped": len(dropped),
        "p_valid": len(succeeded) / len(terminals) if terminals else 0.0,
        "by_type": by_type,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "queue_terminal_candidate_id_set_equal": True,
        "all_seed_events_terminal": True,
        "running_or_ghost_seed_count": 0,
        "unresolved_semantics_changing_suspicious": sum(unresolved_by_seed.values()),
        "telemetry_only_suspicious": len(suspicious_rows),
        "training_validator_failures": 0,
    }
    _write_json(root / "07_tracker/queue_state.json", summary)
    _write_json(root / "03_replay/replay_summary.json", summary)
    shutil.copy2(terminal_all, root / "terminal_results_all.jsonl")
    shutil.copy2(candidate_source, root / "validated_candidates_all.jsonl")
    return summary


def environment_manifest(root: Path) -> dict[str, Any]:
    generator_root = (
        WORKSPACE
        / "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_data_generation_v1"
    )
    hashes: dict[str, str] = {}
    for path in sorted(generator_root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            hashes[str(path.relative_to(WORKSPACE))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    _write_json(
        root / "00_manifest/code_hashes.json",
        {"created_at": _utc_now(), "sha256": hashes},
    )
    inventory = discover_local_inventory()
    env = {
        "created_at": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_visible": os.cpu_count(),
        "observed_hardware": inventory,
        "catalog_parquet": str(CATALOG_PARQUET),
        "model": str(ROOTS.models_root / "gemma-4-31B-it-manual"),
        "formal_training_launched": False,
    }
    _write_json(root / "00_manifest/environment.json", env)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("discover", "offline", "finalize", "environment"))
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "discover":
        result = discover(root)
    elif args.phase == "offline":
        result = offline_revalidate(root)
    elif args.phase == "finalize":
        result = finalize(root)
    else:
        result = environment_manifest(root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
