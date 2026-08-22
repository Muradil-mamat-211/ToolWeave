#!/usr/bin/env python3
"""Revalidate immutable Generator candidates into a separate artifact root."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

from env_tuning.rods_data_generation_v1.config import LLMConfig
from env_tuning.rods_data_generation_v1.function_catalog import FunctionCatalog
from env_tuning.rods_data_generation_v1.llm_backend import VLLMOpenAIBackend
from env_tuning.rods_data_generation_v1.metrics import GeneratorMetrics
from env_tuning.rods_data_generation_v1.models import to_builtin, utc_now
from env_tuning.rods_data_generation_v1.parsing import (
    StructuredParseError,
    parse_planner_response,
)
from env_tuning.rods_data_generation_v1.query_verifier import QueryVerifier
from env_tuning.rods_data_generation_v1.queue import LockedJsonlQueue, atomic_write_json
from env_tuning.rods_data_generation_v1.revalidation import (
    revalidate_candidate_grounding,
)
from env_tuning.rods_data_generation_v1.validation.relational_resolution import (
    relational_resolution_gate,
)
from env_tuning.rods_matchtir_v1.lifecycle import validate_candidate_record
from machine_paths import project_roots


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _planner_narratives(
    paths: list[Path], catalog: FunctionCatalog
) -> dict[str, dict]:
    """Recover latest valid Planner narrative from immutable raw transport."""

    recovered: dict[str, dict] = {}
    for path in paths:
        for record in _read_jsonl(path):
            if record.get("role") != "planner":
                continue
            metadata = record.get("metadata", {})
            seed_id = metadata.get("seed_id") if isinstance(metadata, dict) else None
            attempt_id = metadata.get("attempt_id") if isinstance(metadata, dict) else None
            choices = record.get("response", {}).get("choices", [])
            if not isinstance(seed_id, str) or not choices:
                continue
            text = choices[0].get("message", {}).get("content")
            if not isinstance(text, str):
                continue
            try:
                plan = parse_planner_response(
                    text,
                    allowed_functions=catalog.names(),
                    class_for_function=catalog.class_for_function(),
                )
            except StructuredParseError:
                continue
            numeric_attempt = int(attempt_id or 0)
            prior = recovered.get(seed_id)
            if prior is None or numeric_attempt >= prior["attempt_id"]:
                recovered[seed_id] = {
                    "attempt_id": numeric_attempt,
                    "narrative": plan.narrative,
                    "source_path": str(path.resolve()),
                }
    return recovered


async def main_async(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    by_seed: dict[str, dict] = {}
    source_for_seed: dict[str, str] = {}
    for source in args.candidates:
        for candidate in _read_jsonl(source):
            seed_id = candidate["generation_metadata"]["source_seed_id"]
            by_seed[seed_id] = candidate
            source_for_seed[seed_id] = str(source.resolve())

    catalog = FunctionCatalog.from_training_parquet(args.catalog)
    planner_narratives = _planner_narratives(args.raw_responses, catalog)
    metrics = GeneratorMetrics()
    backend = VLLMOpenAIBackend(
        LLMConfig(
            backend="vllm_openai",
            model=args.model,
            endpoint=args.endpoint,
            api_key="EMPTY",
            temperature=1.0,
            top_p=0.7,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout,
            transport_retries=2,
            concurrency=args.concurrency,
            disable_native_thinking=True,
            raw_response_log_path=str(args.output / "revalidation_raw_responses.jsonl"),
        )
    )
    verifier = QueryVerifier(backend, metrics)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def inspect(seed_id: str, candidate: dict) -> tuple[str, dict, dict | None]:
        review_candidate = copy.deepcopy(candidate)
        recovered_narrative = planner_narratives.get(seed_id)
        if (
            "latent_narrative" not in review_candidate["generation_metadata"]
            and recovered_narrative is not None
        ):
            review_candidate["generation_metadata"]["latent_narrative"] = (
                recovered_narrative["narrative"]
            )
        validate_candidate_record(review_candidate)
        draft, grounding = revalidate_candidate_grounding(
            review_candidate, catalog=catalog
        )
        relational_gate = (
            relational_resolution_gate(draft) if grounding.passed else None
        )
        final_gate = None
        if grounding.passed and relational_gate is not None and relational_gate.passed:
            async with semaphore:
                final_gate = await verifier.verify_final_conversation(draft)
        passed = (
            grounding.passed
            and relational_gate is not None
            and relational_gate.passed
            and final_gate is not None
            and final_gate.passed
        )
        detail = {
            "source_seed_id": seed_id,
            "candidate_id": candidate["candidate_id"],
            "source_candidate_path": source_for_seed[seed_id],
            "status": "PASS" if passed else "QUARANTINED",
            "grounding_gate": to_builtin(asdict(grounding)),
            "relational_resolution_gate": (
                to_builtin(asdict(relational_gate))
                if relational_gate is not None
                else None
            ),
            "final_query_semantic_gate": (
                to_builtin(asdict(final_gate)) if final_gate is not None else None
            ),
            "reviewed_at": utc_now(),
            "planner_narrative_recovery": recovered_narrative,
        }
        if not passed:
            detail["reason"] = (
                grounding.detail
                if not grounding.passed
                else relational_gate.detail
                if relational_gate is not None and not relational_gate.passed
                else final_gate.detail
            )
            return seed_id, detail, None

        output_candidate = review_candidate
        output_candidate["generation_metadata"]["semantic_revalidation"] = {
            "source_status": "PROJECT_SEMANTIC_GUARD",
            "grounding_gate": to_builtin(asdict(grounding)),
            "relational_resolution_gate": to_builtin(asdict(relational_gate)),
            "final_query_semantic_gate": to_builtin(asdict(final_gate)),
            "reviewed_at": detail["reviewed_at"],
        }
        output_candidate["generation_metadata"]["execution_trace"] = [
            {
                "turn_id": turn.turn_id,
                "intentional_missing": turn.is_intentional_missing,
                "missing_kind": turn.missing_kind,
                "records": [to_builtin(asdict(record)) for record in turn.execution_records],
            }
            for turn in draft.turns
        ]
        validate_candidate_record(output_candidate)
        return seed_id, detail, output_candidate

    results = await asyncio.gather(
        *(inspect(seed_id, candidate) for seed_id, candidate in sorted(by_seed.items()))
    )
    await backend.aclose()

    detail_queue = LockedJsonlQueue(
        args.output / args.details_name, key_field="source_seed_id"
    )
    pass_queue = LockedJsonlQueue(
        args.output / args.pass_name, key_field="candidate_id"
    )
    quarantine_queue = LockedJsonlQueue(
        args.output / args.quarantine_name, key_field="source_seed_id"
    )
    details = [detail for _, detail, _ in results]
    passed_candidates = [candidate for _, _, candidate in results if candidate is not None]
    quarantined = [
        {**detail, "candidate": by_seed[seed_id]}
        for seed_id, detail, candidate in results
        if candidate is None
    ]
    detail_queue.append(details)
    pass_queue.append(passed_candidates)
    quarantine_queue.append(quarantined)

    report = {
        "schema_version": "rods_generator_semantic_revalidation.v1",
        "created_at": utc_now(),
        "source_candidate_files": [str(path.resolve()) for path in args.candidates],
        "raw_response_files": [str(path.resolve()) for path in args.raw_responses],
        "unique_candidates": len(by_seed),
        "passed": len(passed_candidates),
        "quarantined": len(quarantined),
        "final_verifier_calls": sum(
            1 for _, detail, _ in results if detail["final_query_semantic_gate"] is not None
        ),
        "metrics": metrics.snapshot(),
        "results": [
            {
                "source_seed_id": seed_id,
                "candidate_id": detail["candidate_id"],
                "status": detail["status"],
                "reason": detail.get("reason", "all semantic gates passed"),
            }
            for seed_id, detail, _ in results
        ],
    }
    atomic_write_json(args.output / "revalidation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    roots = project_roots()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-responses", type=Path, nargs="*", default=[])
    parser.add_argument("--pass-name", default="revalidated_candidates.jsonl")
    parser.add_argument("--quarantine-name", default="quarantined_candidates.jsonl")
    parser.add_argument("--details-name", default="semantic_gate_details.jsonl")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("TOOLWEAVE_GENERATOR_ENDPOINT", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--model",
        default=str(roots.models_root / "gemma-4-31B-it-manual"),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
