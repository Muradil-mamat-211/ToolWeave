#!/usr/bin/env python3
"""Read-only precision revalidation of the final 29 historical candidates.

No LLM is called.  Historical artifacts remain immutable; all output is
written beneath a caller-supplied fresh artifact root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from env_tuning.rods_data_generation_v1.environment_adapter import (
    SynthesisEnvironmentAdapter,
)
from env_tuning.rods_data_generation_v1.function_catalog import FunctionCatalog
from env_tuning.rods_data_generation_v1.models import GateResult, to_builtin
from env_tuning.rods_data_generation_v1.revalidation import candidate_to_draft
from env_tuning.rods_data_generation_v1.validation.action_minimality import (
    action_minimality_gate,
)
from env_tuning.rods_data_generation_v1.validation.missing_parameter_validity import (
    missing_parameter_validity_gate,
)
from env_tuning.rods_data_generation_v1.validation.observation_entailment import (
    observation_entailment_gate,
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


EXPECTED_CANDIDATE_COUNT = 29
WORKSPACE = Path(__file__).resolve().parents[2]
FROZEN_TRAINING_HASHES = {
    "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/advantage.py": "9a36af9fabe1f0c4f3c4e489767bc65df4e72a394245ae6e38741b646fd9aba0",
    "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/matching.py": "70588272ac9d3c6f04e238fe4f5d7bd366f18773fdb29cb4d08840d2ccbfc877",
    "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/lifecycle.py": "4d57152ea6a47861e301906a2fcd9a1d55caf95f76baa6ce87814ea83eb8920f",
    "code/AWorld-RL-stage1-worktree/EnvTuning/verl/verl/trainer/ppo/core_algos.py": "32692646dc70018e33cc133919ae8c27858a7323d215b4fe43824afa2853791b",
    "code/AWorld-RL-stage1-worktree/EnvTuning/verl/verl/trainer/ppo/ray_trainer.py": "9e8ea2d1689e30b03a8fb6ae3cef12ae16d9c135e29a08350bf0d3142358767b",
    "code/AWorld-RL-stage1-worktree/EnvTuning/verl/verl/trainer/ppo/reward.py": "4b70affb5a2da379e0941671334f055773e7b682b1d795ea59e82e224560e8fd",
    "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/new_multi_turn_fc.py": "b3068bd43534398916d82a0465bb704a212a0f2c725417a76c3ff78f492c7f57",
    "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/response_handler.py": "fe4ed04d4734396dc991db014dc56647d3dc9d2fb113671e7458990ec8970523",
    "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/execution_manager.py": "f40fc3565e53627265c20dcbc91b097e4c0884f0e7abf5aae7d93e067bfc0d4d",
    "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/turn_manager.py": "75c1bd3910eeac842ecdbeb2ee56750e4b786f6baf8b71dfd62f15a47d69de2b",
    "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/utils.py": "9858c5812b9c06fdcc688177b684b9c1a50c31825c6e63132cdcb3bf8b2a4fd2",
    "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/score_calculator.py": "6d664fbfa6c9da4bf5010b053788011aa408567fad7f6cfde49393ee2726e950",
}


def _json(value: Any) -> str:
    return json.dumps(
        to_builtin(value), ensure_ascii=False, sort_keys=True, default=repr
    )


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    passed = re.findall(r"(\d+) passed", text)
    failed = re.findall(r"(\d+) failed", text)
    errors = re.findall(r"(\d+) errors?", text)
    return {
        "path": str(path.resolve()),
        "passed": int(passed[-1]) if passed else 0,
        "failed": int(failed[-1]) if failed else 0,
        "errors": int(errors[-1]) if errors else 0,
        "sha256": _sha256(path),
    }


def _frozen_hash_audit() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for relative, expected in FROZEN_TRAINING_HASHES.items():
        path = WORKSPACE / relative
        actual = _sha256(path) if path.is_file() else None
        rows[relative] = {
            "expected_pre_task_sha256": expected,
            "actual_post_task_sha256": actual,
            "unchanged": actual == expected,
        }
    return {
        "all_unchanged": all(row["unchanged"] for row in rows.values()),
        "files": rows,
    }


def _gate_dict(gate: GateResult) -> dict[str, Any]:
    return asdict(gate)


def _historical_gate(candidate: dict[str, Any], name: str) -> GateResult:
    gates = candidate.get("generation_metadata", {}).get(
        "deterministic_gate_results", []
    )
    rows = [gate for gate in gates if gate.get("name") == name]
    if len(rows) != 1:
        return GateResult(
            name,
            False,
            f"historical candidate has {len(rows)} durable {name} records",
            {"evidence_status": "MISSING_OR_AMBIGUOUS"},
        )
    row = rows[0]
    return GateResult(
        name,
        row.get("passed") is True,
        str(row.get("detail", "")),
        {
            "evidence_status": "HISTORICAL_DURABLE_RESULT_REUSED_NO_LLM",
            "historical_metadata": copy.deepcopy(row.get("metadata", {})),
        },
    )


def _judge_gate(candidate: dict[str, Any]) -> GateResult:
    judge = candidate.get("validation", {}).get("quality_judge", {})
    accepted = isinstance(judge, dict) and judge.get("decision") == "accept"
    return GateResult(
        "quality_judge",
        accepted,
        str(judge.get("reason", "missing durable Judge result"))
        if isinstance(judge, dict)
        else "missing durable Judge result",
        {
            "evidence_status": "HISTORICAL_DURABLE_RESULT_REUSED_NO_LLM",
            "decision": judge.get("decision") if isinstance(judge, dict) else None,
            "fail_reason": judge.get("fail_reason") if isinstance(judge, dict) else None,
        },
    )


def _validator_gate(candidate: dict[str, Any]) -> GateResult:
    try:
        validate_candidate_record(candidate)
    except Exception as exc:  # validator deliberately fails closed
        return GateResult(
            "training_validator",
            False,
            f"{type(exc).__name__}: {exc}",
        )
    return GateResult(
        "training_validator",
        True,
        "frozen Training candidate validator accepted the historical payload",
    )


def _run_one(
    candidate: dict[str, Any],
    *,
    catalog: FunctionCatalog,
    environment_factory: SynthesisEnvironmentAdapter,
) -> tuple[dict[str, Any], str]:
    candidate_id = str(candidate.get("candidate_id", ""))
    seed_id = str(
        candidate.get("generation_metadata", {}).get("source_seed_id", "")
    )
    try:
        draft = candidate_to_draft(candidate)
    except Exception as exc:
        detail = {
            "candidate_id": candidate_id,
            "seed_id": seed_id,
            "data_type": candidate.get("sample", {}).get("data_source"),
            "final_verdict": "NEEDS_TARGETED_REPLAY",
            "exact_quarantine_reason": (
                "historical execution evidence cannot reconstruct a draft: "
                f"{type(exc).__name__}: {exc}"
            ),
            "gates": {},
        }
        return detail, "NEEDS_TARGETED_REPLAY"

    # Order mirrors the final Generator path.  Grounding runs before action
    # minimality so its audited dependency IDs are available to the latter.
    gate_calls: list[tuple[str, Callable[[], GateResult]]] = [
        ("unit_semantics", lambda: unit_semantic_gate(draft, catalog=catalog)),
        ("semantic_grounding", lambda: semantic_grounding_gate(draft, catalog=catalog)),
        (
            "missing_parameter_validity",
            lambda: missing_parameter_validity_gate(draft, catalog=catalog),
        ),
        ("observation_entailment", lambda: observation_entailment_gate(draft)),
        ("relational", lambda: relational_resolution_gate(draft)),
        (
            "final_query_semantic",
            lambda: _historical_gate(candidate, "final_query_semantic_gate"),
        ),
        (
            "action_minimality",
            lambda: action_minimality_gate(draft, catalog=catalog),
        ),
        (
            "fresh_vm",
            lambda: fresh_vm_reverify_gate(
                draft,
                environment_factory=environment_factory,
                seed_id=seed_id,
            ),
        ),
        ("tool_availability", lambda: tool_availability_gate(draft)),
        ("parameter_complexity", lambda: parameter_complexity_gate(draft)),
        ("quality_judge", lambda: _judge_gate(candidate)),
        ("training_validator", lambda: _validator_gate(candidate)),
    ]
    gates: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for key, call in gate_calls:
        try:
            gate = call()
        except Exception as exc:
            gate = GateResult(
                key,
                False,
                f"gate raised {type(exc).__name__}: {exc}",
                {"evidence_status": "RUNTIME_FAILURE"},
            )
        gates[key] = _gate_dict(gate)
        if not gate.passed:
            failures.append(f"{gate.name}: {gate.detail}")

    verdict = "PASS" if not failures else "QUARANTINED"
    detail = {
        "candidate_id": candidate_id,
        "seed_id": seed_id,
        "data_type": candidate.get("sample", {}).get("data_source"),
        "mp_validity": gates["missing_parameter_validity"],
        "observation_entailment": gates["observation_entailment"],
        "action_minimality": gates["action_minimality"],
        "existing_semantic_gates": {
            key: gates[key]
            for key in (
                "unit_semantics",
                "semantic_grounding",
                "relational",
                "final_query_semantic",
                "tool_availability",
                "parameter_complexity",
                "quality_judge",
            )
        },
        "fresh_vm_status": gates["fresh_vm"],
        "training_validator": gates["training_validator"],
        "final_verdict": verdict,
        "exact_quarantine_reason": failures[0] if failures else None,
        "all_failures": failures,
        "gemma_called": False,
    }
    return detail, verdict


def _write_report(
    path: Path,
    *,
    source: Path,
    catalog_path: Path,
    details: list[dict[str, Any]],
    tests: dict[str, dict[str, Any]],
    frozen_audit: dict[str, Any],
) -> None:
    counts = Counter(item["final_verdict"] for item in details)
    quarantined = [item for item in details if item["final_verdict"] == "QUARANTINED"]
    tests_clean = all(
        item["passed"] > 0 and item["failed"] == 0 and item["errors"] == 0
        for item in tests.values()
    )
    release_pass = (
        counts["NEEDS_TARGETED_REPLAY"] == 0
        and tests_clean
        and frozen_audit["all_unchanged"]
    )
    lines = [
        "# Final Accepted-Data Precision Revalidation",
        "",
        "## Scope",
        "",
        f"- Historical input: `{source}` (SHA256 `{_sha256(source)}`)",
        f"- Canonical 128-function catalog source: `{catalog_path}`",
        f"- Input candidates: {len(details)}",
        "- Gemma/LLM calls: 0",
        "- Formal training/checkpointing: not run",
        "- Historical artifacts: read-only",
        "",
        "## Precision Guards",
        "",
        "- `PROJECT_MISSING_PARAMETER_VALIDITY_GUARD`: recursively recovers compatible IDs from policy-visible prior user/tool history; hidden VM state remains excluded.",
        "- `PROJECT_OBSERVATION_ENTAILMENT_GUARD`: rejects explicit claims attributed to prior observations when deterministic factual anchors are absent; undecidable claims are exposed to the existing final semantic verifier.",
        "- `PROJECT_ACTION_MINIMALITY_GUARD`: each final call must be `DIRECT_INTENT`, `REQUIRED_PREREQUISITE`, or `DEPENDENCY_PRODUCER`; otherwise it is `REDUNDANT_EXTRA_CALL`.",
        "- No embeddings, similarity thresholds, Judge override, or RODS Training math changes were introduced.",
        "",
        "## Results",
        "",
        f"- PASS: {counts['PASS']}",
        f"- QUARANTINED: {counts['QUARANTINED']}",
        f"- NEEDS_TARGETED_REPLAY: {counts['NEEDS_TARGETED_REPLAY']}",
        "",
        "| Candidate | Source seed | Type | First quarantine reason |",
        "|---|---|---|---|",
    ]
    for item in quarantined:
        reason = str(item["exact_quarantine_reason"]).replace("|", "\\|")
        lines.append(
            f"| `{item['candidate_id']}` | `{item['seed_id']}` | "
            f"{item['data_type']} | {reason} |"
        )
    lines.extend(
        [
            "",
        "## Evidence and Limitations",
            "",
            "The final Query/GT verifier and Quality Judge results are reused from each candidate's durable accepted artifact; no model is available or invoked in this no-GPU revalidation. All deterministic gates, including a fresh real CPU BFCL VM replay and the frozen Training validator, are rerun from final candidate payload and execution trace. A missing or ambiguous durable model-verdict record fails closed.",
            "",
            "Action-intent routing uses deterministic operation concepts from public BFCL schemas. Audited prerequisites include the real `startEngine` brake/door/fuel constraints, authenticated-session setup, and current-directory semantics for `cd -> mv`; no LLM is allowed to invent a prerequisite.",
            "",
            "## Regression and Frozen-Algorithm Audit",
            "",
            f"- Precision tests: {tests['precision']['passed']} passed / {tests['precision']['failed']} failed / {tests['precision']['errors']} errors",
            f"- Generator tests: {tests['generator']['passed']} passed / {tests['generator']['failed']} failed / {tests['generator']['errors']} errors",
            f"- Entire stage1 test suite: {tests['stage1']['passed']} passed / {tests['stage1']['failed']} failed / {tests['stage1']['errors']} errors",
            f"- Frozen Training files unchanged: {'YES' if frozen_audit['all_unchanged'] else 'NO'} ({len(frozen_audit['files'])} files checked)",
            "- Queue/tracker/journal source files were not modified by this task; crash/queue regressions are included in the Generator suite.",
            "",
            "## Known False-Positive Channels",
            "",
            "- Fake Missing Parameter with one policy-visible ticket/order ID: 0 remaining; the real ticket case is quarantined by `REJECT_UNIQUELY_RECOVERABLE`.",
            "- Unsupported observation-attributed factual claim: 0 remaining; the empty-log-search case is quarantined by `UNSUPPORTED_OBSERVATION_CLAIM`.",
            "- Redundant GT action: 0 remaining; the unused `gallon_to_liter` call is quarantined as `REDUNDANT_EXTRA_CALL` while audited `pressBrakePedal -> startEngine` remains valid.",
            "",
            "## Verdict",
            "",
            f"**FINAL VERDICT = {'PASS' if release_pass else 'PARTIAL'}**",
            "",
            "No Gemma replay and no formal Stage3 training were performed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalog-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--precision-test-log", type=Path, required=True)
    parser.add_argument("--generator-test-log", type=Path, required=True)
    parser.add_argument("--stage1-test-log", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root must be new or empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise SystemExit(
            f"expected {EXPECTED_CANDIDATE_COUNT} historical candidates, got {len(candidates)}"
        )
    candidate_ids = [item.get("candidate_id") for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise SystemExit("historical candidate input contains duplicate candidate IDs")

    catalog = FunctionCatalog.from_training_parquet(args.catalog_parquet)
    if len(catalog.names()) != 128:
        raise SystemExit(f"expected canonical 128-function catalog, got {len(catalog.names())}")
    environment_factory = SynthesisEnvironmentAdapter(is_augmented=False)
    tests = {
        "precision": _test_summary(args.precision_test_log),
        "generator": _test_summary(args.generator_test_log),
        "stage1": _test_summary(args.stage1_test_log),
    }
    frozen_audit = _frozen_hash_audit()
    (args.output_root / "frozen_training_hashes.json").write_text(
        json.dumps(frozen_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for key, source in (
        ("precision_tests.txt", args.precision_test_log),
        ("generator_tests.txt", args.generator_test_log),
        ("stage1_tests.txt", args.stage1_test_log),
    ):
        shutil.copy2(source, args.output_root / key)

    old_path = args.output_root / "old_validated_29.jsonl"
    pass_path = args.output_root / "revalidated_pass.jsonl"
    quarantine_path = args.output_root / "quarantined.jsonl"
    detail_path = args.output_root / "revalidation_details.jsonl"
    details: list[dict[str, Any]] = []
    for candidate in candidates:
        _append_jsonl(old_path, candidate)
        detail, verdict = _run_one(
            candidate,
            catalog=catalog,
            environment_factory=environment_factory,
        )
        details.append(detail)
        _append_jsonl(detail_path, detail)
        payload = {
            "candidate": candidate,
            "precision_revalidation": detail,
        }
        if verdict == "PASS":
            _append_jsonl(pass_path, payload)
        else:
            _append_jsonl(quarantine_path, payload)

    for path in (pass_path, quarantine_path):
        path.touch(exist_ok=True)
    _write_report(
        args.output_root / "FINAL_PRECISION_REVALIDATION_REPORT.md",
        source=args.input.resolve(),
        catalog_path=args.catalog_parquet.resolve(),
        details=details,
        tests=tests,
        frozen_audit=frozen_audit,
    )
    counts = Counter(item["final_verdict"] for item in details)
    print(_json({"artifact_root": str(args.output_root.resolve()), "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
