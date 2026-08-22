#!/usr/bin/env python3
"""Generate the final historical Generator replay report from final artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.stat().st_size:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_count(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    passed = 0
    failed = 0
    for line in text.splitlines():
        if " passed" in line:
            token = line.strip().split()[0]
            if token.isdigit():
                passed = max(passed, int(token))
        if " failed" in line:
            token = line.strip().split()[0]
            if token.isdigit():
                failed = max(failed, int(token))
    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    discovery = read_json(root / "00_manifest/discovered_boundary_manifest.json")
    offline = read_json(root / "02_offline_revalidation/revalidation_summary.json")
    replay = read_json(root / "03_replay/replay_summary.json")
    matrix = read_jsonl(root / "08_semantic_audit/validated_candidate_quality_matrix.jsonl")
    suspicious = read_jsonl(root / "08_semantic_audit/unclassified_suspicious_results.jsonl")
    terminals = read_jsonl(root / "04_terminal/terminal_results_all.jsonl")
    statuses = Counter(item["status"] for item in terminals)
    guard_failures = {
        name: sum(not bool(row[name]) for row in matrix)
        for name in (
            "unit_guard",
            "argument_grounding",
            "missing_parameter_validity",
            "execution_semantics",
            "relational_guard",
            "global_semantic",
            "fresh_vm",
            "tool_availability",
            "complexity",
            "judge",
            "training_validator",
        )
    }
    hard_errors = sum(int(row["hard_error_count"]) for row in matrix)
    unresolved = sum(int(row["unresolved_suspicious_count"]) for row in matrix)
    test_files = {
        "precision": root / "09_tests/regression_results.txt",
        "generator": root / "09_tests/full_generator_tests.txt",
        "stage1": root / "09_tests/stage1_full_tests.txt",
    }
    tests = {name: test_count(path) for name, path in test_files.items()}
    frozen_diff = (root / "00_manifest/frozen_training_sha256_diff.txt").read_text().strip()
    pass_conditions = (
        len(terminals) == discovery["unique_real_boundary_events"]
        and statuses.get("SUCCEEDED", 0) + statuses.get("DROPPED", 0) == len(terminals)
        and replay["running_or_ghost_seed_count"] == 0
        and replay["queue_terminal_candidate_id_set_equal"]
        and not any(guard_failures.values())
        and hard_errors == 0
        and unresolved == 0
        and all(failed == 0 and passed > 0 for passed, failed in tests.values())
        and not frozen_diff
    )
    verdict = "PASS" if pass_conditions else "FAIL"

    by_type_lines = []
    for label, values in replay["by_type"].items():
        by_type_lines.append(
            f"| {label} | {values['dispatched']} | {values['validated']} | "
            f"{values['dropped']} | {values['p_valid']:.4f} |"
        )
    failure_lines = [
        f"| `{name}` | {count} |" for name, count in replay["failure_reasons"].items()
    ] or ["| none | 0 |"]
    test_lines = [
        f"| {name} | {passed} | {failed} | {'PASS' if failed == 0 and passed else 'FAIL'} |"
        for name, (passed, failed) in tests.items()
    ]
    known = offline.get("known_precision_patterns", {})
    known_lines = [
        f"| `{seed}` | {item.get('status')} | {item.get('reason')} |"
        for seed, item in sorted(known.items())
    ] or ["| none | N/A | N/A |"]
    guard_lines = [f"| `{name}` | {count} |" for name, count in guard_failures.items()]

    report = f"""# Generator Final Historical Replay Report

## Final Verdict

**{verdict}**

This report is generated from the finalized journal, candidate queue, tracker,
semantic matrices, and test logs under `{root}`. No formal Stage3 training was
started.

## Historical Boundary Discovery

- Raw discovered records: **{discovery['raw_discovered_records']}**
- Artifact duplicate records: **{discovery['artifact_duplicate_records']}**
- Unique real boundary events: **{discovery['unique_real_boundary_events']}**
- Identity: `{discovery['boundary_event_identity']}`

## Offline Historical Candidate Revalidation

- Raw historical validated records: **{offline['raw_historical_validated_records']}**
- Unique training-content candidates: **{offline['unique_training_content_candidates']}**
- Still PASS: **{offline['still_pass']}**
- Quarantined: **{offline['quarantined']}**

| Known pattern | Status | Evidence-driven reason |
|---|---|---|
{chr(10).join(known_lines)}

## Fresh Full Replay

- Dispatched: **{replay['dispatched']}**
- Terminal SUCCEEDED events: **{replay['validated_terminal_events']}**
- Unique validated training contents: **{replay['validated_unique_training_candidates']}**
- Dropped: **{replay['dropped']}**
- End-to-end p_valid: **{replay['p_valid']:.4f}**
- Exact-content duplicates suppressed: **{replay['exact_content_duplicates_suppressed']}**

| BFCL type | Dispatched | Validated | Dropped | p_valid |
|---|---:|---:|---:|---:|
{chr(10).join(by_type_lines)}

## Drop Reasons

| Root bucket | Count |
|---|---:|
{chr(10).join(failure_lines)}

## Precision Certification

| Required validated-candidate property | Failures among accepted candidates |
|---|---:|
{chr(10).join(guard_lines)}
| `semantic_hard_error_count` | {hard_errors} |
| `unresolved_semantics_changing_suspicious` | {unresolved} |

Telemetry-only suspicious observations: **{len(suspicious)}**. They do not
change execution semantics; every `changes_execution_semantics=true` record is
required to be resolved before PASS.

## Queue and Crash Consistency

- All events terminal: `{replay['all_seed_events_terminal']}`
- RUNNING/ghost seeds: `{replay['running_or_ghost_seed_count']}`
- Candidate queue IDs equal terminal SUCCEEDED IDs: `{replay['queue_terminal_candidate_id_set_equal']}`
- Training validator failures: `{replay['training_validator_failures']}`

The durable order remains terminal journal fsync → idempotent candidate queue
append/fsync → tracker terminal state. Exact candidate identity is logically
once-only; physical reconciliation is idempotent.

## Tests and Frozen Training Audit

| Suite | Passed | Failed | Verdict |
|---|---:|---:|---|
{chr(10).join(test_lines)}

Frozen Training hash diff is **{'EMPTY (unchanged)' if not frozen_diff else 'NONEMPTY'}**.
No R_P, A_RODS, MatchTIR, A_local, A_new, PPO/GRPO/KL, rollout, or boundary
algorithm was changed by this Generator certification.

## Limitations

- HIGH_LEVEL → BOTTOM_LEVEL decomposition remains fail-closed because no public
  deterministic mapping is available.
- Generator prompts/guards explicitly marked RECONSTRUCTED or PROJECT_* are not
  claimed as unpublished official RODS source code.
- No minimum yield is imposed; this certification prioritizes precision.
- Formal Stage3 training: **NOT RUN**.
"""
    output = root / "10_reports/GENERATOR_FINAL_HISTORICAL_REPLAY_REPORT.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(json.dumps({"verdict": verdict, "report": str(output), "tests": tests}, indent=2))
    if verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
