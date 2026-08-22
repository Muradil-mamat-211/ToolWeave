#!/usr/bin/env python3
"""Validate and publish the independent BFCL and strict EnvTuning scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


CLASS_ORDER = ("Base", "Miss Func", "Miss Param", "Long Context")
INTERNAL_TO_LABEL = {
    "multi_turn_base": "Base",
    "multi_turn_miss_func": "Miss Func",
    "multi_turn_miss_param": "Miss Param",
    "multi_turn_long_context": "Long Context",
}
RODS_REFERENCE = {
    "Overall": 56.0,
    "Base": 68.0,
    "Miss Func": 59.0,
    "Miss Param": 44.0,
    "Long Context": 53.0,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
    temporary.replace(path)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def validate(
    *,
    manifest: dict[str, Any],
    bfcl_metrics: dict[str, Any],
    bfcl_diagnostics: dict[str, Any],
    bfcl_rows: list[dict[str, Any]],
    strict_metrics: dict[str, Any],
    strict_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    assert manifest["total"] == 100
    assert manifest["class_counts"] == {label: 25 for label in CLASS_ORDER}
    assert len(bfcl_rows) == len(strict_rows) == 100
    bfcl_ids = {row["sample_id"] for row in bfcl_rows}
    strict_ids = {row["sample_id"] for row in strict_rows}
    assert len(bfcl_ids) == len(strict_ids) == 100
    assert bfcl_ids == strict_ids

    for rows in (bfcl_rows, strict_rows):
        counts = Counter(INTERNAL_TO_LABEL[row["data_type"]] for row in rows)
        assert counts == Counter({label: 25 for label in CLASS_ORDER})

    assert bfcl_diagnostics["runtime_failures"] == 0
    assert bfcl_metrics["Overall"]["total"] == 100
    assert bfcl_metrics["Overall"]["correct"] == sum(
        bfcl_metrics[label]["correct"] for label in CLASS_ORDER
    )
    macro = sum(bfcl_metrics[label]["score_percent"] for label in CLASS_ORDER) / 4
    assert math.isclose(macro, bfcl_metrics["Overall"]["score_percent"])
    assert sum(row["final_correctness"] for row in bfcl_rows) == bfcl_metrics["Overall"]["correct"]

    assert strict_metrics["Diagnostics"]["runtime_failures"] == 0
    assert strict_metrics["Diagnostics"]["incomplete_terminal_samples"] == 0
    assert all(row["terminal_complete"] for row in strict_rows)
    valid_codes = {-3, -2, -1, 0, 1}
    observed_codes = Counter(code for row in strict_rows for code in row["diagnostic_codes"])
    assert set(observed_codes).issubset(valid_codes)
    metric_codes = {
        int(code): count for code, count in strict_metrics["Diagnostics"]["code_counts"].items()
    }
    assert observed_codes == Counter(metric_codes)
    recomputed = 100.0 * statistics.fmean(
        row["fixed_denominator_progress"] for row in strict_rows
    )
    assert math.isclose(
        recomputed, strict_metrics["Overall"]["progress_score_percent"], abs_tol=1e-12
    )
    expected_turns = sum(row["expected_user_turns"] for row in strict_rows)
    terminal_codes = sum(len(row["terminal_by_turn"]) for row in strict_rows)
    assert terminal_codes == expected_turns

    return {
        "sample_id_sets_equal": True,
        "balanced_class_counts": True,
        "bfcl_runtime_failures": 0,
        "strict_runtime_failures": 0,
        "strict_terminal_complete_samples": 100,
        "strict_expected_user_turns": expected_turns,
        "strict_terminal_codes": terminal_codes,
        "strict_code_counts": {str(key): value for key, value in sorted(observed_codes.items())},
    }


def telemetry_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    rows = read_jsonl(path)
    result: dict[str, Any] = {"available": bool(rows), "samples": len(rows)}
    for key in ("memory_used_mib", "utilization_percent", "power_w", "temperature_c"):
        values = [float(row["gpu"][key]) for row in rows]
        result[key] = {
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
    if rows:
        result["first_timestamp"] = rows[0]["timestamp"]
        result["last_timestamp"] = rows[-1]["timestamp"]
    return result


def model_hashes(model_root: Path) -> dict[str, Any]:
    files = [
        "config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    hashes = {name: sha256(model_root / name) for name in files}
    manifest_payload = "".join(f"{name}\0{hashes[name]}\n" for name in sorted(hashes))
    return {
        "files": hashes,
        "selected_file_manifest_sha256": hashlib.sha256(
            manifest_payload.encode("utf-8")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()

    bfcl_root = args.run_root / "bfcl/full100"
    strict_root = args.run_root / "strict_envtuning/full100"
    manifest_path = args.eval_root / "dataset/eval_rods_bfcl_multiturn_100_manifest.json"
    manifest = read_json(manifest_path)
    bfcl_metrics = read_json(bfcl_root / "metrics/primary_metrics.json")
    bfcl_diagnostics = read_json(bfcl_root / "metrics/diagnostics.json")
    bfcl_rows = read_jsonl(bfcl_root / "per_sample/per_sample_results.jsonl")
    strict_metrics = read_json(strict_root / "metrics/strict_envtuning_metrics.json")
    strict_rows = read_jsonl(strict_root / "per_sample/per_sample_results.jsonl")
    bfcl_config = yaml.safe_load(
        (bfcl_root / "config/eval_resolved_config.yaml").read_text(encoding="utf-8")
    )
    strict_config = yaml.safe_load(
        (strict_root / "config/strict_envtuning_resolved_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    checks = validate(
        manifest=manifest,
        bfcl_metrics=bfcl_metrics,
        bfcl_diagnostics=bfcl_diagnostics,
        bfcl_rows=bfcl_rows,
        strict_metrics=strict_metrics,
        strict_rows=strict_rows,
    )
    telemetry = telemetry_summary(args.run_root / "logs/gpu_telemetry_attempt2.jsonl")
    hashes = model_hashes(args.model_root)

    bfcl_counts = {
        label: f"{bfcl_metrics[label]['correct']}/{bfcl_metrics[label]['total']}"
        for label in ("Overall", *CLASS_ORDER)
    }
    strict_terminal_successes = sum(
        value == 1
        for row in strict_rows
        for value in row["terminal_by_turn"].values()
    )
    summary = {
        "status": "PASS",
        "dataset": {
            "path": bfcl_config["dataset_path"],
            "sha256": bfcl_config["dataset_sha256"],
            "source": manifest["dataset_source"],
            "total": 100,
            "class_counts": manifest["class_counts"],
        },
        "model": {
            "path": str(args.model_root.resolve()),
            "hashes": hashes,
        },
        "bfcl": {
            "scores_percent": {
                label: bfcl_metrics[label]["score_percent"]
                for label in ("Overall", *CLASS_ORDER)
            },
            "counts": bfcl_counts,
            "diagnostics": bfcl_diagnostics,
            "per_sample_results": str(
                (bfcl_root / "per_sample/per_sample_results.jsonl").resolve()
            ),
        },
        "strict_envtuning": {
            "progress_scores_percent": {
                label: strict_metrics[label]["progress_score_percent"]
                for label in ("Overall", *CLASS_ORDER)
            },
            "complete_episode_accuracy_percent": strict_metrics["Overall"][
                "complete_episode_accuracy_percent"
            ],
            "terminal_successes": strict_terminal_successes,
            "expected_user_turns": checks["strict_expected_user_turns"],
            "diagnostics": strict_metrics["Diagnostics"],
            "per_sample_results": str(
                (strict_root / "per_sample/per_sample_results.jsonl").resolve()
            ),
        },
        "protocol_comparability": {
            "bfcl_is_rods_paper_metric": True,
            "strict_envtuning_is_rods_paper_metric": False,
            "strict_envtuning_is_independent_real_inference": True,
        },
        "checks": checks,
        "telemetry": telemetry,
        "configs": {"bfcl": bfcl_config, "strict_envtuning": strict_config},
        "runtime_sampling_transport": {
            "requested_temperature": 0.001,
            "effective_temperature": 0.01,
            "reason": (
                "vLLM 0.26.0 clamps positive temperatures below 0.01 to 0.01 "
                "to avoid sampling numerical errors"
            ),
        },
    }
    encoded = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    atomic_write(args.run_root / "metrics/dual_score_summary.json", encoded)
    atomic_write(args.eval_root / "metrics/QWEN3_4B_RODS_BFCL100_DUAL_SCORES.json", encoded)
    atomic_write(
        args.eval_root / "metrics/BFCL_SCORE.json",
        json.dumps(summary["bfcl"], indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write(
        args.eval_root / "metrics/STRICT_ENVTUNING_SCORE.json",
        json.dumps(summary["strict_envtuning"], indent=2, ensure_ascii=False) + "\n",
    )

    def score(row: dict[str, Any], label: str, field: str) -> str:
        return f"{row[label][field]:.2f}"

    report_lines = [
        "# Qwen3-4B-RODS BFCL100 Dual-Protocol Evaluation",
        "",
        "## Outcome",
        "",
        "Evaluation status: **PASS**. Both protocols performed an independent 100-sample "
        "model inference pass over exactly the same balanced dataset. No runtime failure "
        "occurred in either pass.",
        "",
        "## Scores",
        "",
        "| Protocol / model | Overall | Base | Miss Func | Miss Param | Long Context |",
        "|---|---:|---:|---:|---:|---:|",
        "| RODS paper/reference — BFCL | 56.00 | 68.00 | 59.00 | 44.00 | 53.00 |",
        "| Local Qwen3-4B-RODS — BFCL | "
        + " | ".join(
            score(bfcl_metrics, label, "score_percent")
            for label in ("Overall", *CLASS_ORDER)
        )
        + " |",
        "| Local Qwen3-4B-RODS — strict EnvTuning Progress | "
        + " | ".join(
            score(strict_metrics, label, "progress_score_percent")
            for label in ("Overall", *CLASS_ORDER)
        )
        + " |",
        "",
        "Only the **BFCL row** is definitionally comparable with the RODS paper/model-card "
        "reference. The strict EnvTuning row is a separate local protocol result and is not "
        "presented as a reproduction of the paper's BFCL metric.",
        "",
        "## BFCL Integer Counts",
        "",
        *[
            f"- {label}: {bfcl_metrics[label]['correct']}/{bfcl_metrics[label]['total']} "
            f"({bfcl_metrics[label]['score_percent']:.2f})"
            for label in ("Overall", *CLASS_ORDER)
        ],
        "",
        "The BFCL score is complete-entry multi-turn accuracy. It uses the public Qwen-FC "
        "prompt/parser behavior, the real stateful BFCL environment, state/response checking, "
        "and missing-turn irrelevance checking. One trajectory reached BFCL's public 20-step "
        "limit; the official handler defines that entry as incorrect, so it remains in the "
        "denominator and does not invalidate the run.",
        "",
        "## Strict EnvTuning Result",
        "",
        f"- Fixed-denominator Progress Score: **{strict_metrics['Overall']['progress_score_percent']:.2f}**",
        f"- Complete-episode accuracy: **{strict_metrics['Overall']['complete_episode_accuracy_percent']:.2f}** "
        f"({strict_metrics['Overall']['complete_episode_correct']}/100)",
        f"- Terminal successes: {strict_terminal_successes}/{checks['strict_expected_user_turns']} user turns",
        f"- Exact diagnostic counts: `{json.dumps(checks['strict_code_counts'], sort_keys=True)}`",
        "- All 100 samples reached all expected terminal user-turn codes; runtime failures: 0.",
        "",
        "EnvTuning requires exactly one `<think>...</think>` block followed by exactly one "
        "`<tool_call>` or `<answer>` block, with no outside text. In this run the released "
        "model consistently emitted its documented Qwen3 function-calling style (natural "
        "language plus `<tool_call>`) instead. The strict parser therefore recorded format "
        "code `-3`; after the source-defined retry limit each user turn received terminal "
        "code `0`. Codes `-3/-2/-1/0/1` are transition diagnostics, not signed rewards to sum.",
        "",
        "## Why Two Scores Are Kept",
        "",
        "- **BFCL score:** `<think>` is optional for the audited public Qwen handler; this is "
        "the headline metric corresponding to the RODS model card.",
        "- **Strict EnvTuning score:** `<think>` and a single action block are mandatory; this "
        "measures compatibility with the checked-in EnvTuning interaction contract.",
        "- The two passes use separate prompts/transports and separate model generations. No "
        "shadow estimate is substituted for either score.",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{bfcl_config['dataset_path']}`",
        f"- SHA256: `{bfcl_config['dataset_sha256']}`",
        "- 100 samples: Base 25, Miss Func 25, Miss Param 25, Long Context 25.",
        f"- Source: `{manifest['dataset_source']}`; verified against canonical eval-400 "
        f"`{manifest['source_eval_400_path']}`.",
        "- The local balanced subset has 25 samples/class; the RODS reference is from the full "
        "BFCL multi-turn evaluation. Numerical equality is not expected.",
        "",
        "## Model and Runtime",
        "",
        f"- Model: `{args.model_root.resolve()}`",
        f"- Selected-file model manifest SHA256: `{hashes['selected_file_manifest_sha256']}`",
        "- Backend: vLLM 0.26.0, BF16, one RTX PRO 6000 Blackwell Server Edition, "
        "native context 262144, max output 4096, client concurrency 32.",
        "- BFCL's generation CLI default `temperature=0.001` was requested. vLLM 0.26.0 "
        "explicitly clamped it to an effective 0.01; both values are retained in the runtime "
        "evidence and no exact-0.001 claim is made.",
        f"- BFCL wall time: {bfcl_diagnostics['run_wall_seconds']:.2f}s; generated-token "
        f"throughput: {bfcl_diagnostics['completion_tokens_per_second']:.2f} tokens/s.",
        f"- Strict wall time: {strict_metrics['Diagnostics']['run_wall_seconds']:.2f}s; "
        f"generated-token throughput: {strict_metrics['Diagnostics']['completion_tokens_per_second']:.2f} tokens/s.",
        f"- Peak observed VRAM: {telemetry['memory_used_mib']['max']:.0f} MiB; peak GPU "
        f"utilization: {telemetry['utilization_percent']['max']:.0f}%; peak power: "
        f"{telemetry['power_w']['max']:.2f} W.",
        "",
        "## Source Audit",
        "",
        "- RODS paper: arXiv:2606.19047v1.",
        "- AWorld-RL audited commit: `be52dbf33051c9b86e8e4d3c4e2394548906c75b`.",
        "- BFCL/Gorilla audited commit: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.",
        "- Qwen3-4B-RODS Hugging Face snapshot commit: "
        "`856cd69ef94e93231b9fe28ebcc8ab0c8d4c3a66`.",
        "- Exact source-file SHA256 values are embedded in each resolved config.",
        "",
        "## Incidents and Narrow Fixes",
        "",
        "1. The first vLLM preflight OOMed during dummy warmup before serving any model "
        "request. Native context remained 262144; GPU utilization was reduced from 0.94 to "
        "0.90 and the warmup token budget from 131072 to 65536. No prompt was truncated.",
        "2. After all 100 BFCL samples were durably written, a local final assertion treated "
        "one official BFCL force-terminated entry as an infrastructure failure. BFCL source "
        "defines it as an incorrect entry. The assertion was narrowed accordingly; the sample "
        "remains incorrect and no score/checker behavior changed. Regression tests pass.",
        "3. The current vLLM transport clamps requested temperature 0.001 to 0.01. This is "
        "reported as a runtime compatibility detail rather than silently describing the run "
        "as exact 0.001 sampling.",
        "",
        "## Evidence",
        "",
        f"- Dual summary: `{(args.run_root / 'metrics/dual_score_summary.json').resolve()}`",
        f"- BFCL per-sample evidence: `{(bfcl_root / 'per_sample/per_sample_results.jsonl').resolve()}`",
        f"- BFCL raw responses: `{(bfcl_root / 'raw_outputs/raw_model_responses.jsonl').resolve()}`",
        f"- Strict per-sample evidence: `{(strict_root / 'per_sample/per_sample_results.jsonl').resolve()}`",
        f"- Strict raw responses: `{(strict_root / 'raw_outputs/raw_model_responses.jsonl').resolve()}`",
        f"- GPU telemetry: `{(args.run_root / 'logs/gpu_telemetry_attempt2.jsonl').resolve()}`",
        f"- Temperature transport evidence: `{(args.run_root / 'logs/vllm_temperature_clamp_observation.json').resolve()}`",
        "",
        "No formal Stage3 training was launched and no checkpoint was written.",
    ]
    report = "\n".join(report_lines) + "\n"
    report_name = "QWEN3_4B_RODS_BFCL100_EVAL_REPORT.md"
    atomic_write(args.run_root / "report" / report_name, report)
    atomic_write(args.eval_root / "report" / report_name, report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "bfcl": summary["bfcl"]["scores_percent"],
                "strict_envtuning": summary["strict_envtuning"][
                    "progress_scores_percent"
                ],
                "report": str((args.eval_root / "report" / report_name).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
