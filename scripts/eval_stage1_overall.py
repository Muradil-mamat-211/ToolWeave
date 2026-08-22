#!/usr/bin/env python3
"""Format-only evaluation for the base Qwen3 model and the Stage 1 model.

This script never trains or modifies model weights.  The ``all`` mode launches
one subprocess per model so vLLM can release its two-GPU tensor-parallel
runtime before the other model is loaded.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRIC_KEYS = (
    "valid_tool_call_block",
    "valid_json_parse",
    "valid_function_name",
    "valid_arguments_object",
    "required_arguments_present",
    "malformed_output",
    "direct_answer_without_tool",
)
OUTPUT_MODE_ORDER = ("deterministic", "stochastic")
DATASET_ORDER = ("train_format", "heldout_base", "heldout_mixed")
MODEL_ORDER = ("base", "stage1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--stage1_model", type=Path, required=True)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--bfcl_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_path", type=Path, default=None)
    parser.add_argument("--max_train_eval", type=int, default=100)
    parser.add_argument("--max_heldout_base_eval", type=int, default=200)
    parser.add_argument("--max_mixed_eval", type=int, default=150)
    parser.add_argument("--stochastic_n", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.94)
    parser.add_argument("--model_to_run", choices=("all", "base", "stage1"), default="all")
    parser.add_argument("--run_tag", choices=("smoke", "full"), default="full")
    parser.add_argument("--summarize_only", action="store_true")
    return parser.parse_args()


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def method_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1]


def load_tool_docs(bfcl_dir: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for path in sorted((bfcl_dir / "multi_turn_func_doc").glob("*.json")):
        for row in read_json_lines(path):
            name = row.get("name")
            if name:
                docs[str(name)] = row
    return docs


def load_ground_truth(bfcl_dir: Path, source_name: str) -> dict[str, Any]:
    path = bfcl_dir / "possible_answer" / source_name
    if not path.is_file():
        return {}
    return {str(row["id"]): row.get("ground_truth") for row in read_json_lines(path) if row.get("id")}


def first_gold_tool_call(ground_truth: Any) -> dict[str, Any] | None:
    """Parse the first keyword-only BFCL reference call as an evaluation proxy."""
    if not isinstance(ground_truth, list):
        return None
    for turn in ground_truth:
        if not isinstance(turn, list):
            continue
        for call_text in turn:
            if not isinstance(call_text, str):
                continue
            try:
                node = ast.parse(call_text, mode="eval").body
            except SyntaxError:
                continue
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
                continue
            arguments: dict[str, Any] = {}
            try:
                for keyword in node.keywords:
                    if keyword.arg is None:
                        raise ValueError("starred keyword argument")
                    arguments[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                continue
            return {"name": node.func.id, "arguments": arguments, "source": call_text}
    return None


def build_record(
    raw: dict[str, Any],
    tools_by_name: dict[str, dict[str, Any]],
    ground_truth: dict[str, Any],
    dataset: str,
    source_file: str,
    render_prompt: Any,
) -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for qualified in raw.get("path", []):
        name = method_name(str(qualified))
        if name in names:
            continue
        names.add(name)
        tools.append(
            tools_by_name.get(
                name,
                {
                    "name": name,
                    "description": "Schema absent from local BFCL documentation.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            )
        )
    gold = ground_truth.get(str(raw.get("id")))
    return {
        "id": str(raw.get("id")),
        "dataset": dataset,
        "source_file": source_file,
        "prompt": render_prompt(raw.get("question", []), tools),
        "tools": tools,
        "gold": {"ground_truth": gold, "first_tool_call": first_gold_tool_call(gold)},
        "raw": raw,
    }


def interleave_harder_sets(groups: list[tuple[str, list[dict[str, Any]]]], total: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    indices = {name: 0 for name, _ in groups}
    while len(selected) < total:
        advanced = False
        for name, rows in groups:
            index = indices[name]
            if index >= len(rows) or index >= 50 or len(selected) >= total:
                continue
            selected.append(rows[index])
            indices[name] += 1
            advanced = True
        if not advanced:
            break
    return selected


def prepare_eval_sets(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    local_dir = Path(__file__).parents[1] / "code" / "rods_stage1_local"
    sys.path.insert(0, str(local_dir))
    from prompt_template import render_prompt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_json_lines(args.train_file)
    train_ids = {str(row["id"]) for row in train_rows}
    base_path = args.bfcl_dir / "BFCL_v3_multi_turn_base.json"
    base_raw = read_json_lines(base_path)
    base_ground_truth = load_ground_truth(args.bfcl_dir, base_path.name)
    docs = load_tool_docs(args.bfcl_dir)
    heldout_base = [
        build_record(row, docs, base_ground_truth, "heldout_base", base_path.name, render_prompt)
        for row in base_raw
        if str(row.get("id")) not in train_ids
    ]
    if not heldout_base:
        raise ValueError("No held-out Base samples remain after excluding Stage 1 training IDs.")

    harder_files = (
        ("missing_function", "BFCL_v3_multi_turn_miss_func.json"),
        ("missing_parameter", "BFCL_v3_multi_turn_miss_param.json"),
        ("long_context", "BFCL_v3_multi_turn_long_context.json"),
    )
    harder_groups: list[tuple[str, list[dict[str, Any]]]] = []
    for label, filename in harder_files:
        source = args.bfcl_dir / filename
        raw_rows = read_json_lines(source)
        ground_truth = load_ground_truth(args.bfcl_dir, filename)
        records = [
            build_record(row, docs, ground_truth, "heldout_mixed", f"{label}:{filename}", render_prompt)
            for row in raw_rows
        ]
        harder_groups.append((label, records))
    heldout_mixed = interleave_harder_sets(harder_groups, total=150)
    if len(heldout_mixed) < 100:
        raise ValueError(f"Only {len(heldout_mixed)} unambiguous harder records were available.")

    # Retain the exact training prompt artifact rather than re-rendering it.
    normalized_train = []
    for row in train_rows:
        row = dict(row)
        row["dataset"] = "train_format"
        row.setdefault("gold", {})["first_tool_call"] = first_gold_tool_call(row.get("gold", {}).get("ground_truth"))
        normalized_train.append(row)

    write_jsonl(args.out_dir / "heldout_base_eval.jsonl", heldout_base)
    write_jsonl(args.out_dir / "heldout_mixed_harder_eval.jsonl", heldout_mixed)
    candidate_report = args.out_dir / "bfcl_candidate_files.md"
    candidate_report.write_text(
        "# BFCL Evaluation Candidate Files\n\n"
        "| Evaluation set | Source file | Available | Selected |\n"
        "| --- | --- | ---: | ---: |\n"
        f"| Stage 1 train format | `{args.train_file}` | {len(normalized_train)} | {len(normalized_train)} |\n"
        f"| Held-out Base | `BFCL_v3_multi_turn_base.json` | {len(base_raw) - len(train_ids)} after excluding training IDs | {len(heldout_base)} |\n"
        f"| Missing Function | `BFCL_v3_multi_turn_miss_func.json` | {len(harder_groups[0][1])} | 50 |\n"
        f"| Missing Parameter | `BFCL_v3_multi_turn_miss_param.json` | {len(harder_groups[1][1])} | 50 |\n"
        f"| Long Context | `BFCL_v3_multi_turn_long_context.json` | {len(harder_groups[2][1])} | 50 |\n\n"
        "All source files are JSONL-formatted local BFCL files. The held-out Base IDs are excluded by exact ID.\n",
        encoding="utf-8",
    )
    return {"train_format": normalized_train, "heldout_base": heldout_base, "heldout_mixed": heldout_mixed}


def parse_tool_call(output: str) -> dict[str, Any] | None:
    local_dir = Path(__file__).parents[1] / "code" / "rods_stage1_local"
    if str(local_dir) not in sys.path:
        sys.path.insert(0, str(local_dir))
    from format_reward import TOOL_CALL_RE

    matches = TOOL_CALL_RE.findall(output or "")
    if len(matches) != 1:
        return None
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def score_first_tool(predicted: dict[str, Any] | None, gold: dict[str, Any] | None) -> dict[str, bool | None]:
    result: dict[str, bool | None] = {
        "first_tool_name_correct": None,
        "first_required_args_correct": None,
        "first_argument_values_correct": None,
    }
    if not gold:
        return result
    if not isinstance(predicted, dict):
        return {key: False for key in result}
    predicted_name = predicted.get("name")
    predicted_args = predicted.get("arguments")
    expected_name = gold.get("name")
    expected_args = gold.get("arguments", {})
    name_ok = predicted_name == expected_name
    result["first_tool_name_correct"] = name_ok
    if not name_ok or not isinstance(predicted_args, dict):
        result["first_required_args_correct"] = False
        result["first_argument_values_correct"] = False
        return result
    result["first_required_args_correct"] = all(key in predicted_args for key in expected_args)
    result["first_argument_values_correct"] = bool(result["first_required_args_correct"]) and all(
        predicted_args[key] == value for key, value in expected_args.items()
    )
    return result


def rendered_prompts(records: list[dict[str, Any]], tokenizer: Any) -> list[str]:
    prompts = []
    for record in records:
        messages = [{"role": "user", "content": record["prompt"]}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(rendered)
    return prompts


def flatten_generation(
    model_label: str,
    dataset: str,
    records: list[dict[str, Any]],
    prompts: list[str],
    outputs: Any,
    mode: str,
) -> list[dict[str, Any]]:
    local_dir = Path(__file__).parents[1] / "code" / "rods_stage1_local"
    if str(local_dir) not in sys.path:
        sys.path.insert(0, str(local_dir))
    from format_reward import score_format

    rows: list[dict[str, Any]] = []
    for record, prompt, request_output in zip(records, prompts, outputs):
        for sample_index, completion in enumerate(request_output.outputs):
            raw_output = completion.text
            parsed = parse_tool_call(raw_output)
            scored = score_format(raw_output, record.get("tools", []))
            first = score_first_tool(parsed, record.get("gold", {}).get("first_tool_call"))
            rows.append(
                {
                    "model": model_label,
                    "dataset": dataset,
                    "sample_id": record["id"],
                    "source_file": record.get("source_file"),
                    "mode": mode,
                    "sample_index": sample_index,
                    "prompt": record["prompt"],
                    "raw_output": raw_output,
                    "parsed_tool_call": parsed,
                    "metrics": {key: bool(scored[key]) for key in METRIC_KEYS},
                    "format_reward": scored["format_reward"],
                    "gold_first_tool_call": record.get("gold", {}).get("first_tool_call"),
                    **first,
                }
            )
    return rows


def output_path(out_dir: Path, model_label: str, run_tag: str) -> Path:
    suffix = "" if run_tag == "full" else f"_{run_tag}"
    return out_dir / f"{model_label}_model_outputs{suffix}.jsonl"


def evaluate_one_model(args: argparse.Namespace, model_label: str, datasets: dict[str, list[dict[str, Any]]]) -> None:
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # vLLM 0.8.5 still reads this pre-Transformers-5 tokenizer property.
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda tokenizer: tokenizer.all_special_tokens)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = args.base_model if model_label == "base" else args.stage1_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        distributed_executor_backend="mp",
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    deterministic = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    stochastic = SamplingParams(
        n=args.stochastic_n,
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_new_tokens,
    )
    output_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        records = datasets[dataset]
        limit = {
            "train_format": args.max_train_eval,
            "heldout_base": args.max_heldout_base_eval,
            "heldout_mixed": args.max_mixed_eval,
        }[dataset]
        records = records[:limit]
        prompts = rendered_prompts(records, tokenizer)
        print(f"model={model_label} dataset={dataset} prompts={len(prompts)} mode=deterministic", flush=True)
        deterministic_outputs = llm.generate(prompts, deterministic, use_tqdm=True)
        output_rows.extend(flatten_generation(model_label, dataset, records, prompts, deterministic_outputs, "deterministic"))
        print(
            f"model={model_label} dataset={dataset} prompts={len(prompts)} mode=stochastic n={args.stochastic_n}",
            flush=True,
        )
        stochastic_outputs = llm.generate(prompts, stochastic, use_tqdm=True)
        output_rows.extend(flatten_generation(model_label, dataset, records, prompts, stochastic_outputs, "stochastic"))
    write_jsonl(output_path(args.out_dir, model_label, args.run_tag), output_rows)
    print(f"wrote={output_path(args.out_dir, model_label, args.run_tag)} rows={len(output_rows)}", flush=True)
    del llm
    gc.collect()


def mean_or_none(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def aggregate_rows(rows: list[dict[str, Any]], dataset: str, model: str, mode: str, stochastic_n: int) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["dataset"] == dataset and row["model"] == model and row["mode"] == mode
    ]
    if not selected:
        raise ValueError(f"No rows for dataset={dataset}, model={model}, mode={mode}")
    result: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "mode": mode,
        "stochastic_n": stochastic_n if mode == "stochastic" else 1,
        "num_outputs": len(selected),
        "num_unique_samples": len({row["sample_id"] for row in selected}),
        "mean_format_reward": mean_or_none([float(row["format_reward"]) for row in selected]),
    }
    for metric in METRIC_KEYS:
        result[f"{metric}_rate"] = mean_or_none([float(bool(row["metrics"][metric])) for row in selected])
    for metric in ("first_tool_name_correct", "first_required_args_correct", "first_argument_values_correct"):
        values = [float(bool(row[metric])) for row in selected if row.get(metric) is not None]
        result[metric.replace("_correct", "_accuracy")] = mean_or_none(values)
        result[f"{metric}_count"] = len(values)
    result["tool_execution_success_rate"] = None
    return result


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append("N/A" if value is None else str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def get_metric(summary_rows: list[dict[str, Any]], dataset: str, model: str, mode: str, metric: str) -> float | None:
    for row in summary_rows:
        if row["dataset"] == dataset and row["model"] == model and row["mode"] == mode:
            return row.get(metric)
    return None


def build_report(args: argparse.Namespace, summary_rows: list[dict[str, Any]], data_counts: dict[str, int]) -> None:
    report_path = args.report_path or args.out_dir / "stage1_overall_eval_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    deterministic = [row for row in summary_rows if row["mode"] == "deterministic"]
    stochastic = [row for row in summary_rows if row["mode"] == "stochastic"]
    first_tool = [row for row in summary_rows if row.get("first_tool_name_correct_count", 0) > 0]
    base_holdout = get_metric(summary_rows, "heldout_base", "base", "deterministic", "mean_format_reward")
    stage1_holdout = get_metric(summary_rows, "heldout_base", "stage1", "deterministic", "mean_format_reward")
    base_direct = get_metric(summary_rows, "heldout_base", "base", "deterministic", "direct_answer_without_tool_rate")
    stage1_direct = get_metric(summary_rows, "heldout_base", "stage1", "deterministic", "direct_answer_without_tool_rate")
    base_malformed = get_metric(summary_rows, "heldout_base", "base", "deterministic", "malformed_output_rate")
    stage1_malformed = get_metric(summary_rows, "heldout_base", "stage1", "deterministic", "malformed_output_rate")
    base_stochastic = get_metric(summary_rows, "heldout_base", "base", "stochastic", "mean_format_reward")
    stage1_stochastic = get_metric(summary_rows, "heldout_base", "stage1", "stochastic", "mean_format_reward")
    generalizes = stage1_holdout is not None and base_holdout is not None and stage1_holdout > base_holdout
    stage2_ready = stage1_holdout is not None and stage1_stochastic is not None and stage1_holdout >= 0.95 and stage1_stochastic >= 0.90
    deterministic_columns = [
        "dataset",
        "model",
        "mean_format_reward",
        "valid_tool_call_block_rate",
        "valid_json_parse_rate",
        "valid_function_name_rate",
        "valid_arguments_object_rate",
        "required_arguments_present_rate",
        "malformed_output_rate",
        "direct_answer_without_tool_rate",
    ]
    stochastic_columns = [
        "dataset",
        "model",
        "stochastic_n",
        "mean_format_reward",
        "valid_tool_call_block_rate",
        "valid_json_parse_rate",
        "valid_function_name_rate",
        "required_arguments_present_rate",
        "malformed_output_rate",
    ]
    first_tool_columns = [
        "dataset",
        "model",
        "mode",
        "first_tool_name_accuracy",
        "first_required_args_accuracy",
        "first_argument_values_accuracy",
        "tool_execution_success_rate",
    ]
    report = [
        "# Stage 1 Overall Evaluation Report",
        "",
        "## Scope",
        "This run only evaluates the base Qwen3-1.7B model and the frozen Stage 1 final model. No training, GRPO, Stage 2, Stage 3, synthesis, or model modification was performed.",
        "",
        "## Evaluation Data",
        f"- train-format: {data_counts['train_format']} samples from the Stage 1 training JSONL.",
        f"- held-out Base: {data_counts['heldout_base']} samples, exact-ID excluded from the 100 Stage 1 training IDs.",
        f"- held-out mixed harder: {data_counts['heldout_mixed']} samples, 50 each from Missing Function, Missing Parameter, and Long Context.",
        "- Prompting reused the Stage 1 format prompt and schema; `enable_thinking=false` was used in the Qwen chat template.",
        f"- Deterministic decoding used temperature 0.0. Stochastic decoding used temperature 0.7, top_p 0.9, and n={args.stochastic_n}.",
        "",
        "## Table 1: Deterministic Format Metrics",
        "",
        markdown_table(deterministic, deterministic_columns),
        "",
        "## Table 2: Stochastic Format Stability",
        "",
        markdown_table(stochastic, stochastic_columns),
        "",
        "## Table 3: First-Tool Proxy and Execution Metrics",
        "",
        markdown_table(first_tool, first_tool_columns),
        "",
        "`first_*` metrics compare only the first keyword-only function call parsed from BFCL `possible_answer` ground truth. Because this evaluation deliberately requests one tool call for a multi-turn sample, they are a first-action proxy rather than end-to-end trajectory success. No local BFCL environment executor was configured, so tool_execution_success_rate is N/A.",
        "",
        "## Analysis",
        f"1. Held-out Base generalization: {'yes' if generalizes else 'not established'}. Deterministic held-out mean format reward is base={base_holdout}, Stage1={stage1_holdout}.",
        f"2. Direct answers without a tool on held-out Base changed from {base_direct} to {stage1_direct}.",
        f"3. Malformed output on held-out Base changed from {base_malformed} to {stage1_malformed}.",
        f"4. Stochastic held-out Base mean format reward changed from {base_stochastic} to {stage1_stochastic}.",
        f"5. Stage 2 recommendation: {'the Stage 1 format gate is strong enough to proceed to Stage 2, subject to treating this result as format-only.' if stage2_ready else 'do not enter Stage 2 until format stability is improved further.'}",
        "6. The model still lacks an evaluated multi-turn reasoning, progress, state-tracking, and end-to-end tool-execution capability. Stage 1 rewards only the single-call format and schema compliance; it does not establish full trajectory correctness.",
        "",
        "## Artifacts",
        f"- Base outputs: `{args.out_dir / 'base_model_outputs.jsonl'}`",
        f"- Stage 1 outputs: `{args.out_dir / 'stage1_model_outputs.jsonl'}`",
        f"- Summary JSON: `{args.out_dir / 'stage1_overall_summary.json'}`",
        f"- Summary CSV: `{args.out_dir / 'stage1_overall_summary.csv'}`",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def summarize(args: argparse.Namespace, data_counts: dict[str, int]) -> None:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        path = output_path(args.out_dir, model, args.run_tag)
        if not path.is_file():
            raise FileNotFoundError(f"Missing model outputs: {path}")
        rows.extend(read_json_lines(path))
    summary_rows = [
        aggregate_rows(rows, dataset, model, mode, args.stochastic_n)
        for mode in OUTPUT_MODE_ORDER
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
    ]
    suffix = "" if args.run_tag == "full" else f"_{args.run_tag}"
    summary_json = args.out_dir / f"stage1_overall_summary{suffix}.json"
    summary_csv = args.out_dir / f"stage1_overall_summary{suffix}.csv"
    summary_json.write_text(
        json.dumps(
            {
                "run_tag": args.run_tag,
                "data_counts": data_counts,
                "stochastic_n": args.stochastic_n,
                "tool_execution_success_rate": None,
                "results": summary_rows,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    fieldnames = list(summary_rows[0])
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    if args.run_tag == "full":
        build_report(args, summary_rows, data_counts)
    print(f"summary_json={summary_json}")
    print(f"summary_csv={summary_csv}")


def child_command(args: argparse.Namespace, model_label: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__)),
        "--base_model",
        str(args.base_model),
        "--stage1_model",
        str(args.stage1_model),
        "--train_file",
        str(args.train_file),
        "--bfcl_dir",
        str(args.bfcl_dir),
        "--out_dir",
        str(args.out_dir),
        "--max_train_eval",
        str(args.max_train_eval),
        "--max_heldout_base_eval",
        str(args.max_heldout_base_eval),
        "--max_mixed_eval",
        str(args.max_mixed_eval),
        "--stochastic_n",
        str(args.stochastic_n),
        "--batch_size",
        str(args.batch_size),
        "--max_prompt_length",
        str(args.max_prompt_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--model_to_run",
        model_label,
        "--run_tag",
        args.run_tag,
    ]
    if args.report_path:
        command.extend(["--report_path", str(args.report_path)])
    return command


def main() -> None:
    args = parse_args()
    if args.stochastic_n < 1:
        raise ValueError("--stochastic_n must be positive")
    datasets = prepare_eval_sets(args)
    data_counts = {name: len(rows) for name, rows in datasets.items()}
    if args.summarize_only:
        summarize(args, data_counts)
        return
    if args.model_to_run == "all":
        for label in MODEL_ORDER:
            subprocess.run(child_command(args, label), check=True)
        summarize(args, data_counts)
        return
    evaluate_one_model(args, args.model_to_run, datasets)


if __name__ == "__main__":
    main()
