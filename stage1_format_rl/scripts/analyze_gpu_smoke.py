#!/usr/bin/env python3
"""Aggregate the no-save Stage 1 GPU smoke evidence."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


WORKSPACE = Path("/root/autodl-tmp/rods-workspace")
ROOT = WORKSPACE / "stage1_format_rl"
ARTIFACT_ROOT = ROOT / "artifacts/gpu_smoke"
LOG_ROOT = ROOT / "logs/gpu_smoke"
DATA_ROOT = ROOT / "data/smoke"
MODEL = WORKSPACE / "models/Qwen3-4B"
OUTPUT = ROOT / "artifacts/gpu_smoke_summary.json"

RUNS = {
    "smoke1": ("smoke1", "smoke1_functional.parquet", 2, 10000, "0.jsonl"),
    "smoke2_case_0": ("smoke2_case_0", "smoke2_extreme_0.parquet", 16, 10000, "0.jsonl"),
    "smoke2_case_1": ("smoke2_case_1", "smoke2_extreme_1.parquet", 16, 10000, "0.jsonl"),
    "smoke2_case_2": ("smoke2_case_2", "smoke2_extreme_2.parquet", 16, 10000, "0.jsonl"),
    "smoke2_case_3": ("smoke2_case_3", "smoke2_extreme_3.parquet", 16, 10000, "0.jsonl"),
    "smoke3": ("smoke3", "smoke3_stress_4.parquet", 16, 10000, "0.jsonl"),
    "smoke4": ("smoke4", "smoke4_train_4.parquet", 16, 10000, "1.jsonl"),
    "length_guard_case_0": (
        "smoke2_case_0_max16384",
        "smoke2_extreme_0.parquet",
        16,
        16384,
        "0.jsonl",
    ),
    "length_guard_case_3": (
        "smoke2_case_3_max16384",
        "smoke2_extreme_3.parquet",
        16,
        16384,
        "0.jsonl",
    ),
}

ACTION_RE = re.compile(
    r"<think>.*?</think>\s*(?:<tool_call>.*?</tool_call>|<answer>.*?</answer>)",
    re.DOTALL,
)


def percentile(values: list[int | float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def as_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def sample_catalog(parquet_path: Path, tokenizer) -> list[dict]:
    frame = pd.read_parquet(parquet_path)
    result = []
    for _, row in frame.iterrows():
        prompt = as_list(row["prompt"])
        rendered = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        result.append(
            {
                "sample_id": row["extra_info"]["index"],
                "first_user_text": prompt[-1]["content"],
                "exact_initial_prompt_tokens": len(
                    tokenizer(rendered, add_special_tokens=False)["input_ids"]
                ),
            }
        )
    return result


def identify_sample(input_text: str, catalog: list[dict]) -> dict:
    matches = [item for item in catalog if item["first_user_text"] in input_text]
    if len(matches) == 1:
        return matches[0]
    return {
        "sample_id": "UNRESOLVED",
        "first_user_text": "",
        "exact_initial_prompt_tokens": None,
    }


def read_gpu_telemetry(path: Path) -> dict:
    per_gpu = defaultdict(lambda: {"memory": [], "util": [], "power": [], "temp": []})
    timestamps = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for fields in reader:
            # date -Ins contains a comma before fractional seconds, adding one CSV field.
            if len(fields) != 9:
                continue
            timestamp_text = fields[0] + "." + fields[1]
            timestamp_text = re.sub(r"(\.\d{6})\d+", r"\1", timestamp_text)
            timestamp = datetime.fromisoformat(timestamp_text)
            gpu = int(fields[2].strip())
            timestamps.append(timestamp)
            per_gpu[gpu]["memory"].append(float(fields[3]))
            per_gpu[gpu]["util"].append(float(fields[5]))
            per_gpu[gpu]["power"].append(float(fields[7]))
            per_gpu[gpu]["temp"].append(float(fields[8]))
    summary = {}
    for gpu, values in sorted(per_gpu.items()):
        summary[str(gpu)] = {
            "peak_memory_mib": max(values["memory"]),
            "peak_utilization_pct": max(values["util"]),
            "mean_utilization_pct": float(np.mean(values["util"])),
            "peak_power_w": max(values["power"]),
            "peak_temperature_c": max(values["temp"]),
        }
    duration = (max(timestamps) - min(timestamps)).total_seconds() if timestamps else None
    return {"duration_seconds": duration, "gpus": summary}


def read_cpu_telemetry(path: Path) -> dict:
    load1 = []
    available = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for fields in reader:
            if len(fields) != 6:
                continue
            load1.append(float(fields[2]))
            available.append(float(fields[5]))
    return {
        "peak_load1": max(load1) if load1 else None,
        "min_available_memory_gib": min(available) / 1024 / 1024 if available else None,
    }


def inferred_termination(output: str) -> str:
    stripped = output.rstrip()
    if stripped.endswith("</answer>"):
        return "complete_answer"
    if stripped.endswith("</tool_call>"):
        return "tool_call_or_environment_limit"
    return "malformed_or_length_limited"


def parse_exact_training_metrics(log_text: str) -> dict:
    step_lines = [line for line in log_text.splitlines() if "step:1 - " in line]
    if not step_lines:
        return {}
    line = step_lines[-1]
    pairs = re.findall(r"([A-Za-z0-9_./-]+):(-?\d+(?:\.\d+)?)", line)
    return {key: float(value) for key, value in pairs}


def run_summary(name: str, spec: tuple, tokenizer) -> dict:
    run_dir, data_file, k, max_response_length, rollout_file = spec
    artifact_dir = ARTIFACT_ROOT / run_dir
    rollout_path = artifact_dir / "rollouts" / rollout_file
    rows = load_jsonl(rollout_path)
    catalog = sample_catalog(DATA_ROOT / data_file, tokenizer)

    records = []
    for rollout_index, row in enumerate(rows):
        sample = identify_sample(row["input"], catalog)
        input_tokens = len(tokenizer(row["input"], add_special_tokens=False)["input_ids"])
        output_tokens = len(tokenizer(row["output"], add_special_tokens=False)["input_ids"])
        actions = ACTION_RE.findall(row["output"])
        action_tokens = [
            len(tokenizer(action, add_special_tokens=False)["input_ids"])
            for action in actions
        ]
        records.append(
            {
                "rollout_index": rollout_index,
                "sample_id": sample["sample_id"],
                "exact_initial_prompt_tokens": sample["exact_initial_prompt_tokens"],
                "dumped_input_tokens": input_tokens,
                "decoded_output_tokens": output_tokens,
                "offline_full_transcript_tokens": input_tokens + output_tokens,
                "max_decoded_assistant_action_tokens": max(action_tokens, default=0),
                "assistant_action_count": len(actions),
                "tool_call_tag_count": row["output"].count("<tool_call>"),
                "answer_tag_count": row["output"].count("<answer>"),
                "termination_inference": inferred_termination(row["output"]),
                "score": float(row["score"]),
                "progress": float(row["progress"]),
                "format_reward": float(row["format_reward"]),
                "tool_call_reward": float(row["tool_call_reward"]),
                "is_tool_call": float(row["is_tool_call"]),
                "interaction_stages": int(row["total_interaction_rounds"]),
            }
        )

    output_tokens = [record["decoded_output_tokens"] for record in records]
    prompt_tokens = [
        record["exact_initial_prompt_tokens"]
        for record in records
        if record["exact_initial_prompt_tokens"] is not None
    ]
    full_tokens = [record["offline_full_transcript_tokens"] for record in records]
    action_tokens = [record["max_decoded_assistant_action_tokens"] for record in records]
    scores = [record["score"] for record in records]
    stages = sum(record["interaction_stages"] for record in records)
    parse_success_stages = sum(
        record["format_reward"] * record["interaction_stages"] for record in records
    )
    log_path = LOG_ROOT / f"{run_dir}.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    exact_metrics = parse_exact_training_metrics(log_text)
    exact_clip_rate = exact_metrics.get("response_length/clip_ratio")

    return {
        "run_id": run_dir,
        "dataset_file": str(DATA_ROOT / data_file),
        "k": k,
        "prompt_count": len(catalog),
        "trajectory_count": len(records),
        "max_response_length": max_response_length,
        "exit_code": int((artifact_dir / "exit_code.txt").read_text().strip()),
        "rewards": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": min(scores),
            "max": max(scores),
            "all_finite": all(math.isfinite(value) for value in scores),
            "mean_format_reward": float(np.mean([r["format_reward"] for r in records])),
            "mean_tool_call_reward": float(np.mean([r["tool_call_reward"] for r in records])),
            "is_tool_call_rate": float(np.mean([r["is_tool_call"] for r in records])),
            "interaction_stage_parser_success_rate": parse_success_stages / stages,
        },
        "lengths": {
            "initial_prompt_tokens_min": min(prompt_tokens),
            "initial_prompt_tokens_mean": float(np.mean(prompt_tokens)),
            "initial_prompt_tokens_max": max(prompt_tokens),
            "decoded_output_tokens_p95": percentile(output_tokens, 95),
            "decoded_output_tokens_p99": percentile(output_tokens, 99),
            "decoded_output_tokens_max": max(output_tokens),
            "offline_full_transcript_tokens_p95": percentile(full_tokens, 95),
            "offline_full_transcript_tokens_p99": percentile(full_tokens, 99),
            "offline_full_transcript_tokens_max": max(full_tokens),
            "max_decoded_assistant_action_tokens": max(action_tokens),
            "mean_interaction_stages": float(
                np.mean([r["interaction_stages"] for r in records])
            ),
            "max_interaction_stages": max(r["interaction_stages"] for r in records),
            "exact_veRL_response_clip_rate": exact_clip_rate,
            "offline_near_cap_count": sum(
                value >= max_response_length - 100 for value in output_tokens
            ),
        },
        "termination_inference": dict(
            Counter(record["termination_inference"] for record in records)
        ),
        "telemetry": read_gpu_telemetry(LOG_ROOT / f"{run_dir}_gpu.csv"),
        "cpu": read_cpu_telemetry(LOG_ROOT / f"{run_dir}_cpu.csv"),
        "exact_training_metrics": exact_metrics,
        "log_flags": {
            "contains_oom": "OutOfMemoryError" in log_text,
            "contains_nan": bool(re.search(r"\bNaN\b", log_text, re.IGNORECASE)),
        },
        "records": records,
    }


def forbidden_artifacts() -> list[str]:
    forbidden = []
    file_names = {"optimizer.pt", "model.safetensors", "pytorch_model.bin"}
    dir_patterns = ("global_step_", "checkpoint-")
    for path in ROOT.rglob("*"):
        if path.is_file() and (path.name in file_names or "trainer_state" in path.name.lower()):
            forbidden.append(str(path))
        if path.is_dir() and (
            path.name in {"actor", "final_model"}
            or any(path.name.startswith(prefix) for prefix in dir_patterns)
        ):
            forbidden.append(str(path))
    return sorted(forbidden)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    runs = {name: run_summary(name, spec, tokenizer) for name, spec in RUNS.items()}
    optimizer_records = []
    for path in sorted(
        (ARTIFACT_ROOT / "smoke4/optimizer_step_audit").glob("*.json")
    ):
        optimizer_records.append(json.loads(path.read_text(encoding="utf-8")))

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "model": str(MODEL),
        "runs": runs,
        "smoke4_attempt1_oom": {
            "log": str(LOG_ROOT / "smoke4_attempt1_oom.log"),
            "failed_in": "loss.backward",
            "optimizer_steps_executed": 0,
            "actor_ppo_max_token_len_per_gpu": 65536,
        },
        "smoke4_success": {
            "actor_ppo_max_token_len_per_gpu": 32768,
            "optimizer_rank_records": optimizer_records,
            "distributed_optimizer_updates": max(
                (record["step_count"] for record in optimizer_records), default=0
            ),
        },
        "forbidden_checkpoint_artifacts": forbidden_artifacts(),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
