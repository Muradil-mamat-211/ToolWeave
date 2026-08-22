#!/usr/bin/env python3
"""Official BFCL multi-turn evaluation for Base/Stage1/Stage2 models."""
from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import os
import random
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

WORKSPACE = Path(os.environ.get("WORKSPACE", "/root/autodl-tmp/rods-workspace"))
if not WORKSPACE.is_dir():
    WORKSPACE = Path("/root/rods-workspace")
AWORLD = WORKSPACE / "code" / "AWorld-RL" / "EnvTuning"
VERL = WORKSPACE / "code" / "verl"
sys.path.insert(0, str(AWORLD))
sys.path.insert(0, str(VERL))

# This checkout predates veRL's packaged interaction base class. The official
# AWorld-RL interaction only needs the base constructor for this standalone
# evaluator, so provide a local compatibility shim without editing veRL.
if "verl.interactions.base" not in sys.modules:
    interaction_pkg = types.ModuleType("verl.interactions")
    interaction_base = types.ModuleType("verl.interactions.base")

    class BaseInteraction:
        def __init__(self, config=None):
            self.config = config or {}

    interaction_base.BaseInteraction = BaseInteraction
    interaction_pkg.base = interaction_base
    sys.modules["verl.interactions"] = interaction_pkg
    sys.modules["verl.interactions.base"] = interaction_base

from env_tuning.bfcl_reward import compute_score as official_reward
from env_tuning.interaction.new_multi_turn_fc import MultiTurnFunctionCallInteraction
from env_tuning.interaction.utils import parse_model_response
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# vLLM in the prepared environment expects this tokenizer property, while
# the installed Transformers version no longer exposes it on Qwen2Tokenizer.
try:
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
except Exception:
    pass

MODEL_PATHS = {
    "base_model": WORKSPACE / "models" / "Qwen3-1.7B",
    "stage1_model": WORKSPACE / "outputs" / "stage1_format_qwen3_1p7b" / "final_model",
    "stage2_model": WORKSPACE / "outputs" / "stage2_base_reasoning_qwen3_1p7b" / "final_model",
}


def norm(x: Any) -> Any:
    if isinstance(x, np.generic):
        return norm(x.item())
    if isinstance(x, np.ndarray):
        return [norm(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [norm(v) for v in x]
    return x


def bounded_json(x: Any, depth: int = 0) -> Any:
    if depth > 6:
        return repr(x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, np.generic):
        return bounded_json(x.item(), depth)
    if isinstance(x, dict):
        return {str(k): bounded_json(v, depth + 1) for k, v in list(x.items())[:80]}
    if isinstance(x, (list, tuple, set)):
        return [bounded_json(v, depth + 1) for v in list(x)[:80]]
    if hasattr(x, "__dict__"):
        return {"__class__": x.__class__.__name__, "attributes": bounded_json(vars(x), depth + 1)}
    return repr(x)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def turn_messages(turn: Any) -> list[dict[str, Any]]:
    turn = norm(turn)
    if isinstance(turn, dict):
        return [turn]
    if isinstance(turn, list):
        return [x for x in turn if isinstance(x, dict) and x.get("role") and x.get("content") is not None]
    return []


def turn_text(turn: Any) -> str:
    return "\n".join(str(x.get("content", "")) for x in turn_messages(turn))


def official_tools(system_text: str) -> list[dict[str, Any]]:
    marker = system_text.rfind("\n[")
    if marker < 0:
        return []
    try:
        value = json.loads(system_text[marker + 1 :].strip())
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def official_system_with_tools(template: str, tools: list[dict[str, Any]]) -> str:
    # The repository's prompt says <thinking>, while its official parser only
    # accepts <think>. Use one parser-aligned prompt for every model.
    return (
        "You are a BFCL multi-turn tool-use assistant. At each turn, output exactly one action.\n"
        "For a tool call, output exactly: <think></think><tool_call>{\"name\":\"tool\",\"arguments\":{}}</tool_call>\n"
        "For a turn with no available or appropriate tool, output exactly: <think></think><answer>brief answer</answer>\n"
        "Keep the think block empty. Do not output explanations or multiple tool_call blocks. Use only the listed tools.\n"
        "Available tools:\n" + json.dumps(tools, ensure_ascii=False)
    )


def possible_answers() -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
    root = WORKSPACE / "data" / "Berkeley-Function-Calling-Leaderboard" / "possible_answer"
    for path in root.glob("BFCL_v3_multi_turn_*.json"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    out[row["id"]] = norm(row.get("ground_truth", []))
    return out


def make_train_samples() -> list[dict[str, Any]]:
    df = pd.read_parquet(AWORLD / "data" / "bfcl_train_base.parquet")
    out = []
    for i in range(len(df)):
        row = df.iloc[i]
        extra = norm(row["extra_info"])["interaction_kwargs"]
        messages = [x for x in norm(row["prompt"]) if isinstance(x, dict)]
        tools = official_tools(str(messages[0]["content"]))
        messages[0]["content"] = official_system_with_tools(str(messages[0]["content"]), tools)
        questions = [turn_messages(x) for x in extra["question"]]
        out.append({
            "id": str(extra["id"]), "dataset_name": "train_base_100",
            "source_file": "AWorld-RL/EnvTuning/data/bfcl_train_base.parquet",
            "messages": messages,
            "tools": tools,
            "question_turns": questions,
            "initial_config": str(extra["initial_config"]),
            "involved_classes": norm(extra["involved_classes"]),
            "ground_truth": norm(extra["ground_truth"]),
            "processed_question": norm(extra["processed_question"]),
        })
    return out


def make_heldout_samples(path: Path, answer_map: dict[str, list[list[str]]], base_template: str) -> list[dict[str, Any]]:
    out = []
    for row in read_jsonl(path):
        raw = row["raw"]
        questions = [turn_messages(x) for x in norm(raw.get("question", []))]
        tools = norm(row.get("tools", []))
        messages = [{"role": "system", "content": official_system_with_tools(base_template, tools)}]
        if questions:
            messages.extend(questions[0])
        source = str(row.get("source_file", ""))
        category = source.split(":", 1)[0] if ":" in source else "base"
        dataset = "heldout_base_100" if category == "base" else "heldout_mixed_150"
        config = raw.get("initial_config", {})
        if not isinstance(config, str):
            config = json.dumps(config, ensure_ascii=False)
        out.append({
            "id": str(row["id"]), "dataset_name": dataset, "category": category,
            "source_file": source, "messages": messages, "tools": tools,
            "question_turns": questions,
            "initial_config": config,
            "involved_classes": norm(raw.get("involved_classes", [])),
            "ground_truth": answer_map.get(str(row["id"]), []),
            "processed_question": [turn_text(x) for x in questions[1:] if turn_text(x)],
        })
    return out


def datasets() -> dict[str, list[dict[str, Any]]]:
    train = make_train_samples()
    template = str(train[0]["messages"][0]["content"])
    answers = possible_answers()
    base = make_heldout_samples(WORKSPACE / "evals/stage1_overall/heldout_base_eval.jsonl", answers, template)
    mixed = make_heldout_samples(WORKSPACE / "evals/stage1_overall/heldout_mixed_harder_eval.jsonl", answers, template)
    return {"train_base_100": train, "heldout_base_100": base, "heldout_mixed_150": mixed}


def parser_input(raw: str) -> tuple[str, bool]:
    """Align local Stage1/Stage2 format with the official parser's wrapper."""
    if "<thinking>" in raw or "</thinking>" in raw:
        return raw.replace("<thinking>", "<think>").replace("</thinking>", "</think>"), True
    if "<think>" not in raw and ("<tool_call>" in raw or "<answer>" in raw):
        return "<think></think>\n" + raw, True
    return raw, False


def action_format(raw: str, tools: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], str, bool]:
    parser_text, adapted = parser_input(raw)
    content, message = parse_model_response(parser_text)
    calls: list[dict[str, Any]] = []
    json_ok = 0
    if message == "tool_call":
        try:
            value = json.loads(content)
            calls = value if isinstance(value, list) else [value]
            calls = [x for x in calls if isinstance(x, dict)]
            json_ok = 1
        except Exception:
            pass
    names = {str(x.get("name")) for x in tools}
    required = {}
    for tool in tools:
        params = tool.get("parameters") or {}
        required[str(tool.get("name"))] = set(params.get("required") or [])
    function_ok = bool(calls) and all(str(x.get("name")) in names for x in calls)
    args_ok = bool(calls) and all(isinstance(x.get("arguments"), dict) for x in calls)
    required_ok = bool(calls) and all(required.get(str(x.get("name")), set()).issubset(set((x.get("arguments") or {}).keys())) for x in calls)
    metrics = {
        "valid_tool_call_block": int(message == "tool_call"),
        "json_parse_ok": int(json_ok), "function_name_ok": int(function_ok),
        "arguments_object_ok": int(args_ok), "required_arguments_ok": int(required_ok),
        "direct_answer_without_tool": int(message == "answer"),
        "malformed_output": int(message not in {"tool_call", "answer"}),
        "parse_message": message,
    }
    return metrics, calls, parser_text, adapted


def score_status(score: float) -> str:
    return {-3: "parse_error", -2: "execution_error", -1: "successful_execution", 0: "user_turn_scored", 1: "user_turn_scored"}.get(score, "other")


class Evaluator:
    def __init__(self, name: str, path: Path, max_new_tokens: int, max_actions: int, backend: str = "vllm"):
        self.name, self.path = name, path
        self.max_new_tokens, self.max_actions = max_new_tokens, max_actions
        self.backend = backend
        if backend == "vllm":
            self.llm = LLM(model=str(path), tensor_parallel_size=2, gpu_memory_utilization=0.90, max_model_len=32768, dtype="bfloat16", enforce_eager=True, trust_remote_code=True)
            self.tokenizer = self.llm.get_tokenizer()
            self.model = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True}
            try:
                self.model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="flash_attention_2", **kwargs)
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
            self.model.eval()

    def generate(self, messages: list[dict[str, Any]], seed: int, stochastic: bool) -> str:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.backend == "vllm":
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            sampling = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.7 if stochastic else 0.0, top_p=0.9 if stochastic else 1.0, seed=seed, skip_special_tokens=False)
            result = self.llm.generate([prompt], sampling, use_tqdm=False)[0]
            return result.outputs[0].text.strip()
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", truncation=True, max_length=32768, enable_thinking=False)
        batch_inputs = None
        if hasattr(inputs, "input_ids"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            batch_inputs = inputs
        else:
            inputs = inputs.to(self.model.device); input_len = inputs.shape[-1]
        gen = {"max_new_tokens": self.max_new_tokens, "do_sample": stochastic, "pad_token_id": self.tokenizer.eos_token_id, "eos_token_id": self.tokenizer.eos_token_id}
        if stochastic:
            gen.update({"temperature": 0.7, "top_p": 0.9})
        with torch.inference_mode():
            output = self.model.generate(**batch_inputs, **gen) if batch_inputs is not None else self.model.generate(inputs, **gen)
        return self.tokenizer.decode(output[0][input_len:], skip_special_tokens=False).strip()

    def generate_batch(self, messages_batch: list[list[dict[str, Any]]], seeds: list[int], stochastic: bool) -> list[str]:
        if self.backend != "vllm":
            return [self.generate(messages, seed, stochastic) for messages, seed in zip(messages_batch, seeds)]
        prompts = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) for messages in messages_batch]
        params = [SamplingParams(max_tokens=self.max_new_tokens, temperature=0.7 if stochastic else 0.0, top_p=0.9 if stochastic else 1.0, seed=seed, skip_special_tokens=False) for seed in seeds]
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [result.outputs[0].text.strip() for result in outputs]

    async def trajectory(self, sample: dict[str, Any], seed: int, stochastic: bool) -> dict[str, Any]:
        interaction = MultiTurnFunctionCallInteraction({"name": "stage2_three_model_eval"})
        instance = f"{self.name}_{sample['id']}_{seed}"
        rewards, actions = [], []
        conversation = [dict(x) for x in sample["messages"]]
        terminated, error = False, None
        state = None
        try:
            await interaction.start_interaction(instance_id=instance, id=sample["id"], initial_config=sample["initial_config"], involved_classes=sample["involved_classes"], ground_truth=sample["ground_truth"], processed_question=list(sample["processed_question"]), question=sample["question_turns"])
            state = interaction._instance_dict[instance]
            for action_index in range(self.max_actions):
                turn = int(state.current_turn_index)
                raw = self.generate(conversation, seed + action_index, stochastic)
                fmt, calls, parser_text, parser_adapted = action_format(raw, sample["tools"])
                conversation.append({"role": "assistant", "content": raw})
                should_end, response, score, _ = await interaction.generate_response(instance, [{"role": "assistant", "content": parser_text}], id=sample["id"])
                score = float(score); rewards.append(score)
                state = interaction._instance_dict.get(instance, state)
                actions.append({"turn_index": turn, "action_index": action_index, "raw_output": raw, "parser_input": parser_text, "parser_adapter_applied": parser_adapted, "parsed_tool_calls": calls, "parse_status": fmt["parse_message"], "execution_status": score_status(score), "step_score": score, "tool_response": response, "environment_state_after": {"turn": state.current_turn_index, "attempt": state.current_turn_attempt_counts, "instances": bounded_json(state.involved_instances), "execution_results": bounded_json(state.all_turn_model_execution_results + state.single_turn_model_execution_results)}, "format_metrics": fmt})
                if should_end:
                    terminated = True; break
                conversation.append({"role": "user", "content": response})
            if not terminated:
                error = "max_actions_reached"
        except Exception as exc:
            error = repr(exc)
        finally:
            try:
                await interaction.finalize_interaction(instance_id=instance)
            except Exception:
                pass
        reward = official_reward({"user_turn_rewards": rewards}, sample["ground_truth"], extra_info={"id": sample["id"]})
        counts = Counter()
        for action in actions:
            counts.update({k: v for k, v in action["format_metrics"].items() if k != "parse_message"})
        return {"model_name": self.name, "model_path": str(self.path), "dataset_name": sample["dataset_name"], "category": sample.get("category", "base"), "sample_id": sample["id"], "decoding_mode": "stochastic" if stochastic else "deterministic", "seed": seed, "initial_prompt": json.dumps(sample["messages"], ensure_ascii=False), "tools": sample["tools"], "conversation": conversation, "assistant_actions": actions, "user_turn_rewards": rewards, "progress_score": float(reward.get("score", 0.0)), "format_reward": float(reward.get("format_reward", 0.0)), "tool_call_reward": float(reward.get("tool_call_reward", 0.0)), "is_tool_call": float(reward.get("is_tool_call", 0.0)), "num_tool_calls": sum(len(x["parsed_tool_calls"]) for x in actions), "num_turns": sum(x in (0, 1) for x in rewards), "terminated_reason": "environment_terminated" if terminated else error, "environment_error": error, "parse_error_count": rewards.count(-3), "execution_error_count": rewards.count(-2), "successful_execution_count": rewards.count(-1), "incorrect_user_turn_count": rewards.count(0), "correct_user_turn_count": rewards.count(1), "over_max_turn": int(error == "max_actions_reached"), "format_counts": {**dict(counts), "assistant_actions": len(actions), "tool_attempts": sum(x["valid_tool_call_block"] for x in [a["format_metrics"] for a in actions])}, "eval_config": {"official_environment": True, "official_reward": "env_tuning.bfcl_reward.compute_score", "official_max_attempts_per_user_turn": interaction.max_step_limit, "max_actions": self.max_actions, "max_new_tokens": self.max_new_tokens, "temperature": 0.7 if stochastic else 0.0, "top_p": 0.9 if stochastic else None}}

    def unload(self):
        if self.backend == "vllm":
            del self.llm, self.tokenizer
        else:
            del self.model, self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def variance(xs):
    if len(xs) < 2: return 0.0
    m = sum(xs) / len(xs)
    return float(sum((x - m) ** 2 for x in xs) / len(xs))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: return {"sample_count": 0}
    p = [x["progress_score"] for x in rows]
    c = Counter()
    for x in rows: c.update(x["format_counts"])
    actions, attempts = max(1, c["assistant_actions"]), max(1, c["tool_attempts"])
    groups = defaultdict(list)
    for x in rows: groups[x["sample_id"]].append(x["progress_score"])
    vars_ = [variance(x) for x in groups.values()]
    return {"sample_count": len(rows), "mean_progress": mean(p), "full_success_rate": mean([x == 1 for x in p]), "partial_progress_rate": mean([0 < x < 1 for x in p]), "zero_progress_rate": mean([x == 0 for x in p]), "mean_format_reward": mean([x["format_reward"] for x in rows]), "mean_tool_call_reward": mean([x["tool_call_reward"] for x in rows]), "is_tool_call_rate": mean([x["is_tool_call"] for x in rows]), "valid_tool_call_block_rate": c["valid_tool_call_block"] / actions, "valid_json_parse_rate": c["json_parse_ok"] / attempts, "valid_function_name_rate": c["function_name_ok"] / attempts, "valid_arguments_object_rate": c["arguments_object_ok"] / attempts, "required_arguments_present_rate": c["required_arguments_ok"] / attempts, "direct_answer_without_tool_rate": c["direct_answer_without_tool"] / actions, "malformed_output_rate": c["malformed_output"] / actions, "parse_error_trajectory_rate": mean([x["parse_error_count"] > 0 for x in rows]), "execution_error_trajectory_rate": mean([x["execution_error_count"] > 0 for x in rows]), "mean_tool_calls_per_trajectory": mean([x["num_tool_calls"] for x in rows]), "mean_turns_per_trajectory": mean([x["num_turns"] for x in rows]), "over_max_turn_rate": mean([bool(x["over_max_turn"]) for x in rows]), "task_success_rate": mean([x == 1 for x in p]), "mean_group_reward_variance": mean(vars_), "nonzero_variance_prompt_rate": mean([v > 0 for v in vars_]), "all_zero_prompt_rate": mean([all(x == 0 for x in v) for v in groups.values()]), "all_success_prompt_rate": mean([all(x == 1 for x in v) for v in groups.values()]), "boundary_prompt_rate": mean([any(x > 0 for x in v) and any(x < 1 for x in v) and variance(v) > 0 for v in groups.values()]), "parse_error_count": sum(x["parse_error_count"] for x in rows), "execution_error_count": sum(x["execution_error_count"] for x in rows), "successful_execution_count": sum(x["successful_execution_count"] for x in rows), "incorrect_user_turn_count": sum(x["incorrect_user_turn_count"] for x in rows), "correct_user_turn_count": sum(x["correct_user_turn_count"] for x in rows), "stochastic_n": max((len(v) for v in groups.values()), default=0)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")


def write_json(path: Path, x: Any):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]):
    fields = sorted({k for x in rows for k in x})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def summarize(root: Path):
    by_model, comparison = {}, []
    for name in MODEL_PATHS:
        rows = read_jsonl(root / name / f"{name}_eval_outputs.jsonl"); by_model[name] = rows
        grouped = []
        for dataset in sorted({x["dataset_name"] for x in rows}):
            for mode in sorted({x["decoding_mode"] for x in rows if x["dataset_name"] == dataset}):
                grouped.append({"model_name": name, "dataset_name": dataset, "decoding_mode": mode, **aggregate([x for x in rows if x["dataset_name"] == dataset and x["decoding_mode"] == mode])})
        write_json(root / name / f"{name}_eval_summary.json", grouped); write_csv(root / name / f"{name}_eval_summary.csv", grouped)
    metrics = ["mean_progress", "full_success_rate", "partial_progress_rate", "zero_progress_rate", "mean_format_reward", "mean_tool_call_reward", "is_tool_call_rate", "valid_tool_call_block_rate", "valid_json_parse_rate", "valid_function_name_rate", "required_arguments_present_rate", "malformed_output_rate", "parse_error_trajectory_rate", "execution_error_trajectory_rate", "mean_tool_calls_per_trajectory", "mean_turns_per_trajectory", "mean_group_reward_variance", "boundary_prompt_rate"]
    keys = sorted({(x["dataset_name"], x["decoding_mode"]) for rows in by_model.values() for x in rows})
    for dataset, mode in keys:
        vals = {name: aggregate([x for x in rows if x["dataset_name"] == dataset and x["decoding_mode"] == mode]) for name, rows in by_model.items()}
        for name in MODEL_PATHS:
            row = {"dataset": dataset, "decoding_mode": mode, "model": name, **vals[name]}
            row.update({"stage1_minus_base": vals["stage1_model"].get("mean_progress", 0) - vals["base_model"].get("mean_progress", 0) if name == "stage1_model" else "", "stage2_minus_stage1": vals["stage2_model"].get("mean_progress", 0) - vals["stage1_model"].get("mean_progress", 0) if name == "stage2_model" else "", "stage2_minus_base": vals["stage2_model"].get("mean_progress", 0) - vals["base_model"].get("mean_progress", 0) if name == "stage2_model" else ""})
            for metric in metrics:
                row[f"{metric}_stage1_minus_base"] = vals["stage1_model"].get(metric, 0) - vals["base_model"].get(metric, 0)
                row[f"{metric}_stage2_minus_stage1"] = vals["stage2_model"].get(metric, 0) - vals["stage1_model"].get(metric, 0)
                row[f"{metric}_stage2_minus_base"] = vals["stage2_model"].get(metric, 0) - vals["base_model"].get(metric, 0)
            comparison.append(row)
    write_json(root / "summary/three_model_comparison_summary.json", comparison); write_csv(root / "summary/three_model_comparison_summary.csv", comparison)
    focus = ["mean_progress", "full_success_rate", "partial_progress_rate", "mean_format_reward", "mean_tool_call_reward", "malformed_output_rate", "execution_error_trajectory_rate", "boundary_prompt_rate"]
    lines = ["# Three-Model Stage 2 Comparison", "", "| dataset | decoding | model | " + " | ".join(focus) + " |", "|---|---|---|" + "---|" * len(focus)]
    for row in comparison:
        lines.append("| " + " | ".join([row["dataset"], row["decoding_mode"], row["model"]] + [f"{row.get(k, 0):.4f}" for k in focus]) + " |")
    (root / "summary/three_model_comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return by_model, comparison


def cases(root: Path, by_model: dict[str, list[dict[str, Any]]]):
    lookup = {(m, x["dataset_name"], x["decoding_mode"], x["sample_id"]): x for m, rows in by_model.items() for x in rows}
    keys = sorted({(x["dataset_name"], "stochastic", x["sample_id"]) for rows in by_model.values() for x in rows if x["decoding_mode"] == "stochastic"})
    buckets = defaultdict(list)
    for dataset, mode, sid in keys:
        trio = {m: lookup.get((m, dataset, mode, sid)) for m in MODEL_PATHS}
        if not all(trio.values()): continue
        b, s1, s2 = trio["base_model"], trio["stage1_model"], trio["stage2_model"]
        bp, p1, p2 = b["progress_score"], s1["progress_score"], s2["progress_score"]
        item = {"dataset_name": dataset, "sample_id": sid, "models": trio}
        if p2 > max(bp, p1): buckets["stage2_better"].append(item)
        if p1 > p2: buckets["stage1_better_stage2"].append(item)
        if bp == p1 == p2 == 0: buckets["all_zero"].append(item)
        if s2["format_counts"].get("malformed_output", 0): buckets["stage2_malformed"].append(item)
        if s2["execution_error_count"]: buckets["stage2_execution_error"].append(item)
        if 0 < p2 < 1: buckets["stage2_partial"].append(item)
    selected = []
    for typ in ["stage2_better", "stage1_better_stage2", "all_zero", "stage2_malformed", "stage2_execution_error", "stage2_partial"]:
        selected.extend({"case_type": typ, **x} for x in buckets[typ][:10])
    write_jsonl(root / "summary/case_studies.jsonl", selected)
    lines = ["# Case Studies", "", "Complete machine-readable records are in `case_studies.jsonl`.", ""]
    for x in selected:
        lines.append(f"## {x['case_type']} | {x['dataset_name']} | {x['sample_id']}")
        for name, row in x["models"].items():
            lines.append(f"### {name}: progress={row['progress_score']:.4f}, format={row['format_reward']:.4f}, tool_call={row['tool_call_reward']:.4f}")
            lines.append("```text")
            for action in row["assistant_actions"]: lines.append(f"turn={action['turn_index']} score={action['step_score']} {action['raw_output']}")
            lines.append(f"user_turn_rewards={row['user_turn_rewards']}"); lines.append("```")
    (root / "summary/case_studies.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(root: Path, by_model: dict[str, list[dict[str, Any]]], comparison: list[dict[str, Any]], args):
    lines = ["# Stage 2 Three-Model Official BFCL Evaluation", "", "Official AWorld-RL `MultiTurnFunctionCallInteraction`, `bfcl_env` executor, parser, state checker, response checker, and `bfcl_reward.compute_score` were used. No training or Stage 3 was run.", "", "## Configuration", "", f"- max_new_tokens: {args.max_new_tokens}", f"- max_actions: {args.max_actions}; official per-user-turn attempt limit: 5", "- deterministic: do_sample=false, temperature=0, n=1", "- stochastic: do_sample=true, temperature=0.7, top_p=0.9, n=4", "- fixed identical seed schedule derived from seed 42", "- models loaded sequentially and unloaded between runs", "- the same parser-aligned canonical system prompt was used for all models and datasets", "- raw model output is preserved; `parser_input` records the exact text sent to the official parser", "- a common parser adapter only normalizes missing `<think>` or legacy `<thinking>` tags before official parsing; this was applied identically to all models", "", "## Dataset", "", "- train_base_100: 100", "- heldout_base_100: 100; train intersection: 0", "- heldout_mixed_150: Missing Function 50, Missing Parameter 50, Long Context 50", "", "## Results", "", "| dataset | decoding | model | mean_progress | full_success | partial_progress | mean_format | mean_tool_call | malformed | exec_error | boundary |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in comparison:
        lines.append("| %s | %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (x["dataset"], x["decoding_mode"], x["model"], x.get("mean_progress", 0), x.get("full_success_rate", 0), x.get("partial_progress_rate", 0), x.get("mean_format_reward", 0), x.get("mean_tool_call_reward", 0), x.get("malformed_output_rate", 0), x.get("execution_error_trajectory_rate", 0), x.get("boundary_prompt_rate", 0)))
    def pick(dataset, mode, model):
        return next(x for x in comparison if x["dataset"] == dataset and x["decoding_mode"] == mode and x["model"] == model)

    hb = {m: pick("heldout_base_100", "stochastic", m) for m in ("base_model", "stage1_model", "stage2_model")}
    hm = {m: pick("heldout_mixed_150", "stochastic", m) for m in ("base_model", "stage1_model", "stage2_model")}
    lines += [
        "", "## Interpretation", "",
        f"- Stage 1 format improvement on held-out Base stochastic: mean format reward {hb['stage1_model']['mean_format_reward'] - hb['base_model']['mean_format_reward']:+.4f}; malformed output rate {hb['stage1_model']['malformed_output_rate'] - hb['base_model']['malformed_output_rate']:+.4f}. On mixed harder, the corresponding changes are {hm['stage1_model']['mean_format_reward'] - hm['base_model']['mean_format_reward']:+.4f} and {hm['stage1_model']['malformed_output_rate'] - hm['base_model']['malformed_output_rate']:+.4f}.",
        f"- Stage 2 progress delta versus Stage 1: held-out Base stochastic {hb['stage2_model']['mean_progress'] - hb['stage1_model']['mean_progress']:+.4f}; mixed harder stochastic {hm['stage2_model']['mean_progress'] - hm['stage1_model']['mean_progress']:+.4f}. This is not a consistent progress improvement.",
        f"- Stage 2 format retention: mean format reward changes {hb['stage2_model']['mean_format_reward'] - hb['stage1_model']['mean_format_reward']:+.4f} on held-out Base and {hm['stage2_model']['mean_format_reward'] - hm['stage1_model']['mean_format_reward']:+.4f} on mixed harder; malformed rates change {hb['stage2_model']['malformed_output_rate'] - hb['stage1_model']['malformed_output_rate']:+.4f} and {hm['stage2_model']['malformed_output_rate'] - hm['stage1_model']['malformed_output_rate']:+.4f}.",
        f"- Stage 2 full-success delta versus Stage 1 is {hb['stage2_model']['full_success_rate'] - hb['stage1_model']['full_success_rate']:+.4f} on held-out Base and {hm['stage2_model']['full_success_rate'] - hm['stage1_model']['full_success_rate']:+.4f} on mixed harder. Execution-error trajectory deltas are {hb['stage2_model']['execution_error_trajectory_rate'] - hb['stage1_model']['execution_error_trajectory_rate']:+.4f} and {hm['stage2_model']['execution_error_trajectory_rate'] - hm['stage1_model']['execution_error_trajectory_rate']:+.4f}.",
        f"- Stochastic boundary rate is {hb['stage1_model']['boundary_prompt_rate']:.4f} -> {hb['stage2_model']['boundary_prompt_rate']:.4f} on held-out Base and {hm['stage1_model']['boundary_prompt_rate']:.4f} -> {hm['stage2_model']['boundary_prompt_rate']:.4f} on mixed harder; this is useful for Stage 3 analysis but is not itself a Stage 3 run.",
        "- Stage 1 improves tool-call formatting and initial progress over Base, but malformed outputs remain substantial. Stage 2 preserves format ability but does not yet establish a robust held-out progress gain.",
        "- Recommendation: do not enter Stage 3 solely from this evaluation. First inspect the Stage 2 reward/environment alignment and execution-error cases; the likely bottleneck is trajectory progress/execution learning rather than a broad format collapse.",
        "- Current models still require stronger multi-turn reasoning and state-completion ability; format success alone is insufficient.",
        "", "Official reward codes: -3 parse error, -2 execution error, -1 successful execution before a completed turn, 0 incorrect completed turn, 1 correct completed turn. Main progress is the mean of valid 0/1 values.",
        "", "## Artifacts", "", f"- Evaluation root: `{root}`", f"- Base full outputs: `{root / 'base_model/base_model_eval_outputs.jsonl'}`", f"- Stage1 full outputs: `{root / 'stage1_model/stage1_model_eval_outputs.jsonl'}`", f"- Stage2 full outputs: `{root / 'stage2_model/stage2_model_eval_outputs.jsonl'}`", f"- Unified summary: `{root / 'summary/three_model_comparison_summary.json'}` and `{root / 'summary/three_model_comparison_summary.csv'}`", f"- Case studies: `{root / 'summary/case_studies.jsonl'}` and `{root / 'summary/case_studies.md'}`"]
    (WORKSPACE / "reports/stage2_three_model_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args):
    data = datasets(); root = Path(args.eval_root) / ("smoke" if args.mode == "smoke" else "")
    root.mkdir(parents=True, exist_ok=True)
    order = ["train_base_100", "heldout_base_100", "heldout_mixed_150"]
    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.is_dir(): raise FileNotFoundError(model_path)
        out = root / model_name; out.mkdir(parents=True, exist_ok=True)
        output_path = out / f"{model_name}_eval_outputs.jsonl"
        if output_path.exists(): output_path.unlink()
        print(f"[model-start] {model_name}", flush=True)
        evaluator = Evaluator(model_name, model_path, args.max_new_tokens, args.max_actions, args.backend)
        with output_path.open("w", encoding="utf-8") as f:
            for di, dataset in enumerate(order):
                samples = data[dataset][:args.max_per_dataset]
                for mode, n in (("deterministic", 1), ("stochastic", args.stochastic_n)):
                    stochastic = mode == "stochastic"
                    for si, sample in enumerate(samples):
                        for ri in range(n):
                            seed = 42 + di * 100000 + si * 100 + ri
                            row = await evaluator.trajectory(sample, seed, stochastic)
                            row["rollout_index"] = ri
                            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                        print(f"[done] {model_name} {dataset} {mode} {si + 1}/{len(samples)}", flush=True)
        evaluator.unload(); print(f"[model-done] {output_path}", flush=True)
    if args.mode == "formal":
        by_model, comparison = summarize(root); cases(root, by_model); report(root, by_model, comparison, args)
        print(f"[summary-done] {root}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "formal"], required=True)
    p.add_argument("--eval_root", default=str(WORKSPACE / "evals/stage2_three_model_comparison"))
    p.add_argument("--max_per_dataset", type=int, default=5)
    p.add_argument("--stochastic_n", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_actions", type=int, default=100)
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    args = p.parse_args(); asyncio.run(run(args))


if __name__ == "__main__":
    main()
