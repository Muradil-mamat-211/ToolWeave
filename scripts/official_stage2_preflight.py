#!/usr/bin/env python3
"""Read-only preflight for the official AWorld-RL EnvTuning Stage 2 path."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import random
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


def plain(value: Any) -> Any:
    """Convert parquet/numpy containers into JSON-serializable Python values."""
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def official_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    return [plain(m) for m in plain(row["prompt"])]


def gt_json_calls(turn_calls: list[str]) -> list[dict[str, Any]]:
    """Translate official BFCL Python-call ground truth to the official JSON action format."""
    from env_tuning.interaction.utils import ast_parse

    source = "[" + ", ".join(str(x) for x in turn_calls) + "]"
    decoded = ast_parse(source)
    calls: list[dict[str, Any]] = []
    for item in decoded:
        for name, arguments in item.items():
            calls.append({"name": name, "arguments": jsonable(arguments)})
    return calls


def action_text(calls: list[dict[str, Any]]) -> str:
    return "<think></think><tool_call>" + json.dumps(calls, ensure_ascii=False) + "</tool_call>"


def answer_text() -> str:
    return "<think></think><answer>done</answer>"


async def gold_one(row: dict[str, Any]) -> dict[str, Any]:
    from env_tuning.bfcl_reward import compute_score
    from env_tuning.interaction.new_multi_turn_fc import MultiTurnFunctionCallInteraction

    extra = plain(row["extra_info"])
    kwargs = dict(extra["interaction_kwargs"])
    interaction = MultiTurnFunctionCallInteraction(config={"name": "multi_turn_tool_call"})
    instance_id = f"preflight-{extra['original_id']}"
    result: dict[str, Any] = {
        "sample_id": str(extra["original_id"]),
        "environment_initialized": False,
        "tool_execution_occurred": False,
        "state_updated": False,
        "user_turn_rewards": [],
        "progress": None,
        "error": None,
    }
    try:
        await interaction.start_interaction(instance_id=instance_id, **kwargs)
        result["environment_initialized"] = True
        current_messages = official_prompt(row)
        ground_truth = plain(kwargs["ground_truth"])
        for turn_calls in ground_truth:
            raw_calls = gt_json_calls(turn_calls)
            current_messages.append({"role": "assistant", "content": action_text(raw_calls)})
            terminated, observation, score, metrics = await interaction.generate_response(
                instance_id, current_messages, **kwargs
            )
            result["user_turn_rewards"].append(float(score))
            result["tool_execution_occurred"] = True
            current_messages.append({"role": "tool", "content": str(observation)})
            if terminated:
                break
            current_messages.append({"role": "assistant", "content": answer_text()})
            terminated, observation, score, metrics = await interaction.generate_response(
                instance_id, current_messages, **kwargs
            )
            result["user_turn_rewards"].append(float(score))
            result["state_updated"] = True
            if terminated:
                break
            current_messages.append({"role": "user", "content": str(observation)})
        reward = compute_score(
            {"user_turn_rewards": result["user_turn_rewards"]},
            ground_truth,
            extra_info=extra,
        )
        result["progress"] = float(reward["score"])
        result["official_reward"] = reward
        result["terminated"] = terminated
        result["conversation_roles"] = [m["role"] for m in current_messages]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=4)
    finally:
        try:
            await interaction.finalize_interaction(instance_id=instance_id)
        except Exception:
            pass
    return result


def parser_smoke() -> list[dict[str, Any]]:
    from env_tuning.interaction.utils import parse_model_response

    cases = {
        "valid_tool_call": "<think>plan</think><tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call>",
        "valid_answer": "<think>done</think><answer>done</answer>",
        "outside_text": "prefix<think>plan</think><tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call>",
        "multiple_tool_calls": "<think>plan</think><tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call>",
        "invalid_json": "<think>plan</think><tool_call>{bad}</tool_call>",
        "unknown_tool": "<think>plan</think><tool_call>{\"name\":\"not_a_tool\",\"arguments\":{}}</tool_call>",
        "next_action": "<think>continue</think><tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call>",
    }
    results = []
    for name, text in cases.items():
        content, flag = parse_model_response(text)
        results.append({"case": name, "accepted": flag in {"tool_call", "answer"}, "flag": flag, "parsed": content})
    return results


def stage1_compatibility(df: pd.DataFrame, model_path: str, count: int, max_new_tokens: int) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from env_tuning.bfcl_reward import compute_score
    from env_tuning.interaction.new_multi_turn_fc import MultiTurnFunctionCallInteraction
    from env_tuning.interaction.utils import parse_model_response

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    for _, series in df.head(count).iterrows():
        row = series.to_dict()
        extra = plain(row["extra_info"])
        kwargs = dict(extra["interaction_kwargs"])
        messages = official_prompt(row)
        record: dict[str, Any] = {
            "sample_id": str(extra["original_id"]),
            "conversation": list(messages),
            "assistant_actions": [],
            "user_turn_rewards": [],
            "progress": None,
            "error": None,
        }
        interaction = MultiTurnFunctionCallInteraction(config={"name": "multi_turn_tool_call"})
        instance_id = f"stage1-preflight-{extra['original_id']}"
        try:
            asyncio.run(interaction.start_interaction(instance_id=instance_id, **kwargs))
            for turn_index in range(8):
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
                parsed, flag = parse_model_response(raw)
                action = {"turn_index": turn_index, "raw_output": raw, "parser_flag": flag, "parsed": parsed}
                if flag not in {"tool_call", "answer"}:
                    record["assistant_actions"].append(action)
                    record["user_turn_rewards"].append(-3.0)
                    messages.append({"role": "assistant", "content": raw})
                    break
                messages.append({"role": "assistant", "content": raw})
                terminated, observation, score, metrics = asyncio.run(
                    interaction.generate_response(instance_id, messages, **kwargs)
                )
                action.update({"score": float(score), "observation": str(observation), "metrics": metrics, "terminated": terminated})
                record["assistant_actions"].append(action)
                record["user_turn_rewards"].append(float(score))
                if terminated:
                    break
                messages.append({"role": "user", "content": str(observation)})
            record["conversation"] = messages
            record["progress"] = float(compute_score({"user_turn_rewards": record["user_turn_rewards"]}, kwargs["ground_truth"], extra_info=extra)["score"])
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc(limit=4)
        finally:
            try:
                asyncio.run(interaction.finalize_interaction(instance_id=instance_id))
            except Exception:
                pass
        rows.append(record)
    del model
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--stage1-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gold-count", type=int, default=5)
    parser.add_argument("--compat-count", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data)
    random.seed(42)
    selected = df.sample(n=min(args.gold_count, len(df)), random_state=42)
    gold = [asyncio.run(gold_one(row.to_dict())) for _, row in selected.iterrows()]
    (out / "gold_environment_smoke.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in gold) + "\n", encoding="utf-8"
    )
    (out / "official_parser_smoke.json").write_text(json.dumps(parser_smoke(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compat = stage1_compatibility(df, args.stage1_model, args.compat_count, args.max_new_tokens)
    (out / "stage1_official_protocol_compatibility.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in compat) + "\n", encoding="utf-8"
    )
    summary = {
        "gold_rows": len(gold),
        "gold_environment_initialized_rate": sum(bool(x["environment_initialized"]) for x in gold) / max(1, len(gold)),
        "gold_tool_execution_rate": sum(bool(x["tool_execution_occurred"]) for x in gold) / max(1, len(gold)),
        "gold_progress_values": [x.get("progress") for x in gold],
        "compat_rows": len(compat),
        "compat_first_action_parser_success_rate": sum(bool(x.get("assistant_actions") and x["assistant_actions"][0].get("parser_flag") in {"tool_call", "answer"}) for x in compat) / max(1, len(compat)),
        "compat_nonzero_progress_rate": sum(float(x.get("progress") or 0) > 0 for x in compat) / max(1, len(compat)),
    }
    (out / "preflight_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
