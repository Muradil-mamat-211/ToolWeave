#!/usr/bin/env python3
"""Evaluate only Stage 1 tool-call format compliance on the local BFCL JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


METRICS = (
    "valid_tool_call_block",
    "valid_json_parse",
    "valid_function_name",
    "valid_arguments_object",
    "required_arguments_present",
    "direct_answer_without_tool",
    "malformed_output",
)


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 evaluation requires a CUDA device.")
    records = load_records(args.data, args.num_samples)
    local_dir = Path(__file__).parents[1] / "code" / "rods_stage1_local"
    sys.path.insert(0, str(local_dir))
    from format_reward import score_format

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()

    totals: dict[str, float] = {metric: 0.0 for metric in METRICS}
    totals["format_reward"] = 0.0
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            messages = [[{"role": "user", "content": record["prompt"]}] for record in batch]
            rendered = [
                tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for message in messages
            ]
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            ).to(model.device)
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            continuation = generated[:, inputs.input_ids.shape[1] :]
            for record, tokens in zip(batch, continuation):
                output = tokenizer.decode(tokens, skip_special_tokens=True)
                scored = score_format(output, record.get("tools", []))
                totals["format_reward"] += float(scored["format_reward"])
                for metric in METRICS:
                    totals[metric] += float(bool(scored[metric]))

    count = len(records)
    result = {
        "model_path": str(args.model),
        "data_path": str(args.data),
        "num_samples": count,
        "generation": {
            "do_sample": False,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
        },
        "mean_format_reward": round(totals["format_reward"] / count, 6),
    }
    result.update({f"{metric}_rate": round(totals[metric] / count, 6) for metric in METRICS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
