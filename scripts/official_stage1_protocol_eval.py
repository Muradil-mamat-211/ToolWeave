#!/usr/bin/env python3
"""Render official EnvTuning prompts and run read-only parser evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def plain(value: Any) -> Any:
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


def messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [plain(x) for x in plain(row["prompt"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rendered-out", required=True)
    parser.add_argument("--outputs-out", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from env_tuning.interaction.utils import parse_model_response

    df = pd.read_parquet(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rendered = []
    selected = df.head(args.max_samples)
    for _, series in selected.iterrows():
        row = series.to_dict()
        extra = plain(row["extra_info"])
        msg = messages(row)
        rendered_prompt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        rendered.append(
            {
                "sample_id": str(extra.get("original_id", extra.get("index"))),
                "messages": msg,
                "rendered_prompt": rendered_prompt,
                "tools": [],
                "chat_template": tokenizer.chat_template,
                "enable_thinking": "tokenizer_default",
                "stop_strings": [tokenizer.eos_token],
            }
        )
    Path(args.rendered_out).write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rendered) + "\n", encoding="utf-8"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    records = []
    for item in rendered:
        inputs = tokenizer(item["rendered_prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        # SGLang interaction text excludes tokenizer control tokens before the
        # official parser sees it; mirror that boundary in this read-only eval.
        raw = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        parsed, flag = parse_model_response(raw)
        records.append(
            {
                "model": args.model,
                "sample_id": item["sample_id"],
                "rendered_prompt": item["rendered_prompt"],
                "raw_output": raw,
                "parser_flag": flag,
                "accepted": flag in {"tool_call", "answer"},
                "action_type": flag if flag in {"tool_call", "answer"} else None,
                "parsed_action": parsed if flag in {"tool_call", "answer"} else None,
                "complete_think_tag": "<think>" in raw and "</think>" in raw,
                "outside_tag_text": flag == "Error: Response must not contain text outside the required XML tags",
                "malformed": flag not in {"tool_call", "answer"},
            }
        )
    Path(args.outputs_out).write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8"
    )
    summary = {
        "model": args.model,
        "samples": len(records),
        "parser_success_rate": sum(x["accepted"] for x in records) / max(1, len(records)),
        "complete_think_tag_rate": sum(x["complete_think_tag"] for x in records) / max(1, len(records)),
        "valid_tool_action_rate": sum(x["action_type"] == "tool_call" for x in records) / max(1, len(records)),
        "valid_answer_action_rate": sum(x["action_type"] == "answer" for x in records) / max(1, len(records)),
        "outside_tag_text_rate": sum(x["outside_tag_text"] for x in records) / max(1, len(records)),
        "malformed_rate": sum(x["malformed"] for x in records) / max(1, len(records)),
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
