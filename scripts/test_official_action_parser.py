#!/usr/bin/env python3
"""Run protocol cases against the official EnvTuning parser without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from env_tuning.interaction.utils import parse_model_response

    cases = {
        "complete_think_tool": "<think>plan</think><tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call>",
        "missing_think": "<tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call>",
        "unclosed_think": "<think>plan<tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call>",
        "unclosed_tool": "<think>plan</think><tool_call>{\"name\":\"pwd\",\"arguments\":{}",
        "outside_text": "prefix<think>plan</think><tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call>",
        "invalid_json": "<think>plan</think><tool_call>{bad}</tool_call>",
        "unknown_tool_name": "<think>plan</think><tool_call>{\"name\":\"not_a_real_tool\",\"arguments\":{}}</tool_call>",
        "multiple_tool_calls": "<think>plan</think><tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call>",
        "valid_answer": "<think>done</think><answer>done</answer>",
        "think_answer": "<think>done</think><answer>result</answer>",
        "next_action": "<think>continue</think><tool_call>{\"name\":\"pwd\",\"arguments\":{}}</tool_call>",
    }
    results = []
    for name, raw in cases.items():
        parsed, flag = parse_model_response(raw)
        accepted = flag in {"tool_call", "answer"}
        results.append(
            {
                "case": name,
                "raw_output": raw,
                "accepted": accepted,
                "parsed_action": parsed if accepted else None,
                "error": None if accepted else flag,
                "parser_flag": flag,
            }
        )
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
