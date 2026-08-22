#!/usr/bin/env python3
"""Capture one fair EnvTuning inference step at four serialization levels."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from transformers import AutoConfig, AutoTokenizer

from run_qwen_rods_bfcl100_official_eval import atomic_text, sample_id, to_builtin
from run_qwen_rods_bfcl100_strict_envtuning_eval import OpenAIChatBackend


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_file_hashes(root: Path) -> dict[str, str]:
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    )
    return {name: file_sha256(root / name) for name in names if (root / name).is_file()}


async def get_server_metadata(endpoint: str) -> dict[str, Any]:
    headers = {"Authorization": "Bearer EMPTY"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        output: dict[str, Any] = {}
        for name, suffix in (("version", "/version"), ("models", "/v1/models")):
            async with session.get(endpoint.rstrip("/") + suffix, headers=headers) as response:
                body = await response.text()
                output[name] = {
                    "status": response.status,
                    "body": json.loads(body) if body.startswith(("{", "[")) else body,
                }
        return output


async def async_main(args: argparse.Namespace) -> None:
    frame = pd.read_parquet(args.dataset)
    records = [to_builtin(row) for row in frame.to_dict(orient="records")]
    matches = [record for record in records if sample_id(record) == args.sample_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {args.sample_id!r} row, found {len(matches)}"
        )
    record = matches[0]
    messages = record["prompt"]
    if not isinstance(messages, list):
        raise TypeError("evaluation prompt is not a message list")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    template_tokenizer = AutoTokenizer.from_pretrained(
        args.chat_template_model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    chat_template = template_tokenizer.chat_template
    if not isinstance(chat_template, str) or not chat_template:
        raise RuntimeError("shared template source has no chat template")
    model_config = AutoConfig.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    native_context = int(model_config.max_position_embeddings)
    if args.max_context_length > native_context:
        raise ValueError(
            f"diagnostic context {args.max_context_length} exceeds native {native_context}"
        )

    async with OpenAIChatBackend(
        endpoint=args.endpoint,
        model=args.served_model_name,
        tokenizer=tokenizer,
        max_context_length=args.max_context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        chat_template=chat_template,
        chat_template_source=str(args.chat_template_model_path.resolve()),
        enable_thinking=args.enable_thinking,
        parser_input_mode="compatible",
        reasoning_parser_name=args.reasoning_parser_name,
        tool_call_parser_name=args.tool_call_parser_name,
        timeout_seconds=args.timeout_seconds,
        transport_retries=0,
    ) as backend:
        generated = await backend.generate(messages)

    evidence = {
        "schema_version": "rods_envtuning_parser_diagnostic.v1",
        "label": args.label,
        "sample_id": args.sample_id,
        "data_type": record["data_source"],
        "model_path": str(args.model_path.resolve()),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "chat_template_model_path": str(args.chat_template_model_path.resolve()),
        "model_native_context": native_context,
        "model_key_file_hashes": key_file_hashes(args.model_path),
        "tokenizer_key_file_hashes": key_file_hashes(args.tokenizer_path),
        "backend_metadata": await get_server_metadata(args.endpoint),
        "generation_parameters": generated["request_generation_parameters"],
        "reasoning_parser_name": args.reasoning_parser_name or None,
        "tool_call_parser_name": args.tool_call_parser_name or None,
        "level_1_rendered_prompt": generated["level_1_rendered_prompt"],
        "level_2_true_decoder_raw_output": generated[
            "level_2_true_decoder_raw_output"
        ],
        "level_3_serving_api_response": generated[
            "level_3_serving_api_response"
        ],
        "level_4_envtuning_parser_input": generated[
            "level_4_envtuning_parser_input"
        ],
        "transport": {
            key: generated[key]
            for key in (
                "request_id",
                "server_request_id",
                "finish_reason",
                "prompt_tokens",
                "completion_tokens",
                "max_tokens",
                "latency_seconds",
            )
        },
    }

    root = args.output_root
    atomic_text(
        root / f"{args.label}_single_step_debug.json",
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_text(
        root / f"{args.label}_true_raw.txt",
        evidence["level_2_true_decoder_raw_output"]["true_raw_decoded_text"],
    )
    atomic_text(
        root / f"{args.label}_rendered_prompt.txt",
        evidence["level_1_rendered_prompt"]["rendered_prompt_text"],
    )
    atomic_text(
        root / f"{args.label}_api_response.json",
        json.dumps(
            evidence["level_3_serving_api_response"]["full_api_response_json"],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )
    atomic_text(
        root / f"{args.label}_envtuning_parser_input.txt",
        evidence["level_4_envtuning_parser_input"]["compatible"][
            "envtuning_parser_input"
        ],
    )
    print(json.dumps(evidence, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=("released", "step25"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--chat-template-model-path", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-context-length", type=int, default=40960)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--reasoning-parser-name", default="")
    parser.add_argument("--tool-call-parser-name", default="")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
