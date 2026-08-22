#!/usr/bin/env python3
"""Lightweight checks for the prepared RODS environment."""

from __future__ import annotations

import importlib
from pathlib import Path


WORKSPACE = Path("/root/autodl-tmp/rods-workspace")
MODEL_PATH = WORKSPACE / "models" / "Qwen3-1.7B"
DATA_PATH = WORKSPACE / "data" / "Berkeley-Function-Calling-Leaderboard"


def report_import(name: str) -> None:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: OK ({version})")
    except Exception as exc:
        print(f"{name}: FAILED ({exc!r})")


def main() -> None:
    print("RODS environment check")
    print(f"workspace: {WORKSPACE}")

    report_import("torch")
    try:
        import torch

        print(f"torch version: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        print(f"gpu count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"gpu 0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch details failed: {exc!r}")

    report_import("transformers")
    report_import("verl")
    report_import("vllm")
    report_import("ray")

    print(f"model path exists: {MODEL_PATH.is_dir()}")
    if MODEL_PATH.is_dir():
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            print(f"tokenizer: OK ({tokenizer.__class__.__name__})")
        except Exception as exc:
            print(f"tokenizer: FAILED ({exc!r})")

    print(f"BFCL data path exists: {DATA_PATH.is_dir()}")
    if DATA_PATH.is_dir():
        files = sorted(
            path
            for path in DATA_PATH.rglob("*")
            if path.is_file()
            and ".cache" not in path.parts
            and path.suffix.lower() in {".json", ".jsonl"}
        )
        print(f"BFCL JSON/JSONL file count: {len(files)}")
        print("BFCL JSON/JSONL files (first 50):")
        for path in files[:50]:
            print(path.relative_to(DATA_PATH))


if __name__ == "__main__":
    main()
