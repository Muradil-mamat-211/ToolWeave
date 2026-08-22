"""Load Qwen through SGLang on one GPU and execute a real decode calibration."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from sglang.srt.entrypoints.engine import Engine


def main() -> None:
    request_count = int(os.environ.get("RODS_SGLANG_CALIBRATION_REQUESTS", "1"))
    max_new_tokens = int(os.environ.get("RODS_SGLANG_CALIBRATION_NEW_TOKENS", "16"))
    sampling_backend = os.environ.get("RODS_SGLANG_SAMPLING_BACKEND", "pytorch")
    max_prefill_tokens = int(os.environ.get("RODS_SGLANG_MAX_PREFILL_TOKENS", "16384"))
    chunked_prefill_size = int(os.environ.get("RODS_SGLANG_CHUNKED_PREFILL_SIZE", "8192"))
    asset_root = Path(
        os.environ.get(
            "TOOLWEAVE_ASSET_ROOT", Path(__file__).resolve().parents[2]
        )
    ).expanduser().resolve()
    engine = Engine(
        model_path=str(asset_root / "stage1_format_rl/artifacts/stage2_eval/merged/global_step_25"),
        dtype="bfloat16",
        mem_fraction_static=0.50,
        tp_size=1,
        attention_backend="triton",
        sampling_backend=sampling_backend,
        disable_cuda_graph=True,
        context_length=32768,
        max_prefill_tokens=max_prefill_tokens,
        chunked_prefill_size=chunked_prefill_size,
        trust_remote_code=True,
        log_level="info",
    )
    try:
        if request_count == 1:
            prompts: str | list[str] = "Return the single word READY."
        else:
            workload = (
                "You are handling a stateful tool-use request. Review the available "
                "function descriptions and preserve all parameter constraints. "
            ) * 64
            prompts = [f"Request {index}: {workload}" for index in range(request_count)]
        started = time.perf_counter()
        result = engine.generate(
            prompt=prompts,
            sampling_params={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": max_new_tokens,
            },
        )
        elapsed = time.perf_counter() - started
        records = result if isinstance(result, list) else [result]
        prompt_tokens = sum(int(item["meta_info"]["prompt_tokens"]) for item in records)
        completion_tokens = sum(int(item["meta_info"]["completion_tokens"]) for item in records)
        print(
            "SGLANG_REAL_CALIBRATION="
            + json.dumps(
                {
                    "request_count": len(records),
                    "sampling_backend": sampling_backend,
                    "max_prefill_tokens": max_prefill_tokens,
                    "chunked_prefill_size": chunked_prefill_size,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "elapsed_seconds": elapsed,
                    "total_tokens_per_second": (prompt_tokens + completion_tokens) / elapsed,
                    "completion_tokens_per_second": completion_tokens / elapsed,
                    "first_text": records[0]["text"],
                },
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
