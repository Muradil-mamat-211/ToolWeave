"""Smoke-only hard guard for distributed AdamW optimizer updates."""

from __future__ import annotations

import os


if os.environ.get("RODS_SGLANG_FORCE_NATIVE_CUSTOM_OPS") == "1":
    # sgl-kernel 0.1.9's published manylinux wheel does not contain sm_120
    # cubins. SGLang 0.4.8 supplies mathematically equivalent PyTorch-native
    # implementations for its CustomOp classes; force that documented fallback
    # in every spawned scheduler process instead of executing incompatible
    # precompiled kernels. Attention remains on the explicitly configured
    # Triton backend, and sampling is selected independently by ServerArgs.
    from sglang.srt.custom_op import CustomOp

    def _dispatch_native_custom_op(self):
        return self.forward_native

    CustomOp.dispatch_forward = _dispatch_native_custom_op
    print("[smoke-blackwell] SGLang CustomOp backend forced to PyTorch native")


_limit_raw = os.environ.get("SMOKE_MAX_OPTIMIZER_STEPS_PER_PROCESS")
if _limit_raw:
    import json
    import threading
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    _limit = int(_limit_raw)
    _audit_dir = Path(os.environ["SMOKE_OPTIMIZER_AUDIT_DIR"])
    _audit_dir.mkdir(parents=True, exist_ok=True)
    _counts: dict[int, int] = {}
    _lock = threading.Lock()
    _original_adamw_step = torch.optim.AdamW.step

    def _guarded_adamw_step(self, *args, **kwargs):
        optimizer_id = id(self)
        with _lock:
            current = _counts.get(optimizer_id, 0)
            if current >= _limit:
                raise RuntimeError(
                    f"Smoke optimizer hard limit exceeded: {current + 1} > {_limit}"
                )

        result = _original_adamw_step(self, *args, **kwargs)

        with _lock:
            count = _counts.get(optimizer_id, 0) + 1
            _counts[optimizer_id] = count
            record = {
                "pid": os.getpid(),
                "rank": os.environ.get("RANK"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "optimizer": type(self).__qualname__,
                "optimizer_object_id": optimizer_id,
                "step_count": count,
                "max_steps_per_process": _limit,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            target = _audit_dir / f"adamw_step_pid_{os.getpid()}_{optimizer_id}.json"
            target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        return result

    torch.optim.AdamW.step = _guarded_adamw_step
    print(f"[smoke-guard] AdamW step limit enabled: {_limit} per optimizer/process")
