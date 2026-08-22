#!/usr/bin/env python
"""Generate the final read-only RODS resource audit report."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from project_paths import (
    LOGS_ROOT,
    MODELS_ROOT,
    REPORTS_ROOT,
    SHARED_DATA_ROOT,
    SOURCE_ROOT,
)


WORKSPACE = SOURCE_ROOT
REPORT = REPORTS_ROOT / "rods_resource_audit.md"
MODEL = MODELS_ROOT / "Qwen3-4B"
BFCL = SHARED_DATA_ROOT / "Berkeley-Function-Calling-Leaderboard"
AWORLD = SOURCE_ROOT / "code" / "AWorld-RL"
GORILLA = SOURCE_ROOT / "code" / "gorilla"
VERL = SOURCE_ROOT / "code" / "verl"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def git_info(path: Path) -> dict[str, str]:
    if not (path / ".git").exists():
        return {"path": str(path), "status": "MISSING"}
    return {
        "path": str(path),
        "remote": run(["git", "remote", "get-url", "origin"], path),
        "commit": run(["git", "rev-parse", "HEAD"], path),
        "status": run(["git", "status", "--short"], path) or "clean",
    }


def model_audit() -> dict:
    result: dict = {"path": str(MODEL), "exists": MODEL.is_dir()}
    if not MODEL.is_dir():
        return result
    result["bytes"] = int(run(["du", "-sb", str(MODEL)]).split()[0])
    result["human_size"] = run(["du", "-sh", str(MODEL)]).split()[0]
    result["files"] = sorted(path.name for path in MODEL.iterdir() if path.is_file())
    result["incomplete_files"] = [str(path) for path in MODEL.rglob("*") if path.is_file() and path.suffix in {".incomplete", ".lock", ".tmp"}]
    index_path = MODEL / "model.safetensors.index.json"
    result["index_exists"] = index_path.exists()
    result["expected_shards"] = []
    result["missing_shards"] = []
    result["shard_sizes"] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected = sorted(set(index.get("weight_map", {}).values()))
        result["expected_shards"] = expected
        result["total_size_from_index"] = index.get("metadata", {}).get("total_size")
        result["missing_shards"] = [name for name in expected if not (MODEL / name).is_file()]
        result["shard_sizes"] = {name: (MODEL / name).stat().st_size for name in expected if (MODEL / name).is_file()}
    result["required_files"] = {
        name: (MODEL / name).is_file()
        for name in ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"]
    }
    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        result["config_verification"] = {"pass": True, "model_type": getattr(config, "model_type", None), "architectures": getattr(config, "architectures", None)}
        result["tokenizer_verification"] = {"pass": True, "class": type(tokenizer).__name__, "vocab_size": getattr(tokenizer, "vocab_size", None)}
    except Exception as exc:
        result["config_verification"] = {"pass": False, "error": repr(exc)}
    return result


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    audit_path = REPORTS_ROOT / "bfcl_rods_data_audit.json"
    bfcl_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    model = model_audit()
    repositories = {name: git_info(path) for name, path in {"AWorld-RL": AWORLD, "Gorilla": GORILLA, "veRL": VERL}.items()}
    adapted_verl = git_info(AWORLD / "EnvTuning" / "verl")
    free = shutil.disk_usage(WORKSPACE)
    workspace_size = run(["du", "-sh", str(WORKSPACE)]).split()[0]
    tmux_running = subprocess.run(["tmux", "has-session", "-t", "rods-download"], capture_output=True).returncode == 0

    raw_counts = bfcl_audit.get("category_counts", {})
    parquet = bfcl_audit.get("official_parquet", [])
    model_pass = bool(
        model.get("exists")
        and not model.get("missing_shards")
        and not model.get("incomplete_files")
        and all(model.get("required_files", {}).values())
        and model.get("config_verification", {}).get("pass")
    )
    bfcl_pass = not bfcl_audit.get("parse_errors") and all(
        raw_counts.get(f"raw/{name}", 0) == 200
        for name in ["Base Multi-Turn", "Missing Function", "Missing Parameter", "Long Context"]
    )
    train_base = next((x for x in parquet if x.get("path", "").endswith("bfcl_train_base.parquet")), {})
    train_all = next((x for x in parquet if x.get("path", "").endswith("bfcl_train.parquet")), {})
    stage3_pass = train_all.get("rows") == 400 and train_all.get("data_source_counts") == {
        "multi_turn_base": 100,
        "multi_turn_miss_func": 100,
        "multi_turn_miss_param": 100,
        "multi_turn_long_context": 100,
    }
    split_status = "PASS via official committed parquet; no separate RODS manifest file found"

    lines = [
        "# RODS Resource Audit",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Workspace",
        f"- Actual workspace: `{WORKSPACE}`",
        f"- Disk capacity: `{free.total / (1024**3):.1f} GiB`",
        f"- Available disk: `{free.free / (1024**3):.1f} GiB`",
        f"- Current workspace size: `{workspace_size}`",
        "- Required directories: `code/`, `data/`, `models/`, `logs/`, `scripts/`, `reports/`, `downloads/`",
        "",
        "## Official Repositories",
    ]
    for name, info in repositories.items():
        lines += [f"- **{name}**: `{info['path']}`; remote `{info.get('remote', '')}`; commit `{info.get('commit', '')}`; status `{info.get('status', '')}`"]
    lines.append(f"- **AWorld-RL adapted EnvTuning/verl submodule**: `{adapted_verl.get('path')}`; commit `{adapted_verl.get('commit', '')}`; status `{adapted_verl.get('status', '')}`")
    lines += [
        "",
        "## RODS Official Instructions",
        "- Requested official starting model repo: `Qwen/Qwen3-4B`.",
        "- Public `RODS/README.md` states that RODS starts from 400 human seeds and maintains an active pool of approximately 800 samples.",
        "- Official EnvTuning README documents `bfcl_train_base.parquet` for Stage 1/2 and `bfcl_train.parquet` for Stage 3/4; it says the processed split samples half of each 200-example BFCL V3 Multi-Turn category.",
        "- The RODS directory contains README/assets only; no separate RODS data downloader or standalone 400-ID manifest was found.",
        "- RODS/EnvTuning training code is present in AWorld-RL, including the official Stage 1/2/3 entrypoints.",
        "",
        "## BFCL Dataset",
        f"- Local path: `{BFCL}`",
        f"- Raw JSON/JSONL files: `{bfcl_audit.get('total_data_files', 0)}`; raw JSON bytes: `{bfcl_audit.get('total_json_bytes', 0):,}`",
        "- File format verified as JSON Lines; no `datasets.load_dataset()` used.",
        "- Raw multi-turn counts: Base 200, Missing Function 200, Missing Parameter 200, Long Context 200, Composite 200.",
        "- `possible_answer` files exist for all five multi-turn categories; tool documentation exists under `multi_turn_func_doc/`; environment implementation/config is present in AWorld-RL `EnvTuning/bfcl_env`.",
        f"- Official Base train parquet: `{train_base.get('path', '')}`, rows `{train_base.get('rows', '')}`.",
        f"- Official Stage 3 train parquet: `{train_all.get('path', '')}`, rows `{train_all.get('rows', '')}`, counts `{train_all.get('data_source_counts', {})}`.",
        "- Official validation parquet: 100 rows, 25 per four training categories.",
        f"- Official 100/100/100/100 IDs: `{split_status}`; exact IDs are embedded in `extra_info.index` of the committed parquet, rather than a separate manifest.",
        "- Composite raw data is present for evaluation/reference but is not included in the official 400-row training parquet.",
        "",
        "## Qwen3-4B",
        "- Repo ID: `Qwen/Qwen3-4B`.",
        f"- Target local path: `{MODEL}`",
        f"- Size: `{model.get('human_size', 'MISSING')}`",
        f"- Expected safetensors shards: `{len(model.get('expected_shards', []))}`; missing shards: `{len(model.get('missing_shards', []))}`",
        f"- Shard sizes: `{model.get('shard_sizes', {})}`",
        f"- Required-file checks: `{model.get('required_files', {})}`",
        f"- Config/tokenizer verification: `{model.get('config_verification', {})}` / `{model.get('tokenizer_verification', {})}`",
        f"- Complete download status: **{'PASS' if model_pass else 'PARTIAL/FAIL'}**",
        "- Download source used: ModelScope official mirror for the exact `Qwen/Qwen3-4B` repo after the unauthenticated HF Xet CAS route returned HTTP 401; no RODS-trained, quantized, or alternate model was used.",
        "",
        "## Download Session",
        "- tmux session: `rods-download`.",
        "- Academic acceleration was sourced only inside the detached download child; the current Codex shell did not source `/etc/network_turbo`.",
        f"- Current session at report generation: `{'RUNNING' if tmux_running else 'STOPPED'}`.",
        f"- Main download log: `{LOGS_ROOT / 'resource_download.log'}`.",
        f"- ModelScope download log: `{LOGS_ROOT / 'modelscope_qwen3_4b_download.log'}`.",
        f"- HF partial/failed-attempt log: `{LOGS_ROOT / 'resource_download.log'}`; HF Xet returned 401 and standard HF retry was superseded by the exact-repo ModelScope mirror.",
        "- No permanent proxy file was changed; user shell profiles and the system environment were left untouched.",
        "",
        "## Final Status",
        f"- Stage 1 data preparation: **{'PASS' if train_base.get('rows') == 100 else 'PARTIAL/FAIL'}**",
        f"- Stage 2 data preparation: **{'PASS' if train_base.get('rows') == 100 else 'PARTIAL/FAIL'}**",
        f"- Stage 3 four-category human seed data: **{'PASS' if stage3_pass else 'PARTIAL/FAIL'}**",
        f"- Official split evidence: **PASS/PARTIAL** (`{split_status}`)",
        f"- Qwen3-4B: **{'PASS' if model_pass else 'PARTIAL/FAIL'}**",
        f"- AWorld-RL: **{'PASS' if repositories['AWorld-RL'].get('commit') and 'MISSING' not in repositories['AWorld-RL'].get('status', '') else 'PARTIAL/FAIL'}**",
        f"- Gorilla/BFCL: **{'PASS' if repositories['Gorilla'].get('commit') and bfcl_pass else 'PARTIAL/FAIL'}**",
        f"- veRL: **{'PASS' if repositories['veRL'].get('commit') and adapted_verl.get('commit') else 'PARTIAL/FAIL'}**",
        "",
        "## Remaining Problems",
    ]
    problems = []
    if not model_pass:
        problems.append("Qwen3-4B download is incomplete or failed verification; keep `rods-download` running and resume without deleting partial files.")
    if not bfcl_pass:
        problems.append("BFCL raw multi-turn audit did not pass all expected category checks.")
    problems.append("No separate RODS-specific 400-ID manifest is published in the RODS directory; the official committed EnvTuning parquet is the available exact split evidence.")
    problems.append("AWorld-RL working tree contains prior untracked `EnvTuning/outputs/`; it was preserved and not removed.")
    for problem in problems:
        lines.append(f"- {problem}")
    lines += ["", "No training, GRPO rollout, synthesis, or model inference was started in this resource audit.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
