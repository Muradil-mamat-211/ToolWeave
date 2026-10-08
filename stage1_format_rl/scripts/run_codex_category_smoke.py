"""Run real dataset seeds through every Generator gate using codex exec.

Smoke seeds preserve source tasks, tools, restoration events and configuration.
Their progress fields are explicitly synthetic placeholders, not RL measurements.
Completed real responses may be reused only for identical role/prompt pairs;
every run reconstructs and executes the actual BFCL VMs.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import pandas as pd

from env_tuning.rods_data_generation_v1.codex_backend import CodexCLIBackend, FORBIDDEN_EVENT_ITEMS
from env_tuning.rods_data_generation_v1.config import GeneratorConfig, LLMConfig, QueueConfig
from env_tuning.rods_data_generation_v1.environment_adapter import SynthesisEnvironmentAdapter
from env_tuning.rods_data_generation_v1.function_catalog import FunctionCatalog
from env_tuning.rods_data_generation_v1.llm_backend import BackendQuotaExceeded, push_request_metadata, pop_request_metadata
from env_tuning.rods_data_generation_v1.models import BackendResponse, SeedRecord, to_builtin
from env_tuning.rods_data_generation_v1.pipeline import RODSDataGenerationPipeline
from env_tuning.rods_matchtir_v1.provenance import extract_available_functions, extract_source_tool_updates


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(value), ensure_ascii=False, indent=2), encoding="utf-8")


def event(**values) -> None:
    print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **values}, ensure_ascii=False), flush=True)


class AuditedCLIBackend(CodexCLIBackend):
    def __init__(self, config: LLMConfig, *, reuse: bool):
        super().__init__(config)
        self.completed = {}
        self.new_requests = 0
        self.reused_requests = 0
        if reuse:
            for job in sorted(self.artifacts.glob("codex-*")):
                try:
                    request = json.loads((job / "request.json").read_text())
                    response = json.loads((job / "response.json").read_text())
                    events = [json.loads(line) for line in (job / "events.jsonl").read_text().splitlines() if line.strip()]
                    if not any(item.get("type") == "turn.completed" for item in events):
                        continue
                    if any((item.get("item") or {}).get("type") in FORBIDDEN_EVENT_ITEMS for item in events):
                        continue
                    if set(response) != {"text"} or not isinstance(response["text"], str) or not response["text"].strip():
                        continue
                    key = self.key(request["logical_role"], request["messages"])
                    self.completed[key] = (job, response)
                except (OSError, ValueError, TypeError, KeyError):
                    continue

    @staticmethod
    def key(role, messages):
        # Preserve exact message text, including rendered input ordering.
        return json.dumps([role, to_builtin(messages)], ensure_ascii=False, sort_keys=True)

    async def complete(self, **kwargs):
        role = kwargs["role"]
        event(event="role_start", role=role, metadata=kwargs.get("metadata"))
        cached = self.completed.get(self.key(role, kwargs["messages"]))
        if cached:
            job, response = cached
            self.reused_requests += 1
            event(event="reuse_completed_exact_prompt", role=role, artifact=str(job))
            return BackendResponse(role=role, text=response["text"], request_id=job.name,
                                   raw_response={**response, "artifact_dir": str(job), "smoke_exact_prompt_reuse": True},
                                   latency_seconds=0)
        self.new_requests += 1
        response = await super().complete(**kwargs)
        event(event="role_done", role=role, request_id=response.request_id,
              latency_seconds=round(response.latency_seconds, 2))
        return response


class AuditedEnvironmentFactory(SynthesisEnvironmentAdapter):
    """Record unchanged VM outcomes even if Query generation is interrupted."""
    def __init__(self, folder: Path):
        super().__init__()
        self.folder = folder

    def create(self, **kwargs):
        session = super().create(**kwargs)
        write_json(self.folder / "vm_sessions" / f"{session.environment_id}.json", {
            "environment_id": session.environment_id, "purpose": kwargs["purpose"],
            "long_context": kwargs["long_context"], "initial_state": session.snapshot(),
        })
        original = session.execute

        def observed(call):
            result = original(call)
            value = {"environment_id": session.environment_id, "purpose": kwargs["purpose"],
                     "call": call.canonical(), "arguments": call.arguments, **asdict(result)}
            with (self.folder / "vm_trace.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(to_builtin(value), ensure_ascii=False) + "\n")
            return result

        session.execute = observed
        return session


def make_seed(row, dataset_hash: str) -> SeedRecord:
    row = to_builtin(row)
    kwargs = row["extra_info"]["interaction_kwargs"]
    return SeedRecord.from_mapping({
        "schema_version": "rods_boundary_seed.v1", "sample_id": kwargs["id"],
        "data_type": row["data_source"], "Q_old": kwargs["question"],
        "GT_old": kwargs["ground_truth"], "available_functions": extract_available_functions(row["prompt"]),
        "initial_config": json.loads(kwargs["initial_config"]),
        "mean_progress": 0.5, "boundary_score_phi": 1.0,
        "training_epoch_or_step": {"epoch": 0, "global_step": 0},
        "generation_metadata": {
            "source": "Real source dataset row; smoke-only progress placeholders, no observed RL boundary selection",
            "source_dataset_sha256": dataset_hash,
            "planner_source": {"tool_updates": extract_source_tool_updates(kwargs)},
        },
    })


def record_drafts(pipeline, folder: Path) -> None:
    # Observation hooks preserve method results and never alter generated data.
    for owner, method, label in [
        (pipeline.execution, "execute", "executed"),
        (pipeline.rewrite, "rewrite", "rewritten"),
        (pipeline.missing_function, "transform", "missing_function"),
        (pipeline.missing_parameter, "transform", "missing_parameter"),
    ]:
        original = getattr(owner, method)
        counter = [0]

        async def observed(*args, _original=original, _counter=counter, _label=label, **kwargs):
            draft = await _original(*args, **kwargs)
            _counter[0] += 1
            write_json(folder / f"draft_{_label}_{_counter[0]}.json", asdict(draft))
            return draft

        setattr(owner, method, observed)


async def run_case(seed, catalog, backend, output, run_id):
    folder = output / seed.sample_id / run_id
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "seed.json", asdict(seed))
    queues = QueueConfig(**{name: str(folder / name) for name in QueueConfig.__dataclass_fields__})
    config = GeneratorConfig(llm=backend.config, queues=queues, dry_run=False, test_mode=False)
    pipeline = RODSDataGenerationPipeline(config=config, backend=backend, catalog=catalog,
                                          environment_factory=AuditedEnvironmentFactory(folder))
    record_drafts(pipeline, folder)
    token = push_request_metadata(smoke_sample_id=seed.sample_id)
    event(event="case_start", sample_id=seed.sample_id, artifact=str(folder))
    try:
        result = await pipeline.generate(seed, checkpoint_callback=lambda value: write_json(folder / "checkpoint.json", value))
        write_json(folder / "result.json", asdict(result))
        if result.candidate is not None:
            write_json(folder / "candidate.json", result.candidate)
            write_json(folder / "training_sample.json", result.candidate["sample"])
            write_json(folder / "validation_manifest.json", result.candidate["validation"])
        summary = {"sample_id": seed.sample_id, "data_type": seed.data_type,
                   "status": result.status, "reason": result.reason,
                   "attempts": result.attempts, "planner_calls": result.planner_calls,
                   "artifact": str(folder)}
    except BackendQuotaExceeded as exc:
        summary = {"sample_id": seed.sample_id, "data_type": seed.data_type,
                   "status": "DEFERRED_BACKEND_QUOTA", "reason": str(exc), "artifact": str(folder)}
        write_json(folder / "quota_deferred.json", summary)
    except Exception as exc:
        summary = {"sample_id": seed.sample_id, "data_type": seed.data_type,
                   "status": "HARNESS_ERROR", "reason": f"{type(exc).__name__}: {exc}", "artifact": str(folder)}
        (folder / "exception.txt").write_text(traceback.format_exc())
    finally:
        pop_request_metadata(token)
        for index, (payload, prompt) in enumerate(zip(pipeline.planner.rendered_inputs, pipeline.planner.rendered_prompts), 1):
            write_json(folder / f"planner_input_attempt_{index}.json", payload)
            (folder / f"planner_prompt_attempt_{index}.txt").write_text(prompt, encoding="utf-8")
    write_json(folder / "summary.json", summary)
    write_json(output / seed.sample_id / "latest.json", summary)
    event(event="case_done", **summary)
    return summary


async def main(args):
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    catalog = FunctionCatalog.from_training_parquet(args.dataset)
    wanted = set(args.sample_ids or [f"multi_turn_{category}_33" for category in ("base", "miss_func", "miss_param", "long_context")])
    seeds = [make_seed(row, dataset_hash) for row in pd.read_parquet(args.dataset).to_dict("records")
             if to_builtin(row)["extra_info"]["interaction_kwargs"]["id"] in wanted]
    if {seed.sample_id for seed in seeds} != wanted:
        raise ValueError("requested source sample IDs are missing")
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2] / "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_data_generation_v1"
    source_hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source.rglob("*")) if path.is_file() and path.suffix in {".py", ".txt"}}
    write_json(args.output / f"{run_id}_manifest.json", {
        "source_dataset": str(args.dataset), "source_dataset_sha256": dataset_hash,
        "sample_ids": sorted(wanted), "generator_source_hashes": source_hashes,
        "transport": "official codex exec with existing ChatGPT login", "progress_is_measured": False,
        "reuse_completed_exact_prompts": args.reuse_completed, "maximum_cli_concurrency": args.concurrency,
    })
    config = LLMConfig(backend="codex_cli", model="", codex_binary=args.codex_binary,
                       codex_artifact_dir=str(args.output / "codex_calls"),
                       timeout_seconds=args.timeout, concurrency=args.concurrency)
    backend = AuditedCLIBackend(config, reuse=args.reuse_completed)
    try:
        summaries = await asyncio.gather(*(run_case(seed, catalog, backend, args.output, run_id) for seed in seeds))
        write_json(args.output / f"{run_id}_results.json", {
            "cases": summaries, "new_real_cli_requests": backend.new_requests,
            "reused_completed_real_responses": backend.reused_requests,
        })
        return all(summary["status"] == "SUCCEEDED" for summary in summaries)
    finally:
        await backend.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-ids", nargs="*")
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--reuse-completed", action="store_true")
    raise SystemExit(0 if asyncio.run(main(parser.parse_args())) else 1)
