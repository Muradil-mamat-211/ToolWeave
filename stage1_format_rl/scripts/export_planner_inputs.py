"""Export actual Planner v2 inputs/prompts without calling an LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from env_tuning.rods_data_generation_v1.environment_adapter import SynthesisEnvironmentAdapter
from env_tuning.rods_data_generation_v1.function_catalog import FunctionCatalog
from env_tuning.rods_data_generation_v1.llm_backend import FakeLLMBackend
from env_tuning.rods_data_generation_v1.metrics import GeneratorMetrics
from env_tuning.rods_data_generation_v1.models import SeedRecord, to_builtin
from env_tuning.rods_data_generation_v1.planner import PlannerAgent
from env_tuning.rods_matchtir_v1.provenance import extract_available_functions, extract_source_tool_updates


def export(dataset: Path, output: Path, sample_ids: set[str] | None = None,
           *, validation_policy: str = "rods") -> int:
    catalog = FunctionCatalog.from_training_parquet(dataset)
    count = 0
    for raw in pd.read_parquet(dataset).to_dict("records"):
        row = to_builtin(raw)
        kwargs = row["extra_info"]["interaction_kwargs"]
        if sample_ids is not None and kwargs["id"] not in sample_ids:
            continue
        seed = SeedRecord.from_mapping({
            "schema_version": "rods_boundary_seed.v1", "sample_id": kwargs["id"],
            "data_type": row["data_source"], "Q_old": kwargs["question"],
            "GT_old": kwargs["ground_truth"], "available_functions": extract_available_functions(row["prompt"]),
            "initial_config": json.loads(kwargs["initial_config"]),
            # Export is not boundary selection or observed progress measurement.
            "mean_progress": 0.5, "boundary_score_phi": 1.0,
            "training_epoch_or_step": {"epoch": 0, "global_step": 0},
            "generation_metadata": {"source": "input export only; progress fields are placeholders",
                                    "planner_source": {"tool_updates": extract_source_tool_updates(kwargs)}},
        })
        planner = PlannerAgent(FakeLLMBackend({}), catalog, GeneratorMetrics(),
                               environment_factory=SynthesisEnvironmentAdapter(),
                               validation_policy=validation_policy)
        prompt, _ = planner._render(seed, failure_history=[], blocked_functions=set(), current_config=seed.initial_config)
        folder = output / seed.sample_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "input.json").write_text(json.dumps(planner.rendered_inputs[-1], ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
        count += 1
    if sample_ids is not None and count != len(sample_ids):
        raise ValueError("some requested sample IDs are missing")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-ids", nargs="*")
    parser.add_argument("--validation-policy", choices=["rods", "strict"], default="rods")
    args = parser.parse_args()
    print(json.dumps({"exported": export(args.dataset, args.output, set(args.sample_ids) if args.sample_ids else None,
                                         validation_policy=args.validation_policy),
                      "validation_policy": args.validation_policy, "model_calls": 0}))
