"""Verify published Codex examples without model calls or VM execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from env_tuning.rods_matchtir_v1.lifecycle import validate_candidate_record


ROOT = Path(__file__).resolve().parents[2]


def verify(bundle: Path) -> list[dict]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "stage1_format_rl/schemas/rods_validated_candidate_v1.schema.json").read_text())
    validator = Draft202012Validator(schema)
    samples = []
    results = []
    for entry in manifest["examples"]:
        path = bundle / entry["candidate_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"candidate digest differs: {path.name}")
        candidate = json.loads(path.read_text(encoding="utf-8"))
        validator.validate(candidate)
        validate_candidate_record(candidate)
        sample = candidate["sample"]
        samples.append(sample)
        kwargs = sample["extra_info"]["interaction_kwargs"]
        gt = kwargs["ground_truth"]
        trace = candidate["generation_metadata"]["execution_trace"]
        if len(gt) != len(trace):
            raise ValueError(f"GT/trace turn count differs: {path.name}")
        missing_turns = []
        call_count = 0
        for index, (calls, turn) in enumerate(zip(gt, trace)):
            records = turn["records"]
            if turn["turn_id"] != index:
                raise ValueError(f"trace turn index differs: {path.name}")
            if turn["intentional_missing"]:
                if calls or records:
                    raise ValueError(f"missing turn contains calls/observations: {path.name}")
                missing_turns.append(index)
            elif calls != [record["canonical_call"] for record in records] or not all(record["success"] for record in records):
                raise ValueError(f"GT differs from successful execution evidence: {path.name}")
            call_count += len(calls)
        if len(gt) != entry["turn_count"] or call_count != entry["call_count"]:
            raise ValueError(f"manifest counts differ: {path.name}")
        if sample["data_source"] != entry["data_type"]:
            raise ValueError(f"category differs: {path.name}")
        results.append({"source_seed_id": entry["source_seed_id"], "data_type": entry["data_type"],
                        "turn_count": len(gt), "call_count": call_count, "missing_turns": missing_turns,
                        "recorded_judge": candidate["validation"]["quality_judge"]["decision"],
                        "schema_and_training_contract": "passed", "gt_matches_recorded_execution": True})
    jsonl = bundle / "training_samples.jsonl"
    if hashlib.sha256(jsonl.read_bytes()).hexdigest() != manifest["training_samples_sha256"]:
        raise ValueError("training JSONL digest differs")
    exported = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if exported != samples:
        raise ValueError("training JSONL differs from original candidate samples")
    if len(results) != 4 or len({result["data_type"] for result in results}) != 4:
        raise ValueError("bundle must contain one example per category")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "data/codex-synthesis")
    args = parser.parse_args()
    print(json.dumps({"verified": verify(args.bundle), "new_model_calls": 0, "vm_replays": 0}, indent=2))
