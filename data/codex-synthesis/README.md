# Real Codex synthesis examples

This bundle publishes one real, accepted example for each BFCL multi-turn
category. Read the [construction framework](../../docs/codex-data-synthesis.md)
for the current method and role cooperation.

| Category | Source seed | Final turns | Calls | Candidate with GT and execution evidence |
| --- | --- | ---: | ---: | --- |
| Base | `multi_turn_base_33` | 4 | 6 | [base.json](base.json) |
| Missing Function | `multi_turn_miss_func_33` | 5 | 5 | [missing_function.json](missing_function.json) |
| Missing Parameter | `multi_turn_miss_param_33` | 5 | 7 | [missing_parameter.json](missing_parameter.json) |
| Long Context | `multi_turn_long_context_33` | 3 | 5 | [long_context.json](long_context.json) |

## Files and provenance

- Each candidate JSON is a byte-for-byte copy of the original successful
  candidate. `sample.extra_info.interaction_kwargs` contains final Questions,
  processed Questions, GT and initial environment configuration.
- `generation_metadata.execution_trace` contains the real calls, Observations
  and pre/post state. `validation` preserves the actual original decisions.
- [training_samples.jsonl](training_samples.jsonl) contains the four original
  `sample` objects in the table's order, ready for inspection with JSONL tools.
- [manifest.json](manifest.json) records source IDs, candidate hashes, counts,
  source dataset identity and the historical review policy.
- [generation_manifest.json](generation_manifest.json) preserves the original
  run's generator-code hashes and CLI transport provenance. Absolute paths in
  original audit records identify the historical machine; they are not runtime
  requirements for loading these examples.

The source parquet was the public AWorld-RL
[BFCL training file](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_train.parquet),
with SHA256 `f6c9311b3b7977b307fcb8d175603e828dbca0542bb3abca4f4bdf922bc8f293`.
The four source IDs come from real rows. Smoke progress values are placeholders,
not measured training rollouts.

All examples belong to `run_20261007T140417_209480Z` and were accepted under the
then-current **strict policy**, with 12 gates, fresh-VM execution and actual
model review. The run made 12 new real CLI requests and reused 63 completed real
responses only for identical roles and prompts. They are historical evidence,
not new outputs of the default policy introduced on 2026-10-08. Exact formatting
and LC extraction conditions reflect that older policy. No Query, GT,
Observation or verdict was edited for publication.

## Inspect the missing turns

The following indices are **zero-based**, as in the stored execution trace:

- **MF**: turn 2 requests sending a note while `send_message` is hidden and has
  GT `[]`. Turn 3 has an empty Question, restores the function through a tool
  update and performs the original call. Turn 4 checks sent history.
- **MP**: turn 2 asks to send an inventory without selecting a colleague and
  has GT `[]`. Turn 3 says “I choose Daniel.” and performs the original lookup
  and send calls. Turn 4 checks sent history.
- **LC**: the first turn reads a real 3349-character tool-returned string;
  the next turn uses its function identifier and final five tokens in a receipt.

In both missing categories, the affected turn has no execution Observation.
The preserved successful calls/results belong to the recovery turn.

## Verify locally

Run from the repository root with `jsonschema` and the project dependencies:

```bash
export PYTHONPATH="$PWD/code/AWorld-RL-stage1-worktree/EnvTuning${PYTHONPATH:+:$PYTHONPATH}"
python stage1_format_rl/scripts/verify_codex_synthesis_examples.py
```

The verifier checks hashes, schema, the frozen Training contract, stored
GT/trace alignment and JSONL equality. It makes zero model calls and does not
repeat VM execution or produce new Judge verdicts. Four related smoke examples
do not estimate population quality or demonstrate a training gain.
