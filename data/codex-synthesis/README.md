# Codex-generated data examples

This bundle contains one generated example for each BFCL multi-turn category.
See the [construction guide](../../docs/codex-data-synthesis.md) for the workflow
and role cooperation.

| Category | Source seed | Final turns | GT calls | Candidate |
| --- | --- | ---: | ---: | --- |
| Base | `multi_turn_base_33` | 4 | 6 | [base.json](base.json) |
| Missing Function | `multi_turn_miss_func_33` | 5 | 5 | [missing_function.json](missing_function.json) |
| Missing Parameter | `multi_turn_miss_param_33` | 5 | 7 | [missing_parameter.json](missing_parameter.json) |
| Long Context | `multi_turn_long_context_33` | 3 | 5 | [long_context.json](long_context.json) |

## File contents

- Candidate JSON files retain the generated Queries, GT, initial environment,
  actual execution evidence and recorded review results.
- `sample.extra_info.interaction_kwargs` contains Questions, processed Questions,
  GT and environment configuration.
- `generation_metadata.execution_trace` contains calls, observations and state
  changes; `validation` contains each candidate's recorded decisions.
- [training_samples.jsonl](training_samples.jsonl) contains the four candidates'
  `sample` objects in the table's order.
- [manifest.json](manifest.json) records source IDs, file hashes and counts.
  [generation_manifest.json](generation_manifest.json) records implementation
  hashes and CLI provenance.

The source tasks come from the public AWorld-RL
[BFCL training dataset](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_train.parquet).
The candidate files and exported samples retain their original contents.

## Category behavior in these examples

Turn indices below start at **0**:

- **Base:** inspect a deployment script, send a summary and check sent history.
- **MF:** turn 2 requests sending a note while `send_message` is unavailable and
  has `GT=[]`. Turn 3 has an empty Question, restores the function through a
  tool update and contains the successful call. Turn 4 checks sent history.
- **MP:** turn 2 asks to send an inventory without choosing a colleague and has
  `GT=[]`. Turn 3 supplies “I choose Daniel.” and contains the lookup and send
  calls. Turn 4 checks sent history.
- **LC:** the first turn reads an extended tool response; the next turn uses
  its function identifier and final five tokens in a receipt.

MF and MP have no execution observations at the missing turn. The successful
calls and observations are assigned to the recovery turn.

## Verify the bundle

Run from the repository root with the project dependencies installed:

```bash
export PYTHONPATH="$PWD/code/AWorld-RL-stage1-worktree/EnvTuning${PYTHONPATH:+:$PYTHONPATH}"
python stage1_format_rl/scripts/verify_codex_synthesis_examples.py
```

The verifier checks hashes, schema, Training compatibility, GT alignment with
recorded execution, and equality of the exported JSONL samples.
