"""Build the exact validated-candidate contract consumed by Training."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from env_tuning.rods_matchtir_v1.lifecycle import validate_candidate_record

from .config import WORKSPACE
from .models import (
    CANDIDATE_SCHEMA_VERSION,
    ConversationDraft,
    ErrorRecord,
    GateResult,
    JudgeResult,
    SeedRecord,
    stable_id,
    to_builtin,
    utc_now,
)
from .source_audit import RODS_ARXIV_VERSION, SOURCE_PROVENANCE


# Reused verbatim from the local executable Stage-1/2 Training contract.  This
# is the Actor protocol, deliberately distinct from Generator-agent <reason>.
ACTOR_PROTOCOL_HEADER = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, make the function/tool calls needed to complete the user's request.

Every assistant action must contain exactly one reasoning block followed by exactly one action block. Do not emit any text outside these XML blocks.

Use exactly one of these two forms:

<think>your step-by-step reasoning</think><tool_call>{\"name\": \"function_name\", \"arguments\": {\"argument_name\": \"value\"}}</tool_call>

<think>your step-by-step reasoning</think><answer>a concise user-facing answer</answer>

The content of <tool_call> must be valid JSON. To invoke multiple APIs in one interaction stage, put a JSON array of call objects inside one and only one <tool_call> block. Never emit multiple <tool_call> blocks in the same assistant action. Use <answer> only when no further tool call is needed or possible.

At each turn, try to complete the current user request. After a tool call, use the environment result to decide whether another tool call is required or whether to finish with <answer>.

"""

FUNCTION_MARKER = "Here is a list of functions in JSON format that you can invoke.\n"
UPDATE_MESSAGE = "I have updated some more functions you can choose from. What about now?"
DEFAULT_SCHEMA_PATH = (
    WORKSPACE / "stage1_format_rl/schemas/rods_validated_candidate_v1.schema.json"
)


def _question_messages(draft: ConversationDraft) -> list[list[dict[str, str]]]:
    output: list[list[dict[str, str]]] = []
    for turn in draft.turns:
        if turn.recovery_tools and not turn.query:
            output.append([])
        else:
            if not turn.query.strip():
                raise ValueError(f"turn {turn.turn_id} has no user query")
            output.append([{"role": "user", "content": turn.query}])
    return output


def _processed_questions(draft: ConversationDraft) -> list[str]:
    output: list[str] = []
    for turn in draft.turns[1:]:
        if turn.recovery_tools:
            output.append(
                json.dumps(turn.recovery_tools, ensure_ascii=False, sort_keys=True)
                + "\n"
                + UPDATE_MESSAGE
            )
        else:
            if not turn.query.strip():
                raise ValueError(f"turn {turn.turn_id} has no processed question")
            output.append(turn.query)
    return output


class CandidateBuilder:
    """Construct, JSON-schema validate, then invoke Training's real validator."""

    def __init__(self, schema_path: str | Path = DEFAULT_SCHEMA_PATH):
        self.schema_path = Path(schema_path)
        schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self._schema_validator = Draft202012Validator(schema)

    def build(
        self,
        *,
        seed: SeedRecord,
        draft: ConversationDraft,
        gates: Sequence[GateResult],
        final_judge: JudgeResult,
        generator_backend: str,
        generator_model: str,
        pipeline_attempts: int,
        planner_calls: int,
        failures: Sequence[ErrorRecord],
        blocklist_history: Sequence[Sequence[str]],
        config_patch_history: Sequence[Mapping[str, Any]],
        refinement_used: bool,
        refinement_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not gates or not all(gate.passed for gate in gates):
            raise ValueError("candidate cannot be built before all deterministic gates pass")
        if not final_judge.accepted:
            raise ValueError("candidate cannot be built without Quality Judge acceptance")

        question = _question_messages(draft)
        processed_question = _processed_questions(draft)
        ground_truth = [turn.ground_truth for turn in draft.turns]
        # PROJECT_NOVELTY_GUARD: exact-content identity deliberately excludes
        # source seed/epoch, timestamps, Judge text, and retry provenance.
        # Thus byte-equivalent Training content generated from different seeds
        # has one stable logical candidate identity.  This is exact canonical
        # deduplication, not an embedding or semantic-similarity threshold.
        canonical_training_content = {
            "data_type": draft.data_type,
            "initial_config": draft.initial_config,
            "visible_tools": draft.initial_tools,
            "question": question,
            "processed_question": processed_question,
            "ground_truth": ground_truth,
        }
        content_fingerprint = stable_id(
            "candidate_content_v2", canonical_training_content
        )
        digest_id = stable_id("candidate", canonical_training_content)
        sample_prefix = draft.data_type.removeprefix("multi_turn_")
        sample_id = f"generated_{sample_prefix}_{digest_id.removeprefix('candidate_')}"

        system_prompt = (
            ACTOR_PROTOCOL_HEADER
            + FUNCTION_MARKER
            + json.dumps(draft.initial_tools, ensure_ascii=False, sort_keys=True)
        )
        sample = {
            "data_source": draft.data_type,
            "prompt": [
                {"role": "system", "content": system_prompt},
                copy.deepcopy(question[0][0]),
            ],
            "ability": "tool",
            "reward_model": {"style": "interaction"},
            "extra_info": {
                "original_id": sample_id,
                "synthetic": True,
                "interaction_kwargs": {
                    "name": "multi_turn_fc",
                    "id": sample_id,
                    "initial_config": json.dumps(
                        draft.initial_config, ensure_ascii=False, sort_keys=True
                    ),
                    "involved_classes": list(draft.involved_classes),
                    "ground_truth": ground_truth,
                    "processed_question": processed_question,
                    "question": question,
                },
            },
        }

        gate_records = [to_builtin(asdict(gate)) for gate in gates]
        execution_trace = [
            {
                "turn_id": turn.turn_id,
                "intentional_missing": turn.is_intentional_missing,
                "missing_kind": turn.missing_kind,
                "records": [to_builtin(asdict(record)) for record in turn.execution_records],
            }
            for turn in draft.turns
        ]
        generation_metadata = {
            "source_seed_id": seed.sample_id,
            "source_epoch": seed.source_epoch,
            "source_global_step": seed.source_global_step,
            "generated_epoch": seed.source_epoch,
            "data_type": draft.data_type,
            # Audit metadata only; deliberately excluded from exact Training
            # content identity because the Actor never consumes it.
            "latent_narrative": draft.narrative,
            "generator_backend": generator_backend,
            "generator_model": generator_model,
            "rods_arxiv_version": f"2606.19047{RODS_ARXIV_VERSION}",
            "pipeline_attempt_count": int(pipeline_attempts),
            "planner_calls": int(planner_calls),
            "failure_history": [error.to_dict() for error in failures],
            "blocklist_history": [list(values) for values in blocklist_history],
            "config_patch_history": to_builtin(config_patch_history),
            "deterministic_gate_results": gate_records,
            "judge_result": to_builtin(asdict(final_judge)),
            "refinement_used": bool(refinement_used),
            "refinement_metadata": to_builtin(refinement_metadata or {}),
            "reconstruction_components_used": sorted(
                name
                for name, status in SOURCE_PROVENANCE.items()
                if status.value in {"RECONSTRUCTED", "PROJECT_SUBSTITUTION"}
            ),
            "structural_profile": to_builtin(draft.structural_profile),
            "execution_trace": execution_trace,
            "synthesis_environment_id": draft.synthesis_environment_id,
            "created_at": utc_now(),
            "content_fingerprint": content_fingerprint,
            "novelty_guard": {
                "source_status": "PROJECT_NOVELTY_GUARD",
                "mode": "exact_canonical_training_content",
                "used_embedding_similarity": False,
            },
        }
        candidate = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "candidate_id": digest_id,
            "validated": True,
            "validation": {
                "passed": True,
                "deterministic_gates": gate_records,
                "quality_judge": to_builtin(asdict(final_judge)),
                "candidate_schema": "passed",
                "training_validator": "pending",
            },
            "generation_metadata": generation_metadata,
            "sample": sample,
        }
        self._schema_validator.validate(candidate)
        candidate["validation"]["training_validator"] = "passed"
        # This is the exact frozen Training Branch validator, not a local copy.
        return validate_candidate_record(candidate)
