"""Immutable source-audit facts for the V1 reconstruction.

These constants distinguish mechanisms printed in the RODS paper, code reused
from AWorld-RL/EnvTuning, and project reconstruction where no official source
was published.  They are metadata, not a claim that private RODS source code is
available.
"""

from __future__ import annotations

from enum import Enum


RODS_ARXIV_ID = "2606.19047"
RODS_ARXIV_VERSION = "v1"
RODS_ARXIV_SUBMITTED = "2026-06-17"
AWORLD_RL_AUDIT_COMMIT = "be52dbf33051c9b86e8e4d3c4e2394548906c75b"
AWORLD_RL_AUDIT_DATE = "2026-06-18"


class SourceStatus(str, Enum):
    """Permitted provenance labels for Generator components."""

    OFFICIAL_PAPER = "OFFICIAL_PAPER"
    OFFICIAL_REUSED_CODE = "OFFICIAL_REUSED_CODE"
    RECONSTRUCTED = "RECONSTRUCTED"
    PROJECT_SUBSTITUTION = "PROJECT_SUBSTITUTION"


SOURCE_PROVENANCE: dict[str, SourceStatus] = {
    "planner_prompt": SourceStatus.OFFICIAL_PAPER,
    "planner_parser_and_orchestration": SourceStatus.RECONSTRUCTED,
    "parameter_generation": SourceStatus.RECONSTRUCTED,
    "function_sampling": SourceStatus.RECONSTRUCTED,
    "high_level_decomposition_interface": SourceStatus.RECONSTRUCTED,
    "bfcl_vm_execution": SourceStatus.OFFICIAL_REUSED_CODE,
    "query_generation": SourceStatus.RECONSTRUCTED,
    "per_class_query_prompt_bodies": SourceStatus.RECONSTRUCTED,
    "query_verification": SourceStatus.RECONSTRUCTED,
    "execution_result_semantics": SourceStatus.RECONSTRUCTED,
    "semantic_grounding_guard": SourceStatus.RECONSTRUCTED,
    "final_query_semantic_guard": SourceStatus.RECONSTRUCTED,
    "exact_content_novelty_guard": SourceStatus.RECONSTRUCTED,
    "config_patch_prompt": SourceStatus.OFFICIAL_PAPER,
    "deep_merge": SourceStatus.RECONSTRUCTED,
    "error_taxonomy": SourceStatus.OFFICIAL_PAPER,
    "feedback_loop": SourceStatus.OFFICIAL_PAPER,
    "coherence_rewrite_prompt": SourceStatus.OFFICIAL_PAPER,
    "missing_function_transform": SourceStatus.RECONSTRUCTED,
    "missing_parameter_transform": SourceStatus.RECONSTRUCTED,
    "deterministic_validation_gates": SourceStatus.OFFICIAL_PAPER,
    "quality_judge_prompt": SourceStatus.OFFICIAL_PAPER,
    "refine_classify_prompt": SourceStatus.OFFICIAL_PAPER,
    "refine_rewrite_prompt": SourceStatus.OFFICIAL_PAPER,
    "filesystem_queue_and_recovery": SourceStatus.RECONSTRUCTED,
    "terminal_result_commit_protocol": SourceStatus.RECONSTRUCTED,
    "structural_alignment_diagnostics": SourceStatus.RECONSTRUCTED,
    "gemma4_backend": SourceStatus.PROJECT_SUBSTITUTION,
}


FILES_INSPECTED = (
    "RODS/README.md",
    "EnvTuning/bfcl_env/multi_turn_utils.py",
    "EnvTuning/bfcl_env/multi_turn_checker.py",
    "EnvTuning/bfcl_env/func_source_code_wo_aug/*.py",
    "EnvTuning/env_tuning/interaction/new_multi_turn_fc.py",
    "EnvTuning/env_tuning/interaction/execution_manager.py",
    "EnvTuning/env_tuning/interaction/turn_manager.py",
    "EnvTuning/env_tuning/rods_matchtir_v1/lifecycle.py",
    "stage1_format_rl/schemas/rods_boundary_seed_v1.schema.json",
    "stage1_format_rl/schemas/rods_validated_candidate_v1.schema.json",
)
