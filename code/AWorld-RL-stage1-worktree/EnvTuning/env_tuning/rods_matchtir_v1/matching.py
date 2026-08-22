"""MatchTIR hard call matching adapted to structured BFCL calls.

Similarity follows the official MatchTIR implementation at commit
975c4535fbb86a49f21ff7d291a1fa822f827684.  Unlike the repository helper
named ``hungarian_assignment`` (which greedily sorts edges), ToolWeave uses an
actual maximum-weight Hungarian assignment, matching the paper-level objective.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from env_tuning.interaction.utils import ast_parse

from .provenance import to_builtin


@dataclass(frozen=True)
class CanonicalToolCall:
    """One individual tool call at matching resolution."""

    name: str
    arguments: Mapping[str, Any]
    valid: bool = True
    call_idx: int = 0

    @classmethod
    def from_prediction(cls, raw: Mapping[str, Any], fallback_idx: int) -> "CanonicalToolCall":
        name = raw.get("name", "")
        arguments = raw.get("arguments", {})
        valid = bool(raw.get("valid", True))
        if not isinstance(name, str):
            name = ""
            valid = False
        if not isinstance(arguments, Mapping):
            arguments = {}
            valid = False
        return cls(
            name=name.strip(),
            arguments=to_builtin(arguments),
            valid=valid and bool(name.strip()),
            call_idx=int(raw.get("call_idx", fallback_idx)),
        )


@dataclass(frozen=True)
class HardMatchResult:
    """One-to-one call rewards and assignment diagnostics."""

    rewards: tuple[float, ...]
    assignments: tuple[int | None, ...]
    similarities: tuple[float, ...]
    score_matrix: tuple[tuple[float, ...], ...]

    @property
    def matched_count(self) -> int:
        return sum(index is not None for index in self.assignments)


def _multiset_jaccard(left: Sequence[Any], right: Sequence[Any]) -> float:
    """Official MatchTIR ``match_score`` semantics, including empty==empty."""

    if list(left) == list(right):
        return 1.0
    if not left or not right:
        return 0.0
    left_count = Counter(left)
    right_count = Counter(right)
    intersection = sum(
        min(left_count[key], right_count[key])
        for key in left_count.keys() & right_count.keys()
    )
    union = len(left) + len(right) - intersection
    return intersection / union if union > 0 else 0.0


def matchtir_similarity(predicted: CanonicalToolCall, ground_truth: CanonicalToolCall) -> float:
    """Compute the official hard-MatchTIR name/name-set/value similarity."""

    if not predicted.valid or not ground_truth.valid:
        return 0.0
    # Official code compares lower-cased names.  The EnvTuning parser already
    # strips predicted names; canonical GT names are stripped as well.
    if predicted.name.lower() != ground_truth.name.lower():
        return 0.0

    pred_args = predicted.arguments
    gt_args = ground_truth.arguments
    name_score = 1.0
    parameter_name_score = _multiset_jaccard(list(gt_args.keys()), list(pred_args.keys()))
    parameter_value_score = sum(
        1.0
        for key, expected in gt_args.items()
        if key in pred_args and pred_args[key] == expected
    )
    denominator = 2.0 + len(gt_args)
    return (name_score + parameter_name_score + parameter_value_score) / denominator


def parse_bfcl_ground_truth(raw_calls: Sequence[Any]) -> list[CanonicalToolCall]:
    """Parse one BFCL user turn with EnvTuning's existing AST parser."""

    canonical: list[CanonicalToolCall] = []
    for raw_call in to_builtin(raw_calls):
        if not isinstance(raw_call, str):
            raise ValueError(f"BFCL ground-truth call must be a string, got {type(raw_call)!r}")
        parsed_calls = ast_parse(raw_call)
        for parsed in parsed_calls:
            if not isinstance(parsed, Mapping) or len(parsed) != 1:
                raise ValueError(f"Unexpected BFCL AST call: {parsed!r}")
            name, arguments = next(iter(parsed.items()))
            if not isinstance(name, str) or not isinstance(arguments, Mapping):
                raise ValueError(f"Unexpected BFCL AST call: {parsed!r}")
            canonical.append(
                CanonicalToolCall(
                    name=name.strip(),
                    arguments=to_builtin(arguments),
                    valid=bool(name.strip()),
                    call_idx=len(canonical),
                )
            )
    return canonical


def hard_match_calls(
    predicted: Sequence[CanonicalToolCall],
    ground_truth: Sequence[CanonicalToolCall],
    *,
    unmatched_penalty: float = 0.0,
) -> HardMatchResult:
    """Run one true maximum-weight Hungarian match over all calls in a scope.

    Zero-similarity assignments are treated as unmatched, matching the official
    implementation's positive-edge gate.  ToolWeave's unmatched penalty is 0.
    """

    pred_count = len(predicted)
    gt_count = len(ground_truth)
    if pred_count == 0:
        return HardMatchResult((), (), (), ())

    matrix = np.zeros((pred_count, gt_count), dtype=np.float64)
    for pred_idx, pred_call in enumerate(predicted):
        for gt_idx, gt_call in enumerate(ground_truth):
            matrix[pred_idx, gt_idx] = matchtir_similarity(pred_call, gt_call)

    assignments: list[int | None] = [None] * pred_count
    similarities = [0.0] * pred_count
    if gt_count > 0:
        row_indices, col_indices = linear_sum_assignment(matrix, maximize=True)
        for pred_idx, gt_idx in zip(row_indices.tolist(), col_indices.tolist()):
            similarity = float(matrix[pred_idx, gt_idx])
            if similarity > 0.0:
                assignments[pred_idx] = gt_idx
                similarities[pred_idx] = similarity

    rewards = [
        similarities[idx] if assignments[idx] is not None else float(unmatched_penalty)
        for idx in range(pred_count)
    ]
    return HardMatchResult(
        rewards=tuple(rewards),
        assignments=tuple(assignments),
        similarities=tuple(similarities),
        score_matrix=tuple(tuple(float(value) for value in row) for row in matrix.tolist()),
    )
