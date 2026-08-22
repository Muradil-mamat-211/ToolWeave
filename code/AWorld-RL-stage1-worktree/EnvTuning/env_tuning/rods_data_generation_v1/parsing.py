"""Fail-closed parsers for all structured synthesis-agent responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .models import JudgeResult, PatchOperation, PlanTurn, PlannerResult


TAG_PATTERN = re.compile(r"<(?P<tag>[a-z_]+)>(?P<body>[\s\S]*?)</(?P=tag)>")


class StructuredParseError(ValueError):
    pass


def _ordered_top_level_tags(text: str) -> list[tuple[str, str]]:
    """Parse non-nested top-level tags and reject all non-whitespace residue."""

    matches = list(TAG_PATTERN.finditer(text))
    if not matches:
        raise StructuredParseError("no parseable tags")
    cursor = 0
    output: list[tuple[str, str]] = []
    for match in matches:
        if text[cursor : match.start()].strip():
            raise StructuredParseError("text exists outside structured tags")
        output.append((match.group("tag"), match.group("body").strip()))
        cursor = match.end()
    if text[cursor:].strip():
        raise StructuredParseError("text exists after structured tags")
    return output


def _require_nonempty(value: str, label: str) -> str:
    if not value.strip():
        raise StructuredParseError(f"{label} cannot be empty")
    return value.strip()


def parse_planner_response(
    text: str,
    *,
    allowed_functions: Iterable[str],
    class_for_function: dict[str, str],
    blocked_functions: Iterable[str] = (),
) -> PlannerResult:
    tags = _ordered_top_level_tags(text)
    if len(tags) < 4 or tags[0][0] != "reason" or tags[1][0] != "narrative":
        raise StructuredParseError("planner requires reason, narrative, then 2-5 turns")
    if any(tag != "turn" for tag, _ in tags[2:]):
        raise StructuredParseError("planner emitted an unexpected tag")
    if not 2 <= len(tags[2:]) <= 5:
        raise StructuredParseError("planner must emit 2-5 turns")

    allowed = set(allowed_functions)
    blocked = set(blocked_functions)
    turns: list[PlanTurn] = []
    for turn_id, (_, body) in enumerate(tags[2:]):
        if body.count(":") != 1:
            raise StructuredParseError("turn must use 'ClassName: func1, func2' format")
        class_name, raw_functions = (part.strip() for part in body.split(":", 1))
        names = tuple(part.strip() for part in raw_functions.split(",") if part.strip())
        if not class_name or not 1 <= len(names) <= 3:
            raise StructuredParseError("each planner turn requires one class and 1-3 functions")
        for name in names:
            if name not in allowed:
                raise StructuredParseError(f"planner used nonexistent function: {name}")
            if name in blocked:
                raise StructuredParseError(f"planner reused blocked function: {name}")
            if class_for_function.get(name) != class_name:
                raise StructuredParseError(
                    f"planner class/function mismatch: {class_name}:{name}"
                )
        turns.append(PlanTurn(turn_id, class_name, names))
    return PlannerResult(
        reason=_require_nonempty(tags[0][1], "planner reason"),
        narrative=_require_nonempty(tags[1][1], "planner narrative"),
        turns=tuple(turns),
    )


def parse_arguments_response(text: str) -> tuple[str, dict[str, Any]]:
    tags = _ordered_top_level_tags(text)
    if [tag for tag, _ in tags] != ["reason", "arguments"]:
        raise StructuredParseError("parameter response requires reason then arguments")
    reason = _require_nonempty(tags[0][1], "parameter reason")
    try:
        arguments = json.loads(tags[1][1])
    except json.JSONDecodeError as exc:
        raise StructuredParseError(f"arguments are not valid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise StructuredParseError("arguments must be a JSON object")
    return reason, arguments


def parse_query_response(text: str) -> tuple[str, str]:
    tags = _ordered_top_level_tags(text)
    if [tag for tag, _ in tags] != ["reason", "query"]:
        raise StructuredParseError("query response requires reason then query")
    return (
        _require_nonempty(tags[0][1], "query reason"),
        _require_nonempty(tags[1][1], "query"),
    )


def parse_verifier_response(text: str) -> tuple[str, bool]:
    tags = _ordered_top_level_tags(text)
    if [tag for tag, _ in tags] != ["reason", "verdict"]:
        raise StructuredParseError("verifier response requires reason then verdict")
    verdict = tags[1][1].strip().lower()
    if verdict not in {"accept", "reject"}:
        raise StructuredParseError("verifier verdict must be accept or reject")
    return _require_nonempty(tags[0][1], "verifier reason"), verdict == "accept"


def parse_rewrite_response(text: str, *, expected_count: int) -> list[str]:
    tags = _ordered_top_level_tags(text)
    if any(tag != "query" for tag, _ in tags) or len(tags) != expected_count:
        raise StructuredParseError(
            f"rewrite must emit exactly {expected_count} query tags"
        )
    return [_require_nonempty(body, "rewritten query") for _, body in tags]


def parse_judge_response(text: str) -> JudgeResult:
    tags = _ordered_top_level_tags(text)
    tag_names = [tag for tag, _ in tags]
    if tag_names not in (
        ["reason", "decision"],
        ["reason", "decision", "fail_reason"],
    ):
        raise StructuredParseError(
            "judge requires reason and decision, plus fail_reason when rejected"
        )
    decision = tags[1][1].strip().lower()
    if decision not in {"accept", "reject"}:
        raise StructuredParseError("judge decision must be accept or reject")
    fail_reason = tags[2][1].strip() if len(tags) == 3 else ""
    if decision == "reject" and not fail_reason:
        raise StructuredParseError("rejected judge result requires fail_reason")
    if decision == "accept" and fail_reason:
        raise StructuredParseError("accepted judge result cannot contain fail_reason")
    return JudgeResult(
        reason=_require_nonempty(tags[0][1], "judge reason"),
        decision=decision,
        fail_reason=fail_reason,
    )


def parse_refine_classification(text: str) -> tuple[str, str]:
    tags = _ordered_top_level_tags(text)
    if [tag for tag, _ in tags] != ["reason", "answer"]:
        raise StructuredParseError("refine classification requires reason then answer")
    answer = tags[1][1].strip()
    if answer not in {"query_fixable", "gt_unfixable"}:
        raise StructuredParseError("invalid refine classification")
    return _require_nonempty(tags[0][1], "refine reason"), answer


def parse_refine_rewrite(text: str) -> str:
    tags = _ordered_top_level_tags(text)
    if [tag for tag, _ in tags] != ["answer"]:
        raise StructuredParseError("refine rewrite must contain only answer")
    return _require_nonempty(tags[0][1], "refined query")


def parse_config_patch_response(text: str) -> tuple[str, list[PatchOperation]]:
    reason_match = re.fullmatch(
        r"\s*<reason>([\s\S]*?)</reason>([\s\S]*)", text
    )
    if reason_match is None:
        raise StructuredParseError("config patch response requires leading reason")
    reason = _require_nonempty(reason_match.group(1), "patch reason")
    remainder = reason_match.group(2)
    patch_matches = list(re.finditer(r"<patch>([\s\S]*?)</patch>", remainder))
    if not patch_matches:
        raise StructuredParseError("config patch response has no patch blocks")
    cursor = 0
    operations: list[PatchOperation] = []
    for match in patch_matches:
        if remainder[cursor : match.start()].strip():
            raise StructuredParseError("text exists outside patch blocks")
        inner = _ordered_top_level_tags(match.group(1))
        if [tag for tag, _ in inner] != ["class", "field", "value"]:
            raise StructuredParseError("patch requires class, field, value")
        raw_value = inner[2][1]
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value.strip()
        operations.append(
            PatchOperation(
                class_name=_require_nonempty(inner[0][1], "patch class"),
                field_path=_require_nonempty(inner[1][1], "patch field"),
                value=value,
            )
        )
        cursor = match.end()
    if remainder[cursor:].strip():
        raise StructuredParseError("text exists after patch blocks")
    return reason, operations


@dataclass(frozen=True)
class MissingFunctionChoice:
    reason: str
    affected_turn: int
    function_name: str


def parse_missing_function_response(text: str) -> MissingFunctionChoice:
    tags = _ordered_top_level_tags(text)
    if [tag for tag, _ in tags] != ["reason", "affected_turn", "missing_function"]:
        raise StructuredParseError("missing-function response has invalid tags")
    try:
        turn = int(tags[1][1])
    except ValueError as exc:
        raise StructuredParseError("affected_turn must be an integer") from exc
    return MissingFunctionChoice(
        _require_nonempty(tags[0][1], "missing-function reason"),
        turn,
        _require_nonempty(tags[2][1], "missing function"),
    )


@dataclass(frozen=True)
class MissingParameterChoice:
    reason: str
    affected_turn: int
    parameter_name: str
    affected_query: str
    recovery_query: str


def parse_missing_parameter_response(text: str) -> MissingParameterChoice:
    tags = _ordered_top_level_tags(text)
    expected = [
        "reason",
        "affected_turn",
        "missing_parameter",
        "affected_query",
        "recovery_query",
    ]
    if [tag for tag, _ in tags] != expected:
        raise StructuredParseError("missing-parameter response has invalid tags")
    try:
        turn = int(tags[1][1])
    except ValueError as exc:
        raise StructuredParseError("affected_turn must be an integer") from exc
    return MissingParameterChoice(
        _require_nonempty(tags[0][1], "missing-parameter reason"),
        turn,
        _require_nonempty(tags[2][1], "missing parameter"),
        _require_nonempty(tags[3][1], "affected query"),
        _require_nonempty(tags[4][1], "recovery query"),
    )
