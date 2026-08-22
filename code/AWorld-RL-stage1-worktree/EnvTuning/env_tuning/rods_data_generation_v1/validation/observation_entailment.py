"""Policy-visible observation entailment for explicit factual claims.

SOURCE_STATUS = PROJECT_OBSERVATION_ENTAILMENT_GUARD

This deterministic precision guard is a project rule, not a published RODS
threshold.  It examines only statements that explicitly attribute a fact to a
prior tool result (for example, "the logs show ...").  It never uses hidden VM
state, embeddings, or fuzzy similarity.  Claims that cannot be decided from
exact lexical/value evidence are surfaced to the existing fail-closed final
semantic verifier instead of being guessed here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

from ..models import ConversationDraft, GateResult
from ..result_semantics import (
    ExecutionSemanticOutcome,
    classify_execution_result,
    normalize_execution_result,
)


SOURCE_STATUS = "PROJECT_OBSERVATION_ENTAILMENT_GUARD"

_ATTRIBUTED_CLAIM_PATTERNS = (
    re.compile(
        r"\b(?P<source>(?:the\s+)?(?:system\s+)?logs?|(?:the\s+)?search\s+results?|"
        r"(?:the\s+)?results?|(?:the\s+)?tool\s+output|(?:the\s+)?output|"
        r"(?:the\s+)?file|(?:the\s+)?report|(?:the\s+)?account|"
        r"(?:the\s+)?booking|(?:the\s+)?record|(?:the\s+)?observation)\s+"
        r"(?P<verb>shows?|showed|indicates?|indicated|says?|said|contains?|contained|"
        r"reveals?|revealed|reports?|reported)\s+(?P<claim>[^.!?;]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<source>I)\s+(?P<verb>found|discovered|observed|learned)\s+"
        r"(?P<claim>[^.!?;]+)",
        re.IGNORECASE,
    ),
)

_ACTION_TAIL = re.compile(
    r"\s+(?:and|then)\s+(?:to\s+)?(?:please\s+)?(?:ask|email|forward|message|notify|post|"
    r"review|send|share|tell|write)\b.*$",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "my", "of", "on", "that", "the",
    "their", "there", "this", "to", "was", "were", "with", "your",
    "show", "shows", "showed", "indicate", "indicates", "indicated",
    "say", "says", "said", "contain", "contains", "contained", "reveal",
    "reveals", "revealed", "report", "reports", "reported", "found",
    "discovered", "observed", "learned",
}
_VAGUE_ANCHORS = {
    "data", "detail", "details", "information", "issue", "result", "results",
    "something", "thing", "things", "update",
}
_NEGATIVE_WORDS = {"empty", "none", "no", "nothing", "not", "zero"}


def _canonical_token(token: str) -> str:
    value = token.casefold()
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and len(value) > 3 and not value.endswith("ss"):
        return value[:-1]
    return value


def _tokens(value: str) -> list[str]:
    return [_canonical_token(token) for token in re.findall(r"[A-Za-z0-9]+", value)]


def _claim_anchors(claim: str) -> list[str]:
    return sorted(
        {
            token
            for token in _tokens(claim)
            if token not in _STOPWORDS and token not in _VAGUE_ANCHORS
        }
    )


def _iter_visible_scalars(value: Any, *, path: str) -> Iterable[tuple[str, Any]]:
    normalized = normalize_execution_result(value)
    if isinstance(normalized, Mapping):
        for key, child in normalized.items():
            yield from _iter_visible_scalars(child, path=f"{path}.{key}")
    elif isinstance(normalized, (list, tuple)):
        for index, child in enumerate(normalized):
            yield from _iter_visible_scalars(child, path=f"{path}[{index}]")
    else:
        yield path, normalized


def _is_empty_visible_result(value: Any) -> bool:
    normalized = normalize_execution_result(value)
    if isinstance(normalized, Mapping):
        return bool(normalized) and all(_is_empty_visible_result(child) for child in normalized.values())
    if isinstance(normalized, (list, tuple)):
        return len(normalized) == 0 or all(_is_empty_visible_result(child) for child in normalized)
    return normalized is None or (isinstance(normalized, str) and not normalized.strip())


def _extract_claims(query: str) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    occupied: set[tuple[int, int]] = set()
    for pattern in _ATTRIBUTED_CLAIM_PATTERNS:
        for match in pattern.finditer(query):
            span = match.span()
            if span in occupied:
                continue
            occupied.add(span)
            raw_claim = _ACTION_TAIL.sub("", match.group("claim")).strip(" ,:-")
            if raw_claim:
                claims.append(
                    {
                        "source_phrase": match.group("source").strip(),
                        "attribution_verb": match.group("verb").strip(),
                        "claim": raw_claim,
                    }
                )
    return claims


def observation_entailment_gate(draft: ConversationDraft) -> GateResult:
    """Reject explicit prior-observation claims unsupported by visible results."""

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(draft.turns):
        for extracted in _extract_claims(turn.query):
            evidence: list[dict[str, Any]] = []
            evidence_tokens: set[str] = set()
            has_empty_result = False
            for prior_turn_index, prior_turn in enumerate(draft.turns[:turn_index]):
                for call_index, record in enumerate(prior_turn.execution_records):
                    semantic = classify_execution_result(
                        record.call.name, record.execution_result
                    )
                    if semantic.outcome == ExecutionSemanticOutcome.HARD_ERROR:
                        continue
                    normalized = normalize_execution_result(record.execution_result)
                    has_empty_result = has_empty_result or _is_empty_visible_result(normalized)
                    scalar_rows = list(
                        _iter_visible_scalars(normalized, path="result")
                    )
                    evidence.append(
                        {
                            "source_turn": prior_turn_index,
                            "source_call": call_index,
                            "function": record.call.name,
                            "semantic_outcome": semantic.outcome.value,
                            "result": normalized,
                            "scalar_paths": [path for path, _ in scalar_rows],
                        }
                    )
                    evidence_tokens.update(_tokens(json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=repr)))

            anchors = _claim_anchors(extracted["claim"])
            missing = [anchor for anchor in anchors if anchor not in evidence_tokens]
            claim_tokens = set(_tokens(extracted["claim"]))
            negative_claim = bool(claim_tokens & _NEGATIVE_WORDS)

            row: dict[str, Any] = {
                "turn_id": turn_index,
                **extracted,
                "claim_anchors": anchors,
                "supported_anchors": [anchor for anchor in anchors if anchor in evidence_tokens],
                "missing_anchors": missing,
                "prior_observations": evidence,
                "hidden_environment_state_used": False,
                "embedding_or_similarity_threshold_used": False,
            }
            if negative_claim and has_empty_result:
                row["status"] = "ENTAILED"
                row["decision_basis"] = "explicit negative claim matches an empty prior visible result"
            elif len(anchors) >= 2 and not missing:
                row["status"] = "ENTAILED"
                row["decision_basis"] = "all deterministic factual anchors occur in prior visible observations"
            elif len(anchors) >= 2:
                row["status"] = "UNSUPPORTED_OBSERVATION_CLAIM"
                row["decision_basis"] = "one or more factual anchors are absent from all prior visible observations"
                failures.append(row)
            else:
                row["status"] = "DEFERRED_TO_FINAL_SEMANTIC_VERIFIER"
                row["decision_basis"] = "claim lacks enough deterministic anchors for a safe exact decision"
                deferred.append(row)
            checks.append(row)

    metadata = {
        "source_status": SOURCE_STATUS,
        "checks": checks,
        "failures": failures,
        "deferred_to_final_semantic_verifier": deferred,
        "policy_visible_prior_observations_only": True,
        "hidden_environment_state_used": False,
        "embedding_or_similarity_threshold_used": False,
    }
    draft.structural_profile.setdefault("project_semantic_guards", {})[
        "observation_entailment"
    ] = metadata
    if failures:
        first = failures[0]
        return GateResult(
            "observation_entailment_gate",
            False,
            "UNSUPPORTED_OBSERVATION_CLAIM at "
            f"turn={first['turn_id']}: {first['claim']!r}; "
            f"missing evidence anchors={first['missing_anchors']}",
            metadata,
        )
    return GateResult(
        "observation_entailment_gate",
        True,
        "all deterministically decidable observation-attributed claims are entailed",
        metadata,
    )
