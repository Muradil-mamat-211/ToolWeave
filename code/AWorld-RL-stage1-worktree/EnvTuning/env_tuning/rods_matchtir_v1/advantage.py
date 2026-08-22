"""Interaction-aware local credit and residual fusion with pure RODS GRPO.

The caller must first run the unchanged RODS/veRL GRPO estimator.  This module
never changes ``token_level_rewards`` and never recomputes the global outcome
advantage.  It only constructs a token-aligned local residual from structured
rollout provenance.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from .matching import CanonicalToolCall, hard_match_calls, parse_bfcl_ground_truth
from .provenance import to_builtin


@dataclass(frozen=True)
class LocalCreditConfig:
    """Configuration for ToolWeave's MatchTIR-inspired local credit only."""

    enabled: bool = True
    weight: float = 1.0
    gamma: float = 0.9
    matching: str = "hard"
    unmatched_penalty: float = 0.0
    min_group_size: int = 2
    epsilon: float = 1.0e-6
    mode: str = "runtime_interaction_final"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "LocalCreditConfig":
        raw = raw or {}
        config = cls(
            enabled=bool(raw.get("enabled", True)),
            weight=float(raw.get("weight", 1.0)),
            gamma=float(raw.get("gamma", 0.9)),
            matching=str(raw.get("matching", "hard")),
            unmatched_penalty=float(raw.get("unmatched_penalty", 0.0)),
            min_group_size=int(raw.get("min_group_size", 2)),
            epsilon=float(raw.get("epsilon", 1.0e-6)),
            mode=str(raw.get("mode", "runtime_interaction_final")),
        )
        if config.matching != "hard":
            raise ValueError("ToolWeave local credit supports matching='hard' only")
        if config.mode != "runtime_interaction_final":
            raise ValueError("matchtir_local.mode must be 'runtime_interaction_final'")
        if not 0.0 <= config.gamma <= 1.0:
            raise ValueError("matchtir_local.gamma must be in [0, 1]")
        if config.min_group_size < 2:
            raise ValueError("matchtir_local.min_group_size must be at least 2")
        if config.epsilon <= 0:
            raise ValueError("matchtir_local.epsilon must be positive")
        return config


@dataclass
class _StepCredit:
    """One real non-answer runtime interaction in a BFCL user turn."""

    batch_index: int
    uid: str
    data_type: str
    user_turn_id: int
    policy_step_id: int
    runtime_interaction_index: int
    # Backward-compatible diagnostic metadata only.  It is never used for
    # ordering, discounting, peer grouping, normalization, or token alignment.
    tool_attempt_index: int | None
    depth: int
    span_start: int
    span_end: int
    actor_span_reliable: bool
    response_type: str
    attempted_action_type: str
    call_parse_reliable: bool
    call_count: int
    call_rewards: list[float]
    step_reward: float
    local_return: float = 0.0
    local_advantage: float = 0.0
    local_active: bool = False
    token_residual_applied: bool = False
    peer_support: int = 0
    peer_mean: float = 0.0
    peer_sample_std: float = 0.0
    follows_missing_turn: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "uid": self.uid,
            "data_type": self.data_type,
            "user_turn_id": self.user_turn_id,
            "policy_step_id": self.policy_step_id,
            "runtime_interaction_index": self.runtime_interaction_index,
            "runtime_depth": self.runtime_interaction_index,
            "tool_attempt_index": self.tool_attempt_index,
            "depth": self.depth,
            "actor_span": {"start": self.span_start, "end": self.span_end},
            "actor_span_reliable": self.actor_span_reliable,
            "response_type": self.response_type,
            "attempted_action_type": self.attempted_action_type,
            "call_parse_reliable": self.call_parse_reliable,
            "call_count": self.call_count,
            "call_rewards": list(self.call_rewards),
            "step_reward": self.step_reward,
            "local_return": self.local_return,
            "local_advantage": self.local_advantage,
            "local_active": self.local_active,
            "token_residual_applied": self.token_residual_applied,
            "peer_support": self.peer_support,
            "peer_mean": self.peer_mean,
            "peer_sample_std": self.peer_sample_std,
            "follows_missing_turn": self.follows_missing_turn,
        }


@dataclass(frozen=True)
class FusionResult:
    """Fused tensors plus CPU-side diagnostics used by tests and logging."""

    advantages: torch.Tensor
    returns: torch.Tensor
    local_advantages: torch.Tensor
    local_token_mask: torch.Tensor
    metrics: dict[str, float]
    step_records: tuple[dict[str, Any], ...]


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(_mean([value * value for value in values])) if values else 0.0


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _rollout_scalar_advantages(
    advantages: torch.Tensor,
    actor_response_mask: torch.Tensor,
) -> list[float]:
    mask = actor_response_mask.bool()
    has_actor_token = mask.any(dim=-1)
    first_indices = mask.to(dtype=torch.int64).argmax(dim=-1, keepdim=True)
    values = advantages.gather(dim=-1, index=first_indices).squeeze(-1)
    values = torch.where(has_actor_token, values, torch.zeros_like(values))
    return [float(value) for value in values.detach().cpu().tolist()]


def _global_metrics(
    *,
    token_level_rewards: torch.Tensor,
    rods_advantages: torch.Tensor,
    actor_response_mask: torch.Tensor,
) -> tuple[dict[str, float], list[float], list[float]]:
    progress = [float(value) for value in token_level_rewards.detach().sum(dim=-1).cpu().tolist()]
    rods_scalars = _rollout_scalar_advantages(rods_advantages.detach(), actor_response_mask.detach())
    return (
        {
            "rods_matchtir_v1/global/progress_reward_mean": _mean(progress),
            "rods_matchtir_v1/global/progress_reward_std": _population_std(progress),
            "rods_matchtir_v1/global/A_RODS_mean": _mean(rods_scalars),
            "rods_matchtir_v1/global/A_RODS_std": _population_std(rods_scalars),
        },
        progress,
        rods_scalars,
    )


def _empty_local_metrics(rods_scalars: Sequence[float]) -> dict[str, float]:
    """Stable metric schema for disabled or provenance-free batches."""

    return {
        "rods_matchtir_v1/call/num_predicted_calls": 0.0,
        "rods_matchtir_v1/call/num_gt_calls": 0.0,
        "rods_matchtir_v1/call/matched_call_count": 0.0,
        "rods_matchtir_v1/call/unmatched_call_count": 0.0,
        "rods_matchtir_v1/call/match_rate": 0.0,
        "rods_matchtir_v1/call/similarity_mean": 0.0,
        "rods_matchtir_v1/call/similarity_max": 0.0,
        "rods_matchtir_v1/step/policy_steps_per_user_turn": 0.0,
        "rods_matchtir_v1/step/calls_per_policy_step": 0.0,
        "rods_matchtir_v1/step/step_reward_mean": 0.0,
        "rods_matchtir_v1/step/local_return_mean": 0.0,
        "rods_matchtir_v1/step/local_return_std": 0.0,
        "rods_matchtir_v1/interaction/non_answer_runtime_interactions": 0.0,
        "rods_matchtir_v1/interaction/tool_attempt_interactions": 0.0,
        "rods_matchtir_v1/interaction/parsed_tool_interactions": 0.0,
        "rods_matchtir_v1/interaction/parse_error_tool_attempts": 0.0,
        "rods_matchtir_v1/interaction/parse_error_runtime_interactions": 0.0,
        "rods_matchtir_v1/interaction/unparsed_runtime_interactions": 0.0,
        "rods_matchtir_v1/interaction/peer_supported_interactions": 0.0,
        "rods_matchtir_v1/interaction/unsupported_local_interactions": 0.0,
        "rods_matchtir_v1/interaction/nonzero_local_advantages": 0.0,
        "rods_matchtir_v1/interaction/parse_error_interactions_receiving_local_residual": 0.0,
        "rods_matchtir_v1/interaction/parse_error_tokens_receiving_local_residual": 0.0,
        "rods_matchtir_v1/normalization/local_support_size": 0.0,
        "rods_matchtir_v1/normalization/singleton_local_count": 0.0,
        "rods_matchtir_v1/normalization/singleton_local_rate": 0.0,
        "rods_matchtir_v1/normalization/zero_variance_local_count": 0.0,
        "rods_matchtir_v1/normalization/local_adv_mean": 0.0,
        "rods_matchtir_v1/normalization/local_adv_std": 0.0,
        "rods_matchtir_v1/normalization/local_coverage": 0.0,
        "rods_matchtir_v1/fusion/A_new_mean": _mean(rods_scalars),
        "rods_matchtir_v1/fusion/A_new_std": _population_std(rods_scalars),
        "rods_matchtir_v1/fusion/RMS_A_RODS": _rms(rods_scalars),
        "rods_matchtir_v1/fusion/RMS_A_local": 0.0,
        "rods_matchtir_v1/fusion/RMS_A_new": _rms(rods_scalars),
        "rods_matchtir_v1/fusion/sign_flip_rate": 0.0,
        "rods_matchtir_v1/missing/missing_turn_local_coverage": 0.0,
        "rods_matchtir_v1/missing/followup_normal_turn_local_coverage": 0.0,
        "rods_matchtir_v1/provenance/unreliable_rollout_count": 0.0,
        "rods_matchtir_v1/provenance/invalid_gt_turn_count": 0.0,
        "rods_matchtir_v1/provenance/unreliable_tool_turn_count": 0.0,
        "rods_matchtir_v1/provenance/span_assignment_failure_count": 0.0,
    }


def _extract_span(step: Mapping[str, Any], response_length: int) -> tuple[int, int] | None:
    span = step.get("actor_span")
    if not isinstance(span, Mapping):
        return None
    try:
        start = int(span["start"])
        end = int(span["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if start < 0 or end <= start or end > response_length:
        return None
    return start, end


def _build_step_credits(
    *,
    uids: Sequence[Any],
    rollout_provenance: Sequence[Any],
    data_sources: Sequence[Any],
    response_length: int,
    config: LocalCreditConfig,
    counters: MutableMapping[str, Any],
) -> list[_StepCredit]:
    """Build credits on the real non-answer runtime-interaction timeline.

    ``runtime_interaction_index`` is the only temporal axis.  The historical
    ``tool_attempt_index`` is copied into diagnostics when present but never
    controls sequence construction, discount distance, normalization, or token
    assignment.
    """

    credits: list[_StepCredit] = []

    for batch_index, raw_provenance in enumerate(rollout_provenance):
        provenance = to_builtin(raw_provenance)
        if not isinstance(provenance, Mapping):
            counters["unreliable_rollouts"] += 1
            continue
        policy_steps = provenance.get("policy_steps", [])
        gt_turns = provenance.get("ground_truth", [])
        if not isinstance(policy_steps, list) or not isinstance(gt_turns, list):
            counters["unreliable_rollouts"] += 1
            continue

        steps_by_turn: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for raw_step in policy_steps:
            if not isinstance(raw_step, Mapping):
                continue
            try:
                user_turn_id = int(raw_step.get("user_turn_id"))
            except (TypeError, ValueError):
                continue
            steps_by_turn[user_turn_id].append(raw_step)

        observed_turn_ids = set(steps_by_turn)
        all_turn_ids = sorted(observed_turn_ids | set(range(len(gt_turns))))
        for user_turn_id in all_turn_ids:
            raw_gt = gt_turns[user_turn_id] if user_turn_id < len(gt_turns) else []
            raw_gt = to_builtin(raw_gt)
            is_missing = not raw_gt
            if is_missing:
                counters["missing_turn_count"] += 1
                counters["missing_turn_actor_steps"] += len(steps_by_turn.get(user_turn_id, []))
                counters["policy_steps_per_turn"].append(0.0)
                continue

            try:
                ground_truth = parse_bfcl_ground_truth(raw_gt)
            except (AssertionError, SyntaxError, TypeError, ValueError):
                counters["invalid_gt_turns"] += 1
                counters["policy_steps_per_turn"].append(0.0)
                continue
            counters["num_gt_calls"] += len(ground_truth)

            def runtime_index(step: Mapping[str, Any]) -> int:
                raw_index = step.get(
                    "runtime_interaction_index", step.get("policy_step_id", -1)
                )
                try:
                    return int(raw_index)
                except (TypeError, ValueError):
                    return -1

            sorted_steps = sorted(
                steps_by_turn.get(user_turn_id, []),
                key=runtime_index,
            )
            runtime_indices = [runtime_index(step) for step in sorted_steps]
            if runtime_indices != list(range(len(sorted_steps))):
                counters["unreliable_tool_turns"] += 1
                counters["policy_steps_per_turn"].append(0.0)
                continue

            # A valid final answer closes the BFCL user turn and is deliberately
            # excluded from the local sequence.  Every preceding non-answer
            # generation remains a real temporal step, including parser errors
            # and malformed/unknown actions with no parsed calls.
            answer_positions = [
                index
                for index, step in enumerate(sorted_steps)
                if str(step.get("response_type", "")) == "answer"
            ]
            if answer_positions:
                if (
                    len(answer_positions) != 1
                    or answer_positions[0] != len(sorted_steps) - 1
                ):
                    counters["unreliable_tool_turns"] += 1
                    counters["policy_steps_per_turn"].append(0.0)
                    continue
                non_answer_steps = sorted_steps[: answer_positions[0]]
            else:
                non_answer_steps = sorted_steps

            # Tuple fields: metadata, runtime index, legacy diagnostic
            # tool-attempt index, actor span (optional), span reliability,
            # parsed calls, response type, attempted action type, and
            # structured-call parse reliability.
            runtime_steps: list[
                tuple[
                    Mapping[str, Any],
                    int,
                    int | None,
                    tuple[int, int] | None,
                    bool,
                    list[CanonicalToolCall],
                    str,
                    str,
                    bool,
                ]
            ] = []
            unreliable_tool_turn = False
            legacy_tool_attempt_index = 0
            for step in non_answer_steps:
                has_split_provenance = (
                    "attempted_action_type" in step
                    or step.get("local_credit_semantics")
                    in {"interaction_aware_v2", "runtime_interaction_final"}
                )
                response_type = str(step.get("response_type", ""))
                if has_split_provenance:
                    attempted_action_type = str(
                        step.get("attempted_action_type", "unknown")
                    )
                    if not bool(step.get("temporal_provenance_reliable", False)):
                        unreliable_tool_turn = True
                        break
                    raw_tool_attempt_index = step.get("tool_attempt_index")
                    if raw_tool_attempt_index is None:
                        tool_attempt_index = None
                    else:
                        try:
                            tool_attempt_index = int(raw_tool_attempt_index)
                        except (TypeError, ValueError):
                            tool_attempt_index = None
                    call_parse_reliable = bool(step.get("call_parse_reliable", False))
                    actor_span_reliable = bool(step.get("actor_span_reliable", False))
                else:
                    # Compatibility for old valid-tool fixtures/artifacts.  A
                    # legacy parser error lacks split temporal provenance and
                    # must be enriched by deterministic parser replay before
                    # it can enter the formal runtime sequence.
                    if not bool(step.get("provenance_reliable", False)):
                        unreliable_tool_turn = True
                        break
                    attempted_action_type = (
                        "tool_call" if response_type == "tool_call" else "unknown"
                    )
                    tool_attempt_index = (
                        legacy_tool_attempt_index
                        if response_type == "tool_call"
                        else None
                    )
                    if response_type == "tool_call":
                        legacy_tool_attempt_index += 1
                    call_parse_reliable = response_type == "tool_call"
                    actor_span_reliable = True

                current_runtime_index = runtime_index(step)
                if current_runtime_index < 0:
                    unreliable_tool_turn = True
                    break

                span = _extract_span(step, response_length)
                if actor_span_reliable and span is None:
                    actor_span_reliable = False
                    counters["span_metadata_failures"] += 1

                raw_calls = step.get("calls", [])
                calls: list[CanonicalToolCall] = []
                if response_type == "tool_call":
                    if not call_parse_reliable or not isinstance(raw_calls, list):
                        # A supposedly parsed tool action with broken structured
                        # metadata is an infrastructure failure, not model r=0.
                        unreliable_tool_turn = True
                        break
                    for fallback_idx, raw_call in enumerate(raw_calls):
                        if not isinstance(raw_call, Mapping):
                            continue
                        canonical = CanonicalToolCall.from_prediction(
                            raw_call, fallback_idx
                        )
                        if canonical.valid:
                            calls.append(canonical)
                    counters["parsed_tool_interactions"] += 1
                else:
                    if call_parse_reliable:
                        unreliable_tool_turn = True
                        break
                    if response_type == "parse_error":
                        counters["parse_error_runtime_interactions"] += 1
                        if attempted_action_type == "tool_call":
                            counters["parse_error_tool_attempts"] += 1
                    if attempted_action_type == "unknown":
                        counters["unknown_action_interactions"] += 1
                    counters["unparsed_runtime_interactions"] += 1

                runtime_steps.append(
                    (
                        step,
                        current_runtime_index,
                        tool_attempt_index,
                        span,
                        actor_span_reliable and span is not None,
                        calls,
                        response_type,
                        attempted_action_type,
                        call_parse_reliable,
                    )
                )

            if unreliable_tool_turn:
                counters["unreliable_tool_turns"] += 1
                counters["policy_steps_per_turn"].append(0.0)
                continue

            counters["policy_steps_per_turn"].append(float(len(runtime_steps)))
            counters["runtime_interactions"] += len(runtime_steps)
            counters["tool_attempt_interactions"] += sum(
                step_fields[7] == "tool_call" for step_fields in runtime_steps
            )
            if not runtime_steps:
                continue

            predicted_calls: list[CanonicalToolCall] = []
            call_to_step: list[int] = []
            for local_step_index, (_, _, _, _, _, calls, _, _, _) in enumerate(
                runtime_steps
            ):
                for call in calls:
                    predicted_calls.append(call)
                    call_to_step.append(local_step_index)
            # Exactly one assignment for every successfully parsed individual
            # call across the complete (prompt, rollout, BFCL user turn) scope.
            match = hard_match_calls(
                predicted_calls,
                ground_truth,
                unmatched_penalty=config.unmatched_penalty,
            )
            counters["num_predicted_calls"] += len(predicted_calls)
            counters["matched_calls"] += match.matched_count
            counters["unmatched_calls"] += len(predicted_calls) - match.matched_count
            counters["matched_similarities"].extend(
                similarity
                for similarity, assignment in zip(match.similarities, match.assignments)
                if assignment is not None
            )

            rewards_by_step: list[list[float]] = [[] for _ in runtime_steps]
            for call_index, reward in enumerate(match.rewards):
                rewards_by_step[call_to_step[call_index]].append(float(reward))

            turn_credits: list[_StepCredit] = []
            follows_missing = (
                user_turn_id > 0
                and user_turn_id - 1 < len(gt_turns)
                and not to_builtin(gt_turns[user_turn_id - 1])
            )
            for (
                step,
                current_runtime_index,
                tool_attempt_index,
                span,
                actor_span_reliable,
                calls,
                response_type,
                attempted_action_type,
                call_parse_reliable,
            ), call_rewards in zip(runtime_steps, rewards_by_step):
                if len(call_rewards) != len(calls):
                    raise AssertionError(
                        "flattened calls did not map back to their runtime interaction"
                    )
                # Calls emitted in one generation are averaged before temporal
                # discounting; they never become separate timesteps.
                step_reward = _mean(call_rewards)
                credit = _StepCredit(
                    batch_index=batch_index,
                    uid=str(uids[batch_index]),
                    data_type=str(data_sources[batch_index]),
                    user_turn_id=user_turn_id,
                    policy_step_id=int(step.get("policy_step_id", current_runtime_index)),
                    runtime_interaction_index=current_runtime_index,
                    tool_attempt_index=tool_attempt_index,
                    depth=current_runtime_index,
                    span_start=span[0] if span is not None else -1,
                    span_end=span[1] if span is not None else -1,
                    actor_span_reliable=actor_span_reliable,
                    response_type=response_type,
                    attempted_action_type=attempted_action_type,
                    call_parse_reliable=call_parse_reliable,
                    call_count=len(calls),
                    call_rewards=call_rewards,
                    step_reward=step_reward,
                    follows_missing_turn=follows_missing,
                )
                turn_credits.append(credit)
                counters["calls_per_step"].append(float(len(calls)))
                counters["step_rewards"].append(step_reward)

            running_return = 0.0
            for credit in reversed(turn_credits):
                running_return = credit.step_reward + config.gamma * running_return
                credit.local_return = running_return
            counters["local_returns"].extend(credit.local_return for credit in turn_credits)
            credits.extend(turn_credits)

    return credits


def fuse_rods_and_local_advantages(
    *,
    rods_advantages: torch.Tensor,
    rods_returns: torch.Tensor,
    token_level_rewards: torch.Tensor,
    actor_response_mask: torch.Tensor,
    uids: Sequence[Any],
    rollout_provenance: Sequence[Any] | None,
    data_sources: Sequence[Any] | None = None,
    config: LocalCreditConfig | Mapping[str, Any] | None = None,
) -> FusionResult:
    """Construct and fuse BFCL user-turn-scoped local advantages.

    Shapes are ``[batch/rollout, response_token]`` for all tensors.  The input
    ``rods_advantages`` must already be the unchanged group-normalized Progress
    Reward advantage.  Local matching is CPU metadata work under ``no_grad``.
    """

    if not isinstance(config, LocalCreditConfig):
        config = LocalCreditConfig.from_mapping(config)
    if rods_advantages.shape != rods_returns.shape:
        raise ValueError("RODS advantages and returns must have identical shapes")
    if rods_advantages.shape != token_level_rewards.shape:
        raise ValueError("token rewards and advantages must have identical [batch, token] shapes")
    if rods_advantages.shape != actor_response_mask.shape:
        raise ValueError("actor_response_mask must align with response-token advantages")
    batch_size, response_length = rods_advantages.shape
    if len(uids) != batch_size:
        raise ValueError("uid count must equal rollout batch size")

    global_metrics, _, rods_scalars = _global_metrics(
        token_level_rewards=token_level_rewards,
        rods_advantages=rods_advantages,
        actor_response_mask=actor_response_mask,
    )
    empty_metrics = _empty_local_metrics(rods_scalars)
    zero_local = torch.zeros_like(rods_advantages)
    zero_mask = torch.zeros_like(actor_response_mask, dtype=torch.bool)

    # Exact baseline fallback: do not clone, renormalize, or otherwise rewrite
    # the tensors produced by the original RODS GRPO path.
    if not config.enabled or config.weight == 0.0:
        return FusionResult(
            advantages=rods_advantages,
            returns=rods_returns,
            local_advantages=zero_local,
            local_token_mask=zero_mask,
            metrics={**global_metrics, **empty_metrics},
            step_records=(),
        )
    if rollout_provenance is None or len(rollout_provenance) != batch_size:
        metrics = {**global_metrics, **empty_metrics}
        metrics["rods_matchtir_v1/provenance/unreliable_rollout_count"] = float(batch_size)
        return FusionResult(
            advantages=rods_advantages,
            returns=rods_returns,
            local_advantages=zero_local,
            local_token_mask=zero_mask,
            metrics=metrics,
            step_records=(),
        )
    if data_sources is None:
        data_sources = ["unknown"] * batch_size
    if len(data_sources) != batch_size:
        raise ValueError("data_sources count must equal rollout batch size")

    counters: MutableMapping[str, Any] = defaultdict(int)
    for list_key in (
        "matched_similarities",
        "policy_steps_per_turn",
        "calls_per_step",
        "step_rewards",
        "local_returns",
    ):
        counters[list_key] = []

    with torch.no_grad():
        credits = _build_step_credits(
            uids=uids,
            rollout_provenance=rollout_provenance,
            data_sources=data_sources,
            response_length=response_length,
            config=config,
            counters=counters,
        )

        groups: dict[tuple[str, int, int], list[_StepCredit]] = defaultdict(list)
        for credit in credits:
            groups[
                (credit.uid, credit.user_turn_id, credit.runtime_interaction_index)
            ].append(credit)

        support_sizes: list[float] = []
        support_by_depth: dict[int, list[float]] = defaultdict(list)
        singleton_affected = 0
        zero_variance_group_count = 0
        for (_, _, runtime_interaction_index), group in groups.items():
            support_size = len(group)
            support_sizes.append(float(support_size))
            support_by_depth[runtime_interaction_index].append(float(support_size))
            values = [credit.local_return for credit in group]
            mean = _mean(values)
            std = 0.0
            if support_size >= 2:
                std = float(torch.tensor(values, dtype=torch.float64).std(unbiased=True).item())
            for credit in group:
                credit.peer_support = support_size
                credit.peer_mean = mean
                credit.peer_sample_std = std if math.isfinite(std) else 0.0
            if support_size < config.min_group_size:
                singleton_affected += support_size
                continue
            if all(value == values[0] for value in values[1:]):
                zero_variance_group_count += 1
                continue
            if not math.isfinite(std) or std == 0.0:
                zero_variance_group_count += 1
                continue
            for credit in group:
                credit.local_advantage = (credit.local_return - mean) / (std + config.epsilon)
                credit.local_active = True

        local_tensor = torch.zeros_like(rods_advantages)
        local_token_mask = torch.zeros_like(actor_response_mask, dtype=torch.bool)
        provenance_assignment_failures = 0
        for credit in credits:
            if not credit.local_active:
                continue
            if not credit.actor_span_reliable:
                provenance_assignment_failures += 1
                continue
            row_mask = actor_response_mask[
                credit.batch_index, credit.span_start : credit.span_end
            ].bool()
            if not bool(row_mask.any()):
                provenance_assignment_failures += 1
                continue
            target = local_tensor[
                credit.batch_index, credit.span_start : credit.span_end
            ]
            target_mask = local_token_mask[
                credit.batch_index, credit.span_start : credit.span_end
            ]
            if bool((target_mask & row_mask).any()):
                raise AssertionError("actor runtime-interaction spans overlap")
            target[row_mask] = credit.local_advantage
            target_mask[row_mask] = True
            credit.token_residual_applied = True

        if bool((local_tensor[~actor_response_mask.bool()] != 0).any()):
            raise AssertionError("local advantage leaked onto environment tokens")

        fused_advantages = rods_advantages + config.weight * local_tensor
        fused_returns = rods_returns + config.weight * local_tensor

    active_credits = [credit for credit in credits if credit.local_active]
    applied_credits = [credit for credit in credits if credit.token_residual_applied]
    local_values = [credit.local_advantage for credit in active_credits]
    new_step_values = [
        rods_scalars[credit.batch_index] + config.weight * credit.local_advantage
        for credit in active_credits
    ]
    sign_flips = [
        _sign(new_value) != _sign(rods_scalars[credit.batch_index])
        for credit, new_value in zip(active_credits, new_step_values)
    ]
    actor_values = fused_advantages.detach()[actor_response_mask.bool()]
    actor_values_list = [float(value) for value in actor_values.cpu().tolist()]
    predicted_count = int(counters["num_predicted_calls"])
    matched_count = int(counters["matched_calls"])
    similarities = list(counters["matched_similarities"])
    parse_error_applied = [
        credit for credit in applied_credits if credit.response_type == "parse_error"
    ]
    parse_error_token_count = sum(
        int(
            local_token_mask[
                credit.batch_index, credit.span_start : credit.span_end
            ].sum().item()
        )
        for credit in parse_error_applied
    )

    metrics = {
        **global_metrics,
        "rods_matchtir_v1/call/num_predicted_calls": float(predicted_count),
        "rods_matchtir_v1/call/num_gt_calls": float(counters["num_gt_calls"]),
        "rods_matchtir_v1/call/matched_call_count": float(matched_count),
        "rods_matchtir_v1/call/unmatched_call_count": float(counters["unmatched_calls"]),
        "rods_matchtir_v1/call/match_rate": matched_count / predicted_count if predicted_count else 0.0,
        "rods_matchtir_v1/call/similarity_mean": _mean(similarities),
        "rods_matchtir_v1/call/similarity_max": max(similarities, default=0.0),
        "rods_matchtir_v1/step/policy_steps_per_user_turn": _mean(counters["policy_steps_per_turn"]),
        "rods_matchtir_v1/step/calls_per_policy_step": _mean(counters["calls_per_step"]),
        "rods_matchtir_v1/step/step_reward_mean": _mean(counters["step_rewards"]),
        "rods_matchtir_v1/step/local_return_mean": _mean(counters["local_returns"]),
        "rods_matchtir_v1/step/local_return_std": _population_std(counters["local_returns"]),
        "rods_matchtir_v1/interaction/non_answer_runtime_interactions": float(
            counters["runtime_interactions"]
        ),
        "rods_matchtir_v1/interaction/tool_attempt_interactions": float(
            counters["tool_attempt_interactions"]
        ),
        "rods_matchtir_v1/interaction/parsed_tool_interactions": float(
            counters["parsed_tool_interactions"]
        ),
        "rods_matchtir_v1/interaction/parse_error_tool_attempts": float(
            counters["parse_error_tool_attempts"]
        ),
        "rods_matchtir_v1/interaction/parse_error_runtime_interactions": float(
            counters["parse_error_runtime_interactions"]
        ),
        "rods_matchtir_v1/interaction/unparsed_runtime_interactions": float(
            counters["unparsed_runtime_interactions"]
        ),
        "rods_matchtir_v1/interaction/peer_supported_interactions": float(
            len(active_credits)
        ),
        "rods_matchtir_v1/interaction/unsupported_local_interactions": float(
            len(credits) - len(active_credits)
        ),
        "rods_matchtir_v1/interaction/nonzero_local_advantages": float(
            sum(credit.local_advantage != 0.0 for credit in active_credits)
        ),
        "rods_matchtir_v1/interaction/parse_error_interactions_receiving_local_residual": float(
            len(parse_error_applied)
        ),
        "rods_matchtir_v1/interaction/parse_error_tokens_receiving_local_residual": float(
            parse_error_token_count
        ),
        "rods_matchtir_v1/normalization/local_support_size": _mean(support_sizes),
        "rods_matchtir_v1/normalization/singleton_local_count": float(singleton_affected),
        "rods_matchtir_v1/normalization/singleton_local_rate": singleton_affected / len(credits) if credits else 0.0,
        "rods_matchtir_v1/normalization/zero_variance_local_count": float(zero_variance_group_count),
        "rods_matchtir_v1/normalization/local_adv_mean": _mean(local_values),
        "rods_matchtir_v1/normalization/local_adv_std": _population_std(local_values),
        "rods_matchtir_v1/normalization/local_coverage": len(active_credits) / len(credits) if credits else 0.0,
        "rods_matchtir_v1/fusion/A_new_mean": _mean(actor_values_list),
        "rods_matchtir_v1/fusion/A_new_std": _population_std(actor_values_list),
        "rods_matchtir_v1/fusion/RMS_A_RODS": _rms(rods_scalars),
        "rods_matchtir_v1/fusion/RMS_A_local": _rms(local_values),
        "rods_matchtir_v1/fusion/RMS_A_new": _rms(new_step_values or rods_scalars),
        "rods_matchtir_v1/fusion/sign_flip_rate": _mean([float(value) for value in sign_flips]),
        # Missing turns never enter ``credits``, so their local coverage is
        # structurally zero.  The next non-empty GT turn is handled independently.
        "rods_matchtir_v1/missing/missing_turn_local_coverage": 0.0,
        "rods_matchtir_v1/missing/followup_normal_turn_local_coverage": (
            _mean([float(credit.local_active) for credit in credits if credit.follows_missing_turn])
        ),
        "rods_matchtir_v1/provenance/unreliable_rollout_count": float(counters["unreliable_rollouts"]),
        "rods_matchtir_v1/provenance/invalid_gt_turn_count": float(counters["invalid_gt_turns"]),
        "rods_matchtir_v1/provenance/unreliable_tool_turn_count": float(counters["unreliable_tool_turns"]),
        "rods_matchtir_v1/provenance/span_assignment_failure_count": float(
            provenance_assignment_failures + counters["span_metadata_failures"]
        ),
    }
    for depth, sizes in sorted(support_by_depth.items()):
        metrics[f"rods_matchtir_v1/normalization/local_support_depth_{depth}"] = _mean(sizes)

    type_name_map = {
        "multi_turn_base": "Base",
        "multi_turn_miss_func": "Missing_Function",
        "multi_turn_miss_param": "Missing_Parameter",
        "multi_turn_long_context": "Long_Context",
    }
    for raw_type in sorted({credit.data_type for credit in credits}):
        typed = [credit for credit in credits if credit.data_type == raw_type]
        label = type_name_map.get(raw_type, raw_type or "unknown")
        metrics[f"rods_matchtir_v1/by_type/{label}/local_coverage"] = _mean(
            [float(credit.local_active) for credit in typed]
        )

    return FusionResult(
        advantages=fused_advantages,
        returns=fused_returns,
        local_advantages=local_tensor,
        local_token_mask=local_token_mask,
        metrics=metrics,
        step_records=tuple(credit.as_dict() for credit in credits),
    )
