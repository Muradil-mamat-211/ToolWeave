#!/usr/bin/env python3
"""Replay a real formal-training K-group through production runtime-depth credit.

The input artifact may use the legacy V1 serialization schema.  Such records
are upgraded only by replaying each raw assistant response through the current
strict parser; no substring heuristic or hand-authored reward is used.  The
formal temporal axis is every non-answer runtime interaction, not the retained
diagnostic ``tool_attempt_index``.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--user-turn-id", type=int, required=True)
    parser.add_argument("--special-rollout-offset", type=int, required=True)
    return parser.parse_args()


def bootstrap(workspace: Path) -> dict[str, Any]:
    envtuning = workspace / "code/AWorld-RL-stage1-worktree/EnvTuning"
    for path in (workspace, envtuning, envtuning / "verl"):
        sys.path.insert(0, str(path))

    from bfcl_env.multi_turn_checker import state_checker
    from env_tuning.interaction.new_multi_turn_fc import MultiTurnFunctionCallInteraction
    from env_tuning.interaction.response_handler import ResponseHandler
    from env_tuning.rods_matchtir_v1.advantage import (
        LocalCreditConfig,
        fuse_rods_and_local_advantages,
    )
    from env_tuning.rods_matchtir_v1.matching import parse_bfcl_ground_truth
    from stage1_format_rl.rewards.rods_stage2_progress_reward import compute_score
    from verl import DataProto
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.trainer.ppo.ray_trainer import compute_advantage

    return {
        "ResponseHandler": ResponseHandler,
        "MultiTurnFunctionCallInteraction": MultiTurnFunctionCallInteraction,
        "LocalCreditConfig": LocalCreditConfig,
        "fuse": fuse_rods_and_local_advantages,
        "parse_gt": parse_bfcl_ground_truth,
        "compute_score": compute_score,
        "compute_global": compute_grpo_outcome_advantage,
        "DataProto": DataProto,
        "AdvantageEstimator": AdvantageEstimator,
        "compute_trainer_advantage": compute_advantage,
        "state_checker": state_checker,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        return [(line_number, json.loads(line)) for line_number, line in enumerate(handle, 1)]


def non_tensor(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record["non_tensor"]
    assert isinstance(value, Mapping)
    return value


def provenance(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = non_tensor(record)["matchtir_provenance"]
    assert isinstance(value, Mapping)
    return value


def enrich_legacy_provenance(
    record: Mapping[str, Any], source: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Upgrade one old rollout by replaying the production parser exactly once."""

    enriched = copy.deepcopy(dict(provenance(record)))
    steps = enriched.get("policy_steps", [])
    assert isinstance(steps, list)
    handler = source["ResponseHandler"]()
    parser_rows: list[dict[str, Any]] = []
    by_turn: dict[int, list[dict[str, Any]]] = {}
    for step in steps:
        assert isinstance(step, dict)
        by_turn.setdefault(int(step["user_turn_id"]), []).append(step)

    for user_turn_id, turn_steps in by_turn.items():
        ordered = sorted(turn_steps, key=lambda row: int(row["policy_step_id"]))
        assert [int(row["policy_step_id"]) for row in ordered] == list(range(len(ordered)))
        next_tool_attempt = 0
        for step in ordered:
            raw_response = step.get("raw_policy_response")
            assert isinstance(raw_response, str)
            parsed = handler.parse_and_validate(
                [{"role": "assistant", "content": raw_response}]
            )
            assert parsed.response_type.value == step["response_type"]
            recorded_calls = [
                {
                    "call_idx": call.get("call_idx"),
                    "name": call.get("name"),
                    "arguments": call.get("arguments"),
                    "valid": call.get("valid"),
                }
                for call in step.get("calls", [])
                if isinstance(call, Mapping)
            ]
            reparsed_calls = [
                {
                    "call_idx": call.get("call_idx"),
                    "name": call.get("name"),
                    "arguments": call.get("arguments"),
                    "valid": call.get("valid"),
                }
                for call in parsed.tool_calls
            ]
            assert reparsed_calls == recorded_calls

            runtime_index = int(step["policy_step_id"])
            span = step.get("actor_span")
            span_reliable = (
                isinstance(span, Mapping)
                and isinstance(span.get("start"), int)
                and isinstance(span.get("end"), int)
                and int(span["start"]) >= 0
                and int(span["end"]) > int(span["start"])
            )
            is_tool_attempt = (
                parsed.action_classification_reliable
                and parsed.attempted_action_type.value == "tool_call"
            )
            tool_attempt_index = next_tool_attempt if is_tool_attempt else None
            if is_tool_attempt:
                next_tool_attempt += 1
            step.update(
                {
                    "local_credit_semantics": "runtime_interaction_final",
                    "runtime_interaction_index": runtime_index,
                    "tool_attempt_index": tool_attempt_index,
                    "attempted_action_type": parsed.attempted_action_type.value,
                    "action_classification_reliable": bool(
                        parsed.action_classification_reliable
                    ),
                    "temporal_provenance_reliable": True,
                    "actor_span_reliable": span_reliable,
                    "call_parse_reliable": bool(parsed.call_parse_reliable),
                    "parser_error_code": parsed.parser_error_code,
                }
            )
            parser_rows.append(
                {
                    "user_turn_id": user_turn_id,
                    "runtime_interaction_index": runtime_index,
                    "tool_attempt_index": tool_attempt_index,
                    "recorded_response_type": step["response_type"],
                    "attempted_action_type": parsed.attempted_action_type.value,
                    "action_classification_reliable": bool(
                        parsed.action_classification_reliable
                    ),
                    "call_parse_reliable": bool(parsed.call_parse_reliable),
                    "parser_error_code": parsed.parser_error_code,
                    "actor_span_reliable": span_reliable,
                }
            )
    enriched["schema_version"] = "rods_matchtir_rollout.v3.offline_replay"
    return enriched, parser_rows


def replay_offset_state(
    record: Mapping[str, Any], user_turn_id: int, source: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay one complete rollout and capture the requested terminal state check."""

    kwargs = dict(non_tensor(record)["extra_info"]["interaction_kwargs"])
    runtime_steps = list(provenance(record)["policy_steps"])

    async def replay() -> dict[str, Any]:
        interaction = source["MultiTurnFunctionCallInteraction"](
            {"name": kwargs["name"], "is_augmented": False}
        )
        instance_id = await interaction.start_interaction(**kwargs)
        state = interaction._instance_dict[instance_id]
        captured: dict[str, Any] = {}
        original_check = interaction.score_calculator._check_state_consistency

        def wrapped_check(
            model_instances: Mapping[str, Any], gt_instances: Mapping[str, Any]
        ) -> bool:
            result = source["state_checker"](model_instances, gt_instances)
            checked_turn = int(state.current_turn_index) - 1
            if checked_turn == user_turn_id and "state_check" not in captured:
                model_travel = vars(model_instances["TravelAPI"])
                gt_travel = vars(gt_instances["TravelAPI"])
                model_ticket = vars(model_instances["TicketAPI"])
                gt_ticket = vars(gt_instances["TicketAPI"])
                card_id = "travel_card_12345"
                captured.update(
                    {
                        "state_check": result,
                        "model_travel_booking_record": model_travel.get(
                            "booking_record", {}
                        ),
                        "ground_truth_travel_booking_ids": sorted(
                            str(key)
                            for key in gt_travel.get("booking_record", {}).keys()
                        ),
                        "model_travel_credit_balance": model_travel.get(
                            "credit_card_list", {}
                        ).get(card_id, {}).get("balance"),
                        "ground_truth_travel_credit_balance": gt_travel.get(
                            "credit_card_list", {}
                        ).get(card_id, {}).get("balance"),
                        "ticket_api_state_matches": {
                            key: model_ticket.get(key) == gt_ticket.get(key)
                            for key in sorted(set(model_ticket) | set(gt_ticket))
                            if not key.startswith("_")
                        },
                    }
                )
            return original_check(model_instances, gt_instances)

        interaction.score_calculator._check_state_consistency = wrapped_check
        terminal_score = None
        for step in runtime_steps:
            result = await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": step["raw_policy_response"]}],
                id=kwargs["id"],
            )
            if int(step["user_turn_id"]) == user_turn_id and step["response_type"] == "answer":
                terminal_score = float(result[2])
        captured["terminal_score"] = terminal_score
        return captured

    captured = asyncio.run(replay())
    actual_turn0_calls = [
        {"name": call.get("name"), "arguments": dict(call.get("arguments", {}))}
        for step in provenance(record)["policy_steps"]
        if int(step["user_turn_id"]) == 0 and step["response_type"] == "tool_call"
        for call in step.get("calls", [])
        if isinstance(call, Mapping)
    ]
    gt_turn0 = source["parse_gt"](provenance(record)["ground_truth"][0])
    actual_cost = next(call for call in actual_turn0_calls if call["name"] == "get_flight_cost")
    expected_cost = next(call for call in gt_turn0 if call.name == "get_flight_cost")
    state_check = captured["state_check"]
    return {
        "rollout_offset": int(provenance(record)["rollout_offset"]),
        "rollout_id": provenance(record)["rollout_id"],
        "runtime_replayed": True,
        "user_turn_id": user_turn_id,
        "terminal_score": captured["terminal_score"],
        "user_turn0_actual_tool_call_names": [row["name"] for row in actual_turn0_calls],
        "user_turn0_actual_get_flight_cost_arguments": actual_cost["arguments"],
        "user_turn0_ground_truth_call_names": [call.name for call in gt_turn0],
        "user_turn0_ground_truth_get_flight_cost_arguments": dict(expected_cost.arguments),
        "state_check_failure_type": state_check.get("error_type"),
        "state_check_difference_fields": sorted(
            state_check.get("details", {}).get("differences", {}).keys()
        ),
        "model_travel_booking_record": captured["model_travel_booking_record"],
        "ground_truth_travel_booking_ids": captured[
            "ground_truth_travel_booking_ids"
        ],
        "model_travel_credit_balance": captured["model_travel_credit_balance"],
        "ground_truth_travel_credit_balance": captured[
            "ground_truth_travel_credit_balance"
        ],
        "ticket_api_state_matches": captured["ticket_api_state_matches"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rollout_offset",
        "rollout_id",
        "interaction_index",
        "runtime_interaction_index",
        "runtime_depth",
        "tool_attempt_index",
        "response_type",
        "parsed_calls",
        "call_rewards",
        "r_j",
        "R_j",
        # Backward-compatible aliases retained for existing HF consumers.
        "r_t",
        "R_t",
        "peer_support",
        "peer_mean_R",
        "peer_sample_std_R",
        "A_local",
        "R_P",
        "A_RODS",
        "A_TW",
        "token_residual_applied",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["parsed_calls"] = json.dumps(row["parsed_calls"], ensure_ascii=False)
            serialized["call_rewards"] = json.dumps(row["call_rewards"])
            writer.writerow(serialized)


def main() -> None:
    args = arguments()
    workspace = args.workspace.resolve()
    raw_jsonl = args.raw_jsonl.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = bootstrap(workspace)
    records = load_jsonl(raw_jsonl)

    sample_rows = [row for row in records if non_tensor(row[1]).get("index") == args.sample_id]
    assert sample_rows
    target_line, first_sample = sample_rows[0]
    group_uid = str(non_tensor(first_sample)["uid"])
    group = [record for _, record in records if str(non_tensor(record)["uid"]) == group_uid]
    group.sort(key=lambda record: int(provenance(record)["rollout_offset"]))
    offsets = [int(provenance(record)["rollout_offset"]) for record in group]
    assert offsets == list(range(len(group)))
    assert len({provenance(record)["rollout_id"] for record in group}) == len(group)

    enriched: list[dict[str, Any]] = []
    parser_audits: list[list[dict[str, Any]]] = []
    for record in group:
        upgraded, parser_rows = enrich_legacy_provenance(record, source)
        enriched.append(upgraded)
        parser_audits.append(parser_rows)

    max_end = max(
        int(step["actor_span"]["end"])
        for item in enriched
        for step in item["policy_steps"]
        if step.get("actor_span_reliable")
    )
    import numpy as np
    import torch

    actor_mask = torch.zeros((len(group), max_end), dtype=torch.float32)
    progress_values: list[float] = []
    progress_diagnostics: list[dict[str, Any]] = []
    for row_index, (record, item) in enumerate(zip(group, enriched)):
        for step in item["policy_steps"]:
            if not step.get("actor_span_reliable"):
                continue
            start = int(step["actor_span"]["start"])
            end = int(step["actor_span"]["end"])
            actor_mask[row_index, start:end] = 1
        score = source["compute_score"](
            reward_scores=dict(non_tensor(record).get("reward_scores", {})),
            ground_truth=list(item["ground_truth"]),
            extra_info=dict(non_tensor(record).get("extra_info", {})),
        )
        progress_values.append(float(score["progress"]))
        progress_diagnostics.append(score)

    token_rewards = torch.zeros_like(actor_mask)
    token_rewards[:, -1] = torch.tensor(progress_values, dtype=token_rewards.dtype)
    global_advantages, global_returns = source["compute_global"](
        token_level_rewards=token_rewards,
        response_mask=actor_mask,
        index=np.asarray([group_uid] * len(group), dtype=object),
        epsilon=1.0e-6,
        norm_adv_by_std_in_grpo=True,
    )
    config = source["LocalCreditConfig"](
        mode="runtime_interaction_final",
        weight=1.0,
        gamma=0.9,
        matching="hard",
        unmatched_penalty=0.0,
        min_group_size=2,
        epsilon=1.0e-6,
    )
    result = source["fuse"](
        rods_advantages=global_advantages,
        rods_returns=global_returns,
        token_level_rewards=token_rewards,
        actor_response_mask=actor_mask,
        uids=[group_uid] * len(group),
        rollout_provenance=enriched,
        data_sources=[str(non_tensor(record).get("data_source", "")) for record in group],
        config=config,
    )
    assert result.advantages.shape == global_advantages.shape
    assert result.returns.shape == global_returns.shape
    assert torch.allclose(result.advantages, global_advantages + result.local_advantages)
    assert torch.allclose(result.returns, global_returns + result.local_advantages)
    assert not bool((result.local_advantages[~actor_mask.bool()] != 0).any())

    # Deterministic CPU tensor-contract replay through the actual trainer entry
    # point. This is not a GPU integration smoke: it launches no optimizer,
    # training process, rollout engine, or checkpoint writer.
    from omegaconf import OmegaConf

    trainer_data = source["DataProto"].from_dict(
        tensors={
            "responses": torch.zeros_like(actor_mask, dtype=torch.long),
            "response_mask": torch.ones_like(actor_mask),
            "loss_mask": actor_mask.clone(),
            "token_level_rewards": token_rewards.clone(),
        },
        non_tensors={
            "uid": np.asarray([group_uid] * len(group), dtype=object),
            "matchtir_provenance": np.asarray(enriched, dtype=object),
            "data_source": np.asarray(
                [str(non_tensor(record).get("data_source", "")) for record in group],
                dtype=object,
            ),
        },
    )
    trainer_config = OmegaConf.create(
        {
            "use_kl_in_reward": False,
            "matchtir_local": {
                "mode": "runtime_interaction_final",
                "enabled": True,
                "weight": 1.0,
                "gamma": 0.9,
                "matching": "hard",
                "unmatched_penalty": 0.0,
                "min_group_size": 2,
                "epsilon": 1.0e-6,
            },
        }
    )
    trainer_output = source["compute_trainer_advantage"](
        trainer_data,
        adv_estimator=source["AdvantageEstimator"].GRPO,
        multi_turn=True,
        config=trainer_config,
    )
    assert torch.allclose(trainer_output.batch["advantages"], result.advantages)
    assert torch.allclose(trainer_output.batch["returns"], result.returns)
    assert torch.allclose(
        trainer_output.batch["matchtir_local_advantages"], result.local_advantages
    )
    assert torch.equal(trainer_output.batch["matchtir_local_mask"], result.local_token_mask)

    global_scalars: list[float] = []
    for row in range(len(group)):
        positions = actor_mask[row].bool()
        unique = torch.unique(global_advantages[row][positions])
        assert unique.numel() == 1
        global_scalars.append(float(unique.item()))

    metadata_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row_index, item in enumerate(enriched):
        for step in item["policy_steps"]:
            if int(step["user_turn_id"]) != args.user_turn_id:
                continue
            if step.get("response_type") == "answer":
                continue
            metadata_by_key[
                (row_index, int(step["runtime_interaction_index"]))
            ] = step

    rows: list[dict[str, Any]] = []
    for record in result.step_records:
        if int(record["user_turn_id"]) != args.user_turn_id:
            continue
        row_index = int(record["batch_index"])
        runtime_depth = int(record["runtime_interaction_index"])
        metadata = metadata_by_key[(row_index, runtime_depth)]
        parsed_calls = [
            call.get("name")
            for call in metadata.get("calls", [])
            if isinstance(call, Mapping) and bool(call.get("valid", True))
        ]
        fused_scalar = global_scalars[row_index] + float(record["local_advantage"])
        if record["token_residual_applied"]:
            start = int(record["actor_span"]["start"])
            end = int(record["actor_span"]["end"])
            mask = actor_mask[row_index, start:end].bool()
            assert torch.allclose(
                result.local_advantages[row_index, start:end][mask],
                torch.full_like(
                    result.local_advantages[row_index, start:end][mask],
                    float(record["local_advantage"]),
                ),
            )
        rows.append(
            {
                "rollout_offset": int(provenance(group[row_index])["rollout_offset"]),
                "rollout_id": str(provenance(group[row_index])["rollout_id"]),
                "interaction_index": runtime_depth,
                "runtime_interaction_index": runtime_depth,
                "runtime_depth": runtime_depth,
                "tool_attempt_index": record["tool_attempt_index"],
                "response_type": record["response_type"],
                "parsed_calls": parsed_calls,
                "call_rewards": list(record["call_rewards"]),
                "r_j": float(record["step_reward"]),
                "R_j": float(record["local_return"]),
                "r_t": float(record["step_reward"]),
                "R_t": float(record["local_return"]),
                "peer_support": int(record["peer_support"]),
                "peer_mean_R": float(record["peer_mean"]),
                "peer_sample_std_R": float(record["peer_sample_std"]),
                "A_local": float(record["local_advantage"]),
                "R_P": progress_values[row_index],
                "A_RODS": global_scalars[row_index],
                "A_TW": fused_scalar,
                "token_residual_applied": bool(record["token_residual_applied"]),
            }
        )
    rows.sort(key=lambda row: (row["rollout_offset"], row["runtime_depth"]))

    by_offset: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_offset.setdefault(row["rollout_offset"], []).append(row)
    for rollout_rows in by_offset.values():
        for index, row in enumerate(rollout_rows):
            expected = row["r_j"]
            if index + 1 < len(rollout_rows):
                expected += 0.9 * rollout_rows[index + 1]["R_j"]
            assert math.isclose(row["R_j"], expected, rel_tol=1e-9, abs_tol=1e-9)
            assert math.isclose(
                row["A_TW"], row["A_RODS"] + row["A_local"], rel_tol=1e-9, abs_tol=1e-9
            )
            if row["peer_support"] < 2:
                assert row["A_local"] == 0.0
                assert row["A_TW"] == row["A_RODS"]
            if row["response_type"] == "parse_error":
                assert row["parsed_calls"] == [] and row["call_rewards"] == []
                assert row["r_j"] == 0.0

    special_rows = by_offset[args.special_rollout_offset]
    assert [row["response_type"] for row in special_rows] == ["parse_error"] * 5 + [
        "tool_call"
    ]
    assert special_rows[-1]["parsed_calls"] == ["ticket_login", "create_ticket"]
    assert special_rows[-1]["call_rewards"] == [1.0, 1.0]
    assert [row["r_j"] for row in special_rows] == [0.0] * 5 + [1.0]
    assert [row["peer_support"] for row in special_rows] == [16, 16, 1, 1, 1, 1]

    # Real-group scientific regression invariants requested by the Stage-3 gate.
    assert all(row["A_local"] > 0 and row["A_TW"] > row["A_RODS"] for row in by_offset[0])
    assert by_offset[14][1]["A_local"] < 0 < by_offset[14][1]["A_RODS"]
    assert by_offset[14][1]["A_TW"] < 0
    assert all(row["A_local"] > 0 > row["A_TW"] for row in by_offset[2])
    assert all(row["A_local"] < 0 for row in special_rows[:2])
    assert all(row["A_local"] == 0.0 for row in special_rows[2:])
    assert all(row["token_residual_applied"] for row in special_rows[:2])

    offset2_record = next(
        record for record in group if int(provenance(record)["rollout_offset"]) == 2
    )
    offset2_state = replay_offset_state(offset2_record, args.user_turn_id, source)
    assert offset2_state["terminal_score"] == 0.0
    assert offset2_state["model_travel_booking_record"] == {}
    assert offset2_state["ground_truth_travel_booking_ids"] == ["3426812"]
    assert offset2_state["model_travel_credit_balance"] == 6000.0
    assert offset2_state["ground_truth_travel_credit_balance"] == 5000.0

    special_record = next(
        record
        for record in group
        if int(provenance(record)["rollout_offset"]) == args.special_rollout_offset
    )
    special_source_line = next(
        line_number
        for line_number, record in records
        if record is special_record
    )
    special_trajectory_index = int(special_record["trajectory_index"])
    assert special_trajectory_index == args.special_rollout_offset
    special_row_index = group.index(special_record)
    special_parser = [
        row
        for row in parser_audits[special_row_index]
        if row["user_turn_id"] == args.user_turn_id
        and row["recorded_response_type"] != "answer"
    ]
    parse_recovery = {
        "sample_id": args.sample_id,
        "group_uid": group_uid,
        "rollout_offset": args.special_rollout_offset,
        "rollout_id": provenance(special_record)["rollout_id"],
        "user_turn_id": args.user_turn_id,
        "parser_replay": special_parser,
        "production_credit_rows": special_rows,
    }
    token_audit = {
        "shape": list(result.local_advantages.shape),
        "outside_actor_nonzero": int(
            torch.count_nonzero(result.local_advantages[~actor_mask.bool()]).item()
        ),
        "parse_error_interactions_receiving_local_residual": result.metrics[
            "rods_matchtir_v1/interaction/parse_error_interactions_receiving_local_residual"
        ],
        "parse_error_tokens_receiving_local_residual": result.metrics[
            "rods_matchtir_v1/interaction/parse_error_tokens_receiving_local_residual"
        ],
        "special_supported_parse_error_rows": special_rows[:2],
        "advantages_shape_preserved": list(result.advantages.shape)
        == list(global_advantages.shape),
        "returns_shape_preserved": list(result.returns.shape) == list(global_returns.shape),
        "additive_fusion_exact": bool(
            torch.allclose(result.advantages, global_advantages + result.local_advantages)
        ),
        "trainer_compute_advantage_tensor_contract_replay": "PASS",
        "trainer_advantages_match_direct_production_fusion": bool(
            torch.allclose(trainer_output.batch["advantages"], result.advantages)
        ),
        "optimizer_compatible_tensor_contract": {
            "advantages_shape": list(trainer_output.batch["advantages"].shape),
            "returns_shape": list(trainer_output.batch["returns"].shape),
            "response_loss_mask_shape": list(actor_mask.shape),
            "finite_advantages": bool(
                torch.isfinite(trainer_output.batch["advantages"]).all()
            ),
            "finite_returns": bool(torch.isfinite(trainer_output.batch["returns"]).all()),
        },
    }
    semantics = {
        "implementation": "runtime_interaction_final",
        "package_path_is_legacy_name": "env_tuning.rods_matchtir_v1",
        "gamma": config.gamma,
        "lambda_local": config.weight,
        "epsilon": config.epsilon,
        "matching": "current ToolWeave similarity plus scipy linear_sum_assignment(maximize=True)",
        "matching_scope": "all successfully parsed calls in one BFCL user turn",
        "parse_error_immediate_reward": 0.0,
        "temporal_axis": "non-answer runtime interaction",
        "tool_attempt_index_role": "backward-compatible diagnostic only",
        "normalization_key": [
            "group_uid",
            "user_turn_id",
            "runtime_interaction_index",
        ],
        "sample_std_unbiased": True,
        "missing_depth_zero_padding": False,
        "singleton_local_advantage": 0.0,
        "fusion": "A_RODS + A_local",
        "post_fusion_normalization": None,
        "global_rods_reimplemented": False,
    }
    summary = {
        "schema_version": "toolweave.runtime-interaction-credit-final.production-replay.v1",
        "source": {
            "raw_jsonl_sha256": sha256(raw_jsonl),
            "raw_trajectory_count": len(records),
            "sample_first_jsonl_line": target_line,
            "special_jsonl_line": special_source_line,
            "special_trajectory_index": special_trajectory_index,
        },
        "sample_id": args.sample_id,
        "group_uid": group_uid,
        "group_size": len(group),
        "user_turn_id": args.user_turn_id,
        "special_rollout_offset": args.special_rollout_offset,
        "special_rollout_id": provenance(special_record)["rollout_id"],
        "progress_values": progress_values,
        "progress_diagnostics": progress_diagnostics,
        "production_metrics": result.metrics,
        "rows": rows,
    }
    dump_json(output_dir / "algorithm_semantics.json", semantics)
    dump_json(output_dir / "k16_user_turn3_regression.json", summary)
    write_csv(output_dir / "k16_user_turn3_regression.csv", rows)
    dump_json(output_dir / "offset2_state_checker_replay.json", offset2_state)
    dump_json(output_dir / "offset9_parse_recovery_replay.json", parse_recovery)
    dump_json(output_dir / "token_broadcast_audit.json", token_audit)
    print(
        json.dumps(
            {
                "status": "PASS",
                "group_size": len(group),
                "interaction_rows": len(rows),
                "special_rollout_id": provenance(special_record)["rollout_id"],
                "special_r": [row["r_j"] for row in special_rows],
                "special_R": [row["R_j"] for row in special_rows],
                "special_A_local": [row["A_local"] for row in special_rows],
                "special_A_RODS": special_rows[0]["A_RODS"],
                "special_A_TW": [row["A_TW"] for row in special_rows],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
