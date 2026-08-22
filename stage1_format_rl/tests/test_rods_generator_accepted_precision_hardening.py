"""Final accepted-data precision regressions for the RODS Generator.

The three gates exercised here are project correctness guards.  They are not
claimed as rules or thresholds published by RODS.
"""

from __future__ import annotations

import json

from env_tuning.rods_data_generation_v1.validation.action_minimality import (
    action_minimality_gate,
)
from env_tuning.rods_data_generation_v1.validation.missing_parameter_validity import (
    missing_parameter_validity_gate,
)
from env_tuning.rods_data_generation_v1.validation.observation_entailment import (
    observation_entailment_gate,
)
from env_tuning.rods_data_generation_v1.validation.semantic_grounding import (
    semantic_context_for_verifier,
    semantic_grounding_gate,
)

from rods_data_generation_v1_fixtures import make_catalog
from test_rods_generator_semantic_hardening import _draft, _record, _turn


def _missing_parameter_draft(
    prior_records,
    *,
    parameter: str,
    target_value,
    recovery_function: str,
    recovery_arguments: dict,
    class_name: str,
):
    affected_turn = 1
    recovery_turn = 2
    draft = _draft(
        [
            _turn(0, class_name, "Show me the relevant open records.", prior_records),
            _turn(
                affected_turn,
                class_name,
                "Please handle my open record.",
                [],
                intentional_missing=True,
                missing_kind="parameter",
            ),
            _turn(
                recovery_turn,
                class_name,
                f"Use {target_value}.",
                [
                    _record(
                        recovery_function,
                        recovery_arguments,
                        {"status": "success"},
                        class_name=class_name,
                        turn_id=recovery_turn,
                        call_id=0,
                    )
                ],
            ),
        ],
        data_type="multi_turn_miss_param",
    )
    draft.structural_profile["adversarial"] = {
        "affected_turn": affected_turn,
        "recovery_turn": recovery_turn,
        "kind": "missing_parameter",
        "missing_parameter": parameter,
    }
    return draft


def test_mp_unique_ticket_inside_list_of_dicts_is_not_missing() -> None:
    prior = _record(
        "get_user_tickets",
        {},
        [{"id": 1, "status": "Open", "title": "Login issue"}],
        class_name="TicketAPI",
        turn_id=0,
        call_id=0,
    )
    draft = _missing_parameter_draft(
        [prior],
        parameter="ticket_id",
        target_value=1,
        recovery_function="resolve_ticket",
        recovery_arguments={"ticket_id": 1, "resolution": "Resolved"},
        class_name="TicketAPI",
    )
    gate = missing_parameter_validity_gate(draft, catalog=make_catalog())
    assert gate.passed is False
    assert gate.metadata["compatible_visible_candidates"] == [1]
    assert gate.metadata["uniquely_recoverable"] is True
    assert gate.metadata["decision"] == "REJECT_UNIQUELY_RECOVERABLE"
    assert gate.metadata["candidate_sources"][0]["source_path"] == "result[0].id"


def test_mp_unique_order_inside_nested_observation_is_not_missing() -> None:
    prior = _record(
        "get_order_history",
        {},
        {"orders": {"open": [{"id": 12446, "symbol": "AAPL"}]}},
        class_name="TradingBot",
        turn_id=0,
        call_id=0,
    )
    draft = _missing_parameter_draft(
        [prior],
        parameter="order_id",
        target_value=12446,
        recovery_function="cancel_order",
        recovery_arguments={"order_id": 12446},
        class_name="TradingBot",
    )
    gate = missing_parameter_validity_gate(draft, catalog=make_catalog())
    assert gate.passed is False
    assert gate.metadata["compatible_visible_candidates"] == [12446]
    assert gate.metadata["candidate_sources"][0]["source_type"] == "PRIOR_TOOL_OUTPUT"


def test_mp_unique_ticket_inside_safe_stringified_container_is_not_missing() -> None:
    prior = _record(
        "get_user_tickets",
        {},
        '[{"id": 7, "status": "Open"}]',
        class_name="TicketAPI",
        turn_id=0,
        call_id=0,
    )
    draft = _missing_parameter_draft(
        [prior],
        parameter="ticket_id",
        target_value=7,
        recovery_function="resolve_ticket",
        recovery_arguments={"ticket_id": 7, "resolution": "Resolved"},
        class_name="TicketAPI",
    )
    gate = missing_parameter_validity_gate(draft, catalog=make_catalog())
    assert gate.passed is False
    assert gate.metadata["compatible_visible_candidates"] == [7]
    assert gate.metadata["candidate_sources"][0]["source_path"] == "result[0].id"


def test_mp_two_compatible_visible_ids_allow_clarification() -> None:
    prior = _record(
        "get_user_tickets",
        {},
        [{"id": 1, "status": "Open"}, {"id": 2, "status": "Open"}],
        class_name="TicketAPI",
        turn_id=0,
        call_id=0,
    )
    draft = _missing_parameter_draft(
        [prior],
        parameter="ticket_id",
        target_value=1,
        recovery_function="resolve_ticket",
        recovery_arguments={"ticket_id": 1, "resolution": "Resolved"},
        class_name="TicketAPI",
    )
    gate = missing_parameter_validity_gate(draft, catalog=make_catalog())
    assert gate.passed is True
    assert gate.metadata["compatible_visible_candidates"] == [1, 2]
    assert gate.metadata["ambiguity_count"] == 2


def test_mp_zero_visible_ids_allows_clarification() -> None:
    draft = _missing_parameter_draft(
        [],
        parameter="ticket_id",
        target_value=1,
        recovery_function="resolve_ticket",
        recovery_arguments={"ticket_id": 1, "resolution": "Resolved"},
        class_name="TicketAPI",
    )
    gate = missing_parameter_validity_gate(draft, catalog=make_catalog())
    assert gate.passed is True
    assert gate.metadata["compatible_visible_candidates"] == []


def test_mp_hidden_environment_only_id_remains_missing() -> None:
    prior = _record(
        "get_user_tickets",
        {},
        [],
        class_name="TicketAPI",
        turn_id=0,
        call_id=0,
        pre_state={"TicketAPI": {"tickets": {1: {"status": "Open"}}}},
        post_state={"TicketAPI": {"tickets": {1: {"status": "Open"}}}},
    )
    draft = _missing_parameter_draft(
        [prior],
        parameter="ticket_id",
        target_value=1,
        recovery_function="resolve_ticket",
        recovery_arguments={"ticket_id": 1, "resolution": "Resolved"},
        class_name="TicketAPI",
    )
    gate = missing_parameter_validity_gate(draft, catalog=make_catalog())
    assert gate.passed is True
    assert gate.metadata["compatible_visible_candidates"] == []
    assert gate.metadata["visible_context_definition"]["raw_environment_state"] is False


def test_empty_matches_cannot_entail_claimed_critical_error() -> None:
    search = _record(
        "find",
        {"path": ".", "name": "log"},
        {"matches": []},
        class_name="GorillaFileSystem",
        turn_id=0,
        call_id=0,
    )
    send = _record(
        "send_message",
        {"receiver_id": "USR002", "message": "Critical error in Q4 reporting."},
        {"sent_status": True},
        class_name="MessageAPI",
        turn_id=1,
        call_id=0,
    )
    draft = _draft(
        [
            _turn(0, "GorillaFileSystem", "Find the system logs.", [search]),
            _turn(
                1,
                "MessageAPI",
                "The system logs indicate a critical error in Q4 reporting; send that to USR002.",
                [send],
            ),
        ]
    )
    gate = observation_entailment_gate(draft)
    assert gate.passed is False
    assert "UNSUPPORTED_OBSERVATION_CLAIM" in gate.detail


def test_visible_observation_entails_claimed_critical_error() -> None:
    search = _record(
        "find",
        {"path": ".", "name": "log"},
        {"matches": ["Q4 reporting: critical error"]},
        class_name="GorillaFileSystem",
        turn_id=0,
        call_id=0,
    )
    draft = _draft(
        [
            _turn(0, "GorillaFileSystem", "Find the system logs.", [search]),
            _turn(
                1,
                "MessageAPI",
                "The logs show a critical error in Q4 reporting and to please review it; notify the team.",
                [],
            ),
        ]
    )
    gate = observation_entailment_gate(draft)
    assert gate.passed is True
    assert gate.metadata["checks"][0]["status"] == "ENTAILED"


def test_new_user_preference_is_not_misclassified_as_observation_claim() -> None:
    draft = _draft(
        [_turn(0, "MessageAPI", "Please send Alice a new weekly status update.", [])]
    )
    gate = observation_entailment_gate(draft)
    assert gate.passed is True
    assert gate.metadata["checks"] == []


def test_final_verifier_context_receives_claim_and_evidence_relation() -> None:
    observation = _record(
        "find",
        {"path": ".", "name": "log"},
        {"matches": ["warning"]},
        class_name="GorillaFileSystem",
        turn_id=0,
        call_id=0,
    )
    draft = _draft(
        [
            _turn(0, "GorillaFileSystem", "Find logs.", [observation]),
            _turn(1, "MessageAPI", "The result says something happened; notify Alice.", []),
        ]
    )
    gate = observation_entailment_gate(draft)
    assert gate.passed is True
    assert gate.metadata["checks"][0]["status"] == "DEFERRED_TO_FINAL_SEMANTIC_VERIFIER"
    context = json.loads(semantic_context_for_verifier(draft))
    check = context["observation_entailment_contract"]["checks"][0]
    assert check["claimed_fact"] == "something happened"
    assert check["evidence_relation"] == "DEFERRED_TO_FINAL_SEMANTIC_VERIFIER"
    assert check["prior_observation_refs"][0]["function"] == "find"


def _vehicle_state(*, brake_status: str = "released", brake_force: float = 0.0):
    return {
        "VehicleControlAPI": {
            "brakePedalStatus": brake_status,
            "brakePedalForce": brake_force,
            "engineState": "stopped",
            "fuelLevel": 20.0,
            "remainingUnlockedDoors": 0,
        }
    }


def test_direct_fuel_action_is_minimal() -> None:
    fill = _record(
        "fillFuelTank",
        {"fuelAmount": 10.0},
        {"fuelLevel": 30.0},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=0,
    )
    gate = action_minimality_gate(
        _draft([_turn(0, "VehicleControlAPI", "Put 10 gallons of fuel in the car.", [fill])]),
        catalog=make_catalog(),
    )
    assert gate.passed is True
    assert gate.metadata["calls"][0]["classification"] == "DIRECT_INTENT"


def test_unused_gallon_conversion_is_redundant() -> None:
    conversion = _record(
        "gallon_to_liter",
        {"gallon": 10.0},
        {"liter": 37.8541},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=0,
    )
    fill = _record(
        "fillFuelTank",
        {"fuelAmount": 10.0},
        {"fuelLevel": 30.0},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=1,
    )
    draft = _draft(
        [_turn(0, "VehicleControlAPI", "Put 10 gallons of fuel in the car.", [conversion, fill])]
    )
    semantic_grounding_gate(draft, catalog=make_catalog())
    gate = action_minimality_gate(draft, catalog=make_catalog())
    assert gate.passed is False
    assert gate.metadata["calls"][0]["classification"] == "REDUNDANT_EXTRA_CALL"


def test_real_brake_precondition_for_start_engine_is_preserved() -> None:
    before = _vehicle_state()
    braked = _vehicle_state(brake_status="pressed", brake_force=1000.0)
    press = _record(
        "pressBrakePedal",
        {"pedalPosition": 1.0},
        {"brakePedalStatus": "pressed", "brakePedalForce": 1000.0},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=0,
        pre_state=before,
        post_state=braked,
    )
    start = _record(
        "startEngine",
        {"ignitionMode": "START"},
        {"engineState": "running", "fuelLevel": 20.0, "batteryVoltage": 12.6},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=1,
        pre_state=braked,
        post_state={"VehicleControlAPI": {**braked["VehicleControlAPI"], "engineState": "running"}},
    )
    gate = action_minimality_gate(
        _draft([_turn(0, "VehicleControlAPI", "Start the engine.", [press, start])]),
        catalog=make_catalog(),
    )
    assert gate.passed is True
    assert gate.metadata["calls"][0]["classification"] == "REQUIRED_PREREQUISITE"
    assert gate.metadata["calls"][1]["classification"] == "DIRECT_INTENT"


def test_call_consumed_by_later_required_argument_is_dependency_producer() -> None:
    lookup = _record(
        "get_user_id",
        {"user": "Alice"},
        {"user_id": "USR003"},
        class_name="MessageAPI",
        turn_id=0,
        call_id=0,
    )
    send = _record(
        "send_message",
        {"receiver_id": "USR003", "message": "Project ready"},
        {"sent_status": True},
        class_name="MessageAPI",
        turn_id=0,
        call_id=1,
    )
    draft = _draft(
        [_turn(0, "MessageAPI", "Find Alice and send her the message Project ready.", [lookup, send])]
    )
    assert semantic_grounding_gate(draft, catalog=make_catalog()).passed is True
    gate = action_minimality_gate(draft, catalog=make_catalog())
    assert gate.passed is True
    assert gate.metadata["calls"][0]["classification"] == "DEPENDENCY_PRODUCER"


def test_unrelated_decorative_call_is_redundant() -> None:
    speed = _record(
        "get_current_speed",
        {},
        {"currentSpeed": 0.0},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=0,
    )
    fill = _record(
        "fillFuelTank",
        {"fuelAmount": 10.0},
        {"fuelLevel": 30.0},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=1,
    )
    gate = action_minimality_gate(
        _draft([_turn(0, "VehicleControlAPI", "Put 10 gallons of fuel in the car.", [speed, fill])]),
        catalog=make_catalog(),
    )
    assert gate.passed is False
    assert gate.metadata["calls"][0]["classification"] == "REDUNDANT_EXTRA_CALL"


def test_same_retrieval_verb_does_not_justify_unrelated_object() -> None:
    speed = _record(
        "get_current_speed",
        {},
        {"currentSpeed": 0.0},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=0,
    )
    account = _record(
        "get_account_info",
        {},
        {"account_id": 12345, "balance": 10000.0},
        class_name="TradingBot",
        turn_id=0,
        call_id=1,
    )
    gate = action_minimality_gate(
        _draft(
            [
                _turn(
                    0,
                    "TradingBot",
                    "Show me my current trading account balance.",
                    [speed, account],
                )
            ]
        ),
        catalog=make_catalog(),
    )
    assert gate.passed is False
    assert gate.metadata["calls"][0]["classification"] == "REDUNDANT_EXTRA_CALL"
    assert gate.metadata["calls"][1]["classification"] == "DIRECT_INTENT"
