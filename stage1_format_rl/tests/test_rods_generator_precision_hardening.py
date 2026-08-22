"""Final deterministic precision regressions for the Generator.

The guards in this module are project correctness rules.  They are not
claimed as unpublished RODS implementation details.
"""

from __future__ import annotations

import asyncio
import copy

import pytest

from env_tuning.rods_data_generation_v1.adversarial.missing_parameter import (
    MissingParameterTransformer,
)
from env_tuning.rods_data_generation_v1.environment_adapter import VMCallResult
from env_tuning.rods_data_generation_v1.llm_backend import FakeLLMBackend
from env_tuning.rods_data_generation_v1.parsing import StructuredParseError
from env_tuning.rods_data_generation_v1.result_semantics import (
    ExecutionSemanticOutcome,
    classify_execution_result,
    normalize_execution_result,
)
from env_tuning.rods_data_generation_v1.validation.unit_semantics import (
    audit_catalog_unit_contracts,
    convert_unit_value,
    unit_semantic_gate,
)
from env_tuning.rods_data_generation_v1.validation.missing_parameter_validity import (
    missing_parameter_validity_gate,
)
from env_tuning.rods_data_generation_v1.validation.vm_reverify import (
    fresh_vm_reverify_gate,
)

from rods_data_generation_v1_fixtures import make_catalog
from test_rods_generator_semantic_hardening import _draft, _record, _turn


def _unit_draft(function: str, arguments: dict, query: str):
    if function == "fillFuelTank":
        result = {"fuelLevel": arguments["fuelAmount"]}
    else:
        result = {
            "cruiseStatus": "active",
            "currentSpeed": arguments["speed"],
            "distanceToNextVehicle": arguments["distanceToNextVehicle"],
        }
    record = _record(
        function,
        arguments,
        result,
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=0,
    )
    return _draft([_turn(0, "VehicleControlAPI", query, [record])])


def test_unit_a_liter_literal_cannot_ground_same_numeric_gallon() -> None:
    gate = unit_semantic_gate(
        _unit_draft(
            "fillFuelTank",
            {"fuelAmount": 50.0},
            "Fill the fuel tank with 50 liters.",
        ),
        catalog=make_catalog(),
    )
    assert gate.passed is False
    assert "FAIL_UNIT_MISMATCH" in gate.detail


def test_unit_b_audited_liter_to_gallon_tool_chain_passes() -> None:
    conversion = _record(
        "liter_to_gallon",
        {"liter": 50.0},
        {"gallon": 13.2086},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=0,
    )
    fill = _record(
        "fillFuelTank",
        {"fuelAmount": 13.2086},
        {"fuelLevel": 13.2086},
        class_name="VehicleControlAPI",
        turn_id=0,
        call_id=1,
    )
    draft = _draft(
        [
            _turn(
                0,
                "VehicleControlAPI",
                "Convert 50 liters to gallons and add that converted amount to the tank.",
                [conversion, fill],
            )
        ]
    )
    gate = unit_semantic_gate(draft, catalog=make_catalog())
    assert gate.passed is True
    fill_check = next(
        row
        for row in gate.metadata["checks"]
        if row["function"] == "fillFuelTank"
    )
    assert fill_check["source_type"] == "PRIOR_TOOL_OUTPUT"
    assert fill_check["transformation"] == "UNIT_CONVERSION"


def test_unit_c_foot_literal_cannot_ground_same_numeric_meter() -> None:
    gate = unit_semantic_gate(
        _unit_draft(
            "setCruiseControl",
            {"speed": 100.0, "activate": True, "distanceToNextVehicle": 50.0},
            "Set cruise control to 100 with a 50 foot gap.",
        ),
        catalog=make_catalog(),
    )
    assert gate.passed is False
    assert "foot" in gate.detail and "meter" in gate.detail


def test_unit_d_same_meter_quantity_passes() -> None:
    gate = unit_semantic_gate(
        _unit_draft(
            "setCruiseControl",
            {"speed": 100.0, "activate": True, "distanceToNextVehicle": 50.0},
            "Set cruise control to 100 with a 50 meter gap.",
        ),
        catalog=make_catalog(),
    )
    assert gate.passed is True


def test_active_catalog_unit_audit_is_explicit_and_does_not_guess_ambiguous_speed() -> None:
    audit = audit_catalog_unit_contracts(make_catalog())
    indexed = {(row["function"], row["parameter"]): row for row in audit}
    assert indexed[("fillFuelTank", "fuelAmount")]["canonical_unit"] == "gallon"
    assert indexed[("setCruiseControl", "distanceToNextVehicle")]["canonical_unit"] == "meter"
    assert indexed[("setCruiseControl", "speed")]["status"] == "UNIT_UNKNOWN"


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (1.0, "liter", "gallon", 0.264172),
        (1.0, "gallon", "liter", 3.78541),
        (1.0, "meter", "foot", 3.28084),
        (1.0, "foot", "meter", 0.3048),
        (1.0, "kilometer", "mile", 0.621371),
        (1.0, "mile", "kilometer", 1.60934),
        (1.0, "meter", "centimeter", 100.0),
        (1.0, "centimeter", "inch", 0.393701),
        (1.0, "inch", "centimeter", 2.54),
        (1.0, "meter", "yard", 1.09361),
        (1.0, "yard", "meter", 0.9144),
        (1.0, "kilogram", "pound", 2.20462),
        (1.0, "pound", "kilogram", 0.453592),
        (0.0, "celsius", "fahrenheit", 32.0),
        (32.0, "fahrenheit", "celsius", 0.0),
        (100.0, "kilometer_per_hour", "mile_per_hour", 62.1371),
    ],
)
def test_all_audited_deterministic_conversions(
    value: float, source: str, target: str, expected: float
) -> None:
    assert convert_unit_value(value, source, target) == pytest.approx(
        expected, rel=2e-5, abs=1e-8
    )


def _mp_backend(
    *, affected_turn: int, parameter: str, affected_query: str, recovery_query: str
) -> FakeLLMBackend:
    return FakeLLMBackend(
        {
            "missing_parameter": [
                "<reason>Construct the requested clarification fixture.</reason>"
                f"<affected_turn>{affected_turn}</affected_turn>"
                f"<missing_parameter>{parameter}</missing_parameter>"
                f"<affected_query>{affected_query}</affected_query>"
                f"<recovery_query>{recovery_query}</recovery_query>"
            ]
        }
    )


def _cancel_record(order_id: int, *, turn_id: int, pre_state: dict | None = None):
    return _record(
        "cancel_order",
        {"order_id": order_id},
        {"order_id": order_id, "status": "Cancelled"},
        class_name="TradingBot",
        turn_id=turn_id,
        call_id=0,
        pre_state=pre_state,
    )


def _placed_order(order_id: int, *, turn_id: int, call_id: int = 0):
    return _record(
        "place_order",
        {"symbol": "AAPL", "price": 227.16, "amount": 10, "order_type": "Buy"},
        {"order_id": order_id, "status": "Pending"},
        class_name="TradingBot",
        turn_id=turn_id,
        call_id=call_id,
    )


def test_mp_e_unique_policy_visible_prior_order_is_not_genuinely_missing() -> None:
    draft = _draft(
        [
            _turn(0, "TradingBot", "Buy ten Apple shares.", [_placed_order(12446, turn_id=0)]),
            _turn(1, "TradingBot", "Please cancel my order.", [_cancel_record(12446, turn_id=1)]),
        ],
        data_type="multi_turn_miss_param",
    )
    transformer = MissingParameterTransformer(
        _mp_backend(
            affected_turn=1,
            parameter="order_id",
            affected_query="Please cancel my order.",
            recovery_query="The Apple order, order ID 12446.",
        ),
        make_catalog(),
    )
    with pytest.raises(StructuredParseError, match="uniquely recoverable"):
        asyncio.run(transformer.transform(draft))


def test_mp_f_two_policy_visible_orders_allow_clarification() -> None:
    first = _placed_order(101, turn_id=0, call_id=0)
    second = _placed_order(102, turn_id=0, call_id=1)
    draft = _draft(
        [
            _turn(0, "TradingBot", "Place these two Apple orders.", [first, second]),
            _turn(1, "TradingBot", "Please cancel my order.", [_cancel_record(101, turn_id=1)]),
        ],
        data_type="multi_turn_miss_param",
    )
    transformer = MissingParameterTransformer(
        _mp_backend(
            affected_turn=1,
            parameter="order_id",
            affected_query="Please cancel my order.",
            recovery_query="Cancel order ID 101.",
        ),
        make_catalog(),
    )
    output = asyncio.run(transformer.transform(draft))
    audit = output.structural_profile["adversarial"]["missing_parameter_validity"]
    assert audit["ambiguity_count"] == 2
    assert audit["uniquely_recoverable"] is False
    assert audit["decision"] == "GENUINE_MISSING_PARAMETER"


def test_mp_g_no_policy_visible_value_allows_clarification() -> None:
    draft = _draft(
        [_turn(0, "TradingBot", "Please cancel my order.", [_cancel_record(101, turn_id=0)])],
        data_type="multi_turn_miss_param",
    )
    transformer = MissingParameterTransformer(
        _mp_backend(
            affected_turn=0,
            parameter="order_id",
            affected_query="Please cancel my order.",
            recovery_query="Cancel order ID 101.",
        ),
        make_catalog(),
    )
    output = asyncio.run(transformer.transform(draft))
    audit = output.structural_profile["adversarial"]["missing_parameter_validity"]
    assert audit["ambiguity_count"] == 0
    assert audit["decision"] == "GENUINE_MISSING_PARAMETER"


def test_mp_h_hidden_environment_id_is_not_policy_visible() -> None:
    draft = _draft(
        [
            _turn(
                0,
                "TradingBot",
                "Please cancel my order.",
                [
                    _cancel_record(
                        101,
                        turn_id=0,
                        pre_state={"TradingBot": {"orders": {"101": {"status": "Open"}}}},
                    )
                ],
            )
        ],
        data_type="multi_turn_miss_param",
    )
    transformer = MissingParameterTransformer(
        _mp_backend(
            affected_turn=0,
            parameter="order_id",
            affected_query="Please cancel my order.",
            recovery_query="Cancel order ID 101.",
        ),
        make_catalog(),
    )
    output = asyncio.run(transformer.transform(draft))
    audit = output.structural_profile["adversarial"]["missing_parameter_validity"]
    assert audit["compatible_visible_candidates"] == []
    assert all(source["source_type"] != "ENV_STATE" for source in audit["candidate_sources"])


def test_historical_mp_metadata_is_recovered_only_from_explicit_recovery_value() -> None:
    draft = _draft(
        [_turn(0, "TradingBot", "Please cancel my order.", [_cancel_record(101, turn_id=0)])],
        data_type="multi_turn_miss_param",
    )
    transformer = MissingParameterTransformer(
        _mp_backend(
            affected_turn=0,
            parameter="order_id",
            affected_query="Please cancel my order.",
            recovery_query="Cancel order ID 101.",
        ),
        make_catalog(),
    )
    transformed = asyncio.run(transformer.transform(draft))
    transformed.structural_profile.pop("adversarial", None)
    gate = missing_parameter_validity_gate(transformed, catalog=make_catalog())
    assert gate.passed is True
    assert gate.metadata["parameter"] == "order_id"
    assert gate.metadata["recovery_supplies_value"] is True


def test_historical_mp_metadata_recovery_fails_closed_when_value_is_not_explicit() -> None:
    affected = _turn(0, "TradingBot", "Please cancel my order.", [_cancel_record(101, turn_id=0)])
    affected.is_intentional_missing = True
    affected.missing_kind = "parameter"
    recovery = _turn(1, "TradingBot", "Cancel the one I meant.", [_cancel_record(101, turn_id=1)])
    draft = _draft([affected, recovery], data_type="multi_turn_miss_param")
    gate = missing_parameter_validity_gate(draft, catalog=make_catalog())
    assert gate.passed is False
    assert gate.metadata["metadata_recovery"] == "FAILED_CLOSED"


@pytest.mark.parametrize(
    "payload",
    [
        [{"error": "User not authenticated"}],
        {"result": [{"error": "User not authenticated"}]},
        '[{"error":"User not authenticated"}]',
        "[{'error': 'User not authenticated'}]",
    ],
)
def test_result_i_j_nested_and_stringified_hard_errors(payload: object) -> None:
    normalized = normalize_execution_result(payload)
    assert classify_execution_result("get_user_tickets", normalized).outcome == (
        ExecutionSemanticOutcome.HARD_ERROR
    )


def test_result_k_legitimate_domain_negative_remains_non_hard_error() -> None:
    outcome = classify_execution_result(
        "get_symbol_by_name", '{"symbol":"Stock not found"}'
    ).outcome
    assert outcome == ExecutionSemanticOutcome.DOMAIN_NEGATIVE


def test_result_l_ordinary_error_word_is_success() -> None:
    outcome = classify_execution_result(
        "send_message", "The error rate is now zero."
    ).outcome
    assert outcome == ExecutionSemanticOutcome.SUCCESS


class _InconsistentFreshSession:
    environment_id = "fresh-result-semantics"

    def execute(self, call):
        return VMCallResult(
            result="[{'error': 'User not authenticated'}]",
            success=True,
            error_detail=None,
            pre_state={},
            post_state={},
            semantic_outcome="SUCCESS",
            semantic_detail="deliberately stale fixture flag",
        )

    def close(self) -> None:
        return None


class _InconsistentFreshFactory:
    created_environment_ids: list[str] = []

    def create(self, **kwargs):
        return _InconsistentFreshSession()


def test_result_m_fresh_vm_reclassifies_with_shared_semantics() -> None:
    record = _record(
        "get_user_tickets",
        {},
        [],
        class_name="TicketAPI",
        turn_id=0,
        call_id=0,
    )
    draft = _draft([_turn(0, "TicketAPI", "Show my tickets.", [record])])
    draft.synthesis_environment_id = "synthesis-result-semantics"
    gate = fresh_vm_reverify_gate(
        draft,
        environment_factory=_InconsistentFreshFactory(),
        seed_id="result-semantics",
    )
    assert gate.passed is False
    assert "User not authenticated" in gate.detail
