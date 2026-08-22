"""Residual Generator semantic-quality regressions.

All hard rules here are project guards, not claimed RODS paper mechanisms.
"""

from __future__ import annotations

import asyncio
import copy

from env_tuning.rods_data_generation_v1.llm_backend import FakeLLMBackend
from env_tuning.rods_data_generation_v1.metrics import GeneratorMetrics
from env_tuning.rods_data_generation_v1.query_verifier import QueryVerifier
from env_tuning.rods_data_generation_v1.result_semantics import (
    ExecutionSemanticOutcome,
    classify_execution_result,
    find_unclassified_suspicious_results,
)
from env_tuning.rods_data_generation_v1.validation.relational_resolution import (
    relational_resolution_gate,
)
from env_tuning.rods_data_generation_v1.validation.semantic_grounding import (
    semantic_grounding_gate,
)

from rods_data_generation_v1_fixtures import VERIFY_ACCEPT, make_catalog
from test_rods_generator_semantic_hardening import _draft, _record, _turn


def _booking_draft(*, dates: dict[str, str], selected: str, query: str):
    history = _record(
        "get_booking_history",
        {"access_token": "token"},
        {
            "booking_history": {
                booking_id: {
                    "booking_id": booking_id,
                    "travel_date": date,
                    "travel_cost": 100.0 + index,
                }
                for index, (booking_id, date) in enumerate(dates.items())
            }
        },
        class_name="TravelAPI",
        turn_id=0,
        call_id=0,
        pre_state={"TravelAPI": {"access_token": "token"}},
    )
    invoice = _record(
        "retrieve_invoice",
        {"access_token": "token", "booking_id": selected},
        {"invoice": {"booking_id": selected}},
        class_name="TravelAPI",
        turn_id=1,
        call_id=0,
        pre_state={"TravelAPI": {"access_token": "token"}},
    )
    return _draft(
        [
            _turn(0, "TravelAPI", "Show my booking history.", [history]),
            _turn(1, "TravelAPI", query, [invoice]),
        ]
    )


def test_most_recent_unique_value_passes() -> None:
    draft = _booking_draft(
        dates={"booking_1": "2024-01-01", "booking_2": "2024-12-05"},
        selected="booking_2",
        query="Get the invoice for my most recent trip.",
    )
    gate = relational_resolution_gate(draft)
    assert gate.passed is True
    assert gate.metadata["checks"][-1]["status"] == "RESOLVED"


def test_most_recent_tied_maximum_fails_closed() -> None:
    draft = _booking_draft(
        dates={"booking_400": "2024-12-05", "booking_900": "2024-12-05"},
        selected="booking_400",
        query="Get the invoice for my most recent trip.",
    )
    gate = relational_resolution_gate(draft)
    assert gate.passed is False
    assert "AMBIGUOUS_TIE" in gate.detail
    assert gate.metadata["arbitrary_tie_break_used"] is False


def test_relational_tie_with_explicit_booking_id_passes() -> None:
    draft = _booking_draft(
        dates={"booking_400": "2024-12-05", "booking_900": "2024-12-05"},
        selected="booking_400",
        query=(
            "The two most recent trips share a date; get the invoice for "
            "booking_400 specifically."
        ),
    )
    gate = relational_resolution_gate(draft)
    assert gate.passed is True
    assert gate.metadata["checks"][-1]["disambiguation"]["value"] == "booking_400"


def test_highest_and_lowest_unique_values_pass() -> None:
    stocks = _record(
        "get_watchlist",
        {},
        {"stocks": {"LOW": {"price": 10.0}, "HIGH": {"price": 99.0}}},
        class_name="TradingBot",
        turn_id=0,
        call_id=0,
    )
    highest = _record(
        "add_to_watchlist",
        {"stock": "HIGH"},
        {"symbol": ["HIGH"]},
        class_name="TradingBot",
        turn_id=1,
        call_id=0,
    )
    high_draft = _draft(
        [
            _turn(0, "TradingBot", "Show candidate stocks.", [stocks]),
            _turn(1, "TradingBot", "Add the stock with the highest price.", [highest]),
        ]
    )
    assert relational_resolution_gate(high_draft).passed is True

    lowest = copy.deepcopy(highest)
    lowest = _record(
        "add_to_watchlist",
        {"stock": "LOW"},
        {"symbol": ["LOW"]},
        class_name="TradingBot",
        turn_id=1,
        call_id=0,
    )
    low_draft = _draft(
        [
            _turn(0, "TradingBot", "Show candidate stocks.", [copy.deepcopy(stocks)]),
            _turn(1, "TradingBot", "Add the stock with the lowest price.", [lowest]),
        ]
    )
    assert relational_resolution_gate(low_draft).passed is True


def test_unrelated_multiturn_topic_stitching_is_rejected_by_final_verifier() -> None:
    turns = [
        _turn(0, "TradingBot", "Show technology stocks.", []),
        _turn(1, "TradingBot", "What time is it?", []),
        _turn(2, "MessageAPI", "List my contacts.", []),
        _turn(3, "MessageAPI", "Show message statistics.", []),
    ]
    draft = _draft(turns)
    draft.narrative = "Find stocks and share them with a colleague."
    backend = FakeLLMBackend(
        {
            "final_query_verifier": [
                "<reason>Per-turn mappings are plausible, but GLOBAL_COHERENCE "
                "fails: time and contact/stat turns are unrelated filler and "
                "the promised share action never occurs.</reason><verdict>reject</verdict>"
            ]
        }
    )
    gate = asyncio.run(
        QueryVerifier(backend, GeneratorMetrics()).verify_final_conversation(draft)
    )
    assert gate.passed is False
    prompt = backend.calls[0]["messages"][0]["content"]
    assert "GLOBAL_COHERENCE" in prompt
    assert draft.narrative in prompt
    assert "unrelated" in prompt and "filler turns are forbidden" in prompt


def test_coherent_multiturn_dependency_chain_passes_final_verifier() -> None:
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
        turn_id=1,
        call_id=0,
    )
    draft = _draft(
        [
            _turn(0, "MessageAPI", "Find Alice's user ID.", [lookup]),
            _turn(
                1,
                "MessageAPI",
                "Use that ID to send Alice the message Project ready.",
                [send],
            ),
        ]
    )
    draft.narrative = "Resolve Alice's ID and use it to send one status message."
    backend = FakeLLMBackend({"final_query_verifier": [VERIFY_ACCEPT]})
    gate = asyncio.run(
        QueryVerifier(backend, GeneratorMetrics()).verify_final_conversation(draft)
    )
    assert gate.passed is True


def _tail_gate(arguments: dict, query: str):
    record = _record(
        "tail",
        arguments,
        {"last_lines": "fixture"},
        class_name="GorillaFileSystem",
        turn_id=0,
        call_id=0,
    )
    return semantic_grounding_gate(
        _draft([_turn(0, "GorillaFileSystem", query, [record])]),
        catalog=make_catalog(),
    )


def test_optional_supplied_argument_user_context_passes() -> None:
    gate = _tail_gate(
        {"file_name": "notes.txt", "lines": 5},
        "Show the last 5 lines of notes.txt.",
    )
    assert gate.passed is True
    optional = next(
        item for item in gate.metadata["parameter_provenance"] if item["parameter"] == "lines"
    )
    assert optional["required"] is False
    assert optional["source_type"] == "USER_CONTEXT"


def test_optional_supplied_argument_prior_output_passes() -> None:
    calculation = _record(
        "add",
        {"a": 2.0, "b": 3.0},
        5.0,
        class_name="MathAPI",
        turn_id=0,
        call_id=0,
    )
    tail = _record(
        "tail",
        {"file_name": "notes.txt", "lines": 5},
        {"last_lines": "fixture"},
        class_name="GorillaFileSystem",
        turn_id=1,
        call_id=0,
    )
    draft = _draft(
        [
            _turn(0, "MathAPI", "Calculate two plus three.", [calculation]),
            _turn(
                1,
                "GorillaFileSystem",
                "Show that many lines from the end of notes.txt.",
                [tail],
            ),
        ]
    )
    gate = semantic_grounding_gate(draft, catalog=make_catalog())
    assert gate.passed is True
    optional = next(
        item
        for item in gate.metadata["parameter_provenance"]
        if item["function"] == "tail" and item["parameter"] == "lines"
    )
    assert optional["source_type"] == "PRIOR_TOOL_OUTPUT"


def test_optional_supplied_schema_default_passes() -> None:
    gate = _tail_gate(
        {"file_name": "notes.txt", "lines": 10},
        "Show the end of notes.txt.",
    )
    assert gate.passed is True
    optional = next(
        item for item in gate.metadata["parameter_provenance"] if item["parameter"] == "lines"
    )
    assert optional["source_type"] == "SCHEMA_DEFAULT"


def test_optional_supplied_without_source_fails() -> None:
    gate = _tail_gate(
        {"file_name": "notes.txt", "lines": 7},
        "Show the end of notes.txt.",
    )
    assert gate.passed is False
    unsupported = next(
        item for item in gate.metadata["unsupported"] if item["parameter"] == "lines"
    )
    assert unsupported["required"] is False


def test_optional_argument_not_explicitly_provided_needs_no_provenance() -> None:
    gate = _tail_gate({"file_name": "notes.txt"}, "Show the end of notes.txt.")
    assert gate.passed is True
    assert {item["parameter"] for item in gate.metadata["parameter_provenance"]} == {
        "file_name"
    }


def test_suspicious_success_is_telemetry_only() -> None:
    result = {"business_status": "Unknown new sentinel"}
    assert (
        classify_execution_result("future_audited_function", result).outcome
        == ExecutionSemanticOutcome.SUCCESS
    )
    observations = find_unclassified_suspicious_results(
        "future_audited_function", result
    )
    assert observations == [
        {
            "function": "future_audited_function",
            "path": "result.business_status",
            "value": "Unknown new sentinel",
            "reason": "unclassified suspicious string sentinel",
            "source_status": "PROJECT_SEMANTIC_GUARD_TELEMETRY_ONLY",
            "changes_execution_semantics": False,
        }
    ]
