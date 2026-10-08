"""Counterexamples for exact text labels, history bypass and LC coverage."""
import copy

import pytest

from env_tuning.rods_data_generation_v1.validation.query_contract import query_contract_gate
from env_tuning.rods_data_generation_v1.validation.action_minimality import action_minimality_gate
from env_tuning.rods_data_generation_v1.validation.semantic_grounding import semantic_grounding_gate
from rods_data_generation_v1_fixtures import make_catalog
from test_rods_generator_semantic_hardening import _draft, _record, _turn


CONTENT = "def deploy(): # update the server\n" + "unrelated financial prose " * 150 + "for future growth and profitability."
TEMPLATE = "Function: {function}; Ending: {ending}"
MESSAGE = "Function: deploy; Ending: for future growth and profitability."


def example(query=None, message=MESSAGE, *, kind="multi_turn_long_context", content=CONTENT):
    read = _record("cat", {"file_name": "deploy.py"}, {"file_content": content},
                   class_name="GorillaFileSystem", turn_id=0, call_id=0)
    send = _record("send_message", {"receiver_id": "USR003", "message": message},
                   {"sent_status": True}, class_name="MessageAPI", turn_id=1, call_id=0)
    query = query or f"Send Catherine a receipt in exact format `{TEMPLATE}`. Use the earlier declared function name and final five whitespace-separated words, preserving punctuation."
    return _draft([_turn(0, "GorillaFileSystem", "Read deploy.py.", [read]),
                   _turn(1, "MessageAPI", query, [send])], data_type=kind)


def test_exact_history_template_uses_distinct_distant_actor_visible_facts():
    result = query_contract_gate(example())
    assert result.passed, result.detail
    assert result.metadata["cross_turn_binding_count"] >= 2
    pair = result.metadata["long_context_pairs"][0]
    assert pair["span_fraction"] > 0.8 and pair["late_fraction"] > 0.9
    assert result.metadata["training_reward_changed"] is False


@pytest.mark.parametrize("query,message", [
    ("Send Catherine a summary containing its function and ending.", MESSAGE),
    ("Send Catherine exactly this: " + MESSAGE, MESSAGE),
    (f"Send this: {MESSAGE}. Use exact format `{TEMPLATE}`.", MESSAGE),
    (f"Use exact format `{TEMPLATE}`.", "Function: deploy; Ending: fabricated ending."),
    (f"Use exact format `{TEMPLATE}`.", MESSAGE + "."),
    ("Use exact format `{a}{b}`.", MESSAGE),
    ("Use exact format `{a}; {a}`.", "deploy; deploy"),
])
def test_free_summary_filled_answer_unsupported_slot_and_wrong_punctuation_reject(query, message):
    assert not query_contract_gate(example(query, message)).passed


def test_long_output_with_all_required_facts_in_short_prefix_is_rejected():
    draft = example("Use exact format `Function: {function}; Update: {update}`.",
                    "Function: deploy; Update: # update the server")
    result = query_contract_gate(draft)
    assert not result.passed
    assert result.metadata["checks"][0]["mode"] == "EXACT_HISTORY_TEMPLATE"
    assert result.metadata["long_context_pairs"] == []


def test_hidden_snapshot_cannot_supply_template_slots():
    draft = example(message="Function: deploy; Ending: hidden secret")
    draft.turns[0].execution_records[0].pre_state["secret"] = "hidden secret"
    assert not query_contract_gate(draft).passed


def test_wrong_function_name_cannot_use_an_arbitrary_visible_substring():
    assert not query_contract_gate(example(message="Function: server; Ending: for future growth and profitability.")).passed


@pytest.mark.parametrize("unit", ["whitespace-separated words", "whitespace-separated tokens",
                                 "nonempty whitespace-separated tokens"])
def test_ending_selection_checks_the_requested_word_count(unit):
    query = f"Use exact format `{TEMPLATE}`. Use the declared function name and last four {unit}."
    assert not query_contract_gate(example(query)).passed
    assert query_contract_gate(example(query, "Function: deploy; Ending: future growth and profitability.")).passed


def test_observable_array_length_is_a_deterministic_binding():
    read = _record("grep", {"file_name": "deploy.py", "pattern": "update"},
                   {"matching_lines": ["# update the server"]}, class_name="GorillaFileSystem", turn_id=0, call_id=0)
    send = _record("send_message", {"receiver_id": "USR003", "message": "Lines: 1"},
                   {"sent_status": True}, class_name="MessageAPI", turn_id=1, call_id=0)
    draft = _draft([_turn(0, "GorillaFileSystem", "Find the exact text `update` in deploy.py.", [read]),
                    _turn(1, "MessageAPI", "Send Catherine a receipt in exact format `Lines: {lines}` using the number of matching lines.", [send])])
    result = query_contract_gate(draft)
    assert result.passed
    assert result.metadata["checks"][0]["bindings"][0]["sources"][0]["value_transform"] == "ARRAY_LENGTH"


def test_argument_grounding_recognizes_exact_computed_array_length():
    read = _record("grep", {"file_name": "deploy.py", "pattern": "update"},
                   {"matching_lines": ["# update the server"]}, class_name="GorillaFileSystem", turn_id=0, call_id=0)
    lookup = _record("get_user_id", {"user": "Catherine"}, {"user_id": "USR003"},
                     class_name="MessageAPI", turn_id=1, call_id=0)
    send = _record("send_message", {"receiver_id": "USR003", "message": "Lines: 1"},
                   {"sent_status": True}, class_name="MessageAPI", turn_id=1, call_id=1)
    draft = _draft([_turn(0, "GorillaFileSystem", "Find the exact text `update` in deploy.py.", [read]),
                    _turn(1, "MessageAPI", "Look up Catherine and send her a receipt in exact format `Lines: {lines}` using the number of matching lines.", [lookup, send])])
    result = semantic_grounding_gate(draft, catalog=make_catalog())
    assert result.passed, result.detail
    evidence = next(row for row in result.metadata["parameter_provenance"] if row["parameter"] == "message")
    assert evidence["match"] == "exact template with supported observable transformations"
    assert evidence["exact_text_binding"]["bindings"][0]["sources"][0]["value_transform"] == "ARRAY_LENGTH"


@pytest.mark.parametrize("characters,lines,slot", [
    ("98", "2", "update_lines"),
    ("1", "1", "update_lines"),
    ("98", "98", "lines"),
    ("1", "1", "lines"),
])
def test_counts_cannot_borrow_other_units_or_unrelated_array_lengths(characters, lines, slot):
    records = [
        _record("ls", {}, {"contents": ["deploy.py", "backups"]},
                class_name="GorillaFileSystem", turn_id=0, call_id=0),
        _record("wc", {"file_name": "deploy.py", "mode": "c"}, {"count": 98},
                class_name="GorillaFileSystem", turn_id=0, call_id=1),
        _record("wc", {"file_name": "deploy.py"}, {"count": 1},
                class_name="GorillaFileSystem", turn_id=0, call_id=2),
        _record("grep", {"file_name": "deploy.py", "pattern": "update"},
                {"matching_lines": ["# update the server"]},
                class_name="GorillaFileSystem", turn_id=0, call_id=3),
    ]
    query = f"Send Catherine a receipt in exact format `Characters: {{characters}}; Lines: {{{slot}}}` using its character count and "
    query += "the number of matching lines." if slot == "update_lines" else "its file line count."
    send = _record("send_message", {"receiver_id": "USR003", "message": f"Characters: {characters}; Lines: {lines}"},
                   {"sent_status": True}, class_name="MessageAPI", turn_id=1, call_id=0)
    draft = _draft([_turn(0, "GorillaFileSystem", "List entries, count file lines and characters, and search for `update`.", records),
                    _turn(1, "MessageAPI", query, [send])])
    assert not query_contract_gate(draft).passed


def test_intentional_missing_records_never_supply_template_slots():
    draft = example()
    draft.turns[0].is_intentional_missing = True
    assert not query_contract_gate(draft).passed


def test_missing_parameter_recovery_can_reuse_an_exact_format_from_affected_query():
    draft = example(kind="multi_turn_miss_param")
    missing = copy.deepcopy(draft.turns[1])
    missing.is_intentional_missing = True
    missing.missing_kind = "parameter"
    missing.execution_records = []
    recovered = draft.turns[1]
    recovered.turn_id = 2
    recovered.query = "Send that receipt to Catherine."
    from dataclasses import replace
    recovered.execution_records = [replace(record, turn_id=2) for record in recovered.execution_records]
    draft.turns = [draft.turns[0], missing, recovered]
    assert query_contract_gate(draft).passed


@pytest.mark.parametrize("query,pattern,passed", [
    ("Show its function declaration.", "def ", False),
    ("Find the exact text `def`.", "def", True),
    ("Find the exact text `def`.", "def ", False),
    ("Find the exact text `def `, including its trailing space.", "def ", True),
])
def test_precise_search_text_preserves_meaningful_whitespace(query, pattern, passed):
    record = _record("grep", {"file_name": "deploy.py", "pattern": pattern},
                     {"matching_lines": ["def deploy(): pass"]},
                     class_name="GorillaFileSystem", turn_id=0, call_id=0)
    assert query_contract_gate(_draft([_turn(0, "GorillaFileSystem", query, [record])])).passed is passed


def test_explicit_new_message_without_file_inspection_is_deterministic():
    record = _record("send_message", {"receiver_id": "USR003", "message": "Meet at noon."},
                     {"sent_status": True}, class_name="MessageAPI", turn_id=0, call_id=0)
    assert query_contract_gate(_draft([_turn(0, "MessageAPI", "Send Catherine this: Meet at noon.", [record])])).passed


@pytest.mark.parametrize("query,passed", [("Print the receipt in the terminal.", True),
                                         ("Read the report.", False)])
def test_echo_terminal_output_has_a_specific_user_intent(query, passed):
    record = _record("echo", {"content": "receipt"}, {"terminal_output": "receipt"},
                     class_name="GorillaFileSystem", turn_id=0, call_id=0)
    result = action_minimality_gate(_draft([_turn(0, "GorillaFileSystem", query, [record])]), catalog=make_catalog())
    assert result.passed is passed
