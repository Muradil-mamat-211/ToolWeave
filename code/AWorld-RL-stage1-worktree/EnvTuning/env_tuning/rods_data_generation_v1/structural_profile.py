"""Non-gating diagnostics for RODS structural-alignment evidence.

The public RODS sources do not publish a deterministic Phi extractor,
distance, or acceptance threshold.  These profiles therefore report only
facts recoverable from seed GT syntax and executed draft provenance.
"""

from __future__ import annotations

import ast
from typing import Any, Sequence

from .function_catalog import FunctionCatalog
from .models import SeedRecord, SynthesizedTurn


NOT_RECOVERABLE = "NOT_RECOVERABLE"


def _call_name_from_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _parse_call(raw: str) -> ast.Call | None:
    try:
        node = ast.parse(raw.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    return node if isinstance(node, ast.Call) else None


def _call_name(raw: str) -> str | None:
    node = _parse_call(raw)
    return _call_name_from_node(node) if node is not None else None


def _nested_edges(node: ast.Call) -> list[tuple[str, str]]:
    """Return deterministically explicit inner-call -> consumer-call edges."""

    outer_name = _call_name_from_node(node) or "<unknown>"
    edges: list[tuple[str, str]] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            child_name = _call_name_from_node(child) or "<unknown>"
            edges.append((child_name, outer_name))
            edges.extend(_nested_edges(child))
        else:
            for nested in ast.walk(child):
                if isinstance(nested, ast.Call):
                    child_name = _call_name_from_node(nested) or "<unknown>"
                    edges.append((child_name, outer_name))
                    edges.extend(_nested_edges(nested))
                    break
    return edges


def _nested_depth(node: ast.Call) -> int:
    nested_calls = [child for child in ast.iter_child_nodes(node) if isinstance(child, ast.Call)]
    if not nested_calls:
        # Calls nested below containers/keywords remain explicit AST children
        # but are not direct children; walk each argument conservatively.
        nested_calls = [
            nested
            for child in ast.iter_child_nodes(node)
            for nested in ast.walk(child)
            if isinstance(nested, ast.Call)
        ]
    return 1 + max((_nested_depth(child) for child in nested_calls), default=0)


def seed_structural_profile(seed: SeedRecord, catalog: FunctionCatalog) -> dict[str, Any]:
    call_counts: list[int] = []
    classes: list[list[str]] = []
    explicit_nested_edges: list[dict[str, Any]] = []
    nested_depths: list[int] = []
    unparseable_calls = 0
    for turn_id, raw_turn in enumerate(seed.GT_old):
        calls = raw_turn if isinstance(raw_turn, list) else []
        call_counts.append(len(calls))
        turn_classes: set[str] = set()
        for call_index, raw_call in enumerate(calls):
            if not isinstance(raw_call, str):
                unparseable_calls += 1
                continue
            node = _parse_call(raw_call)
            name = _call_name(raw_call)
            if name is not None:
                try:
                    turn_classes.add(catalog.get(name).class_name)
                except ValueError:
                    pass
            if node is None:
                unparseable_calls += 1
                continue
            nested_depths.append(_nested_depth(node))
            for producer, consumer in _nested_edges(node):
                explicit_nested_edges.append(
                    {
                        "turn_id": turn_id,
                        "call_index": call_index,
                        "producer": producer,
                        "consumer": consumer,
                    }
                )
        classes.append(sorted(turn_classes))
    return {
        "num_user_turns": len(seed.Q_old),
        "gt_call_count_per_turn": call_counts,
        "tool_classes_per_turn": classes,
        "class_sequence": classes,
        "explicit_nested_dependency_edges": explicit_nested_edges,
        "recoverable_nested_dependency_depth": (
            max(nested_depths) if nested_depths else 0
        ),
        "unparseable_gt_call_count": unparseable_calls,
        "cross_turn_dependencies": {
            "status": NOT_RECOVERABLE,
            "reason": (
                "Seed contract has calls and questions but no deterministic execution dependency trace; "
                "shared literals are not treated as causal evidence."
            ),
        },
        "state_changes": {
            "status": NOT_RECOVERABLE,
            "reason": "Seed contract contains initial state but no per-call pre/post snapshots.",
        },
        "used_for_acceptance": False,
    }


def draft_structural_profile(turns: Sequence[SynthesizedTurn]) -> dict[str, Any]:
    records = [record for turn in turns for record in turn.execution_records]
    state_chain_edges: list[dict[str, Any]] = []
    state_changes: list[dict[str, Any]] = []
    explicit_dependency_edges: list[dict[str, Any]] = []
    previous = None
    for record in records:
        record_key = {"turn_id": record.turn_id, "call_id": record.call_id}
        state_changes.append(
            {
                **record_key,
                "changed": record.pre_state != record.post_state,
            }
        )
        if previous is not None and previous.post_state == record.pre_state:
            state_chain_edges.append(
                {
                    "from": {"turn_id": previous.turn_id, "call_id": previous.call_id},
                    "to": record_key,
                    "relation": "exact_post_state_to_next_pre_state",
                }
            )
        dependencies = record.dependency_provenance.get(
            "resolved_dependency_call_ids", []
        )
        if isinstance(dependencies, list):
            for dependency in dependencies:
                if isinstance(dependency, dict):
                    explicit_dependency_edges.append(
                        {"from": dict(dependency), "to": record_key}
                    )
        previous = record

    dependency_depth: int | str = NOT_RECOVERABLE
    dependency_trace_complete = all(
        isinstance(record.dependency_provenance.get("resolved_dependency_call_ids"), list)
        and record.dependency_provenance.get("parameter_dependency_status")
        in {"GROUNDED", "INTENTIONAL_MISSING_SKIPPED"}
        for record in records
    )
    if dependency_trace_complete:
        # PROJECT_STRUCTURAL_GUIDANCE (NON-GATING): once the semantic guard has
        # recorded every parameter source, the longest explicit producer ->
        # consumer chain is deterministically recoverable.  An edge-free trace
        # has depth one when it contains calls.  Invalid/cyclic references are
        # reported as not recoverable; they are never guessed or accepted via
        # an invented structural threshold.
        node_keys = {(record.turn_id, record.call_id) for record in records}
        dependencies_by_node: dict[tuple[int, int], set[tuple[int, int]]] = {
            key: set() for key in node_keys
        }
        graph_valid = True
        for edge in explicit_dependency_edges:
            source = edge["from"]
            target = edge["to"]
            source_key = (source.get("turn_id"), source.get("call_id"))
            target_key = (target.get("turn_id"), target.get("call_id"))
            if source_key not in node_keys or target_key not in node_keys:
                graph_valid = False
                break
            dependencies_by_node[target_key].add(source_key)

        visiting: set[tuple[int, int]] = set()
        memo: dict[tuple[int, int], int] = {}

        def depth(node: tuple[int, int]) -> int:
            nonlocal graph_valid
            if node in memo:
                return memo[node]
            if node in visiting:
                graph_valid = False
                return 0
            visiting.add(node)
            value = 1 + max(
                (depth(parent) for parent in dependencies_by_node[node]),
                default=0,
            )
            visiting.remove(node)
            memo[node] = value
            return value

        if graph_valid:
            recovered = max((depth(node) for node in node_keys), default=0)
            dependency_depth = recovered if graph_valid else NOT_RECOVERABLE

    cross_turn_dependencies: list[dict[str, Any]] | dict[str, str]
    if dependency_trace_complete:
        cross_turn_dependencies = [
            edge
            for edge in explicit_dependency_edges
            if edge["from"].get("turn_id") != edge["to"]["turn_id"]
        ]
    else:
        cross_turn_dependencies = {
            "status": NOT_RECOVERABLE,
            "reason": "Complete parameter-to-prior-result provenance is unavailable.",
        }

    return {
        "num_user_turns": len(turns),
        "gt_call_count_per_turn": [len(turn.calls) for turn in turns],
        "tool_classes_per_turn": [[turn.class_name] for turn in turns],
        "class_sequence": [turn.class_name for turn in turns],
        "state_changes": state_changes,
        "state_chain_edges": state_chain_edges,
        "explicit_dependency_edges": explicit_dependency_edges,
        "recoverable_dependency_depth": dependency_depth,
        "cross_turn_dependencies": cross_turn_dependencies,
        "used_for_acceptance": False,
    }


def structural_alignment_diagnostics(
    seed_profile: dict[str, Any], draft_profile: dict[str, Any]
) -> dict[str, Any]:
    seed_counts = list(seed_profile.get("gt_call_count_per_turn", []))
    draft_counts = list(draft_profile.get("gt_call_count_per_turn", []))
    width = max(len(seed_counts), len(draft_counts))
    deltas = [
        (draft_counts[index] if index < len(draft_counts) else None,
         seed_counts[index] if index < len(seed_counts) else None)
        for index in range(width)
    ]
    count_delta = [
        None if draft is None or seed is None else draft - seed
        for draft, seed in deltas
    ]
    seed_classes = seed_profile.get("tool_classes_per_turn")
    draft_classes = draft_profile.get("tool_classes_per_turn")
    class_relation = (
        "EXACT" if seed_classes == draft_classes else "DIFFERENT_OR_REORDERED"
    )
    seed_depth = seed_profile.get("recoverable_nested_dependency_depth")
    draft_depth = draft_profile.get("recoverable_dependency_depth")
    depth_delta: int | str = NOT_RECOVERABLE
    if isinstance(seed_depth, int) and isinstance(draft_depth, int):
        depth_delta = draft_depth - seed_depth
    return {
        "turn_count_equal": seed_profile.get("num_user_turns")
        == draft_profile.get("num_user_turns"),
        "per_turn_call_count_delta_draft_minus_seed": count_delta,
        "class_sequence_relation": class_relation,
        "recoverable_dependency_depth_delta": depth_delta,
        "used_for_acceptance": False,
        "acceptance_threshold": "NOT_DEFINED_BY_PUBLIC_RODS_SOURCES",
    }
