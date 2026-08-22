from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from datasets import Dataset

from env_tuning.rods_matchtir_v1.lifecycle import (
    BFCL_DATA_TYPES,
    CANDIDATE_SCHEMA_VERSION,
    LifecycleConfig,
    RODSStage3Lifecycle,
    SEED_SCHEMA_VERSION,
    classify_progress,
    validate_candidate_record,
)
from env_tuning.rods_matchtir_v1.provenance import extract_available_functions


def _quotas(**overrides: int) -> dict[str, int]:
    quotas = {data_type: 0 for data_type in BFCL_DATA_TYPES}
    quotas.update(overrides)
    return quotas


def _selection_lifecycle(
    tmp_path,
    *,
    max_seeds: int,
    quotas: dict[str, int],
    cooldown: int = 3,
) -> RODSStage3Lifecycle:
    return RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
            max_seeds_per_selection=max_seeds,
            seed_type_quotas=quotas,
            seed_cooldown_steps=cooldown,
        )
    )


def _context(
    sample: dict,
    *,
    sample_id: str | None = None,
    data_type: str | None = None,
    **extra,
) -> dict:
    kwargs = sample["extra_info"]["interaction_kwargs"]
    context = {
        "prompt_id": sample_id or kwargs["id"],
        "data_type": data_type or sample["data_source"],
        "questions": kwargs["question"],
        "ground_truth": kwargs["ground_truth"],
        "available_functions": extract_available_functions(sample["prompt"]),
        "initial_config": json.loads(kwargs["initial_config"]),
        "context_reliable": True,
        "policy_steps": [],
    }
    context.update(extra)
    return context


def _minimal_sample(sample_id: str, data_type: str = "multi_turn_base") -> dict:
    function_marker = "Here is a list of functions in JSON format that you can invoke.\n"
    system = (
        "Use <think>reason</think>, tool calls when needed, and <answer>done</answer>.\n"
        + function_marker
        + json.dumps([{"name": "fixture_tool", "parameters": {"type": "dict"}}])
    )
    return {
        "data_source": data_type,
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": "fixture question"},
        ],
        "ability": "tool",
        "reward_model": {"style": "interaction"},
        "extra_info": {
            "original_id": sample_id,
            "interaction_kwargs": {
                "name": "multi_turn_fc",
                "id": sample_id,
                "initial_config": "{}",
                "involved_classes": [],
                "ground_truth": [["fixture_tool()"]],
                "processed_question": ["fixture question"],
                "question": ["fixture question"],
            },
        },
    }


def _generated_state_item(sample_id: str, progress: float | None = None) -> dict:
    item = {
        "sample_id": sample_id,
        "sample": _minimal_sample(sample_id),
        "generation_metadata": {},
        "validation": {"passed": True},
        "ingested_epoch": 0,
        "ingested_step": 0,
        "last_observed_epoch": None,
        "last_observed_step": None,
        "progress_observations": [],
    }
    if progress is not None:
        item["progress_observations"] = [progress]
        item["last_observed_epoch"] = 0
        item["last_observed_step"] = 1
    return item


def test_13_boundary_detection_is_independent_of_any_local_signal():
    progress = [0.0, 0.2, 0.8, 1.0]
    mean_progress = sum(progress) / len(progress)
    baseline = classify_progress(mean_progress)
    for hypothetical_local in (-100.0, 0.0, 100.0):
        # No local quantity is accepted by the detector API.
        decision = classify_progress(mean_progress)
        assert decision == baseline
        assert hypothetical_local not in (decision.mean_progress, decision.boundary_score_phi)
    assert baseline.classification == "boundary"
    assert baseline.boundary_score_phi == pytest.approx(1.0)


def _candidate(sample: dict, candidate_id: str = "candidate-1") -> dict:
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "validated": True,
        "validation": {"passed": True, "suite": "mock-complete-validator"},
        "generation_metadata": {"generator": "fixture-only"},
        "sample": sample,
    }


def test_candidate_interface_fails_closed_without_validation(validation_rows):
    sample = copy.deepcopy(validation_rows[0])
    record = _candidate(sample)
    record["validated"] = False
    with pytest.raises(ValueError, match="validated"):
        validate_candidate_record(record)
    record["validated"] = True
    record["validation"]["passed"] = False
    with pytest.raises(ValueError, match="validation.passed"):
        validate_candidate_record(record)


def test_boundary_interface_json_schemas_are_machine_readable(stage_root):
    schema_dir = stage_root / "schemas"
    seed_schema = json.loads((schema_dir / "rods_boundary_seed_v1.schema.json").read_text())
    candidate_schema = json.loads(
        (schema_dir / "rods_validated_candidate_v1.schema.json").read_text()
    )
    assert seed_schema["$id"] == "rods_boundary_seed.v1"
    assert candidate_schema["$id"] == "rods_validated_candidate.v1"
    assert candidate_schema["properties"]["validated"] == {"const": True}


def test_epoch_boundary_ingestion_and_generated_retirement_protect_originals(
    tmp_path, validation_rows
):
    original = copy.deepcopy(validation_rows[0])
    generated = copy.deepcopy(validation_rows[1])
    kwargs = generated["extra_info"]["interaction_kwargs"]
    kwargs["id"] = "synthetic_validated_001"
    generated["extra_info"]["original_id"] = "synthetic_validated_001"

    dataset = SimpleNamespace(dataframe=Dataset.from_list([original]))
    candidate_path = tmp_path / "validated_candidates.jsonl"
    candidate_path.write_text(json.dumps(_candidate(generated)) + "\n", encoding="utf-8")
    config = LifecycleConfig(
        enabled=True,
        seed_output_path=str(tmp_path / "seeds.jsonl"),
        candidate_input_path=str(candidate_path),
        state_path=str(tmp_path / "state.json"),
        generated_pool_cap=1,
        injection_ratio=1.0,
    )
    lifecycle = RODSStage3Lifecycle(config)
    metrics = lifecycle.on_epoch_boundary(dataset, epoch=0, global_step=1)
    assert metrics["rods_stage3_lifecycle/ingested_candidate_count"] == 1.0
    assert len(dataset.dataframe) == 2
    assert lifecycle.active_generated_count == 1

    generated_context = _context(generated, sample_id="synthetic_validated_001")
    lifecycle.observe_and_emit(
        progress_rewards=[0.10],
        uids=["generated-uid"],
        rollout_provenance=[generated_context],
        epoch=0,
        global_step=2,
    )
    metrics = lifecycle.on_epoch_boundary(dataset, epoch=1, global_step=3)
    assert metrics["rods_stage3_lifecycle/retired_too_hard_count"] == 1.0
    assert lifecycle.active_generated_count == 0
    assert len(dataset.dataframe) == 1
    remaining_id = dataset.dataframe[0]["extra_info"]["interaction_kwargs"]["id"]
    assert remaining_id == original["extra_info"]["interaction_kwargs"]["id"]


def test_boundary_seed_manifest_contains_complete_training_generator_package(
    tmp_path, validation_rows
):
    sample = validation_rows[0]
    kwargs = sample["extra_info"]["interaction_kwargs"]
    seed_path = tmp_path / "seeds.jsonl"
    quotas = _quotas(**{sample["data_source"]: 1})
    lifecycle = _selection_lifecycle(
        tmp_path,
        max_seeds=1,
        quotas=quotas,
    )
    context = _context(sample)
    metrics = lifecycle.observe_and_emit(
        progress_rewards=[0.25, 0.75],
        uids=["same", "same"],
        rollout_provenance=[context, context],
        epoch=2,
        global_step=17,
    )
    assert metrics["rods_boundary/seed_emitted_count"] == 1.0
    seed = json.loads(seed_path.read_text(encoding="utf-8").strip())
    assert seed["schema_version"] == SEED_SCHEMA_VERSION
    assert seed["sample_id"] == kwargs["id"]
    assert seed["Q_old"] == kwargs["question"]
    assert seed["GT_old"] == kwargs["ground_truth"]
    assert seed["available_functions"]
    assert seed["initial_config"]
    assert seed["mean_progress"] == 0.5
    assert seed["boundary_score_phi"] == 1.0
    assert seed["training_epoch_or_step"] == {"epoch": 2, "global_step": 17}
    assert seed["generation_metadata"]["progress_source"] == "R_P_only"


def test_generated_pool_cap_defers_valid_candidates(tmp_path, validation_rows):
    original = copy.deepcopy(validation_rows[0])
    candidates = []
    for index in (1, 2):
        sample = copy.deepcopy(validation_rows[index])
        sample_id = f"synthetic_cap_{index}"
        sample["extra_info"]["interaction_kwargs"]["id"] = sample_id
        sample["extra_info"]["original_id"] = sample_id
        candidates.append(_candidate(sample, candidate_id=f"candidate-{index}"))
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(record) + "\n" for record in candidates), encoding="utf-8"
    )
    dataset = SimpleNamespace(dataframe=Dataset.from_list([original]))
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(candidate_path),
            state_path=str(tmp_path / "state.json"),
            generated_pool_cap=1,
            injection_ratio=1.0,
        )
    )
    metrics = lifecycle.on_epoch_boundary(dataset, epoch=0, global_step=0)
    assert metrics["rods_stage3_lifecycle/ingested_candidate_count"] == 1.0
    assert metrics["rods_stage3_lifecycle/deferred_candidate_count"] == 1.0
    assert lifecycle.active_generated_count == 1


def test_a_phi_ranking_prefers_capability_midpoint(tmp_path, validation_rows):
    sample = validation_rows[0]
    lifecycle = _selection_lifecycle(
        tmp_path,
        max_seeds=1,
        quotas=_quotas(multi_turn_base=1),
    )
    contexts = [
        _context(sample, sample_id=f"phi-{progress}", data_type="multi_turn_base")
        for progress in (0.21, 0.50, 0.80)
    ]
    lifecycle.observe_and_emit(
        progress_rewards=[0.21, 0.50, 0.80],
        uids=["u-021", "u-050", "u-080"],
        rollout_provenance=contexts,
        epoch=0,
        global_step=10,
    )
    emitted = [json.loads(line) for line in (tmp_path / "seeds.jsonl").read_text().splitlines()]
    assert [item["sample_id"] for item in emitted] == ["phi-0.5"]
    assert emitted[0]["boundary_score_phi"] == pytest.approx(1.0)


def test_b_per_type_quota_is_a_hard_upper_bound(tmp_path, validation_rows):
    sample = validation_rows[0]
    lifecycle = _selection_lifecycle(
        tmp_path,
        max_seeds=1,
        quotas=_quotas(multi_turn_base=1),
    )
    lifecycle.observe_and_emit(
        progress_rewards=[0.40, 0.50, 0.60],
        uids=["base-a", "base-b", "base-c"],
        rollout_provenance=[
            _context(sample, sample_id=f"base-{index}", data_type="multi_turn_base")
            for index in range(3)
        ],
        epoch=0,
        global_step=10,
    )
    emitted = (tmp_path / "seeds.jsonl").read_text().splitlines()
    assert len(emitted) == 1


def test_c_total_seed_budget_is_a_hard_upper_bound(tmp_path, validation_rows):
    sample = validation_rows[0]
    lifecycle = _selection_lifecycle(
        tmp_path,
        max_seeds=2,
        quotas=_quotas(multi_turn_base=1, multi_turn_miss_func=1),
    )
    types = [
        "multi_turn_base",
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_func",
    ]
    metrics = lifecycle.observe_and_emit(
        progress_rewards=[0.50] * len(types),
        uids=[f"uid-{index}" for index in range(len(types))],
        rollout_provenance=[
            _context(sample, sample_id=f"total-{index}", data_type=data_type)
            for index, data_type in enumerate(types)
        ],
        epoch=0,
        global_step=10,
    )
    assert metrics["rods_boundary/seed_emitted_count"] == 2.0
    assert len((tmp_path / "seeds.jsonl").read_text().splitlines()) == 2


def test_d_cooldown_blocks_same_sample_inside_window(tmp_path, validation_rows):
    sample = validation_rows[0]
    lifecycle = _selection_lifecycle(
        tmp_path,
        max_seeds=1,
        quotas=_quotas(multi_turn_base=1),
        cooldown=3,
    )
    context = _context(sample, sample_id="cooldown", data_type="multi_turn_base")
    first = lifecycle.observe_and_emit(
        progress_rewards=[0.50],
        uids=["uid"],
        rollout_provenance=[context],
        epoch=0,
        global_step=100,
    )
    second = lifecycle.observe_and_emit(
        progress_rewards=[0.50],
        uids=["uid"],
        rollout_provenance=[context],
        epoch=0,
        global_step=101,
    )
    assert first["rods_boundary/seed_emitted_count"] == 1.0
    assert second["rods_boundary/seed_emitted_count"] == 0.0
    assert second["rods_boundary/cooldown_filtered_count"] == 1.0


def test_e_cooldown_expires_at_configured_distance(tmp_path, validation_rows):
    sample = validation_rows[0]
    lifecycle = _selection_lifecycle(
        tmp_path,
        max_seeds=1,
        quotas=_quotas(multi_turn_base=1),
        cooldown=3,
    )
    context = _context(sample, sample_id="cooldown-expiry", data_type="multi_turn_base")
    for step in (100, 103):
        metrics = lifecycle.observe_and_emit(
            progress_rewards=[0.50],
            uids=["uid"],
            rollout_provenance=[context],
            epoch=0,
            global_step=step,
        )
        assert metrics["rods_boundary/seed_emitted_count"] == 1.0
    assert len((tmp_path / "seeds.jsonl").read_text().splitlines()) == 2


def test_f_seed_selection_ignores_local_and_fused_advantages(tmp_path, validation_rows):
    sample = validation_rows[0]
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left = _selection_lifecycle(
        left_dir,
        max_seeds=1,
        quotas=_quotas(multi_turn_base=1),
    )
    right = _selection_lifecycle(
        right_dir,
        max_seeds=1,
        quotas=_quotas(multi_turn_base=1),
    )
    baseline_context = _context(
        sample,
        sample_id="local-independent",
        data_type="multi_turn_base",
        A_local=-1000.0,
        A_new=-999.0,
    )
    perturbed_context = _context(
        sample,
        sample_id="local-independent",
        data_type="multi_turn_base",
        A_local=1000.0,
        A_new=1001.0,
    )
    for lifecycle, context in ((left, baseline_context), (right, perturbed_context)):
        lifecycle.observe_and_emit(
            progress_rewards=[0.50],
            uids=["uid"],
            rollout_provenance=[context],
            epoch=0,
            global_step=10,
        )
    left_seed = json.loads((left_dir / "seeds.jsonl").read_text())
    right_seed = json.loads((right_dir / "seeds.jsonl").read_text())
    assert left_seed == right_seed


def test_g_twenty_percent_epoch_injection_cap(tmp_path, monkeypatch):
    originals = [_minimal_sample(f"original-{index}") for index in range(400)]
    records = [
        _candidate(_minimal_sample(f"generated-{index}"), candidate_id=f"candidate-{index}")
        for index in range(300)
    ]
    dataset = SimpleNamespace(dataframe=originals)
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
            injection_ratio=0.20,
            generated_pool_cap=400,
        )
    )
    monkeypatch.setattr(lifecycle.candidate_queue, "read", lambda: records)
    monkeypatch.setattr(lifecycle, "_replace_dataset", lambda _: None)
    metrics = lifecycle.on_epoch_boundary(dataset, epoch=1, global_step=10)
    assert metrics["rods_stage3_lifecycle/active_pool_before_injection"] == 400.0
    assert metrics["rods_stage3_lifecycle/max_new_this_epoch"] == 80.0
    assert metrics["rods_stage3_lifecycle/ingested_candidate_count"] == 80.0
    assert metrics["rods_stage3_lifecycle/deferred_candidate_count"] == 220.0


def test_h_injection_and_generated_pool_caps_apply_simultaneously(tmp_path, monkeypatch):
    originals = [_minimal_sample(f"original-{index}") for index in range(30)]
    records = [
        _candidate(_minimal_sample(f"incoming-{index}"), candidate_id=f"incoming-{index}")
        for index in range(300)
    ]
    dataset = SimpleNamespace(dataframe=originals)
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
            injection_ratio=0.20,
            generated_pool_cap=400,
        )
    )
    lifecycle._state["active_candidates"] = {
        f"active-{index}": _generated_state_item(f"active-generated-{index}")
        for index in range(370)
    }
    monkeypatch.setattr(lifecycle.candidate_queue, "read", lambda: records)
    monkeypatch.setattr(lifecycle, "_replace_dataset", lambda _: None)
    metrics = lifecycle.on_epoch_boundary(dataset, epoch=1, global_step=10)
    assert metrics["rods_stage3_lifecycle/max_new_this_epoch"] == 80.0
    assert metrics["rods_stage3_lifecycle/ingested_candidate_count"] == 30.0
    assert lifecycle.active_generated_count == 400


def test_i_deferred_candidates_persist_for_future_epoch(tmp_path, monkeypatch):
    originals = [_minimal_sample(f"original-{index}") for index in range(5)]
    records = [
        _candidate(_minimal_sample(f"deferred-{index}"), candidate_id=f"deferred-{index}")
        for index in range(2)
    ]
    dataset = SimpleNamespace(dataframe=originals)
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
            injection_ratio=0.20,
            generated_pool_cap=400,
        )
    )
    monkeypatch.setattr(lifecycle.candidate_queue, "read", lambda: records)
    monkeypatch.setattr(lifecycle, "_replace_dataset", lambda _: None)
    first = lifecycle.on_epoch_boundary(dataset, epoch=1, global_step=10)
    second = lifecycle.on_epoch_boundary(dataset, epoch=2, global_step=20)
    assert first["rods_stage3_lifecycle/ingested_candidate_count"] == 1.0
    assert first["rods_stage3_lifecycle/deferred_candidate_count"] == 1.0
    assert second["rods_stage3_lifecycle/ingested_candidate_count"] == 1.0
    assert second["rods_stage3_lifecycle/deferred_candidate_count"] == 0.0
    assert lifecycle.active_generated_count == 2


def test_epoch_n_candidate_is_not_injected_until_epoch_n_plus_one(tmp_path, monkeypatch):
    originals = [_minimal_sample(f"original-{index}") for index in range(5)]
    record = _candidate(_minimal_sample("next-epoch"), candidate_id="next-epoch")
    record["generation_metadata"]["generated_epoch"] = 1
    dataset = SimpleNamespace(dataframe=originals)
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
            injection_ratio=1.0,
        )
    )
    monkeypatch.setattr(lifecycle.candidate_queue, "read", lambda: [record])
    monkeypatch.setattr(lifecycle, "_replace_dataset", lambda _: None)
    current_epoch = lifecycle.on_epoch_boundary(dataset, epoch=1, global_step=10)
    next_epoch = lifecycle.on_epoch_boundary(dataset, epoch=2, global_step=20)
    assert current_epoch["rods_stage3_lifecycle/ingested_candidate_count"] == 0.0
    assert current_epoch["rods_stage3_lifecycle/deferred_candidate_count"] == 1.0
    assert next_epoch["rods_stage3_lifecycle/ingested_candidate_count"] == 1.0


def test_j_original_rows_are_never_retired(tmp_path):
    original = _minimal_sample("protected-original")
    dataset = SimpleNamespace(dataframe=Dataset.from_list([original]))
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
            injection_ratio=0.0,
        )
    )
    lifecycle.on_epoch_boundary(dataset, epoch=0, global_step=0)
    lifecycle.observe_and_emit(
        progress_rewards=[0.99],
        uids=["original-uid"],
        rollout_provenance=[_context(original)],
        epoch=0,
        global_step=1,
    )
    metrics = lifecycle.on_epoch_boundary(dataset, epoch=1, global_step=2)
    assert metrics["rods_stage3_lifecycle/original_protected_count"] == 1.0
    assert len(dataset.dataframe) == 1
    assert dataset.dataframe[0]["extra_info"]["interaction_kwargs"]["id"] == "protected-original"


def test_k_mastered_generated_sample_is_retired(tmp_path):
    original = _minimal_sample("original")
    generated = _minimal_sample("generated-mastered")
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(json.dumps(_candidate(generated)) + "\n", encoding="utf-8")
    dataset = SimpleNamespace(dataframe=Dataset.from_list([original]))
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(candidate_path),
            state_path=str(tmp_path / "state.json"),
            injection_ratio=1.0,
        )
    )
    lifecycle.on_epoch_boundary(dataset, epoch=0, global_step=0)
    lifecycle.observe_and_emit(
        progress_rewards=[0.96],
        uids=["generated-uid"],
        rollout_provenance=[_context(generated)],
        epoch=0,
        global_step=1,
    )
    metrics = lifecycle.on_epoch_boundary(dataset, epoch=1, global_step=2)
    assert metrics["rods_stage3_lifecycle/retired_mastered_count"] == 1.0
    assert lifecycle.active_generated_count == 0


def test_l_first_too_hard_generated_observation_triggers_trial_eviction(tmp_path):
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
            injection_ratio=0.0,
        )
    )
    lifecycle._state["active_candidates"] = {
        "too-hard": _generated_state_item("generated-too-hard", progress=0.19)
    }
    dataset = SimpleNamespace(dataframe=Dataset.from_list([_minimal_sample("original")]))
    metrics = lifecycle.on_epoch_boundary(dataset, epoch=1, global_step=2)
    assert metrics["rods_stage3_lifecycle/retired_too_hard_count"] == 1.0
    assert lifecycle.active_generated_count == 0


def test_m_pool_overflow_prunes_lowest_phi_generated_sample(tmp_path):
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
            injection_ratio=0.0,
            generated_pool_cap=3,
        )
    )
    lifecycle._state["active_candidates"] = {
        "midpoint": _generated_state_item("generated-midpoint", progress=0.50),
        "near-midpoint": _generated_state_item("generated-near-midpoint", progress=0.60),
        "moderate": _generated_state_item("generated-moderate", progress=0.30),
        "low-phi": _generated_state_item("generated-low-phi", progress=0.90),
    }
    dataset = SimpleNamespace(dataframe=Dataset.from_list([_minimal_sample("original")]))
    metrics = lifecycle.on_epoch_boundary(dataset, epoch=1, global_step=2)
    assert metrics["rods_stage3_lifecycle/priority_pruned_count"] == 1.0
    assert "low-phi" not in lifecycle._state["active_candidates"]
    assert "midpoint" in lifecycle._state["active_candidates"]
    assert lifecycle.active_generated_count == 3


def test_seed_selection_requires_complete_explicit_m_quota_and_c_config():
    with pytest.raises(ValueError, match="configured together"):
        LifecycleConfig.from_mapping({"max_seeds_per_selection": 4})
    with pytest.raises(ValueError, match="must equal"):
        LifecycleConfig.from_mapping(
            {
                "max_seeds_per_selection": 1,
                "seed_type_quotas": _quotas(multi_turn_base=2),
                "seed_cooldown_steps": 3,
            }
        )


def test_seed_1_equal_total_and_four_type_quotas_are_valid():
    config = LifecycleConfig.from_mapping(
        {
            "max_seeds_per_selection": 8,
            "seed_type_quotas": {data_type: 2 for data_type in BFCL_DATA_TYPES},
            "seed_cooldown_steps": 3,
        }
    )
    assert config.max_seeds_per_selection == 8
    assert sum(config.seed_type_quotas.values()) == 8


def test_seed_2_unequal_quota_budget_is_rejected():
    with pytest.raises(ValueError, match="must equal"):
        LifecycleConfig.from_mapping(
            {
                "max_seeds_per_selection": 10,
                "seed_type_quotas": {data_type: 2 for data_type in BFCL_DATA_TYPES},
                "seed_cooldown_steps": 3,
            }
        )


def test_seed_3_and_4_production_requires_all_quotas_and_cooldown(tmp_path):
    common = {
        "enabled": True,
        "require_seed_selection_config": True,
        "seed_output_path": str(tmp_path / "seeds.jsonl"),
        "candidate_input_path": str(tmp_path / "candidates.jsonl"),
        "state_path": str(tmp_path / "state.json"),
        "max_seeds_per_selection": 8,
    }
    missing_quota = {data_type: 2 for data_type in BFCL_DATA_TYPES}
    missing_quota.pop("multi_turn_long_context")
    with pytest.raises(ValueError, match="exactly"):
        LifecycleConfig.from_mapping(
            {**common, "seed_type_quotas": missing_quota, "seed_cooldown_steps": 3}
        )
    with pytest.raises(ValueError, match="configured together"):
        LifecycleConfig.from_mapping(
            {
                **common,
                "seed_type_quotas": {
                    data_type: 2 for data_type in BFCL_DATA_TYPES
                },
                "seed_cooldown_steps": None,
            }
        )


def test_seed_5_generator_disabled_allows_paper_unspecified_nulls():
    config = LifecycleConfig.from_mapping(
        {
            "enabled": False,
            "require_seed_selection_config": True,
            "max_seeds_per_selection": None,
            "seed_type_quotas": {data_type: None for data_type in BFCL_DATA_TYPES},
            "seed_cooldown_steps": None,
        }
    )
    assert config.seed_selection_configured is False


def test_seed_production_null_selection_config_fails_fast(tmp_path):
    with pytest.raises(ValueError, match="paper-unspecified project hyperparameters"):
        LifecycleConfig(
            enabled=True,
            require_seed_selection_config=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
        )


def test_seed_6_unused_type_quota_is_not_redistributed(tmp_path, validation_rows):
    sample = validation_rows[0]
    lifecycle = _selection_lifecycle(
        tmp_path,
        max_seeds=2,
        quotas=_quotas(multi_turn_base=1, multi_turn_miss_func=1),
    )
    metrics = lifecycle.observe_and_emit(
        progress_rewards=[0.50, 0.49, 0.48],
        uids=["base-a", "base-b", "base-c"],
        rollout_provenance=[
            _context(sample, sample_id=f"base-only-{index}", data_type="multi_turn_base")
            for index in range(3)
        ],
        epoch=0,
        global_step=10,
    )
    assert metrics["rods_boundary/seed_emitted_count"] == 1.0
    assert metrics["rods_boundary/seed_emitted_multi_turn_miss_func"] == 0.0


def test_unconfigured_paper_seed_constants_fail_closed(tmp_path, validation_rows):
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
        )
    )
    metrics = lifecycle.observe_and_emit(
        progress_rewards=[0.50],
        uids=["uid"],
        rollout_provenance=[_context(validation_rows[0])],
        epoch=0,
        global_step=1,
    )
    assert metrics["rods_boundary/seed_selection_configured"] == 0.0
    assert metrics["rods_boundary/seed_emitted_count"] == 0.0
    assert not (tmp_path / "seeds.jsonl").exists()


def test_optional_stale_hook_is_disabled_by_default_and_configurable(tmp_path):
    lifecycle = RODSStage3Lifecycle(
        LifecycleConfig(
            enabled=True,
            seed_output_path=str(tmp_path / "seeds.jsonl"),
            candidate_input_path=str(tmp_path / "candidates.jsonl"),
            state_path=str(tmp_path / "state.json"),
            injection_ratio=0.0,
            stale_after_steps=5,
        )
    )
    lifecycle._state["active_candidates"] = {
        "stale": _generated_state_item("generated-stale")
    }
    dataset = SimpleNamespace(dataframe=Dataset.from_list([_minimal_sample("original")]))
    metrics = lifecycle.on_epoch_boundary(dataset, epoch=1, global_step=5)
    assert metrics["rods_stage3_lifecycle/retired_stale_count"] == 1.0
    assert lifecycle.active_generated_count == 0
