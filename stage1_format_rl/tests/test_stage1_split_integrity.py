from __future__ import annotations

import json


def ids(rows):
    return {str(row["extra_info"]["original_id"]) for row in rows}


def test_split_ids_are_unique_and_disjoint(train_rows, validation_rows):
    train = ids(train_rows)
    validation = ids(validation_rows)
    assert len(train) == 100
    assert len(validation) == 400
    assert not train & validation


def test_manifest_matches_parquets(stage_root, train_rows, validation_rows):
    manifest = json.loads(
        (stage_root / "data" / "stage1_split_manifest.json").read_text()
    )
    assert manifest["exact_public_envtuning_split"] is True
    assert manifest["exact_rods_experiment_id_manifest_independently_published"] is False
    assert set(manifest["train_ids"]) == ids(train_rows)
    manifest_val = set().union(*map(set, manifest["validation_ids"].values()))
    assert manifest_val == ids(validation_rows)


def test_public_envtuning_split_partitions_all_raw_ids(stage_root):
    manifest = json.loads(
        (stage_root / "data" / "stage1_split_manifest.json").read_text()
    )
    for data_type, train_ids in manifest[
        "official_train_seed_ids_all_four_types"
    ].items():
        held_ids = manifest["validation_ids"][data_type]
        assert len(train_ids) == 100
        assert len(held_ids) == 100
        assert not set(train_ids) & set(held_ids)

