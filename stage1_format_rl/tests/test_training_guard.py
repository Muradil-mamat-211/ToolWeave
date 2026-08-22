from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.parametrize(
    "script_name",
    [
        "run_stage1_qwen3_4b_repo_aligned.sh",
        "run_stage1_qwen3_4b_paper_aligned.sh",
    ],
)
def test_training_script_exits_before_any_training(stage_root, script_name):
    env = os.environ.copy()
    env.pop("ALLOW_STAGE1_TRAINING", None)
    result = subprocess.run(
        [str(stage_root / "scripts" / script_name)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Stage 1 training is disabled" in result.stdout


def test_stage3_training_script_requires_explicit_formal_guard(stage_root):
    env = os.environ.copy()
    env.pop("ALLOW_RODS_MATCHTIR_STAGE3_TRAINING", None)
    result = subprocess.run(
        [str(stage_root / "scripts" / "run_stage3_rods_matchtir_v1.sh"), "--full"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Stage 3 formal training is disabled" in result.stdout
