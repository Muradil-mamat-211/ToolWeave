from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = ROOT / "environment/gemma-synthesis"


def test_gemma_synthesis_lock_contract_is_self_consistent() -> None:
    result = subprocess.run(
        [sys.executable, str(ENVIRONMENT / "verify.py"), "--lock-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS (lock contract)" in result.stdout


def test_gemma_synthesis_contract_contains_no_machine_local_paths() -> None:
    public_files = (
        ENVIRONMENT / "manifest.json",
        ENVIRONMENT / "conda-linux-64.explicit",
        ENVIRONMENT / "requirements-pypi-linux-x86_64.lock",
        ENVIRONMENT / "requirements-pytorch-cu129-linux-x86_64.lock",
        ENVIRONMENT / "README.md",
        ENVIRONMENT / "create.sh",
        ENVIRONMENT / "verify.py",
    )
    for path in public_files:
        content = path.read_text(encoding="utf-8")
        assert "/root/" not in content
        assert "autodl-tmp" not in content
