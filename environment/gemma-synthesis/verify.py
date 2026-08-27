#!/usr/bin/env python3
"""Validate the audited ToolWeave Gemma synthesis environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pip_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError(f"{path.name}:{line_number}: expected one exact == pin")
        name, version = line.split("==", 1)
        pins[_canonical_name(name)] = version
    return pins


def _conda_records(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("https://")
    ]


def _validate_locks(manifest: dict) -> list[str]:
    errors: list[str] = []
    artifacts = manifest["lock_artifacts"]
    for filename, contract in artifacts.items():
        path = ROOT / filename
        if not path.is_file():
            errors.append(f"missing lock artifact: {filename}")
            continue
        actual_digest = _sha256(path)
        if actual_digest != contract["sha256"]:
            errors.append(
                f"{filename}: sha256 {actual_digest} != {contract['sha256']}"
            )

    pypi_path = ROOT / "requirements-pypi-linux-x86_64.lock"
    pytorch_path = ROOT / "requirements-pytorch-cu129-linux-x86_64.lock"
    conda_path = ROOT / "conda-linux-64.explicit"
    pypi_pins = _pip_pins(pypi_path) if pypi_path.is_file() else {}
    pytorch_pins = _pip_pins(pytorch_path) if pytorch_path.is_file() else {}
    expected_pytorch_names = {"torch", "torchaudio", "torchcodec", "torchvision"}
    if set(pytorch_pins) != expected_pytorch_names:
        errors.append(
            "PyTorch index lock names do not match contract: "
            f"{sorted(pytorch_pins)} != {sorted(expected_pytorch_names)}"
        )
    overlap = set(pypi_pins).intersection(pytorch_pins)
    if overlap:
        errors.append(f"duplicate Pip lock entries: {sorted(overlap)}")
    pins = {**pypi_pins, **pytorch_pins}
    conda_records = _conda_records(conda_path) if conda_path.is_file() else []
    expected_pip_records = sum(
        artifacts[filename]["record_count"]
        for filename in (
            "requirements-pypi-linux-x86_64.lock",
            "requirements-pytorch-cu129-linux-x86_64.lock",
        )
    )
    if len(pins) != expected_pip_records:
        errors.append(
            f"Pip lock record count does not match manifest: "
            f"{len(pins)} != {expected_pip_records}"
        )
    expected_conda_records = artifacts["conda-linux-64.explicit"]["record_count"]
    if len(conda_records) != expected_conda_records:
        errors.append(
            f"Conda lock record count does not match manifest: "
            f"{len(conda_records)} != {expected_conda_records}"
        )
    if not any(
        f"/python-{manifest['python_version']}-" in record for record in conda_records
    ):
        errors.append("Conda lock does not contain the manifest Python version")
    for name, expected in manifest["expected_distributions"].items():
        actual = pins.get(_canonical_name(name))
        if actual != expected:
            errors.append(f"Pip lock {name}: {actual!r} != {expected!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock-only",
        action="store_true",
        help="validate committed lock integrity without inspecting this interpreter",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="also require at least one CUDA device to be visible",
    )
    args = parser.parse_args()
    if args.lock_only and args.require_gpu:
        parser.error("--lock-only and --require-gpu cannot be combined")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    errors = _validate_locks(manifest)
    if not args.lock_only:
        if platform.python_version() != manifest["python_version"]:
            errors.append(
                f"Python {platform.python_version()} != {manifest['python_version']}"
            )
        expected_architecture = manifest["platform"]["architecture"]
        if sys.platform != "linux" or platform.machine() != expected_architecture:
            errors.append(
                f"platform {sys.platform}/{platform.machine()} != "
                f"linux/{expected_architecture}"
            )
        for name, expected in manifest["expected_distributions"].items():
            try:
                actual = metadata.version(name)
            except metadata.PackageNotFoundError:
                errors.append(f"distribution not installed: {name}")
                continue
            if actual != expected:
                errors.append(f"{name} {actual} != {expected}")

        try:
            import torch
        except Exception as exc:  # pragma: no cover - rebuilt environment only
            errors.append(f"torch import failed: {type(exc).__name__}: {exc}")
        else:
            compiled_cuda = str(torch.version.cuda)
            expected_cuda = manifest["torch_compiled_cuda_version"]
            if compiled_cuda != expected_cuda:
                errors.append(
                    f"torch compiled CUDA {compiled_cuda} != {expected_cuda}"
                )
            if args.require_gpu and not torch.cuda.is_available():
                errors.append("CUDA GPU is required but torch.cuda.is_available() is false")
            elif args.require_gpu:
                capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
                print("GPU:", torch.cuda.get_device_name(0), "capability", capability)

    if errors:
        print("Gemma synthesis environment verification: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    mode = "lock contract" if args.lock_only else "installed environment"
    print(f"Gemma synthesis environment verification: PASS ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
