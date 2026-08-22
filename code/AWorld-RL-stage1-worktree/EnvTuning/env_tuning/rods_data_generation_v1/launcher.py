"""Safe CLI entrypoint; production generation is opt-in twice."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re

import yaml

from .config import ARTIFACTS_ROOT, ASSET_ROOT, DATA_ROOT, MODELS_ROOT, WORKSPACE, GeneratorConfig
from .daemon import GENERATION_GUARD_ENV, daemon_from_config


def load_config(path: str | Path) -> GeneratorConfig:
    variables = {
        "TOOLWEAVE_SOURCE_ROOT": str(WORKSPACE),
        "TOOLWEAVE_ASSET_ROOT": str(ASSET_ROOT),
        "TOOLWEAVE_MODELS_ROOT": str(MODELS_ROOT),
        "TOOLWEAVE_DATA_ROOT": str(DATA_ROOT),
        "TOOLWEAVE_ARTIFACTS_ROOT": str(ARTIFACTS_ROOT),
    }
    variables.update(os.environ)
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            raise ValueError(f"Unresolved machine-local variable in Generator config: {key}")
        return str(variables[key])

    text = pattern.sub(replace, Path(path).read_text(encoding="utf-8"))
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError("Generator YAML root must be an object")
    return GeneratorConfig.from_mapping(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--allow-generation",
        action="store_true",
        help=f"also requires {GENERATION_GUARD_ENV}=1 when dry_run=false",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    daemon = daemon_from_config(config)
    if args.once or config.dry_run:
        metrics = asyncio.run(
            daemon.run_once(allow_generation=args.allow_generation)
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        asyncio.run(daemon.run_forever(allow_generation=args.allow_generation))


if __name__ == "__main__":
    main()
