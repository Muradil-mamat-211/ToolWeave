"""Safe CLI entrypoint; production generation is opt-in twice."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from .config import GeneratorConfig
from .daemon import GENERATION_GUARD_ENV, daemon_from_config


def load_config(path: str | Path) -> GeneratorConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
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
