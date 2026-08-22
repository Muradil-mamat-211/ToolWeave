#!/usr/bin/env python3
"""Copy an explicit sample-ID subset into a new durable JSONL queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from env_tuning.rods_data_generation_v1.queue import LockedJsonlQueue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    args = parser.parse_args()
    requested = set(args.sample_id)
    records = [
        json.loads(line)
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [record for record in records if record.get("sample_id") in requested]
    found = {record["sample_id"] for record in selected}
    if found != requested:
        raise ValueError(f"missing requested seed IDs: {sorted(requested - found)}")
    queue = LockedJsonlQueue(args.output, key_field="sample_id")
    accepted, duplicates = queue.append(selected)
    print(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "output": str(args.output.resolve()),
                "requested": len(requested),
                "accepted": accepted,
                "duplicates": duplicates,
                "sample_ids": sorted(found),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
