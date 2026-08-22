#!/usr/bin/env python3
"""Resume a large HTTP range download with independent retriable chunks."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import time
from pathlib import Path

import requests


def download_part(args: tuple[str, int, int, Path, int]) -> tuple[int, int]:
    url, start, end, part, retries = args
    expected = end - start + 1
    part.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        current = part.stat().st_size if part.exists() else 0
        if current > expected:
            raise RuntimeError(f"oversized part {part}: {current}>{expected}")
        if current == expected:
            return start, current
        req_start = start + current
        response = None
        temp = part.with_suffix(part.suffix + ".tmp")
        try:
            session = requests.Session()
            # Direct connections are the default. Set RESUME_USE_ENV_PROXY=1
            # only when the caller has measured a faster configured route.
            session.trust_env = os.environ.get("RESUME_USE_ENV_PROXY") == "1"
            response = session.get(
                url,
                headers={"Range": f"bytes={req_start}-{end}"},
                stream=True,
                timeout=(30, 300),
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
            content_range = response.headers.get("Content-Range", "")
            expected_prefix = f"bytes {req_start}-"
            if not content_range.startswith(expected_prefix):
                raise RuntimeError(f"unexpected Content-Range {content_range!r}")
            written = 0
            with temp.open("wb") as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        stream.write(block)
                        written += len(block)
            if written <= 0:
                raise RuntimeError("empty response body")
            with temp.open("rb") as source, part.open("ab") as target:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
            temp.unlink(missing_ok=True)
        except Exception:
            if temp.exists():
                temp.unlink()
            if attempt + 1 == retries:
                raise
            time.sleep(min(2 ** attempt, 30))
        finally:
            if response is not None:
                response.close()
            if "session" in locals():
                session.close()
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    args.parts_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, start in enumerate(range(0, args.size, args.chunk_size)):
        end = min(args.size - 1, start + args.chunk_size - 1)
        jobs.append((args.url, start, end, args.parts_dir / f"part_{index:03d}", 12))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_part, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            start, size = future.result()
            print(f"completed range starting {start}: {size} bytes", flush=True)

    with args.output.open("wb") as target:
        for _, _, _, part, _ in jobs:
            expected = min(args.chunk_size, args.size - int(part.stem.split("_")[-1]) * args.chunk_size)
            if part.stat().st_size != expected:
                raise RuntimeError(f"bad part size {part}: {part.stat().st_size}!={expected}")
            with part.open("rb") as source:
                while block := source.read(1024 * 1024):
                    target.write(block)
    if args.output.stat().st_size != args.size:
        raise RuntimeError(f"bad output size: {args.output.stat().st_size}!={args.size}")
    hasher = hashlib.sha256()
    with args.output.open("rb") as source:
        while block := source.read(1024 * 1024):
            hasher.update(block)
    digest = hasher.hexdigest()
    print(f"sha256={digest}", flush=True)
    if digest != args.sha256:
        raise RuntimeError(f"sha256 mismatch: {digest} != {args.sha256}")


if __name__ == "__main__":
    main()
