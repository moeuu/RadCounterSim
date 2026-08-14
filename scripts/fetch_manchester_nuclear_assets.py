#!/usr/bin/env python3
"""Fetch the CC BY 4.0 Manchester nuclear simulation assets from Figshare.

Run with: uv run python scripts/fetch_manchester_nuclear_assets.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ARTICLE_API = "https://api.figshare.com/v2/articles/25224974"
DEFAULT_FILE = "500L_Drum_Store.zip"


def _download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "RadCounterSim/1.0"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    partial.replace(destination)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/datasets/manchester-nuclear-assets"),
    )
    args = parser.parse_args()

    with urllib.request.urlopen(ARTICLE_API) as response:
        metadata = json.load(response)
    selected = next((item for item in metadata["files"] if item["name"] == args.file), None)
    if selected is None:
        available = ", ".join(item["name"] for item in metadata["files"])
        raise SystemExit(f"{args.file!r} is unavailable; article contains: {available}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / selected["name"]
    expected_md5 = selected["computed_md5"]
    if not archive.is_file() or _md5(archive) != expected_md5:
        _download(selected["download_url"], archive)
    actual_md5 = _md5(archive)
    if actual_md5 != expected_md5:
        raise SystemExit(f"checksum mismatch: expected {expected_md5}, got {actual_md5}")

    destination = output_dir / archive.stem
    marker = destination / ".figshare.json"
    if not marker.is_file():
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as package:
            package.extractall(destination)
        marker.write_text(
            json.dumps(
                {
                    "article_id": metadata["id"],
                    "doi": metadata["doi"],
                    "license": metadata["license"]["name"],
                    "file": selected["name"],
                    "md5": actual_md5,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(destination)


if __name__ == "__main__":
    main()
