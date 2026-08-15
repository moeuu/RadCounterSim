#!/usr/bin/env python3
"""Download and verify the product's pinned Qwen3 GGUF model."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen3-4B-Q4_K_M.gguf"
MODEL_REVISION = "bc640142c66e1fdd12af0bd68f40445458f3869b"
MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/"
    f"{MODEL_REVISION}/{MODEL_NAME}?download=true"
)
MODEL_SHA256 = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
LICENSE_URL = (
    "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/"
    f"{MODEL_REVISION}/LICENSE"
)
LICENSE_SHA256 = "5de36594c10839788a8c589443a8ef9d8b8d17c65a1b5807206ae037fc36c6bd"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT / "runtime/llm/models" / MODEL_NAME,
    )
    return parser.parse_args()


def _install_license(runtime_root: Path) -> None:
    destination = runtime_root / "licenses/Qwen3-Apache-2.0.txt"
    if destination.is_file() and _sha256(destination) == LICENSE_SHA256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(LICENSE_URL, timeout=30) as response:  # noqa: S310
        payload = response.read()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != LICENSE_SHA256:
        raise RuntimeError(
            f"Qwen license SHA-256 mismatch: expected {LICENSE_SHA256}, got {actual}"
        )
    destination.write_bytes(payload)


def main() -> int:
    args = _arguments()
    destination = args.destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual = _sha256(destination)
        if actual == MODEL_SHA256:
            _install_license(destination.parents[1])
            print(f"Model is already verified: {destination}")
            return 0
        raise RuntimeError(
            f"refusing to replace an existing model with SHA-256 {actual}: {destination}"
        )

    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.is_file() else 0
    request = Request(MODEL_URL, headers={"User-Agent": "RadCounterSim/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    response = urlopen(request, timeout=120)  # noqa: S310
    if offset and response.status != 206:
        offset = 0
    mode = "ab" if offset else "wb"
    with response, temporary.open(mode) as output:
        total_header = response.headers.get("Content-Range") or response.headers.get(
            "Content-Length"
        )
        print(f"Downloading {MODEL_NAME} ({total_header or 'size unknown'})")
        received = offset
        next_report = received + 128 * 1024 * 1024
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            received += len(chunk)
            if received >= next_report:
                print(f"  {received / (1024**3):.2f} GiB", flush=True)
                next_report += 128 * 1024 * 1024

    actual = _sha256(temporary)
    if actual != MODEL_SHA256:
        temporary.unlink()
        raise RuntimeError(f"model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual}")
    temporary.replace(destination)
    _install_license(destination.parents[1])
    print(f"Installed verified model: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Download interrupted; the partial file can be resumed.", file=sys.stderr)
        raise SystemExit(130) from None
