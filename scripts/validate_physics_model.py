#!/usr/bin/env python3
"""Evaluate source and shield physics against a structured dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radcounter.validation.physics import (
    load_validation_dataset,
    validate_physics_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--require-measured", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(".cache/physics-validation.json"))
    args = parser.parse_args()
    observations, metadata = load_validation_dataset(args.observations, args.metadata)
    result = validate_physics_dataset(
        observations,
        metadata,
        require_measured=args.require_measured,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
