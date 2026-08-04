#!/usr/bin/env python3
"""Require physical GPU identity before accepting a 600-second endurance result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radcounter.validation.hardware import evaluate_hardware_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vram-class", required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--expected-os")
    parser.add_argument("--driver-pattern")
    parser.add_argument("--required-duration-s", type=float, default=600.0)
    parser.add_argument("--minimum-gpu-count", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(".cache/hardware-gate.json"))
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = evaluate_hardware_gate(
        required_vram_class=args.vram_class,
        endurance_metrics=metrics,
        expected_os=args.expected_os,
        driver_pattern=args.driver_pattern,
        required_duration_s=args.required_duration_s,
        minimum_gpu_count=args.minimum_gpu_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.passed and result.qualified_evidence else 2


if __name__ == "__main__":
    raise SystemExit(main())
