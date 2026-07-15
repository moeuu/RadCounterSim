#!/usr/bin/env python3
"""Run portable core-kernel and experiment timing benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from radcounter.core.experiments.cases import BUILTIN_CASES


def _radiation_kernel_benchmark(seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    detector_positions = rng.uniform(-10.0, 10.0, size=(20_000, 3))
    source_positions = rng.uniform(-3.0, 3.0, size=(128, 3))
    strengths = rng.uniform(1.0e3, 1.0e6, size=128)
    started = time.perf_counter()
    squared_distance = np.sum(
        (detector_positions[:, None, :] - source_positions[None, :, :]) ** 2,
        axis=2,
    )
    dose = np.sum(strengths[None, :] / np.maximum(squared_distance, 0.01), axis=1)
    elapsed_s = time.perf_counter() - started
    return {
        "detector_count": len(detector_positions),
        "source_count": len(source_positions),
        "elapsed_s": elapsed_s,
        "detector_source_pairs_per_s": len(detector_positions) * len(source_positions) / elapsed_s,
        "checksum": float(np.sum(dose)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    case_timings: dict[str, dict[str, float]] = {}
    for case_id, case_type in BUILTIN_CASES.items():
        case = case_type()
        samples = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            case.run(seed=args.seed, planner_id="closed_loop_residual")
            samples.append(time.perf_counter() - started)
        case_timings[case_id] = {
            "minimum_s": min(samples),
            "median_s": float(np.median(samples)),
            "maximum_s": max(samples),
        }
    payload = {
        "seed": args.seed,
        "repeat": args.repeat,
        "radiation_kernel": _radiation_kernel_benchmark(args.seed),
        "experiment_cases": case_timings,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
