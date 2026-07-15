#!/usr/bin/env python3
"""Check deterministic experiment metrics against versioned acceptance bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radcounter.core.experiments.cases import BUILTIN_CASES


def _satisfies(value: Any, constraint: dict[str, Any]) -> bool:
    if "equals" in constraint and value != constraint["equals"]:
        return False
    if "min" in constraint and float(value) < float(constraint["min"]):
        return False
    return not ("max" in constraint and float(value) > float(constraint["max"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/regression_baseline.json"),
    )
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    observed: dict[str, dict[str, Any]] = {}
    for case_id, constraints in baseline.items():
        result = BUILTIN_CASES[case_id]().run(
            seed=args.seed,
            planner_id="closed_loop_residual",
        )
        observed[case_id] = dict(result.metrics)
        for metric, constraint in constraints.items():
            value = result.metrics.get(metric)
            if value is None or not _satisfies(value, constraint):
                failures.append(
                    {
                        "case": case_id,
                        "metric": metric,
                        "observed": value,
                        "constraint": constraint,
                    }
                )
    print(
        json.dumps(
            {"passed": not failures, "failures": failures, "observed": observed},
            indent=2,
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
