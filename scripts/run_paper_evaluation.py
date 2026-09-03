#!/usr/bin/env python3
"""Collect strict multi-seed physical-run evidence for paper tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from radcounter.core.experiments.paper import (
    EvaluationDataMode,
    collect_physical_paper_evaluation,
    validate_paper_physics_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument(
        "--stage", type=Path, default=ROOT / "assets/environments/radcounter_vertical_slice.usda"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/scenarios/vertical_slice.runtime.json",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/paper-evaluation")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--data-mode",
        choices=tuple(EvaluationDataMode),
        type=EvaluationDataMode,
        default=EvaluationDataMode.SYNTHETIC_VALIDATION,
    )
    parser.add_argument("--material", type=Path, action="append", default=[])
    parser.add_argument("--isotope", type=Path, action="append", default=[])
    parser.add_argument("--detector", type=Path, action="append", default=[])
    parser.add_argument("--buildup", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    physics = validate_paper_physics_evidence(
        arguments.config,
        mode=arguments.data_mode,
        material_paths=arguments.material,
        isotope_paths=arguments.isotope,
        detector_paths=arguments.detector,
        buildup_path=arguments.buildup,
    )
    summary = collect_physical_paper_evaluation(
        output_root=arguments.output_root,
        run_id=arguments.run_id,
        stage_path=arguments.stage,
        config_path=arguments.config,
        artifact_paths=arguments.artifact,
        physics=physics,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
