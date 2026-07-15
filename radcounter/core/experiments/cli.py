"""uv-managed experiment and report entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from radcounter.core.experiments.cases import BUILTIN_CASES
from radcounter.core.experiments.report import render_run_report
from radcounter.core.experiments.runner import BatchRunner


def experiments_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RadCounterSim experiment cases")
    parser.add_argument("--case", choices=sorted(BUILTIN_CASES), action="append")
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--planner", action="append", default=[])
    parser.add_argument("--run-root", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    case_ids = args.case or list(BUILTIN_CASES)
    cases = [BUILTIN_CASES[case_id]() for case_id in case_ids]
    outputs = BatchRunner(Path.cwd(), args.run_root).run(
        cases,
        seeds=args.seed or [42],
        planner_ids=args.planner or ["closed_loop_residual"],
    )
    for output in outputs:
        print(output.artifacts.root)
    return 0


def report_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-render a RadCounterSim HTML report")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args(argv)
    output = render_run_report(args.run_directory, args.run_directory / "report.html")
    print(output)
    return 0
