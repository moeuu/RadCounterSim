#!/usr/bin/env python3
"""Run the ROS 2 Jazzy benchmark in the pinned official ROS image."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROS_IMAGE = (
    "ros:jazzy-ros-base@"
    "sha256:31daab66eef9139933379fb67159449944f4e2dcf2e22c2d12cc715f29873e0f"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/ros2-bandwidth"))
    parser.add_argument("--minimum-delivery-ratio", type=float, default=0.95)
    parser.add_argument("--maximum-p99-latency-ms", type=float, default=250.0)
    parser.add_argument(
        "--reliability",
        choices=("reliable", "best_effort"),
        default="reliable",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    benchmark = root / "validation/ros2/benchmark.py"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-e",
        "ROS_DOMAIN_ID=187",
        "-v",
        f"{benchmark}:/benchmark.py:ro",
        "-v",
        f"{output}:/output",
        ROS_IMAGE,
        "bash",
        "-lc",
        (
            "source /opt/ros/jazzy/setup.bash && "
            "python3 /benchmark.py coordinator "
            f"--duration-s {args.duration_s} "
            f"--minimum-delivery-ratio {args.minimum_delivery_ratio} "
            f"--maximum-p99-latency-ms {args.maximum_p99_latency_ms} "
            f"--reliability {args.reliability}"
        ),
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
