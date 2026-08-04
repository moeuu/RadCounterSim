"""Preprocess a normalized environment manifest into an out-of-core cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import EnvironmentStreamingConfig
from .pipeline import load_environment_manifest
from .streaming import LargeEnvironmentBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile-size", type=float, default=25.0)
    parser.add_argument("--physics-radius", type=float, default=40.0)
    parser.add_argument("--visual-radius", type=float, default=250.0)
    args = parser.parse_args()

    loaded = load_environment_manifest(args.manifest)
    scene = getattr(loaded, "scene", loaded)
    config = EnvironmentStreamingConfig(
        enabled=True,
        tile_size_m=args.tile_size,
        physics_radius_m=args.physics_radius,
        visual_radius_m=args.visual_radius,
    )
    index = LargeEnvironmentBuilder().build(scene, config, args.output)
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
