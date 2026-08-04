"""Command-line entry point for simulator-independent environment conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radcounter.core.environment.models import (
    AxisDirection,
    CoordinateSystemConfig,
    EnvironmentFormat,
    EnvironmentImportConfig,
    LengthUnit,
    load_environment_descriptor,
)
from radcounter.core.environment.pipeline import EnvironmentImportPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a 3D environment for RadCounterSim, PhysX, and Embree"
    )
    parser.add_argument("source", type=Path, help="3D file or environment descriptor YAML/JSON")
    parser.add_argument(
        "--format", choices=[item.value for item in EnvironmentFormat], default="auto"
    )
    parser.add_argument("--units", choices=[item.value for item in LengthUnit], default="auto")
    parser.add_argument("--up-axis", choices=[item.value for item in AxisDirection], default="auto")
    parser.add_argument(
        "--forward-axis", choices=[item.value for item in AxisDirection], default="auto"
    )
    parser.add_argument("--material", default="concrete")
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    if source.suffix.lower() in {".yaml", ".yml", ".json"}:
        config = load_environment_descriptor(source)
        base_directory = source.parent
    else:
        config = EnvironmentImportConfig(
            environment_id=source.stem,
            uri=str(source),
            format=EnvironmentFormat(args.format),
            coordinate_system=CoordinateSystemConfig(
                units=LengthUnit(args.units),
                up_axis=AxisDirection(args.up_axis),
                forward_axis=AxisDirection(args.forward_axis),
            ),
            default_material_id=args.material,
        )
        base_directory = source.parent
    result = EnvironmentImportPipeline().import_environment(config, base_directory=base_directory)
    print(
        json.dumps(
            {
                "environment_id": result.scene.environment_id,
                "source_format": result.scene.source_format.value,
                "vertices": result.scene.vertex_count,
                "triangles": result.scene.triangle_count,
                "bounds_m": result.scene.bounds_m,
                "manifest": str(result.manifest_path),
                "normalized_meshes": (
                    str(result.normalized_mesh_path) if result.normalized_mesh_path else None
                ),
                "warnings": result.scene.warnings,
            },
            sort_keys=True,
        )
    )
    return 0
