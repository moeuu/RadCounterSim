import json
from pathlib import Path

import numpy as np
import pytest

from radcounter.core.environment import (
    CoordinateSystemConfig,
    EnvironmentFormat,
    EnvironmentImportConfig,
    EnvironmentImportError,
    EnvironmentImportPipeline,
    LengthUnit,
    MaterialRuleConfig,
    load_environment_descriptor,
    load_environment_manifest,
)


def test_obj_is_normalized_to_metres_and_cached(tmp_path: Path) -> None:
    source = tmp_path / "millimetre-room.obj"
    source.write_text(
        "v 0 0 0\nv 1000 0 0\nv 0 1000 0\nf 1 2 3\n",
        encoding="ascii",
    )
    config = EnvironmentImportConfig(
        environment_id="room",
        uri=str(source),
        format=EnvironmentFormat.OBJ,
        coordinate_system=CoordinateSystemConfig(units=LengthUnit.MM),
        material_rules=(MaterialRuleConfig(pattern="*", material_id="steel"),),
        cache_directory=str(tmp_path / "cache"),
    )
    result = EnvironmentImportPipeline().import_environment(config)
    assert result.scene.triangle_count == 1
    assert result.scene.meshes[0].material_id == "steel"
    assert result.scene.bounds_m[1][0] == pytest.approx(1.0)
    assert result.manifest_path.is_file()
    loaded = load_environment_manifest(result.manifest_path)
    assert np.array_equal(
        loaded.scene.meshes[0].triangles,
        result.scene.meshes[0].triangles,
    )


def test_urdf_primitives_become_collision_and_radiation_meshes(tmp_path: Path) -> None:
    source = tmp_path / "room.urdf"
    source.write_text(
        """<robot name="room">
  <link name="wall">
    <visual><geometry><box size="1 2 3"/></geometry></visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    result = EnvironmentImportPipeline().import_environment(
        EnvironmentImportConfig(
            uri=str(source),
            format=EnvironmentFormat.URDF,
            cache_directory=str(tmp_path / "cache"),
        )
    )
    mesh = result.scene.meshes[0]
    assert len(mesh.triangles) == 12
    assert mesh.visual_enabled and mesh.collision_enabled and mesh.radiation_enabled
    assert result.scene.bounds_m == ((-0.5, -1.0, -1.5), (0.5, 1.0, 1.5))


def test_sdf_world_pose_is_composed(tmp_path: Path) -> None:
    source = tmp_path / "room.world"
    source.write_text(
        """<sdf version="1.9"><world name="default"><model name="wall">
  <pose>2 3 0 0 0 0</pose><link name="link"><collision name="shape">
    <geometry><box><size>1 1 1</size></box></geometry>
  </collision></link>
</model></world></sdf>
""",
        encoding="utf-8",
    )
    result = EnvironmentImportPipeline().import_environment(
        EnvironmentImportConfig(
            uri=str(source),
            cache_directory=str(tmp_path / "cache"),
        )
    )
    lower, upper = result.scene.bounds_m
    assert lower == pytest.approx((1.5, 2.5, -0.5))
    assert upper == pytest.approx((2.5, 3.5, 0.5))


def test_xyz_point_cloud_creates_exterior_voxel_faces(tmp_path: Path) -> None:
    source = tmp_path / "map.xyz"
    source.write_text("0 0 0\n0.01 0.01 0.01\n", encoding="ascii")
    result = EnvironmentImportPipeline().import_environment(
        EnvironmentImportConfig(
            uri=str(source),
            point_cloud_voxel_size_m=0.1,
            cache_directory=str(tmp_path / "cache"),
        )
    )
    assert result.scene.triangle_count == 12
    assert result.scene.vertex_count == 24


def test_descriptor_inside_scenario_and_digest_guard(tmp_path: Path) -> None:
    source = tmp_path / "map.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")
    descriptor = tmp_path / "scenario.yaml"
    descriptor.write_text(
        "schema_version: '1.0'\nenvironment:\n"
        f"  uri: {source.name}\n  expected_sha256: {'0' * 64}\n",
        encoding="utf-8",
    )
    config = load_environment_descriptor(descriptor)
    with pytest.raises(EnvironmentImportError, match="SHA256 mismatch"):
        EnvironmentImportPipeline().import_environment(
            config,
            base_directory=descriptor.parent,
        )
    assert json.loads(config.model_dump_json())["uri"] == source.name
