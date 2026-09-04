"""Portable environment-import contracts shared by every simulator adapter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EnvironmentFormat(StrEnum):
    AUTO = "auto"
    USD = "usd"
    GLTF = "gltf"
    GLB = "glb"
    OBJ = "obj"
    STL = "stl"
    PLY = "ply"
    DAE = "dae"
    THREE_MF = "3mf"
    OFF = "off"
    FBX = "fbx"
    SDF = "sdf"
    URDF = "urdf"
    XACRO = "xacro"
    STEP = "step"
    IGES = "iges"
    BREP = "brep"
    PCD = "pcd"
    XYZ = "xyz"
    IFC = "ifc"
    E57 = "e57"
    LAS = "las"
    LAZ = "laz"
    OCTOMAP = "octomap"


class LengthUnit(StrEnum):
    AUTO = "auto"
    M = "m"
    CM = "cm"
    MM = "mm"
    IN = "in"
    FT = "ft"

    @property
    def metres(self) -> float:
        return {
            LengthUnit.M: 1.0,
            LengthUnit.CM: 0.01,
            LengthUnit.MM: 0.001,
            LengthUnit.IN: 0.0254,
            LengthUnit.FT: 0.3048,
        }.get(self, 1.0)


class AxisDirection(StrEnum):
    AUTO = "auto"
    POS_X = "+X"
    NEG_X = "-X"
    POS_Y = "+Y"
    NEG_Y = "-Y"
    POS_Z = "+Z"
    NEG_Z = "-Z"


class Handedness(StrEnum):
    AUTO = "auto"
    RIGHT = "right"
    LEFT = "left"


class CollisionGeometrySource(StrEnum):
    AUTO = "auto"
    VISUAL = "visual"
    COLLISION = "collision"
    BOTH = "both"


class CollisionApproximation(StrEnum):
    TRIANGLE_MESH = "triangle_mesh"
    CONVEX_HULL = "convex_hull"
    NONE = "none"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CoordinateSystemConfig(_FrozenModel):
    units: LengthUnit = LengthUnit.AUTO
    up_axis: AxisDirection = AxisDirection.AUTO
    forward_axis: AxisDirection = AxisDirection.AUTO
    handedness: Handedness = Handedness.AUTO

    @model_validator(mode="after")
    def validate_axes(self) -> CoordinateSystemConfig:
        if self.up_axis == AxisDirection.AUTO or self.forward_axis == AxisDirection.AUTO:
            return self
        if self.up_axis.value[-1] == self.forward_axis.value[-1]:
            raise ValueError("up_axis and forward_axis must refer to different axes")
        return self


class EnvironmentCollisionConfig(_FrozenModel):
    enabled: bool = True
    geometry_source: CollisionGeometrySource = CollisionGeometrySource.AUTO
    approximation: CollisionApproximation = CollisionApproximation.TRIANGLE_MESH
    show_collision_geometry: bool = False


class MaterialRuleConfig(_FrozenModel):
    pattern: str = Field(min_length=1)
    material_id: str = Field(min_length=1)


class ExternalConverterConfig(_FrozenModel):
    """A no-shell converter command with ``{input}`` and ``{output}`` placeholders."""

    command: tuple[str, ...] = Field(min_length=1)
    output_format: EnvironmentFormat
    timeout_s: float = Field(default=600.0, gt=0.0)

    @model_validator(mode="after")
    def validate_command(self) -> ExternalConverterConfig:
        joined = "\0".join(self.command)
        if "{input}" not in joined or "{output}" not in joined:
            raise ValueError("external converter command requires {input} and {output}")
        if self.output_format in {EnvironmentFormat.AUTO, EnvironmentFormat.USD}:
            raise ValueError("external converter output must be a portable mesh format")
        return self


class EnvironmentStreamingConfig(_FrozenModel):
    """Runtime working-set limits for environments larger than memory."""

    enabled: bool = False
    tile_size_m: float = Field(default=25.0, gt=0.0)
    physics_radius_m: float = Field(default=40.0, gt=0.0)
    visual_radius_m: float = Field(default=250.0, gt=0.0)
    lod_distances_m: tuple[float, ...] = (40.0, 100.0, 200.0)
    lod_vertex_cluster_m: tuple[float, ...] = (0.0, 0.05, 0.25, 1.0)
    unload_hysteresis_m: float = Field(default=20.0, ge=0.0)
    max_loaded_tiles: int = Field(default=160, gt=0)
    max_loaded_triangles: int = Field(default=8_000_000, gt=0)
    max_tile_triangles: int = Field(default=250_000, gt=0)
    max_changes_per_update: int = Field(default=4, gt=0)
    update_interval_frames: int = Field(default=10, gt=0)
    initial_focus_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radiation_segment_padding_m: float = Field(default=0.25, ge=0.0)

    @model_validator(mode="after")
    def validate_lods(self) -> EnvironmentStreamingConfig:
        if self.visual_radius_m < self.physics_radius_m:
            raise ValueError("visual_radius_m must be at least physics_radius_m")
        if tuple(sorted(self.lod_distances_m)) != self.lod_distances_m:
            raise ValueError("lod_distances_m must be sorted")
        if len(self.lod_vertex_cluster_m) != len(self.lod_distances_m) + 1:
            raise ValueError("lod_vertex_cluster_m needs one more entry than lod_distances_m")
        if not self.lod_vertex_cluster_m or self.lod_vertex_cluster_m[0] != 0.0:
            raise ValueError("LOD0 vertex cluster size must be 0")
        return self


class EnvironmentImportConfig(_FrozenModel):
    """One external environment and all deterministic import decisions."""

    environment_id: str = Field(default="environment", min_length=1)
    uri: str = Field(min_length=1)
    format: EnvironmentFormat = EnvironmentFormat.AUTO
    expected_sha256: str | None = None
    coordinate_system: CoordinateSystemConfig = CoordinateSystemConfig()
    scale: float = Field(default=1.0, gt=0.0)
    translation_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_world_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_material_id: str = Field(default="concrete", min_length=1)
    material_rules: tuple[MaterialRuleConfig, ...] = ()
    collision: EnvironmentCollisionConfig = EnvironmentCollisionConfig()
    point_cloud_voxel_size_m: float = Field(default=0.10, gt=0.0)
    max_point_cloud_voxels: int = Field(default=500_000, gt=0)
    max_triangles: int = Field(default=5_000_000, gt=0)
    streaming: EnvironmentStreamingConfig = EnvironmentStreamingConfig()
    cad_mesh_size_m: float = Field(default=0.05, gt=0.0)
    model_search_paths: tuple[str, ...] = ()
    package_search_paths: tuple[str, ...] = ()
    cache_directory: str = "~/.cache/radinteract/environments"
    external_converter: ExternalConverterConfig | None = None

    @field_validator("expected_sha256")
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
        return value.lower() if value is not None else None


class EnvironmentDescriptorConfig(_FrozenModel):
    schema_version: Literal["1.0"]
    environment: EnvironmentImportConfig


@dataclass(frozen=True)
class CoordinateDefaults:
    unit_scale_m: float = 1.0
    up_axis: AxisDirection = AxisDirection.POS_Z
    forward_axis: AxisDirection = AxisDirection.POS_X
    handedness: Handedness = Handedness.RIGHT


@dataclass(frozen=True)
class RawEnvironmentMesh:
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    source_material: str | None = None
    visual_enabled: bool = True
    collision_enabled: bool = True
    radiation_enabled: bool = True
    display_color_rgb: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        triangles = np.asarray(self.triangles, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("environment vertices must be finite Nx3 values")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("environment triangles must be Mx3 values")
        if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
            raise ValueError("environment triangle index lies outside the vertex array")
        vertices = np.ascontiguousarray(vertices)
        triangles = np.ascontiguousarray(triangles, dtype=np.uint32)
        vertices.setflags(write=False)
        triangles.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)


@dataclass(frozen=True)
class RawEnvironmentScene:
    meshes: tuple[RawEnvironmentMesh, ...]
    coordinate_defaults: CoordinateDefaults = CoordinateDefaults()
    dependencies: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentMesh:
    mesh_id: str
    name: str
    vertices_m: np.ndarray
    triangles: np.ndarray
    material_id: str
    visual_enabled: bool = True
    collision_enabled: bool = True
    radiation_enabled: bool = True
    display_color_rgb: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        raw = RawEnvironmentMesh(self.name, self.vertices_m, self.triangles)
        object.__setattr__(self, "vertices_m", raw.vertices)
        object.__setattr__(self, "triangles", raw.triangles)
        if not self.mesh_id or not self.material_id:
            raise ValueError("environment mesh ID and material ID are required")

    @property
    def bounds_m(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if len(self.vertices_m) == 0:
            zero = (0.0, 0.0, 0.0)
            return zero, zero
        return (
            tuple(float(value) for value in self.vertices_m.min(axis=0)),
            tuple(float(value) for value in self.vertices_m.max(axis=0)),
        )


@dataclass(frozen=True)
class EnvironmentScene:
    environment_id: str
    source_uri: str
    source_sha256: str
    source_format: EnvironmentFormat
    meshes: tuple[EnvironmentMesh, ...] = ()
    dependencies: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    native_stage_uri: str | None = None
    import_config: EnvironmentImportConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.environment_id or len(self.source_sha256) != 64:
            raise ValueError("environment ID and source digest are required")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def triangle_count(self) -> int:
        return sum(len(mesh.triangles) for mesh in self.meshes)

    @property
    def vertex_count(self) -> int:
        return sum(len(mesh.vertices_m) for mesh in self.meshes)

    @property
    def bounds_m(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        populated = [mesh.vertices_m for mesh in self.meshes if len(mesh.vertices_m)]
        if not populated:
            zero = (0.0, 0.0, 0.0)
            return zero, zero
        lower = np.min(np.vstack([vertices.min(axis=0) for vertices in populated]), axis=0)
        upper = np.max(np.vstack([vertices.max(axis=0) for vertices in populated]), axis=0)
        return (
            tuple(float(value) for value in lower),
            tuple(float(value) for value in upper),
        )


@dataclass(frozen=True)
class EnvironmentImportResult:
    scene: EnvironmentScene
    cache_key: str
    output_directory: Path
    manifest_path: Path
    normalized_mesh_path: Path | None


def load_environment_descriptor(path: str | Path) -> EnvironmentImportConfig:
    """Load either an environment descriptor or a scenario containing ``environment``."""

    descriptor_path = Path(path).expanduser().resolve()
    text = descriptor_path.read_text(encoding="utf-8")
    raw = json.loads(text) if descriptor_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("environment descriptor root must be a mapping")
    if "environment" in raw:
        environment = raw["environment"]
        if isinstance(environment, dict) and environment.get("expected_sha256") == 0:
            environment = dict(environment)
            environment["expected_sha256"] = "0" * 64
        return EnvironmentImportConfig.model_validate(environment)
    return EnvironmentImportConfig.model_validate(raw)
