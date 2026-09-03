"""Typed, Isaac-independent representation of RadCounterSim USD metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from radcounter.core.scene.activity_map import SurfaceActivityMap


class RadiationRole(StrEnum):
    SOURCE = "source"
    SHIELD = "shield"
    CONTAMINATED_SURFACE = "contaminated_surface"
    DETECTOR = "detector"
    ROBOT = "robot"
    OBSTACLE = "obstacle"
    WASTE = "waste"


class SourceType(StrEnum):
    POINT = "point"
    SURFACE = "surface"
    VOLUME = "volume"


class MaterialMode(StrEnum):
    SOLID = "solid"
    THIN_SHEET = "thin_sheet"


class UsdRadiationAttributes:
    """Canonical custom attribute names authored on USD prims."""

    ROLE = "rad:role"
    SOURCE_TYPE = "rad:source:type"
    SOURCE_ISOTOPE_ID = "rad:source:isotopeId"
    SOURCE_ACTIVITY_BQ = "rad:source:activityBq"
    SOURCE_SURFACE_ACTIVITY_BQ_M2 = "rad:source:surfaceActivityBqM2"
    SOURCE_ACTIVITY_MAP_URI = "rad:source:activityMapUri"
    SOURCE_ACTIVITY_MAP_SHA256 = "rad:source:activityMapSha256"
    SOURCE_SAMPLING_MODE = "rad:source:samplingMode"
    SOURCE_SAMPLES_PER_TRIANGLE = "rad:source:samplesPerTriangle"
    SOURCE_MAX_SAMPLES_PER_TRIANGLE = "rad:source:maxSamplesPerTriangle"
    SOURCE_VOLUME_DISTRIBUTION = "rad:source:volumeDistribution"
    SOURCE_SAMPLE_COUNT = "rad:source:sampleCount"
    SOURCE_HIDDEN_FROM_ESTIMATOR = "rad:source:hiddenFromEstimator"
    SOURCE_MOVABLE_WITH_PRIM = "rad:source:movableWithPrim"
    SOURCE_ENABLED = "rad:source:enabled"
    MATERIAL_ID = "rad:material:id"
    MATERIAL_MODE = "rad:material:mode"
    MATERIAL_THICKNESS_M = "rad:material:thicknessM"
    MATERIAL_ATTENUATION_URI = "rad:material:attenuationUri"
    MATERIAL_CONTAINER_ONLY = "rad:material:containerOnly"
    SHIELD_MOVABLE = "rad:shield:movable"
    SHIELD_RESOURCE_UNITS = "rad:shield:resourceUnits"
    DECON_ENABLED = "rad:decon:enabled"
    DECON_ACTIVITY_MAP_URI = "rad:decon:activityMapUri"
    DECON_ACTIVITY_MAP_SHA256 = "rad:decon:activityMapSha256"
    DECON_SUBSTRATE_MATERIAL_ID = "rad:decon:substrateMaterialId"
    DECON_TREATMENT_MODEL_URI = "rad:decon:treatmentModelUri"
    DECON_TREATMENT_MODEL_SHA256 = "rad:decon:treatmentModelSha256"
    DECON_MIN_TOOL_DWELL_S = "rad:decon:minToolDwellS"
    MANIPULATION_MOVABLE = "rad:manipulation:movable"
    MANIPULATION_REMOVABLE = "rad:manipulation:removable"
    MANIPULATION_GRASP_FRAME = "rad:manipulation:graspFrame"
    MANIPULATION_DISPOSAL_CLASS = "rad:manipulation:disposalClass"
    DETECTOR_ID = "rad:detector:id"


IDENTITY_MATRIX = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


@dataclass(frozen=True)
class SourceDescriptor:
    prim_path: str
    source_type: SourceType
    isotope_id: str
    activity_bq: float = 0.0
    surface_activity_bq_m2: float = 0.0
    activity_map_uri: str | None = None
    activity_map_sha256: str | None = None
    hidden_from_estimator: bool = False
    movable_with_prim: bool = False
    enabled: bool = True
    world_transform: tuple[float, ...] = IDENTITY_MATRIX

    def __post_init__(self) -> None:
        if not self.prim_path.startswith("/") or not self.isotope_id:
            raise ValueError("source prim path and isotope ID are required")
        if self.activity_bq < 0.0 or self.surface_activity_bq_m2 < 0.0:
            raise ValueError("source activity must be nonnegative")
        if len(self.world_transform) != 16:
            raise ValueError("source world transform must contain 16 values")
        if self.activity_map_sha256 is not None and self.activity_map_uri is None:
            raise ValueError("an activity-map digest requires an activity-map URI")


@dataclass(frozen=True)
class MaterialDescriptor:
    prim_path: str
    material_id: str
    mode: MaterialMode = MaterialMode.SOLID
    thickness_m: float | None = None
    attenuation_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.prim_path.startswith("/") or not self.material_id:
            raise ValueError("material prim path and ID are required")
        if self.mode == MaterialMode.THIN_SHEET and (
            self.thickness_m is None or self.thickness_m <= 0.0
        ):
            raise ValueError("thin-sheet materials require positive thickness")


@dataclass(frozen=True)
class ShieldDescriptor:
    prim_path: str
    material_id: str
    movable: bool
    resource_units: int
    world_transform: tuple[float, ...] = IDENTITY_MATRIX

    def __post_init__(self) -> None:
        if not self.prim_path.startswith("/") or not self.material_id:
            raise ValueError("shield prim path and material ID are required")
        if self.resource_units < 0 or len(self.world_transform) != 16:
            raise ValueError("shield resource units or transform are invalid")


@dataclass(frozen=True)
class DecontaminationSurfaceDescriptor:
    prim_path: str
    enabled: bool
    activity_map_uri: str | None
    activity_map_sha256: str | None
    substrate_material_id: str
    treatment_model_uri: str
    treatment_model_sha256: str
    min_tool_dwell_s: float

    def __post_init__(self) -> None:
        if not self.prim_path.startswith("/"):
            raise ValueError("decontamination prim path must be absolute")
        if not self.substrate_material_id or not self.treatment_model_uri:
            raise ValueError("decontamination substrate and treatment model are required")
        if len(self.treatment_model_sha256) != 64:
            raise ValueError("decontamination treatment model requires a SHA256 digest")
        if self.min_tool_dwell_s < 0.0:
            raise ValueError("minimum tool dwell must be nonnegative")


@dataclass(frozen=True)
class ManipulationDescriptor:
    prim_path: str
    movable: bool
    removable: bool
    grasp_frame: str | None
    disposal_class: str | None

    def __post_init__(self) -> None:
        if not self.prim_path.startswith("/"):
            raise ValueError("manipulation prim path must be absolute")


@dataclass(frozen=True)
class RadiationSceneSnapshot:
    """One immutable registry view built from a USD stage."""

    sources: Mapping[str, SourceDescriptor] = field(default_factory=dict)
    materials: Mapping[str, MaterialDescriptor] = field(default_factory=dict)
    shields: Mapping[str, ShieldDescriptor] = field(default_factory=dict)
    decontamination_surfaces: Mapping[str, DecontaminationSurfaceDescriptor] = field(
        default_factory=dict
    )
    manipulable_prims: Mapping[str, ManipulationDescriptor] = field(default_factory=dict)
    activity_maps: Mapping[str, SurfaceActivityMap] = field(default_factory=dict)
    geometry_prim_paths: frozenset[str] = frozenset()
    detector_prim_paths: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in (
            "sources",
            "materials",
            "shields",
            "decontamination_surfaces",
            "manipulable_prims",
            "activity_maps",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def estimator_visible_sources(self) -> Mapping[str, SourceDescriptor]:
        """Return a read-only source view that cannot expose hidden truth sources."""

        return MappingProxyType(
            {
                path: source
                for path, source in self.sources.items()
                if source.enabled and not source.hidden_from_estimator
            }
        )
