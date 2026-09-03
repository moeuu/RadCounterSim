"""Portable scene metadata, activity maps, and revision classification."""

from radcounter.core.scene.activity_map import (
    ActivityMapIntegrityError,
    SurfaceActivityMap,
    VolumeActivityMap,
    resolve_asset_uri,
    sha256_file,
)
from radcounter.core.scene.metadata import (
    DecontaminationSurfaceDescriptor,
    ManipulationDescriptor,
    MaterialDescriptor,
    MaterialMode,
    RadiationRole,
    RadiationSceneSnapshot,
    ShieldDescriptor,
    SourceDescriptor,
    SourceType,
    UsdRadiationAttributes,
)
from radcounter.core.scene.revision import RevisionDelta, classify_stage_changes

__all__ = [
    "ActivityMapIntegrityError",
    "DecontaminationSurfaceDescriptor",
    "ManipulationDescriptor",
    "MaterialDescriptor",
    "MaterialMode",
    "RadiationRole",
    "RadiationSceneSnapshot",
    "RevisionDelta",
    "ShieldDescriptor",
    "SourceDescriptor",
    "SourceType",
    "SurfaceActivityMap",
    "VolumeActivityMap",
    "UsdRadiationAttributes",
    "classify_stage_changes",
    "resolve_asset_uri",
    "sha256_file",
]
