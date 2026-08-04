"""Simulator-neutral 3D environment ingestion API."""

from radcounter.core.environment.importers import (
    EnvironmentDependencyError,
    EnvironmentImporter,
    EnvironmentImporterRegistry,
    EnvironmentImportError,
)
from radcounter.core.environment.models import (
    AxisDirection,
    CollisionApproximation,
    CollisionGeometrySource,
    CoordinateSystemConfig,
    EnvironmentCollisionConfig,
    EnvironmentDescriptorConfig,
    EnvironmentFormat,
    EnvironmentImportConfig,
    EnvironmentImportResult,
    EnvironmentMesh,
    EnvironmentScene,
    ExternalConverterConfig,
    Handedness,
    LengthUnit,
    MaterialRuleConfig,
    load_environment_descriptor,
)
from radcounter.core.environment.pipeline import (
    EnvironmentAssetResolver,
    EnvironmentImportPipeline,
    detect_environment_format,
    load_environment_manifest,
)

__all__ = [
    "AxisDirection",
    "CollisionApproximation",
    "CollisionGeometrySource",
    "CoordinateSystemConfig",
    "EnvironmentAssetResolver",
    "EnvironmentCollisionConfig",
    "EnvironmentDependencyError",
    "EnvironmentDescriptorConfig",
    "EnvironmentFormat",
    "EnvironmentImportConfig",
    "EnvironmentImportError",
    "EnvironmentImportPipeline",
    "EnvironmentImportResult",
    "EnvironmentImporter",
    "EnvironmentImporterRegistry",
    "EnvironmentMesh",
    "EnvironmentScene",
    "ExternalConverterConfig",
    "Handedness",
    "LengthUnit",
    "MaterialRuleConfig",
    "detect_environment_format",
    "load_environment_descriptor",
    "load_environment_manifest",
]
