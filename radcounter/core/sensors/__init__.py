"""Radiation sensor state machines."""

from radcounter.core.sensors.detector import MeasurementState, OmnidirectionalCounter
from radcounter.core.sensors.rotating import (
    DoseRateMeter,
    PhysicalGeometryBinding,
    RotatingShieldConfiguration,
    RotatingShieldCounter,
    RotatingShieldMode,
    ShieldProgram,
    ShieldProgramMeasurement,
)

__all__ = [
    "CallbackDetectorModel",
    "DeadTimeModel",
    "DetectorArray",
    "DetectorDescriptor",
    "DetectorFamily",
    "DetectorOutput",
    "DetectorPose",
    "DetectorReading",
    "DetectorRegistry",
    "DoseRateMeter",
    "Directionality",
    "ExternalDetectorModel",
    "ExternalDetectorReadingBuffer",
    "IncidentParticleFluence",
    "MeasurementState",
    "OmnidirectionalCounter",
    "ParametricDetectorModel",
    "ParticleResponse",
    "PhysicalGeometryBinding",
    "RadiationType",
    "ResponseCurve",
    "RotatingShieldCounter",
    "RotatingShieldConfiguration",
    "RotatingShieldMode",
    "ShieldProgram",
    "ShieldProgramMeasurement",
    "contribution_weighted_sample_without_replacement",
    "load_detector_descriptor",
    "popular_detector_catalog",
]

# Generic multi-detector and plugin API.
from .catalog import popular_detector_catalog
from .plugins import (
    CallbackDetectorModel,
    DetectorRegistry,
    ExternalDetectorModel,
    ExternalDetectorReadingBuffer,
    load_detector_descriptor,
)
from .universal import (
    DeadTimeModel,
    DetectorArray,
    DetectorDescriptor,
    DetectorFamily,
    DetectorOutput,
    DetectorPose,
    DetectorReading,
    Directionality,
    IncidentParticleFluence,
    ParametricDetectorModel,
    ParticleResponse,
    RadiationType,
    ResponseCurve,
    contribution_weighted_sample_without_replacement,
)
