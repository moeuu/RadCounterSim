"""Radiation transport and forward models."""

from radcounter.core.radiation.backend import AnalyticSlab, AnalyticTransportBackend
from radcounter.core.radiation.data import (
    LoadedPhysicsData,
    PhysicsDataError,
    ValidatedPhysicsBundle,
    load_detector_data,
    load_isotope_data,
    load_material_data,
    validate_research_evaluation_data,
)
from radcounter.core.radiation.embree_native import (
    EmbreeTransportBackend,
    RadiationTriangleMesh,
)
from radcounter.core.radiation.forward import CountRatePrediction, RadiationForwardModel
from radcounter.core.radiation.materials import MaterialTable, interpolate_attenuation_m_inv
from radcounter.core.radiation.particle_transport import (
    MultiParticleTransport,
    ParticleEmissionSample,
    ParticleTransportData,
    load_particle_transport_data,
)
from radcounter.core.radiation.sampled_forward import (
    SampleCountRatePrediction,
    SampledRadiationForwardModel,
)
from radcounter.core.radiation.sampling import SourceSampleBatch, sample_surface_triangles
from radcounter.core.radiation.scatter import (
    LoadedBuildupData,
    MaterialBuildupSurface,
    PhotonBuildupModel,
    PrimaryOnlyPhotonModel,
    ReferenceCalibratedBuildupModel,
    load_buildup_data,
)
from radcounter.core.radiation.transfer import TransferMatrixCache, TransferMatrixKey

__all__ = [
    "AnalyticSlab",
    "AnalyticTransportBackend",
    "CountRatePrediction",
    "EmbreeTransportBackend",
    "LoadedPhysicsData",
    "LoadedBuildupData",
    "MaterialTable",
    "MaterialBuildupSurface",
    "MultiParticleTransport",
    "ParticleEmissionSample",
    "ParticleTransportData",
    "PhysicsDataError",
    "PhotonBuildupModel",
    "PrimaryOnlyPhotonModel",
    "ValidatedPhysicsBundle",
    "RadiationForwardModel",
    "RadiationTriangleMesh",
    "ReferenceCalibratedBuildupModel",
    "SampleCountRatePrediction",
    "SampledRadiationForwardModel",
    "SourceSampleBatch",
    "TransferMatrixCache",
    "TransferMatrixKey",
    "load_detector_data",
    "load_buildup_data",
    "load_isotope_data",
    "load_material_data",
    "load_particle_transport_data",
    "interpolate_attenuation_m_inv",
    "sample_surface_triangles",
    "validate_research_evaluation_data",
]
