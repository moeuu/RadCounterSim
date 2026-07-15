"""Radiation transport and forward models."""

from radcounter.core.radiation.backend import AnalyticSlab, AnalyticTransportBackend
from radcounter.core.radiation.embree_native import (
    EmbreeTransportBackend,
    RadiationTriangleMesh,
)
from radcounter.core.radiation.forward import CountRatePrediction, RadiationForwardModel
from radcounter.core.radiation.materials import MaterialTable
from radcounter.core.radiation.sampled_forward import (
    SampleCountRatePrediction,
    SampledRadiationForwardModel,
)
from radcounter.core.radiation.sampling import SourceSampleBatch, sample_surface_triangles
from radcounter.core.radiation.transfer import TransferMatrixCache, TransferMatrixKey

__all__ = [
    "AnalyticSlab",
    "AnalyticTransportBackend",
    "CountRatePrediction",
    "EmbreeTransportBackend",
    "MaterialTable",
    "RadiationForwardModel",
    "RadiationTriangleMesh",
    "SampleCountRatePrediction",
    "SampledRadiationForwardModel",
    "SourceSampleBatch",
    "TransferMatrixCache",
    "TransferMatrixKey",
    "sample_surface_triangles",
]
