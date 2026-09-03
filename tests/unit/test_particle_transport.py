from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from radcounter.core.models.radiation import MaterialSpec
from radcounter.core.radiation import (
    MaterialTable,
    MultiParticleTransport,
    ParticleEmissionSample,
    ParticleTransportData,
    load_particle_transport_data,
)
from radcounter.core.radiation.backend import PathLengthBatch
from radcounter.core.sensors import RadiationType, ResponseCurve

ROOT = Path(__file__).resolve().parents[2]


class _Paths:
    def __init__(self, lead_path_m: float, *, error: bool = False) -> None:
        self.lead_path_m = lead_path_m
        self.error = error

    def trace_path_lengths(
        self, origins_m: np.ndarray, targets_m: np.ndarray
    ) -> PathLengthBatch:
        assert origins_m.shape == targets_m.shape
        return PathLengthBatch(
            ("lead",),
            np.full((len(origins_m), 1), self.lead_path_m),
            np.full(len(origins_m), self.error),
        )


def _curve(value: float) -> ResponseCurve:
    return ResponseCurve((0.00001, 10_000.0), (value, value))


def _data(*, neutron: bool = True) -> ParticleTransportData:
    photons = MaterialTable(
        tuple(
            MaterialSpec(
                material_id,
                np.asarray((1.0, 10_000.0)),
                np.asarray((10.0, 10.0)),
            )
            for material_id in ("concrete", "lead", "steel")
        )
    )
    charged = {
        RadiationType.ALPHA: {"air": _curve(0.05), "lead": _curve(1.0e-5)},
        RadiationType.BETA: {"air": _curve(5.0), "lead": _curve(1.0e-3)},
    }
    return ParticleTransportData(
        photons,
        {"lead": _curve(5.0)} if neutron else {},
        charged,
    )


def _emissions() -> tuple[ParticleEmissionSample, ...]:
    return tuple(
        ParticleEmissionSample(
            (0.0, 0.0, 0.0),
            1.0e6,
            energy,
            radiation_type,
            radiation_type.value,
        )
        for radiation_type, energy in (
            (RadiationType.GAMMA, 662.0),
            (RadiationType.X_RAY, 100.0),
            (RadiationType.NEUTRON, 1000.0),
            (RadiationType.ALPHA, 5500.0),
            (RadiationType.BETA, 1000.0),
        )
    )


def test_multi_particle_transport_uses_one_material_path_for_every_particle_type() -> None:
    transport = MultiParticleTransport(_Paths(0.05), _data())
    incident = transport.transport(_emissions(), {"d": (2.0, 0.0, 0.0)})["d"]
    by_type = {item.radiation_type: item for item in incident}
    free_space = 1.0e6 / (4.0 * math.pi * 4.0)
    assert by_type[RadiationType.GAMMA].fluence_rate_m2_s == pytest.approx(
        free_space * math.exp(-0.5)
    )
    assert by_type[RadiationType.X_RAY].fluence_rate_m2_s == pytest.approx(
        free_space * math.exp(-0.5)
    )
    assert by_type[RadiationType.NEUTRON].fluence_rate_m2_s == pytest.approx(
        free_space * math.exp(-0.25)
    )
    assert by_type[RadiationType.ALPHA].fluence_rate_m2_s == 0.0
    assert by_type[RadiationType.BETA].fluence_rate_m2_s == 0.0
    assert by_type[RadiationType.GAMMA].arrival_direction_world == (-1.0, -0.0, -0.0)


def test_multi_particle_transport_fails_on_missing_kernel_or_trace_error() -> None:
    neutron = (_emissions()[2],)
    with pytest.raises(ValueError, match="no neutron removal data"):
        MultiParticleTransport(_Paths(0.05), _data(neutron=False)).transport(
            neutron, {"d": (2.0, 0.0, 0.0)}
        )
    with pytest.raises(ValueError, match="trace errors"):
        MultiParticleTransport(_Paths(0.05, error=True), _data()).transport(
            neutron, {"d": (2.0, 0.0, 0.0)}
        )


def test_response_curves_never_extrapolate() -> None:
    response = ResponseCurve((100.0, 1000.0), (1.0e-4, 2.0e-5))
    with pytest.raises(ValueError, match="does not cover"):
        response.at(99.0)


def test_particle_transport_tables_are_versioned_and_fail_closed() -> None:
    path = ROOT / "configs/materials/particle_transport.synthetic.yaml"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_particle_transport_data(
        path,
        _data().photon_materials,
        expected_file_sha256=digest,
        required_solid_material_ids={"concrete", "lead", "steel"},
    )
    assert loaded.data_status == "synthetic_validation_only"
    assert loaded.neutron_removal_m_inv["lead"].at(1000.0) == pytest.approx(3.5)
    assert len(loaded.numeric_sha256) == 64
    with pytest.raises(ValueError, match="coverage"):
        load_particle_transport_data(
            path,
            _data().photon_materials,
            expected_file_sha256=digest,
            required_solid_material_ids={"lead"},
        )
