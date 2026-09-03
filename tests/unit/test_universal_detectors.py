from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from radcounter.core.models.radiation import MaterialSpec
from radcounter.core.radiation import (
    MaterialTable,
    MultiParticleTransport,
    ParticleEmissionSample,
    ParticleTransportData,
)
from radcounter.core.radiation.backend import PathLengthBatch
from radcounter.core.sensors.catalog import popular_detector_catalog
from radcounter.core.sensors.plugins import (
    DetectorRegistry,
    ExternalDetectorModel,
    ExternalDetectorReadingBuffer,
)
from radcounter.core.sensors.universal import (
    DetectorArray,
    DetectorOutput,
    DetectorPose,
    DetectorReading,
    Directionality,
    IncidentParticleFluence,
    MeasurementRequest,
    ParametricDetectorModel,
    RadiationType,
)


def _incident(
    fluence_rate: float = 1.0e6,
    direction: tuple[float, float, float] = (-1.0, 0.0, 0.0),
) -> IncidentParticleFluence:
    return IncidentParticleFluence(
        RadiationType.GAMMA,
        662.0,
        fluence_rate,
        direction,
        "Cs-137",
    )


def test_popular_catalog_covers_survey_spectroscopy_dose_neutron_and_imaging() -> None:
    catalog = popular_detector_catalog()
    required = {
        "gm_tube",
        "ion_chamber",
        "nai_tl",
        "csi_tl",
        "plastic_scintillator",
        "hpge",
        "czt",
        "h3d_h100_omni",
        "cdte",
        "proportional_counter",
        "pancake_probe",
        "he3_neutron",
        "li6_zns_neutron",
        "neutron_rem_meter",
        "electronic_dosimeter",
        "collimated_nai",
        "rotating_shield",
        "coded_aperture_czt",
        "compton_camera",
    }
    assert required <= catalog.keys()
    assert {item.response_data_status for item in catalog.values()} == {"synthetic_validation_only"}


def test_h3d_h100_profile_keeps_real_specifications_separate_from_synthetic_response() -> None:
    descriptor = popular_detector_catalog()["h3d_h100_omni"]
    assert descriptor.directionality is Directionality.OMNIDIRECTIONAL
    assert DetectorOutput.DIRECTION not in descriptor.outputs
    assert DetectorOutput.IMAGE not in descriptor.outputs
    assert DetectorOutput.DOSE_RATE in descriptor.outputs
    assert descriptor.energy_resolution_fwhm_fraction_at_662kev == pytest.approx(0.011)
    assert descriptor.response_for(RadiationType.GAMMA).effective_area_m2.at(
        661.657
    ) == pytest.approx(5.6e-5)
    assert descriptor.metadata["czt_crystal_volume_cm3"] == pytest.approx(6.0)
    assert descriptor.metadata["radiation_fov_sr"] == pytest.approx(4.0 * np.pi)
    assert descriptor.metadata["manufacturer_specification_url"] == (
        "https://h3dgamma.com/H100Specs.pdf"
    )
    assert descriptor.response_data_status == "synthetic_validation_only"
    assert "not an H3D calibration" in descriptor.response_provenance["calibration_method"]


def test_h3d_h100_scalar_measurement_tracks_decontaminated_activity() -> None:
    model = ParametricDetectorModel(popular_detector_catalog()["h3d_h100_omni"])
    pose = DetectorPose("h100", (0.0, 0.0, 0.0))

    def reading(fluence_rate: float, seed: int):
        return model.measure(
            MeasurementRequest(
                pose=pose,
                incident_fluence=(_incident(fluence_rate),),
                integration_time_s=1.0,
                rng=np.random.default_rng(seed),
            )
        )

    before = reading(1.0e6, 4)
    after = reading(1.0e5, 5)
    assert after.expected_count_rate_cps < before.expected_count_rate_cps * 0.11
    assert after.dose_rate_usv_h == pytest.approx(before.dose_rate_usv_h * 0.1, rel=2e-4)
    assert before.estimated_direction_world is None
    assert after.estimated_direction_world is None


def test_directional_detector_uses_transport_supplied_arrival_direction() -> None:
    model = ParametricDetectorModel(popular_detector_catalog()["collimated_nai"])
    array = DetectorArray()
    array.add(model, DetectorPose("front", (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    array.add(
        ParametricDetectorModel(popular_detector_catalog()["collimated_nai"]),
        DetectorPose("back", (2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    readings = array.measure({"front": (_incident(),), "back": (_incident(),)}, seed=2)
    assert (
        readings["front"].expected_count_rate_cps > 50.0 * readings["back"].expected_count_rate_cps
    )
    assert readings["front"].metadata["input_representation"] == ("transported_particle_fluence")


class _DirectionalLeadPaths:
    def __init__(self, shielded: bool) -> None:
        self.shielded = shielded

    def trace_path_lengths(self, origins_m: np.ndarray, targets_m: np.ndarray) -> PathLengthBatch:
        lengths = np.zeros((len(origins_m), 1))
        if self.shielded:
            lengths[targets_m[:, 0] > 1.0, 0] = 0.05
        return PathLengthBatch(("lead",), lengths, np.zeros(len(origins_m), dtype=np.bool_))


def _transport(shielded: bool) -> MultiParticleTransport:
    materials = MaterialTable(
        (MaterialSpec("lead", np.asarray((30.0, 3000.0)), np.asarray((120.0, 120.0))),)
    )
    return MultiParticleTransport(
        _DirectionalLeadPaths(shielded),
        ParticleTransportData(materials, {}, {}),
    )


def _emission(rate: float) -> tuple[ParticleEmissionSample, ...]:
    return (ParticleEmissionSample((0.0, 0.0, 0.0), rate, 662.0, RadiationType.GAMMA, "Cs-137"),)


def test_shield_and_source_reduction_are_transported_once_before_detector_response() -> None:
    catalog = popular_detector_catalog()
    array = DetectorArray()
    array.add(
        ParametricDetectorModel(catalog["gm_tube"]),
        DetectorPose("east", (2.0, 0.0, 0.5)),
    )
    array.add(
        ParametricDetectorModel(catalog["hpge"]),
        DetectorPose("north", (0.0, 2.0, 0.5)),
    )
    baseline_field = _transport(False).transport(_emission(1.0e8), array.detector_positions)
    shielded_field = _transport(True).transport(_emission(1.0e8), array.detector_positions)
    reduced_field = _transport(True).transport(_emission(3.0e7), array.detector_positions)
    baseline = array.measure(baseline_field, seed=3)
    shielded = array.measure(shielded_field, seed=3)
    decontaminated = array.measure(reduced_field, seed=3)
    assert shielded["east"].expected_count_rate_cps < baseline["east"].expected_count_rate_cps * 0.1
    assert (
        shielded["north"].expected_count_rate_cps > baseline["north"].expected_count_rate_cps * 0.95
    )
    assert (
        decontaminated["north"].expected_count_rate_cps
        < shielded["north"].expected_count_rate_cps * 0.4
    )
    assert len(baseline["north"].spectrum_counts) > 0


def test_detector_array_requires_a_field_for_every_detector() -> None:
    array = DetectorArray()
    array.add(
        ParametricDetectorModel(popular_detector_catalog()["gm_tube"]),
        DetectorPose("d", (0.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="do not match"):
        array.measure({})


def test_yaml_csv_custom_detector_and_external_buffer() -> None:
    root = Path(__file__).resolve().parents[2]
    registry = DetectorRegistry()
    descriptor = registry.load_descriptor(root / "configs/detectors/custom_detector.example.yaml")
    assert descriptor.model_id == "laboratory_custom_czt"
    assert descriptor.particle_responses[0].effective_area_m2.at(662.0) == pytest.approx(0.000056)
    assert registry.create(descriptor.model_id).descriptor is descriptor

    buffer = ExternalDetectorReadingBuffer(maximum_age_s=1.0)
    reading = DetectorReading(
        detector_id="hardware",
        model_id=descriptor.model_id,
        integration_time_s=1.0,
        expected_count_rate_cps=12.0,
        observed_counts=11,
    )
    buffer.ingest_json(
        '{"detector_id":"hardware","model_id":"laboratory_custom_czt",'
        '"integration_time_s":1.0,"expected_count_rate_cps":12.0,'
        '"observed_counts":11}'
    )
    model = ExternalDetectorModel(descriptor, buffer)
    array = DetectorArray()
    array.add(model, DetectorPose("connected_custom", (0.0, 0.0, 0.0)))
    result = array.measure({"connected_custom": ()}, seed=1)
    assert result["connected_custom"].observed_counts == reading.observed_counts


def test_h3d_h100_yaml_profile_loads_as_omnidirectional() -> None:
    root = Path(__file__).resolve().parents[2]
    descriptor = DetectorRegistry(include_popular=False).load_descriptor(
        root / "configs/detectors/h3d_h100_omni.synthetic.yaml"
    )
    assert descriptor.model_id == "h3d_h100_omni"
    assert descriptor.directionality is Directionality.OMNIDIRECTIONAL
    assert descriptor.metadata["product"] == "H100 Gamma-Ray Imaging Spectrometer"
    assert descriptor.response_data_status == "synthetic_validation_only"
