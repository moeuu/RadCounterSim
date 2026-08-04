from pathlib import Path

from radcounter.core.sensors.catalog import popular_detector_catalog
from radcounter.core.sensors.plugins import (
    DetectorRegistry,
    ExternalDetectorModel,
    ExternalDetectorReadingBuffer,
)
from radcounter.core.sensors.universal import (
    DetectorArray,
    DetectorPose,
    DetectorReading,
    ParametricDetectorModel,
    RadiationSample,
    RadiationType,
    ResponseCurve,
    ShieldPanel,
)


def _sample(rate: float = 1.0e8) -> RadiationSample:
    return RadiationSample((0.0, 0.0, 0.0), rate, 662.0)


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


def test_directional_detector_rejects_a_source_behind_it() -> None:
    model = ParametricDetectorModel(popular_detector_catalog()["collimated_nai"])
    array = DetectorArray()
    array.add(model, DetectorPose("front", (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    array.add(
        ParametricDetectorModel(popular_detector_catalog()["collimated_nai"]),
        DetectorPose("back", (2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    readings = array.measure([_sample()], seed=2)
    assert (
        readings["front"].expected_count_rate_cps > 50.0 * readings["back"].expected_count_rate_cps
    )


def test_shield_and_source_reduction_are_observed_by_detector_array() -> None:
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
    shield = ShieldPanel(
        "lead",
        center_world_m=(1.0, 0.0, 0.4),
        normal_world=(1.0, 0.0, 0.0),
        up_world=(0.0, 0.0, 1.0),
        width_m=1.0,
        height_m=1.2,
        thickness_m=0.05,
        attenuation_by_radiation={
            RadiationType.GAMMA: ResponseCurve((100.0, 3000.0), (120.0, 120.0))
        },
    )
    baseline = array.measure([_sample()], seed=3)
    shielded = array.measure([_sample()], shields=[shield], seed=3)
    decontaminated = array.measure([_sample(3.0e7)], shields=[shield], seed=3)
    assert shielded["east"].expected_count_rate_cps < baseline["east"].expected_count_rate_cps * 0.1
    assert (
        shielded["north"].expected_count_rate_cps > baseline["north"].expected_count_rate_cps * 0.95
    )
    assert (
        decontaminated["north"].expected_count_rate_cps
        < shielded["north"].expected_count_rate_cps * 0.4
    )
    assert len(baseline["north"].spectrum_counts) > 0


def test_yaml_csv_custom_detector_and_external_buffer() -> None:
    root = Path(__file__).resolve().parents[2]
    registry = DetectorRegistry()
    descriptor = registry.load_descriptor(root / "configs/detectors/custom_detector.example.yaml")
    assert descriptor.model_id == "laboratory_custom_czt"
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
    result = array.measure([], seed=1)
    assert result["connected_custom"].observed_counts == reading.observed_counts
