"""Physical rotating-shield USD/Embree synchronization gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "source/extensions/radcounter.isaac")]


def _attribute(prim: object, sdf: object, name: str, value_type: object, value: object) -> None:
    prim.CreateAttribute(name, value_type, custom=True).Set(value)


def main() -> int:
    from pxr import Gf, Sdf, Usd, UsdGeom
    from radcounter.isaac.runtime import (
        IsaacPhysicalRotatingShield,
        IsaacRadiationSimulation,
        physical_rotating_shield_payload,
    )
    from radcounter.isaac.runtime.simulation import (
        RuntimeConfiguration,
        RuntimeDetector,
        RuntimeLine,
    )

    from radcounter.core.sensors import RotatingShieldConfiguration

    shield_configuration = RotatingShieldConfiguration.from_yaml(
        ROOT / "configs/detectors/rotating_shield_counter.physical.synthetic.yaml"
    )
    detector_spec = shield_configuration.detector
    runtime_detector = RuntimeDetector(
        detector_id=detector_spec.detector_id,
        energy_bin_edges_keV=detector_spec.energy_bin_edges_keV.copy(),
        response_energy_keV=detector_spec.response_energy_keV.copy(),
        effective_area_m2_per_bin=detector_spec.effective_area_m2_per_bin.copy(),
        background_cps_per_bin=detector_spec.background_cps_per_bin.copy(),
        dead_time_s=detector_spec.dead_time_s,
    )
    runtime_configuration = RuntimeConfiguration(
        materials={
            "lead": (
                np.asarray([80.0, 300.0, 661.657, 1500.0]),
                np.asarray([640.0, 210.0, 125.0, 70.0]),
            )
        },
        isotopes={"Cs-137": (RuntimeLine(661.657, 0.851),)},
        detectors={detector_spec.detector_id: runtime_detector},
        duration_s=1.0,
        minimum_distance_m=0.01,
        seed=31,
    )

    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    source = UsdGeom.Xform.Define(stage, "/World/Source")
    source.AddTranslateOp().Set(Gf.Vec3d(3.0, 0.0, 0.0))
    source_prim = source.GetPrim()
    _attribute(source_prim, Sdf, "rad:role", Sdf.ValueTypeNames.String, "source")
    _attribute(source_prim, Sdf, "rad:source:type", Sdf.ValueTypeNames.String, "point")
    _attribute(
        source_prim,
        Sdf,
        "rad:source:isotopeId",
        Sdf.ValueTypeNames.String,
        "Cs-137",
    )
    _attribute(source_prim, Sdf, "rad:source:activityBq", Sdf.ValueTypeNames.Double, 1.0e12)

    UsdGeom.Xform.Define(stage, "/World/RotatingShieldRig")
    detector = UsdGeom.Xform.Define(stage, "/World/RotatingShieldRig/Detector")
    detector_prim = detector.GetPrim()
    _attribute(detector_prim, Sdf, "rad:role", Sdf.ValueTypeNames.String, "detector")
    _attribute(
        detector_prim,
        Sdf,
        "rad:detector:id",
        Sdf.ValueTypeNames.String,
        detector_spec.detector_id,
    )

    rotor = UsdGeom.Xform.Define(stage, "/World/RotatingShieldRig/ShieldRotor")
    rotor.AddRotateZOp().Set(-45.0)
    plate = UsdGeom.Cube.Define(stage, "/World/RotatingShieldRig/ShieldRotor/LeadPlate")
    plate.AddTranslateOp().Set(Gf.Vec3d(0.55, 0.0, 0.0))
    plate.AddScaleOp().Set(Gf.Vec3f(0.05, 0.55, 0.55))
    _attribute(
        plate.GetPrim(),
        Sdf,
        "rad:material:id",
        Sdf.ValueTypeNames.String,
        "lead",
    )

    simulation = IsaacRadiationSimulation(stage, runtime_configuration)
    controller = IsaacPhysicalRotatingShield(
        simulation,
        shield_configuration,
        np.random.default_rng(22),
    )
    result = controller.measure()
    payload = physical_rotating_shield_payload(result)
    measurement = result.measurement
    expected_rates = (
        np.sum(measurement.expected_counts_per_posture_bin, axis=1)
        / measurement.dwell_s
    )

    assert result.evidence_class.value == "kinematic_scene_edit"
    assert measurement.mode.value == "physical_geometry"
    assert result.attenuation_geometry_paths == (
        "/World/RotatingShieldRig/ShieldRotor/LeadPlate",
    )
    assert all(
        paths == result.attenuation_geometry_paths
        for paths in result.changed_geometry_paths_per_posture
    )
    assert expected_rates[0] < 0.05 * min(expected_rates[1:]), expected_rates
    assert expected_rates[1] > 1000.0
    payload["expected_rate_cps_per_posture"] = expected_rates.tolist()
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
