#!/usr/bin/env python3
"""Validate multiple omni/directional/custom detectors across shield and decon stages."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/extensions/radcounter.isaac"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".cache/detector-array-validation",
    )
    return parser.parse_args()


ARGS = parse_arguments()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 800,
        "renderer": "RaytracedLighting",
        "window_title": "RadInterAct Multi-Detector Shield and Decon Validation",
    }
)


import omni.kit.viewport.utility as viewport_utility
import omni.ui as ui
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdLux
from radcounter.isaac.runtime.simulation import NativeStageTransport, RuntimeConfiguration

from radcounter.core.models.radiation import MaterialSpec
from radcounter.core.radiation import (
    MaterialTable,
    MultiParticleTransport,
    ParticleEmissionSample,
    ParticleTransportData,
)
from radcounter.core.sensors.plugins import DetectorRegistry
from radcounter.core.sensors.universal import (
    DetectorArray,
    DetectorPose,
    RadiationType,
)
from radcounter.core.surface_decontamination import (
    DecontaminationTool,
    SurfaceSourceGrid,
)

DT_S = 1.0 / 60.0
SOURCE_PATH = "/World/SurfaceSource"
ROBOT_PATH = "/World/DeconRobot"


def set_color(geometry, rgb: tuple[float, float, float]) -> None:
    geometry.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def normalized_toward_origin(position: tuple[float, float, float]) -> tuple[float, float, float]:
    vector = -np.asarray(position, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    return tuple(float(value) for value in vector)


def create_source(stage) -> tuple[SurfaceSourceGrid, list]:
    cells_x, cells_y = 16, 10
    xx, yy = np.meshgrid(
        np.linspace(-1.0, 1.0, cells_x),
        np.linspace(-1.0, 1.0, cells_y),
        indexing="xy",
    )
    hotspot = np.exp(-((xx - 0.25) ** 2 + (yy + 0.18) ** 2) / 0.20)
    activity = (220_000.0 * (1.0 + 1.8 * hotspot)).reshape(-1)
    grid = SurfaceSourceGrid(
        cells_x=cells_x,
        cells_y=cells_y,
        size_x_m=2.1,
        size_y_m=1.3,
        center_world_m=(0.0, 0.0, 0.025),
        activity_bq_per_cell=activity,
    )
    root = UsdGeom.Xform.Define(stage, SOURCE_PATH).GetPrim()
    root.CreateAttribute("rad:role", Sdf.ValueTypeNames.String).Set("contaminated_surface")
    root.CreateAttribute("rad:source:type", Sdf.ValueTypeNames.String).Set("surface")
    color_attributes = []
    for index, center in enumerate(grid.centers_world_m):
        cell = UsdGeom.Cube.Define(stage, f"{SOURCE_PATH}/Cell_{index:03d}")
        cell.CreateSizeAttr(1.0)
        cell.AddTranslateOp().Set(Gf.Vec3d(*center))
        cell.AddScaleOp().Set(Gf.Vec3f(grid.cell_size_x_m * 0.47, grid.cell_size_y_m * 0.47, 0.018))
        color_attributes.append(cell.CreateDisplayColorAttr([Gf.Vec3f(*grid.color_rgb()[index])]))
        prim = cell.GetPrim()
        prim.CreateAttribute("rad:role", Sdf.ValueTypeNames.String).Set("source")
        prim.CreateAttribute("rad:source:activityBq", Sdf.ValueTypeNames.Double).Set(
            float(grid.activity_bq[index])
        )
    return grid, color_attributes


def particle_emissions(grid: SurfaceSourceGrid) -> tuple[ParticleEmissionSample, ...]:
    samples = []
    for index, (position, activity) in enumerate(
        zip(grid.centers_world_m, grid.activity_bq, strict=True)
    ):
        samples.append(
            ParticleEmissionSample(
                tuple(float(value) for value in position),
                float(activity * 0.851),
                662.0,
                RadiationType.GAMMA,
                f"cell_{index:03d}_cs137",
            )
        )
        samples.append(
            ParticleEmissionSample(
                tuple(float(value) for value in position),
                float(activity * 0.055),
                32.0,
                RadiationType.GAMMA,
                f"cell_{index:03d}_xray",
            )
        )
    return tuple(samples)


def create_shield(stage) -> object:
    center_world_m = (1.30, 0.20, 0.55)
    width_m = 2.4
    height_m = 1.4
    thickness_m = 0.05
    cube = UsdGeom.Cube.Define(stage, "/World/LeadShield")
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center_world_m))
    cube.AddScaleOp().Set(
        Gf.Vec3f(thickness_m * 0.5, width_m * 0.5, height_m * 0.5)
    )
    set_color(cube, (0.18, 0.20, 0.23))
    prim = cube.GetPrim()
    prim.CreateAttribute("rad:role", Sdf.ValueTypeNames.String).Set("shield")
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def detector_layout(registry: DetectorRegistry):
    custom = registry.load_descriptor(ROOT / "configs/detectors/custom_detector.example.yaml")
    entries = [
        ("gm_east", "gm_tube", (2.55, 0.0, 0.58), (-1.0, 0.0, 0.0)),
        ("nai_north", "nai_tl", (0.0, 2.45, 0.62), (0.0, -1.0, 0.0)),
        ("hpge_west", "hpge", (-2.45, 0.0, 0.70), (1.0, 0.0, 0.0)),
        ("ion_south", "ion_chamber", (0.0, -2.45, 0.62), (0.0, 1.0, 0.0)),
        (
            "collimated_aimed",
            "collimated_nai",
            (2.20, 1.45, 0.75),
            normalized_toward_origin((2.20, 1.45, 0.75)),
        ),
        (
            "collimated_away",
            "collimated_nai",
            (2.20, -1.45, 0.75),
            tuple(-value for value in normalized_toward_origin((2.20, -1.45, 0.75))),
        ),
        (
            "coded_camera",
            "coded_aperture_czt",
            (-1.85, 1.85, 0.90),
            normalized_toward_origin((-1.85, 1.85, 0.90)),
        ),
        (
            "custom_czt",
            custom.model_id,
            (-1.85, -1.85, 0.78),
            normalized_toward_origin((-1.85, -1.85, 0.78)),
        ),
    ]
    array = DetectorArray()
    for detector_id, model_id, position, forward in entries:
        array.add(
            registry.create(model_id),
            DetectorPose(detector_id, position, forward),
        )
    return array, entries


def create_detector_visuals(stage, entries, registry: DetectorRegistry) -> None:
    colors = {
        "gm_tube": (0.20, 0.75, 0.92),
        "nai_tl": (0.20, 0.85, 0.35),
        "hpge": (0.55, 0.42, 0.92),
        "ion_chamber": (0.95, 0.62, 0.10),
        "collimated_nai": (0.95, 0.22, 0.12),
        "coded_aperture_czt": (0.96, 0.82, 0.10),
        "laboratory_custom_czt": (0.12, 0.82, 0.74),
    }
    for detector_id, model_id, position, forward in entries:
        root_path = f"/World/Detectors/{detector_id}"
        root = UsdGeom.Xform.Define(stage, root_path)
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        body = UsdGeom.Cylinder.Define(stage, f"{root_path}/Body")
        body.CreateAxisAttr("Z")
        body.CreateRadiusAttr(0.16 if "ion" not in detector_id else 0.22)
        body.CreateHeightAttr(0.42)
        set_color(body, colors[model_id])
        prim = root.GetPrim()
        descriptor = registry.descriptor(model_id)
        prim.CreateAttribute("rad:detector:id", Sdf.ValueTypeNames.String).Set(detector_id)
        prim.CreateAttribute("rad:detector:model", Sdf.ValueTypeNames.String).Set(model_id)
        prim.CreateAttribute("rad:detector:directionality", Sdf.ValueTypeNames.String).Set(
            descriptor.directionality.value
        )
        if descriptor.directionality.value != "omnidirectional":
            end = np.asarray(position) + np.asarray(forward) * 0.85
            curve = UsdGeom.BasisCurves.Define(stage, f"/World/DetectorRays/{detector_id}")
            curve.CreateTypeAttr("linear")
            curve.CreateCurveVertexCountsAttr([2])
            curve.CreatePointsAttr([Gf.Vec3f(*position), Gf.Vec3f(*end)])
            curve.CreateWidthsAttr([0.028, 0.012])
            set_color(curve, colors[model_id])


def create_decon_robot(stage):
    root = UsdGeom.Xform.Define(stage, ROBOT_PATH)
    translate = root.AddTranslateOp()
    translate.Set(Gf.Vec3d(-1.0, -0.56, 0.25))
    body = UsdGeom.Cube.Define(stage, f"{ROBOT_PATH}/Body")
    body.CreateSizeAttr(1.0)
    body.AddScaleOp().Set(Gf.Vec3f(0.30, 0.22, 0.15))
    set_color(body, (0.04, 0.34, 0.52))
    tool = UsdGeom.Cube.Define(stage, f"{ROBOT_PATH}/Tool")
    tool.CreateSizeAttr(1.0)
    tool.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.21))
    tool.AddScaleOp().Set(Gf.Vec3f(0.23, 0.17, 0.02))
    set_color(tool, (0.02, 0.82, 0.90))
    return translate


class DetectorValidationWindow:
    def __init__(self, entries, registry: DetectorRegistry) -> None:
        self.window = ui.Window("Detector Array Verification", width=650, height=520)
        self.phase = None
        self.activity = None
        self.rows = {}
        with self.window.frame, ui.VStack(spacing=6):
            ui.Label("MULTI-DETECTOR: BASELINE -> 3D LEAD SHIELD -> DECONTAMINATION")
            self.phase = ui.Label("Phase: baseline")
            self.activity = ui.Label("Surface activity: waiting")
            ui.Label("Detector / type / expected cps / baseline ratio")
            for detector_id, model_id, _position, _forward in entries:
                descriptor = registry.descriptor(model_id)
                label = ui.Label(f"{detector_id:18s} {descriptor.directionality.value:18s} waiting")
                self.rows[detector_id] = label
            ui.Label("Directional rays are colored lines. Dark slab is the lead shield.")

    def update(self, phase: str, activity_bq: float, readings, baseline) -> None:
        self.phase.text = f"Phase: {phase}"
        self.activity.text = f"Surface activity: {activity_bq / 1e6:.2f} MBq"
        for detector_id, reading in readings.items():
            ratio = (
                reading.expected_count_rate_cps / baseline[detector_id].expected_count_rate_cps
                if baseline and baseline[detector_id].expected_count_rate_cps > 0.0
                else 1.0
            )
            self.rows[detector_id].text = (
                f"{detector_id:18s} {reading.model_id:22s} "
                f"{reading.expected_count_rate_cps:9.2f} cps  x{ratio:6.3f}"
            )


def request_capture(path: Path) -> None:
    path.unlink(missing_ok=True)
    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport is unavailable")
    viewport_utility.capture_viewport_to_file(viewport, str(path))


def ratios(numerator, denominator) -> dict[str, float]:
    return {
        detector_id: numerator[detector_id].expected_count_rate_cps
        / max(denominator[detector_id].expected_count_rate_cps, 1e-12)
        for detector_id in numerator
    }


def main() -> int:
    ARGS.output.mkdir(parents=True, exist_ok=True)
    result_path = ARGS.output / "result.json"
    baseline_image = ARGS.output / "baseline.png"
    shield_image = ARGS.output / "shielded.png"
    decon_image = ARGS.output / "decontaminated.png"
    result: dict[str, object] = {"passed": False}
    world = None
    try:
        world = World(stage_units_in_meters=1.0, physics_dt=DT_S, rendering_dt=DT_S)
        world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        grid, color_attributes = create_source(stage)
        shield_prim = create_shield(stage)
        registry = DetectorRegistry()
        array, entries = detector_layout(registry)
        material_spec = MaterialSpec(
            "lead",
            np.asarray((30.0, 100.0, 662.0, 1500.0, 3000.0)),
            np.asarray((950.0, 520.0, 120.0, 75.0, 52.0)),
        )
        material_table = MaterialTable((material_spec,))
        transport_backend = NativeStageTransport(
            stage,
            RuntimeConfiguration(
                materials={
                    "lead": (material_spec.energies_keV, material_spec.linear_attenuation_m_inv)
                },
                isotopes={},
                detectors={},
                duration_s=1.0,
                minimum_distance_m=0.01,
                seed=1,
            ),
        )
        particle_transport = MultiParticleTransport(
            transport_backend,
            ParticleTransportData(material_table, {}, {}),
        )
        create_detector_visuals(stage, entries, registry)
        robot_translate = create_decon_robot(stage)
        light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        light.CreateIntensityAttr(2_100.0)
        light.AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 25.0))
        panel = DetectorValidationWindow(entries, registry)
        set_camera_view(
            eye=np.asarray([5.4, 5.2, 6.2]),
            target=np.asarray([0.0, 0.0, 0.15]),
            camera_prim_path="/OmniverseKit_Persp",
        )
        world.reset()

        decon_tool = DecontaminationTool(
            length_m=0.46,
            width_m=0.34,
            rate_constant_s_inv=4.0,
            max_contact_distance_m=0.10,
            max_surface_speed_m_s=2.0,
        )
        initial_activity = grid.total_activity_bq
        baseline = None
        shielded = None
        current = None
        measurement_index = 0
        next_measurement_s = 0.0
        baseline_captured = False
        shield_captured = False
        start_wall = time.monotonic()
        next_frame = start_wall
        last_log_second = -1
        shield_transport_enabled = False

        while simulation_app.is_running():
            elapsed = time.monotonic() - start_wall
            if elapsed >= ARGS.duration:
                break
            shield_active = elapsed >= 4.0
            if shield_active and not shield_transport_enabled:
                UsdGeom.Imageable(shield_prim).MakeVisible()
                shield_prim.CreateAttribute(
                    "rad:material:id", Sdf.ValueTypeNames.String
                ).Set("lead")
                transport_backend.synchronize_transforms()
                shield_transport_enabled = True
            phase = "baseline"
            if shield_active:
                phase = "lead shield installed"
            if elapsed >= 8.0:
                phase = "robot decontaminating"
                decon_elapsed = elapsed - 8.0
                row_duration = 2.0
                row_index = min(int(decon_elapsed / row_duration), 4)
                row_progress = min((decon_elapsed % row_duration) / row_duration, 1.0)
                x_start, x_end = (-1.0, 1.0) if row_index % 2 == 0 else (1.0, -1.0)
                x = x_start + (x_end - x_start) * row_progress
                y = -0.56 + row_index * 0.28
                yaw = 0.0 if x_end > x_start else math.pi
                robot_translate.Set(Gf.Vec3d(x, y, 0.25))
                step = grid.apply_tool(
                    decon_tool,
                    tool_center_world_m=(x, y, 0.045),
                    tool_yaw_rad=yaw,
                    surface_speed_m_s=1.0,
                    dt_s=DT_S,
                )
                if step.contacted_cells:
                    colors = grid.color_rgb()
                    for index in step.contacted_cells:
                        color_attributes[index].Set([Gf.Vec3f(*colors[index])])
                        stage.GetPrimAtPath(f"{SOURCE_PATH}/Cell_{index:03d}").GetAttribute(
                            "rad:source:activityBq"
                        ).Set(float(grid.activity_bq[index]))

            if elapsed >= next_measurement_s or current is None:
                incident = particle_transport.transport(
                    particle_emissions(grid), array.detector_positions
                )
                current = array.measure(
                    incident,
                    integration_time_s=1.0,
                    seed=10_000 + measurement_index,
                )
                measurement_index += 1
                next_measurement_s = elapsed + 0.5
            if baseline is None and elapsed >= 2.5:
                baseline = current
            if shielded is None and elapsed >= 6.5:
                shielded = current
            panel.update(phase, grid.total_activity_bq, current, baseline)
            world.step(render=True)

            if baseline is not None and not baseline_captured:
                request_capture(baseline_image)
                baseline_captured = True
            if shielded is not None and not shield_captured:
                request_capture(shield_image)
                shield_captured = True
            second = int(elapsed)
            if second != last_log_second:
                last_log_second = second
                gm_rate = current["gm_east"].expected_count_rate_cps
                nai_rate = current["nai_north"].expected_count_rate_cps
                aimed_rate = current["collimated_aimed"].expected_count_rate_cps
                print(
                    f"DETECTOR_TELEMETRY t={elapsed:.1f} phase={phase!r} "
                    f"activity_mbq={grid.total_activity_bq / 1e6:.3f} "
                    f"gm_east_cps={gm_rate:.3f} nai_north_cps={nai_rate:.3f} "
                    f"aimed_cps={aimed_rate:.3f}",
                    flush=True,
                )
            next_frame += DT_S
            sleep_s = next_frame - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        final = array.measure(
            particle_transport.transport(
                particle_emissions(grid), array.detector_positions
            ),
            integration_time_s=1.0,
            seed=99_999,
        )
        panel.update("verification complete", grid.total_activity_bq, final, baseline)
        request_capture(decon_image)
        for _ in range(40):
            world.step(render=True)
            if decon_image.exists():
                break
        if baseline is None or shielded is None:
            raise RuntimeError("baseline or shield measurement was not captured")
        shield_ratios = ratios(shielded, baseline)
        final_to_shield = ratios(final, shielded)
        aimed_to_away = baseline["collimated_aimed"].expected_count_rate_cps / max(
            baseline["collimated_away"].expected_count_rate_cps, 1e-12
        )
        activity_reduction = 1.0 - grid.total_activity_bq / initial_activity
        passed = (
            shield_ratios["gm_east"] < 0.20
            and shield_ratios["hpge_west"] > 0.85
            and final_to_shield["nai_north"] < 0.75
            and final_to_shield["hpge_west"] < 0.75
            and aimed_to_away > 20.0
            and activity_reduction > 0.30
            and sum(baseline["hpge_west"].spectrum_counts) > 0
            and "custom_czt" in final
            and baseline_image.exists()
            and shield_image.exists()
            and decon_image.exists()
        )
        result = {
            "passed": passed,
            "detector_count": len(array.detector_ids),
            "detector_ids": array.detector_ids,
            "initial_activity_bq": initial_activity,
            "final_activity_bq": grid.total_activity_bq,
            "activity_reduction_fraction": activity_reduction,
            "shield_to_baseline_ratios": shield_ratios,
            "decon_to_shield_ratios": final_to_shield,
            "aimed_to_away_directional_ratio": aimed_to_away,
            "baseline_expected_cps": {
                key: value.expected_count_rate_cps for key, value in baseline.items()
            },
            "shielded_expected_cps": {
                key: value.expected_count_rate_cps for key, value in shielded.items()
            },
            "final_expected_cps": {
                key: value.expected_count_rate_cps for key, value in final.items()
            },
            "hpge_spectrum_total_counts": sum(baseline["hpge_west"].spectrum_counts),
            "baseline_image": str(baseline_image.resolve()),
            "shield_image": str(shield_image.resolve()),
            "decontaminated_image": str(decon_image.resolve()),
        }
        print(
            "DETECTOR_VALIDATION_RESULT " + json.dumps(result, sort_keys=True),
            flush=True,
        )
        return 0 if passed else 2
    except Exception as exc:
        result = {
            "passed": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print("DETECTOR_VALIDATION_ERROR " + json.dumps(result), flush=True)
        return 1
    finally:
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if world is not None:
            world.stop()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
