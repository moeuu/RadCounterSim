#!/usr/bin/env python3
"""Generate the self-contained RadInterAct vertical-slice assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from radcounter.core.surface_decontamination import irregular_deposition_field

ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_PATH = ROOT / "assets/contaminated_objects/floor_activity.npz"
STAGE_PATH = ROOT / "assets/environments/radcounter_vertical_slice.usda"
CONFIG_PATH = ROOT / "configs/scenarios/vertical_slice.runtime.json"
TREATMENT_MODEL_PATH = ROOT / "configs/decontamination/concrete_surface.synthetic.yaml"
TREATMENT_MODEL_URI = "../../configs/decontamination/concrete_surface.synthetic.yaml"


def _irregular_floor_source() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical visible floor triangles and activity for active cells only."""

    cells_x, cells_y = 48, 28
    width_m, height_m = 4.0, 3.0
    field = irregular_deposition_field(cells_x, cells_y).reshape(cells_y, cells_x)
    active_cells = np.argwhere(field > 0.0)
    cell_width = width_m / cells_x
    cell_height = height_m / cells_y
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    raw_activity: list[float] = []
    colors: list[tuple[float, float, float]] = []
    maximum = float(field.max())
    host_color = np.asarray((0.27, 0.29, 0.31), dtype=np.float64)
    for row, column in active_cells:
        x_center = (float(column) + 0.5) * cell_width - width_m * 0.5
        y_center = (float(row) + 0.5) * cell_height - height_m * 0.5
        half_x = cell_width * 0.515
        half_y = cell_height * 0.515
        start = len(points)
        points.extend(
            (
                (x_center - half_x, y_center - half_y, 0.015),
                (x_center + half_x, y_center - half_y, 0.015),
                (x_center + half_x, y_center + half_y, 0.015),
                (x_center - half_x, y_center + half_y, 0.015),
            )
        )
        triangles.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))
        raw_activity.extend((float(field[row, column]) * 0.5,) * 2)
        fraction = float(field[row, column]) / maximum
        contamination_color = np.asarray(
            (0.48 + 0.34 * fraction, 0.08 + 0.07 * fraction, 0.025), dtype=np.float64
        )
        blended = 0.34 * contamination_color + 0.66 * host_color
        colors.extend((tuple(blended), tuple(blended)))
    activity = np.asarray(raw_activity, dtype=np.float64)
    activity *= 8.0e5 / float(activity.sum())
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(triangles, dtype=np.int64),
        activity,
        np.asarray(colors, dtype=np.float64),
    )


def _write_activity_map(activity_bq: np.ndarray) -> str:
    ACTIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ACTIVITY_PATH,
        triangle_indices=np.arange(len(activity_bq), dtype=np.int64),
        activity_bq=activity_bq,
        cumulative_treatment_exposure=np.zeros(len(activity_bq), dtype=np.float64),
        last_treated_step=np.full(len(activity_bq), -1, dtype=np.int64),
        verified_contact_dwell_s=np.zeros(len(activity_bq), dtype=np.float64),
    )
    return hashlib.sha256(ACTIVITY_PATH.read_bytes()).hexdigest()


def _write_runtime_config() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 2,
        "scene": "assets/environments/radcounter_vertical_slice.usda",
        "physics_data_class": "synthetic_validation_only",
        "photon_buildup": {"mode": "primary_only"},
        "materials": {
            "concrete": {
                "energy_keV": [80.0, 300.0, 661.657, 1500.0],
                "mu_m_inv": [34.0, 23.0, 18.0, 12.0],
            },
            "lead": {
                "energy_keV": [80.0, 300.0, 661.657, 1500.0],
                "mu_m_inv": [640.0, 210.0, 125.0, 70.0],
            },
            "steel": {
                "energy_keV": [80.0, 300.0, 661.657, 1500.0],
                "mu_m_inv": [180.0, 82.0, 58.0, 36.0],
            },
        },
        "isotopes": {
            "Cs-137": {"lines": [{"energy_keV": 661.657, "yield_per_decay": 0.851}]},
        },
        "detectors": {
            "gamma-counter": {
                "energy_bin_edges_keV": [0.0, 400.0, 600.0, 1000.0, 2000.0],
                "response_energy_keV": [80.0, 300.0, 661.657, 1500.0],
                "effective_area_m2_per_bin": [
                    [0.00018, 0.0, 0.0, 0.0],
                    [0.00011, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.000075, 0.0],
                    [0.0, 0.0, 0.0, 0.000035],
                ],
                "background_cps_per_bin": [0.5, 0.5, 0.5, 0.5],
                "dead_time_s": 1.0e-6,
            }
        },
        "measurement": {"duration_s": 2.0, "minimum_distance_m": 0.05, "seed": 7},
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _write_stage(
    activity_sha256: str,
    treatment_model_sha256: str,
    points: np.ndarray,
    triangles: np.ndarray,
    colors: np.ndarray,
) -> None:
    STAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    point_text = ",\n            ".join(
        f"({x:.9g}, {y:.9g}, {z:.9g})" for x, y, z in points
    )
    count_text = ", ".join("3" for _ in triangles)
    index_text = ", ".join(str(int(value)) for value in triangles.reshape(-1))
    color_text = ",\n            ".join(
        f"({red:.9g}, {green:.9g}, {blue:.9g})" for red, green, blue in colors
    )
    active_cells = len(triangles) // 2
    deposition_model = "gaussian_lobes_correlated_roughness_holes_satellite_droplets"
    stage = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}

    def Xform "Environment"
    {{
        def Cube "FloorCollision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            custom string rad:material:id = "concrete"
            custom string rad:material:mode = "solid"
            bool physics:collisionEnabled = true
            double size = 2
            double3 xformOp:scale = (6, 4, 0.1)
            double3 xformOp:translate = (0, 0, -0.1)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Cube "WallNorth" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "concrete"
            bool physics:collisionEnabled = true
            double3 xformOp:scale = (6, 0.1, 1.5)
            double3 xformOp:translate = (0, 4, 1.5)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Cube "WallSouth" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "concrete"
            bool physics:collisionEnabled = true
            double3 xformOp:scale = (6, 0.1, 1.5)
            double3 xformOp:translate = (0, -4, 1.5)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Cube "WallEast" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "concrete"
            bool physics:collisionEnabled = true
            double3 xformOp:scale = (0.1, 4, 1.5)
            double3 xformOp:translate = (6, 0, 1.5)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Cube "WallWest" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "concrete"
            bool physics:collisionEnabled = true
            double3 xformOp:scale = (0.1, 4, 1.5)
            double3 xformOp:translate = (-6, 0, 1.5)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
    }}

    def Mesh "ContaminatedFloor" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {{
        custom string rad:role = "contaminated_surface"
        custom string rad:source:type = "surface"
        custom string rad:source:isotopeId = "Cs-137"
        custom string rad:source:activityMapUri = "../contaminated_objects/floor_activity.npz"
        custom string rad:source:activityMapSha256 = "{activity_sha256}"
        custom string rad:source:samplingMode = "centroid"
        custom int rad:source:samplesPerTriangle = 4
        custom int rad:source:maxSamplesPerTriangle = 16
        custom bool rad:source:hiddenFromEstimator = false
        custom bool rad:source:movableWithPrim = false
        custom bool rad:source:enabled = true
        custom bool rad:decon:enabled = true
        custom string rad:decon:activityMapUri = "../contaminated_objects/floor_activity.npz"
        custom string rad:decon:activityMapSha256 = "{activity_sha256}"
        custom string rad:decon:substrateMaterialId = "concrete"
        custom string rad:decon:treatmentModelUri = "{TREATMENT_MODEL_URI}"
        custom string rad:decon:treatmentModelSha256 = "{treatment_model_sha256}"
        custom double rad:decon:minToolDwellS = 0.4
        custom bool rad:source:irregularMask = true
        custom int rad:source:candidateCellCount = 1344
        custom int rad:source:activeCellCount = {active_cells}
        custom int rad:source:activeFaceCount = {len(triangles)}
        custom string rad:source:depositionModel = "{deposition_model}"
        custom string rad:source:geometry = "irregular_masked_triangle_activity_map"
        bool physics:collisionEnabled = true
        int[] faceVertexCounts = [{count_text}]
        int[] faceVertexIndices = [{index_text}]
        point3f[] points = [
            {point_text}
        ]
        color3f[] primvars:displayColor = [
            {color_text}
        ] (
            interpolation = "uniform"
        )
        uniform token subdivisionScheme = "none"
    }}

    def Xform "HiddenContaminatedDrum" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "MassAPI"])
    {{
        custom string rad:role = "source"
        custom string rad:source:type = "point"
        custom string rad:source:isotopeId = "Cs-137"
        custom double rad:source:activityBq = 5.0e7
        custom bool rad:source:hiddenFromEstimator = true
        custom bool rad:source:movableWithPrim = true
        custom bool rad:source:enabled = true
        custom bool rad:manipulation:movable = true
        custom bool rad:manipulation:removable = true
        custom string rad:manipulation:graspFrame = "GraspFrame"
        custom string rad:manipulation:placementReference = "root"
        custom double3 rad:manipulation:parkingOffsetM = (1.6, -1.8, 0)
        custom double rad:manipulation:baseStandOffM = 0.9
        custom string rad:manipulation:disposalClass = "solid_waste"
        bool physics:rigidBodyEnabled = true
        float physics:mass = 35
        double3 xformOp:translate = (2.4, 0, 0.55)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cylinder "Drum" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "steel"
            bool physics:collisionEnabled = true
            double height = 1.0
            double radius = 0.32
        }}
        def Xform "GraspFrame"
        {{
            double3 xformOp:translate = (0, 0, 0.05)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}

    def Xform "LeadShield" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "MassAPI"])
    {{
        custom string rad:role = "shield"
        custom string rad:material:id = "lead"
        custom bool rad:material:containerOnly = true
        custom string rad:material:mode = "solid"
        custom bool rad:shield:movable = true
        custom int rad:shield:resourceUnits = 1
        custom bool rad:manipulation:movable = true
        custom bool rad:manipulation:removable = false
        custom string rad:manipulation:graspFrame = "ShieldGraspFrame"
        bool physics:rigidBodyEnabled = true
        float physics:mass = 80
        double3 xformOp:translate = (4.8, 2.8, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Plate" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "lead"
            custom string rad:material:mode = "solid"
            bool physics:collisionEnabled = true
            double3 xformOp:scale = (0.04, 0.75, 0.9)
            double3 xformOp:translate = (0, 0, 0.9)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Cube "Base" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "lead"
            custom string rad:material:mode = "solid"
            bool physics:collisionEnabled = true
            double3 xformOp:scale = (0.38, 0.8, 0.08)
            double3 xformOp:translate = (0, 0, 0.08)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Xform "ShieldGraspFrame"
        {{
            double3 xformOp:translate = (0, 0, 0.6)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}

    def Xform "MovableObstacle" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "MassAPI"])
    {{
        custom string rad:material:id = "steel"
        custom bool rad:material:containerOnly = true
        custom bool rad:manipulation:movable = true
        custom bool rad:manipulation:removable = false
        custom string rad:manipulation:graspFrame = "ObstacleGraspFrame"
        custom string rad:manipulation:placementReference = "root"
        custom double3 rad:manipulation:parkingOffsetM = (5.2, -1.8, 0)
        custom double rad:manipulation:baseStandOffM = 0.72
        bool physics:rigidBodyEnabled = true
        float physics:mass = 18
        double3 xformOp:translate = (0, 3.0, 0.65)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Body" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "steel"
            custom string rad:material:mode = "solid"
            bool physics:collisionEnabled = true
            double3 xformOp:scale = (0.45, 0.45, 0.65)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }}
        def Cube "Base" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "steel"
            custom string rad:material:mode = "solid"
            bool physics:collisionEnabled = true
            double3 xformOp:scale = (0.65, 0.65, 0.08)
            double3 xformOp:translate = (0, 0, -0.57)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Xform "ObstacleGraspFrame"
        {{
            double3 xformOp:translate = (0, 0, -0.05)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}

    def Xform "MeasurementRobot" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "MassAPI"])
    {{
        custom string rad:role = "measurement_robot"
        bool physics:rigidBodyEnabled = true
        float physics:mass = 22
        double3 xformOp:translate = (-4.5, -2.5, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Base" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            bool physics:collisionEnabled = true
            double3 xformOp:scale = (0.45, 0.32, 0.18)
            double3 xformOp:translate = (0, 0, 0.18)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Xform "Detector"
        {{
            custom string rad:role = "detector"
            custom string rad:detector:id = "gamma-counter"
            double3 xformOp:translate = (0, 0, 0.82)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}

    def Xform "CountermeasureRobot" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "MassAPI"])
    {{
        custom string rad:role = "countermeasure_robot"
        bool physics:rigidBodyEnabled = true
        float physics:mass = 32
        double3 xformOp:translate = (-4.5, 2.5, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Base" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            bool physics:collisionEnabled = true
            double3 xformOp:scale = (0.5, 0.38, 0.22)
            double3 xformOp:translate = (0, 0, 0.22)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Xform "DeconTool"
        {{
            custom string rad:role = "decon_tool"
            custom double3 rad:decon:toolAxis = (0, 0, -1)
            custom double rad:decon:maxContactDistanceM = 0.035
            custom double rad:decon:maxNormalAngleDeg = 25
            custom double rad:decon:maxSurfaceSpeedMS = 0.3
            double3 xformOp:translate = (1.2, 0, 0.6)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "DeconContactTool"
        {{
            custom string rad:role = "decon_tool"
            custom double3 rad:decon:toolAxis = (0, 0, -1)
            custom double rad:decon:maxContactDistanceM = 0.035
            custom double rad:decon:maxNormalAngleDeg = 25
            custom double rad:decon:maxSurfaceSpeedMS = 0.3
            double3 xformOp:translate = (1.0, 0, 0.04)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            def Cube "Pad"
            {{
                color3f[] primvars:displayColor = [(0.15, 0.65, 0.85)]
                double3 xformOp:scale = (0.18, 0.12, 0.02)
                uniform token[] xformOpOrder = ["xformOp:scale"]
            }}
        }}
    }}

    def Xform "DetectorStations"
    {{
        def Xform "Protected"
        {{
            custom string rad:role = "detector_station"
            custom string rad:detector:id = "gamma-counter"
            double3 xformOp:translate = (0, 0, 0.8)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "SouthWest"
        {{
            custom string rad:role = "detector_station"
            custom string rad:detector:id = "gamma-counter"
            double3 xformOp:translate = (-3.2, -2.4, 0.8)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "NorthWest"
        {{
            custom string rad:role = "detector_station"
            custom string rad:detector:id = "gamma-counter"
            double3 xformOp:translate = (-3.2, 2.4, 0.8)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "SouthEast"
        {{
            custom string rad:role = "detector_station"
            custom string rad:detector:id = "gamma-counter"
            double3 xformOp:translate = (3.2, -2.4, 0.8)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}

    def Xform "DisposalStorage"
    {{
        custom string rad:role = "shielded_storage"
        custom string rad:disposal:accessSide = "east"
        custom string rad:material:id = "lead"
        custom bool rad:material:containerOnly = true
        custom string rad:material:mode = "solid"
        double3 xformOp:translate = (-4.6, 2.6, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Bottom" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "lead"
            custom string rad:material:mode = "solid"
            bool physics:collisionEnabled = true
            color3f[] primvars:displayColor = [(0.22, 0.24, 0.28)]
            double3 xformOp:scale = (0.9, 1.3, 0.02)
            double3 xformOp:translate = (0, 0, -0.02)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Cube "WestWall" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "lead"
            custom string rad:material:mode = "solid"
            bool physics:collisionEnabled = true
            color3f[] primvars:displayColor = [(0.22, 0.24, 0.28)]
            double3 xformOp:scale = (0.05, 1.2, 0.7)
            double3 xformOp:translate = (-0.85, 0, 0.7)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Cube "SouthWall" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "lead"
            custom string rad:material:mode = "solid"
            bool physics:collisionEnabled = true
            color3f[] primvars:displayColor = [(0.22, 0.24, 0.28)]
            double3 xformOp:scale = (0.9, 0.05, 0.7)
            double3 xformOp:translate = (0, -1.25, 0.7)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Cube "NorthWall" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string rad:material:id = "lead"
            custom string rad:material:mode = "solid"
            bool physics:collisionEnabled = true
            color3f[] primvars:displayColor = [(0.22, 0.24, 0.28)]
            double3 xformOp:scale = (0.9, 0.05, 0.7)
            double3 xformOp:translate = (0, 1.25, 0.7)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
    }}

    def Cube "DisposalZone"
    {{
        custom string rad:role = "disposal_zone"
        custom string rad:disposal:acceptsClass = "solid_waste"
        custom string rad:disposal:accessSide = "east"
        custom string rad:disposal:disposition = "shielded_storage"
        custom string rad:disposal:storagePrimPath = "/World/DisposalStorage"
        custom bool rad:disposal:outsideEvaluationDomain = false
        color3f[] primvars:displayColor = [(0.12, 0.42, 0.22)]
        double3 xformOp:scale = (0.8, 1.2, 0.02)
        double3 xformOp:translate = (-4.6, 2.6, 0.02)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
}}
'''
    STAGE_PATH.write_text(stage, encoding="utf-8")


def main() -> None:
    points, triangles, activity, colors = _irregular_floor_source()
    digest = _write_activity_map(activity)
    treatment_digest = hashlib.sha256(TREATMENT_MODEL_PATH.read_bytes()).hexdigest()
    _write_runtime_config()
    _write_stage(digest, treatment_digest, points, triangles, colors)
    print(
        json.dumps({"stage": str(STAGE_PATH), "activity_map": str(ACTIVITY_PATH), "sha256": digest})
    )


if __name__ == "__main__":
    main()
