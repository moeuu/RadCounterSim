#!/usr/bin/env python3
"""Generate the self-contained RadCounterSim vertical-slice assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_PATH = ROOT / "assets/contaminated_objects/floor_activity.npz"
STAGE_PATH = ROOT / "assets/environments/radcounter_vertical_slice.usda"
CONFIG_PATH = ROOT / "configs/scenarios/vertical_slice.runtime.json"


def _write_activity_map() -> str:
    ACTIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ACTIVITY_PATH,
        triangle_indices=np.array([0, 1], dtype=np.int64),
        activity_bq=np.array([4.0e5, 4.0e5], dtype=np.float64),
        cumulative_treatment_exposure=np.zeros(2, dtype=np.float64),
        last_treated_step=np.full(2, -1, dtype=np.int64),
    )
    return hashlib.sha256(ACTIVITY_PATH.read_bytes()).hexdigest()


def _write_runtime_config() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "scene": "assets/environments/radcounter_vertical_slice.usda",
        "physics_data_class": "synthetic_validation_only",
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
                "efficiency_energy_keV": [80.0, 300.0, 661.657, 1500.0],
                "intrinsic_efficiency": [0.18, 0.11, 0.075, 0.035],
                "background_cps": 2.0,
                "dead_time_s": 1.0e-6,
            }
        },
        "measurement": {"duration_s": 2.0, "minimum_distance_m": 0.05, "seed": 7},
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _write_stage(activity_sha256: str) -> None:
    STAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
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
        custom bool rad:source:hiddenFromEstimator = false
        custom bool rad:source:movableWithPrim = false
        custom bool rad:source:enabled = true
        custom bool rad:decon:enabled = true
        custom string rad:decon:activityMapUri = "../contaminated_objects/floor_activity.npz"
        custom string rad:decon:activityMapSha256 = "{activity_sha256}"
        custom double rad:decon:efficiencyMean = 0.86
        custom double rad:decon:efficiencyStd = 0.08
        custom double rad:decon:minToolDwellS = 0.4
        bool physics:collisionEnabled = true
        int[] faceVertexCounts = [3, 3]
        int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
        point3f[] points = [
            (-2.0, -1.5, 0.015),
            (2.0, -1.5, 0.015),
            (2.0, 1.5, 0.015),
            (-2.0, 1.5, 0.015)
        ]
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
        custom bool rad:manipulation:movable = true
        custom bool rad:manipulation:removable = false
        custom string rad:manipulation:graspFrame = "ObstacleGraspFrame"
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
            custom double rad:decon:rateConstantSInv = 0.9
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
            custom double rad:decon:rateConstantSInv = 0.9
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

    def Cube "DisposalZone"
    {{
        custom string rad:role = "disposal_zone"
        custom string rad:disposal:acceptsClass = "solid_waste"
        double3 xformOp:scale = (0.8, 0.8, 0.02)
        double3 xformOp:translate = (-4.6, 2.6, 0.02)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
}}
'''
    STAGE_PATH.write_text(stage, encoding="utf-8")


def main() -> None:
    digest = _write_activity_map()
    _write_runtime_config()
    _write_stage(digest)
    print(
        json.dumps({"stage": str(STAGE_PATH), "activity_map": str(ACTIVITY_PATH), "sha256": digest})
    )


if __name__ == "__main__":
    main()
