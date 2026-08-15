"""Real Isaac Sim robot assets and articulated countermeasure motions.

This module deliberately keeps Isaac imports inside runtime methods so the
extension package remains importable by ordinary unit tests.  Task motion is
performed through articulation targets and PhysX constraints; no operation
uses USD transform teleportation after physics starts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FrameCallback = Callable[[float, int], None]
NavigationProgressCallback = Callable[
    [int, tuple[float, float, float], tuple[float, float], float], None
]


@dataclass(frozen=True, slots=True)
class RealRobotAssetConfig:
    countermeasure_asset: str = (
        "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
    )
    measurement_asset: str = "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
    countermeasure_root: str = "/World/CountermeasureRobot"
    measurement_root: str = "/World/MeasurementRobot"
    measurement_articulation: str = "/World/MeasurementRobot/chassis_link"
    detector_path: str = "/World/MeasurementRobot/chassis_link/Detector"
    panda_base_path: str = "/World/CountermeasureRobot/panda_link0"
    panda_hand_path: str = "/World/CountermeasureRobot/panda_hand"
    decon_tool_path: str = (
        "/World/CountermeasureRobot/panda_hand/RadCounterDeconTool/ContactPad"
    )
    shield_path: str = "/World/LeadShield"
    shield_grasp_path: str = "/World/LeadShield/ShieldGraspFrame"
    shield_grasp_offset_m: tuple[float, float, float] = (-0.24, 0.0, 0.59)
    decon_surface_path: str = "/World/DeconWorkSurface"
    decon_workbench_center_m: tuple[float, float, float] = (14.39, 0.80, 1.15)
    shield_initial_position_m: tuple[float, float, float] = (0.78, -0.62, 0.0)


@dataclass(frozen=True, slots=True)
class FacilityRoomSpec:
    """Auditable room bounds for the deterministic validation facility."""

    room_id: str
    display_name: str
    center_m: tuple[float, float, float]
    half_extent_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class FacilityCorridorSpec:
    """One physically connected passage between two declared rooms."""

    corridor_id: str
    from_room_id: str
    to_room_id: str
    center_m: tuple[float, float, float]
    half_extent_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class FacilityPrimitiveSpec:
    """Pure-data primitive used to author and regression-test the facility."""

    path: str
    category: str
    center_m: tuple[float, float, float]
    half_extent_m: tuple[float, float, float]
    color: tuple[float, float, float]
    shape: str = "cube"
    axis: str = "Z"
    collision: bool = True
    room_id: str | None = None
    corridor_id: str | None = None
    equipment_id: str | None = None
    equipment_type: str | None = None
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class FacilityRouteSpec:
    """Reserved robot-base centerline kept clear by the authored layout."""

    route_id: str
    waypoints_m: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class FacilityLayoutManifest:
    """Isaac-independent manifest for a deterministic decommissioning layout."""

    layout_id: str
    root_path: str
    rooms: tuple[FacilityRoomSpec, ...]
    corridors: tuple[FacilityCorridorSpec, ...]
    primitives: tuple[FacilityPrimitiveSpec, ...]
    reserved_routes: tuple[FacilityRouteSpec, ...]
    secondary_shield_path: str
    secondary_shield_position_m: tuple[float, float, float]

    @property
    def equipment_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                primitive.equipment_id
                for primitive in self.primitives
                if primitive.equipment_id is not None
            )
        )


@dataclass(frozen=True, slots=True)
class HandMotionResult:
    success: bool
    steps: int
    target_position_m: tuple[float, float, float]
    final_position_m: tuple[float, float, float]
    position_error_m: float


@dataclass(frozen=True, slots=True)
class DecontaminationMotionReport:
    success: bool
    accepted_contacts: int
    rejected_contacts: int
    removed_activity_bq: float
    activity_before_bq: float
    activity_after_bq: float
    removed_fraction: float
    treated_triangle_indices: tuple[int, ...]
    coverage_fraction: float
    waypoint_errors_m: tuple[float, ...]
    tool_path_length_m: float


@dataclass(frozen=True, slots=True)
class ShieldMotionReport:
    success: bool
    failed_phase: str | None
    grasp_distance_m: float
    finger_aperture_open_m: float
    finger_aperture_closed_m: float
    lift_height_m: float
    placement_error_m: float
    initial_position_m: tuple[float, float, float]
    final_position_m: tuple[float, float, float]
    base_waypoint_errors: tuple[float, ...]
    placement_hand_steps: tuple[int, ...]
    placement_hand_errors_m: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MeasurementMotionReport:
    success: bool
    steps: int
    displacement_m: float
    initial_position_m: tuple[float, float, float]
    final_position_m: tuple[float, float, float]

    @property
    def message(self) -> str:
        return (
            "wheel-joint navigation completed"
            if self.success
            else "wheel-joint navigation failed"
        )

    @property
    def state(self) -> str:
        return "complete" if self.success else "failed"


@dataclass(frozen=True, slots=True)
class ArticulatedTaskReport:
    """Workflow-compatible audit for an arm, gripper, and base task."""

    state: str
    success: bool
    steps: int
    message: str
    object_path: str | None = None
    phases: tuple[str, ...] = ()
    grasp_distance_m: float | None = None
    placement_error_m: float | None = None
    finger_aperture_open_m: float | None = None
    finger_aperture_closed_m: float | None = None
    arm_joint_excursion_rad: float = 0.0
    target_position_m: tuple[float, float, float] | None = None
    release_position_m: tuple[float, float, float] | None = None
    final_position_m: tuple[float, float, float] | None = None
    release_orientation_wxyz: tuple[float, float, float, float] | None = None
    final_orientation_wxyz: tuple[float, float, float, float] | None = None


def enable_real_robot_extensions() -> None:
    """Enable the two optional Isaac extensions used by the real controllers."""

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.robot_motion.motion_generation")
    enable_extension("isaacsim.robot.wheeled_robots")


def create_decontamination_activity_map(
    path: str | Path,
    *,
    triangle_count: int | None = None,
    total_activity_bq: float = 2.0e7,
) -> Path:
    """Create a writable surface map whose rows match the authored work surface."""

    from radcounter.core.surface_decontamination import irregular_deposition_field

    destination = Path(path).expanduser().resolve()
    if triangle_count is not None and triangle_count <= 0:
        raise ValueError("triangle_count must be positive")
    if not math.isfinite(total_activity_bq) or total_activity_bq < 0.0:
        raise ValueError("total_activity_bq must be finite and nonnegative")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if triangle_count is None:
        active_cell_activity = irregular_deposition_field()
        active_cell_activity = active_cell_activity[active_cell_activity > 0.0]
        # The rendered collision mesh has two triangles per active cell.  Split
        # the cell activity between them so truth and visible geometry match.
        raw_weights = np.repeat(active_cell_activity * 0.5, 2)
        triangle_count = len(raw_weights)
    else:
        triangle_indices_for_weights = np.arange(triangle_count, dtype=np.int64)
        raw_weights = 0.12 + np.square(
            np.sin(0.83 + triangle_indices_for_weights.astype(np.float64) * 1.71)
        )
    triangle_indices = np.arange(triangle_count, dtype=np.int64)
    activity_bq = total_activity_bq * raw_weights / float(np.sum(raw_weights))
    np.savez_compressed(
        destination,
        triangle_indices=triangle_indices,
        activity_bq=activity_bq,
        cumulative_treatment_exposure=np.zeros(triangle_count, dtype=np.float64),
        last_treated_step=np.full(triangle_count, -1, dtype=np.int64),
    )
    return destination


def decommissioning_facility_layout() -> FacilityLayoutManifest:
    """Return the deterministic, Isaac-independent complex facility manifest."""

    wall_color = (0.20, 0.27, 0.32)
    branch_color = (0.27, 0.34, 0.38)
    floor_color = (0.18, 0.19, 0.20)
    steel_color = (0.22, 0.27, 0.29)
    equipment_blue = (0.12, 0.31, 0.39)
    service_yellow = (0.90, 0.56, 0.08)

    rooms = (
        FacilityRoomSpec(
            "original_cell",
            "Original reactor service cell",
            (0.0, 0.0, 1.50),
            (6.0, 4.0, 1.50),
        ),
        FacilityRoomSpec(
            "remote_decon_room",
            "Remote wall-decontamination room",
            (12.0, 0.50, 1.50),
            (2.50, 3.0, 1.50),
        ),
        FacilityRoomSpec(
            "reactor_service_room",
            "Reactor auxiliary service room",
            (11.0, 7.50, 1.50),
            (3.50, 2.50, 1.50),
        ),
        FacilityRoomSpec(
            "shield_staging_room",
            "Shield and waste staging room",
            (19.50, 7.50, 1.50),
            (2.50, 2.50, 1.50),
        ),
    )
    corridors = (
        FacilityCorridorSpec(
            "original_to_decon_transfer",
            "original_cell",
            "remote_decon_room",
            (7.75, -1.50, 1.50),
            (1.75, 1.0, 1.50),
        ),
        FacilityCorridorSpec(
            "decon_to_reactor_service",
            "remote_decon_room",
            "reactor_service_room",
            (11.20, 4.25, 1.50),
            (1.0, 0.75, 1.50),
        ),
        FacilityCorridorSpec(
            "reactor_to_shield_staging",
            "reactor_service_room",
            "shield_staging_room",
            (15.75, 8.0, 1.50),
            (1.25, 1.0, 1.50),
        ),
    )

    def equipment(
        equipment_id: str,
        equipment_type: str,
        part_name: str,
        center_m: tuple[float, float, float],
        half_extent_m: tuple[float, float, float],
        color: tuple[float, float, float],
        *,
        room_id: str,
        shape: str = "cube",
        axis: str = "Z",
        display_name: str | None = None,
    ) -> FacilityPrimitiveSpec:
        return FacilityPrimitiveSpec(
            path=f"Equipment/{equipment_id}/{part_name}",
            category="equipment",
            center_m=center_m,
            half_extent_m=half_extent_m,
            color=color,
            shape=shape,
            axis=axis,
            room_id=room_id,
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            display_name=display_name,
        )

    primitives = (
        # Floors retain the original remote-room geometry and add two rooms and
        # two turning passages. Their upper faces stay at z=0, matching the
        # original concrete floor without creating a wheel-catching step.
        FacilityPrimitiveSpec(
            "CorridorFloor",
            "floor",
            (7.75, -1.50, -0.10),
            (1.75, 1.0, 0.10),
            floor_color,
            corridor_id="original_to_decon_transfer",
        ),
        FacilityPrimitiveSpec(
            "DeconRoomFloor",
            "floor",
            (12.0, 0.50, -0.10),
            (2.50, 3.0, 0.10),
            floor_color,
            room_id="remote_decon_room",
        ),
        FacilityPrimitiveSpec(
            "ShieldServiceAlcoveFloor",
            "floor",
            (9.25, 0.50, -0.10),
            (0.25, 0.90, 0.10),
            floor_color,
            room_id="remote_decon_room",
        ),
        FacilityPrimitiveSpec(
            "ServiceAccessFloor",
            "floor",
            (11.20, 4.25, -0.10),
            (1.0, 0.75, 0.10),
            floor_color,
            corridor_id="decon_to_reactor_service",
        ),
        FacilityPrimitiveSpec(
            "ReactorServiceRoomFloor",
            "floor",
            (11.0, 7.50, -0.10),
            (3.50, 2.50, 0.10),
            floor_color,
            room_id="reactor_service_room",
        ),
        FacilityPrimitiveSpec(
            "ShieldTransferFloor",
            "floor",
            (15.75, 8.0, -0.10),
            (1.25, 1.0, 0.10),
            floor_color,
            corridor_id="reactor_to_shield_staging",
        ),
        FacilityPrimitiveSpec(
            "ShieldStagingRoomFloor",
            "floor",
            (19.50, 7.50, -0.10),
            (2.50, 2.50, 0.10),
            floor_color,
            room_id="shield_staging_room",
        ),
        # Original doorway, transfer corridor, and remote decon room. The
        # remote north wall is split only to expose the new service-room door;
        # its east wall and irregular contamination geometry remain unchanged.
        FacilityPrimitiveSpec(
            "OriginalEastWallNorth",
            "wall",
            (6.0, 1.75, 1.50),
            (0.10, 2.25, 1.50),
            wall_color,
        ),
        FacilityPrimitiveSpec(
            "OriginalEastWallSouth",
            "wall",
            (6.0, -3.25, 1.50),
            (0.10, 0.75, 1.50),
            wall_color,
        ),
        FacilityPrimitiveSpec(
            "CorridorNorthWall",
            "wall",
            (7.75, -0.50, 1.50),
            (1.75, 0.10, 1.50),
            branch_color,
            corridor_id="original_to_decon_transfer",
        ),
        FacilityPrimitiveSpec(
            "CorridorSouthWall",
            "wall",
            (7.75, -2.50, 1.50),
            (1.75, 0.10, 1.50),
            branch_color,
            corridor_id="original_to_decon_transfer",
        ),
        FacilityPrimitiveSpec(
            "DeconRoomWestWallSouthJamb",
            "wall",
            (9.50, -0.35, 1.50),
            (0.10, 0.15, 1.50),
            wall_color,
            room_id="remote_decon_room",
        ),
        FacilityPrimitiveSpec(
            "DeconRoomWestWallNorth",
            "wall",
            (9.50, 2.35, 1.50),
            (0.10, 1.15, 1.50),
            wall_color,
            room_id="remote_decon_room",
        ),
        FacilityPrimitiveSpec(
            "ShieldServiceAlcoveWestWall",
            "wall",
            (9.0, 0.50, 1.50),
            (0.10, 0.90, 1.50),
            branch_color,
            room_id="remote_decon_room",
        ),
        FacilityPrimitiveSpec(
            "ShieldServiceAlcoveSouthWall",
            "wall",
            (9.25, -0.40, 1.50),
            (0.25, 0.10, 1.50),
            branch_color,
            room_id="remote_decon_room",
        ),
        FacilityPrimitiveSpec(
            "ShieldServiceAlcoveNorthWall",
            "wall",
            (9.25, 1.40, 1.50),
            (0.25, 0.10, 1.50),
            branch_color,
            room_id="remote_decon_room",
        ),
        FacilityPrimitiveSpec(
            "DeconRoomNorthWallWest",
            "wall",
            (9.85, 3.50, 1.50),
            (0.35, 0.10, 1.50),
            wall_color,
            room_id="remote_decon_room",
        ),
        FacilityPrimitiveSpec(
            "DeconRoomNorthWallEast",
            "wall",
            (13.35, 3.50, 1.50),
            (1.15, 0.10, 1.50),
            wall_color,
            room_id="remote_decon_room",
        ),
        FacilityPrimitiveSpec(
            "DeconRoomSouthWall",
            "wall",
            (12.0, -2.50, 1.50),
            (2.50, 0.10, 1.50),
            wall_color,
            room_id="remote_decon_room",
        ),
        FacilityPrimitiveSpec(
            "DeconRoomEastWall",
            "wall",
            (14.50, 0.50, 1.50),
            (0.10, 3.0, 1.50),
            wall_color,
            room_id="remote_decon_room",
        ),
        # North service corridor and reactor auxiliary room.
        FacilityPrimitiveSpec(
            "ServiceAccessWestWall",
            "wall",
            (10.20, 4.25, 1.50),
            (0.10, 0.75, 1.50),
            branch_color,
            corridor_id="decon_to_reactor_service",
        ),
        FacilityPrimitiveSpec(
            "ServiceAccessEastWall",
            "wall",
            (12.20, 4.25, 1.50),
            (0.10, 0.75, 1.50),
            branch_color,
            corridor_id="decon_to_reactor_service",
        ),
        FacilityPrimitiveSpec(
            "ReactorServiceWestWall",
            "wall",
            (7.50, 7.50, 1.50),
            (0.10, 2.50, 1.50),
            wall_color,
            room_id="reactor_service_room",
        ),
        FacilityPrimitiveSpec(
            "ReactorServiceNorthWall",
            "wall",
            (11.0, 10.0, 1.50),
            (3.50, 0.10, 1.50),
            wall_color,
            room_id="reactor_service_room",
        ),
        FacilityPrimitiveSpec(
            "ReactorServiceSouthWallWest",
            "wall",
            (8.85, 5.0, 1.50),
            (1.35, 0.10, 1.50),
            wall_color,
            room_id="reactor_service_room",
        ),
        FacilityPrimitiveSpec(
            "ReactorServiceSouthWallEast",
            "wall",
            (13.35, 5.0, 1.50),
            (1.15, 0.10, 1.50),
            wall_color,
            room_id="reactor_service_room",
        ),
        FacilityPrimitiveSpec(
            "ReactorServiceEastWallSouth",
            "wall",
            (14.50, 6.0, 1.50),
            (0.10, 1.0, 1.50),
            wall_color,
            room_id="reactor_service_room",
        ),
        FacilityPrimitiveSpec(
            "ReactorServiceEastWallNorth",
            "wall",
            (14.50, 9.50, 1.50),
            (0.10, 0.50, 1.50),
            wall_color,
            room_id="reactor_service_room",
        ),
        # Turning passage and shield-staging room.
        FacilityPrimitiveSpec(
            "ShieldTransferSouthWall",
            "wall",
            (15.75, 7.0, 1.50),
            (1.25, 0.10, 1.50),
            branch_color,
            corridor_id="reactor_to_shield_staging",
        ),
        FacilityPrimitiveSpec(
            "ShieldTransferNorthWall",
            "wall",
            (15.75, 9.0, 1.50),
            (1.25, 0.10, 1.50),
            branch_color,
            corridor_id="reactor_to_shield_staging",
        ),
        FacilityPrimitiveSpec(
            "ShieldStagingWestWallSouth",
            "wall",
            (17.0, 6.0, 1.50),
            (0.10, 1.0, 1.50),
            wall_color,
            room_id="shield_staging_room",
        ),
        FacilityPrimitiveSpec(
            "ShieldStagingWestWallNorth",
            "wall",
            (17.0, 9.50, 1.50),
            (0.10, 0.50, 1.50),
            wall_color,
            room_id="shield_staging_room",
        ),
        FacilityPrimitiveSpec(
            "ShieldStagingSouthWall",
            "wall",
            (19.50, 5.0, 1.50),
            (2.50, 0.10, 1.50),
            wall_color,
            room_id="shield_staging_room",
        ),
        FacilityPrimitiveSpec(
            "ShieldStagingNorthWall",
            "wall",
            (19.50, 10.0, 1.50),
            (2.50, 0.10, 1.50),
            wall_color,
            room_id="shield_staging_room",
        ),
        FacilityPrimitiveSpec(
            "ShieldStagingEastWall",
            "wall",
            (22.0, 7.50, 1.50),
            (0.10, 2.50, 1.50),
            wall_color,
            room_id="shield_staging_room",
        ),
        # Reactor-service equipment. Composite entries share one equipment_id,
        # so audit counts represent actual equipment, not the number of meshes.
        equipment(
            "reactor_service_vessel",
            "reactor_service_vessel",
            "Body",
            (8.50, 8.45, 1.15),
            (0.55, 0.55, 1.15),
            steel_color,
            room_id="reactor_service_room",
            shape="cylinder",
            display_name="Drained reactor service vessel",
        ),
        equipment(
            "reactor_service_vessel",
            "reactor_service_vessel",
            "Pedestal",
            (8.50, 8.45, 0.12),
            (0.72, 0.72, 0.12),
            (0.12, 0.14, 0.15),
            room_id="reactor_service_room",
        ),
        equipment(
            "primary_loop_pipe_rack",
            "process_pipe_rack",
            "Header",
            (9.0, 6.35, 2.15),
            (0.10, 0.78, 0.10),
            equipment_blue,
            room_id="reactor_service_room",
            shape="cylinder",
            axis="Y",
            display_name="Isolated primary-loop pipe rack",
        ),
        equipment(
            "primary_loop_pipe_rack",
            "process_pipe_rack",
            "SupportWest",
            (9.0, 5.72, 1.08),
            (0.08, 0.08, 1.08),
            steel_color,
            room_id="reactor_service_room",
        ),
        equipment(
            "primary_loop_pipe_rack",
            "process_pipe_rack",
            "SupportEast",
            (9.0, 6.98, 1.08),
            (0.08, 0.08, 1.08),
            steel_color,
            room_id="reactor_service_room",
        ),
        equipment(
            "valve_manifold",
            "isolated_valve_manifold",
            "Cabinet",
            (13.55, 6.05, 0.72),
            (0.42, 0.34, 0.72),
            equipment_blue,
            room_id="reactor_service_room",
            display_name="Tagged isolation-valve manifold",
        ),
        equipment(
            "valve_manifold",
            "isolated_valve_manifold",
            "ValveStem",
            (13.55, 6.05, 1.58),
            (0.10, 0.10, 0.18),
            service_yellow,
            room_id="reactor_service_room",
            shape="cylinder",
        ),
        equipment(
            "coolant_pump_skid",
            "drained_coolant_pump",
            "Skid",
            (8.35, 5.55, 0.12),
            (0.62, 0.36, 0.12),
            (0.11, 0.12, 0.13),
            room_id="reactor_service_room",
            display_name="Drained coolant pump skid",
        ),
        equipment(
            "coolant_pump_skid",
            "drained_coolant_pump",
            "Pump",
            (8.35, 5.55, 0.52),
            (0.31, 0.31, 0.40),
            steel_color,
            room_id="reactor_service_room",
            shape="cylinder",
        ),
        equipment(
            "heat_exchanger",
            "isolated_heat_exchanger",
            "Shell",
            (13.15, 9.25, 1.05),
            (0.72, 0.30, 0.30),
            steel_color,
            room_id="reactor_service_room",
            shape="cylinder",
            axis="X",
            display_name="Isolated residual-heat exchanger",
        ),
        equipment(
            "heat_exchanger",
            "isolated_heat_exchanger",
            "Saddle",
            (13.15, 9.25, 0.48),
            (0.62, 0.22, 0.18),
            (0.12, 0.14, 0.15),
            room_id="reactor_service_room",
        ),
        equipment(
            "electrical_switchgear",
            "legacy_switchgear",
            "Cabinet",
            (8.0, 9.48, 0.85),
            (0.30, 0.28, 0.85),
            (0.16, 0.25, 0.20),
            room_id="reactor_service_room",
            display_name="De-energized legacy switchgear",
        ),
        equipment(
            "cable_tray",
            "overhead_cable_tray",
            "Tray",
            (11.25, 9.62, 2.55),
            (1.55, 0.10, 0.08),
            (0.30, 0.24, 0.14),
            room_id="reactor_service_room",
            display_name="Overhead legacy cable tray",
        ),
        equipment(
            "scaffold_tower",
            "maintenance_scaffold",
            "Platform",
            (12.35, 5.55, 1.72),
            (0.40, 0.34, 0.08),
            service_yellow,
            room_id="reactor_service_room",
            display_name="Maintenance scaffold tower",
        ),
        equipment(
            "scaffold_tower",
            "maintenance_scaffold",
            "LegWest",
            (12.05, 5.55, 0.86),
            (0.05, 0.05, 0.86),
            steel_color,
            room_id="reactor_service_room",
        ),
        equipment(
            "scaffold_tower",
            "maintenance_scaffold",
            "LegEast",
            (12.65, 5.55, 0.86),
            (0.05, 0.05, 0.86),
            steel_color,
            room_id="reactor_service_room",
        ),
        # Staging-room inventory, waste, and dismantling debris remain along
        # the room perimeter, leaving the center aisle and shield pickup clear.
        equipment(
            "sealed_waste_drums",
            "sealed_waste_drum_bank",
            "DrumA",
            (17.75, 5.75, 0.45),
            (0.25, 0.25, 0.45),
            (0.43, 0.31, 0.08),
            room_id="shield_staging_room",
            shape="cylinder",
            display_name="Sealed low-level waste drum bank",
        ),
        equipment(
            "sealed_waste_drums",
            "sealed_waste_drum_bank",
            "DrumB",
            (18.35, 5.75, 0.45),
            (0.25, 0.25, 0.45),
            (0.38, 0.28, 0.07),
            room_id="shield_staging_room",
            shape="cylinder",
        ),
        equipment(
            "debris_pallet",
            "segmented_pipe_debris_pallet",
            "Pallet",
            (19.65, 5.65, 0.18),
            (0.70, 0.40, 0.18),
            (0.29, 0.19, 0.10),
            room_id="shield_staging_room",
            display_name="Segmented pipe debris pallet",
        ),
        equipment(
            "debris_pallet",
            "segmented_pipe_debris_pallet",
            "Pipe",
            (19.65, 5.65, 0.46),
            (0.62, 0.09, 0.09),
            steel_color,
            room_id="shield_staging_room",
            shape="cylinder",
            axis="X",
        ),
        equipment(
            "ventilation_duct",
            "temporary_ventilation_duct",
            "Duct",
            (19.45, 9.62, 2.45),
            (1.20, 0.16, 0.16),
            (0.36, 0.38, 0.40),
            room_id="shield_staging_room",
            display_name="Temporary filtered ventilation duct",
        ),
        equipment(
            "sampling_cabinet",
            "radiological_sampling_cabinet",
            "Cabinet",
            (21.35, 5.80, 0.82),
            (0.32, 0.30, 0.82),
            equipment_blue,
            room_id="shield_staging_room",
            display_name="Radiological sampling cabinet",
        ),
        equipment(
            "tool_crib",
            "remote_tool_crib",
            "Crib",
            (17.75, 9.50, 0.72),
            (0.42, 0.30, 0.72),
            (0.18, 0.22, 0.24),
            room_id="shield_staging_room",
            display_name="Remote tooling crib",
        ),
        equipment(
            "shield_storage_rack",
            "shield_storage_rack",
            "Shelf",
            (21.30, 9.55, 1.05),
            (0.42, 0.25, 0.08),
            service_yellow,
            room_id="shield_staging_room",
            display_name="Empty shield storage rack",
        ),
        equipment(
            "shield_storage_rack",
            "shield_storage_rack",
            "PostWest",
            (20.98, 9.55, 0.53),
            (0.05, 0.05, 0.53),
            steel_color,
            room_id="shield_staging_room",
        ),
        equipment(
            "shield_storage_rack",
            "shield_storage_rack",
            "PostEast",
            (21.62, 9.55, 0.53),
            (0.05, 0.05, 0.53),
            steel_color,
            room_id="shield_staging_room",
        ),
    )
    reserved_routes = (
        FacilityRouteSpec(
            "primary_decon_route",
            (
                (0.0, -1.50, 0.28),
                (5.20, -1.50, 0.28),
                (8.90, -1.50, 0.28),
                (10.30, -1.50, 0.28),
                (10.30, 0.80, 0.28),
                (13.49, 0.80, 0.28),
            ),
        ),
        FacilityRouteSpec(
            "staging_room_service_route",
            (
                (10.30, 0.80, 0.28),
                (11.20, 4.25, 0.28),
                (11.20, 7.95, 0.28),
                (15.75, 8.0, 0.28),
                (18.20, 8.20, 0.28),
                (20.15, 8.45, 0.28),
            ),
        ),
        FacilityRouteSpec(
            "primary_shield_25_service_route",
            (
                (3.84, 2.80, 0.28),
                (5.20, -1.50, 0.28),
                (8.90, -1.50, 0.28),
                (10.30, -1.50, 0.28),
                (10.30, 0.42755624655, 0.28),
                (9.8825, 0.42755624655, 0.28),
            ),
        ),
        FacilityRouteSpec(
            "primary_shield_65_service_route",
            (
                (9.8825, 0.42755624655, 0.28),
                (10.30, 0.42755624655, 0.28),
                (10.30, -1.50, 0.28),
                (8.90, -1.50, 0.28),
                (5.20, -1.50, 0.28),
                (4.1265, 0.19952624839, 0.28),
            ),
        ),
    )
    return FacilityLayoutManifest(
        layout_id="four_room_decommissioning_cell_v1",
        root_path="/World/RemoteDeconFacility",
        rooms=rooms,
        corridors=corridors,
        primitives=primitives,
        reserved_routes=reserved_routes,
        secondary_shield_path="/World/StagingLeadShield",
        secondary_shield_position_m=(21.15, 8.45, 0.0),
    )


def facility_route_clearance_m(
    layout: FacilityLayoutManifest,
    route_id: str,
    *,
    sample_spacing_m: float = 0.05,
) -> float:
    """Return planar clearance from a reserved route to walls and equipment."""

    if not math.isfinite(sample_spacing_m) or sample_spacing_m <= 0.0:
        raise ValueError("sample_spacing_m must be finite and positive")
    route = next(
        (item for item in layout.reserved_routes if item.route_id == route_id),
        None,
    )
    if route is None:
        raise KeyError(f"unknown facility route: {route_id}")
    obstacles = tuple(
        primitive
        for primitive in layout.primitives
        if primitive.collision and primitive.category in {"wall", "equipment"}
    )
    minimum = math.inf
    for start, end in zip(
        route.waypoints_m[:-1], route.waypoints_m[1:], strict=True
    ):
        distance = math.dist(start[:2], end[:2])
        sample_count = max(1, int(math.ceil(distance / sample_spacing_m)))
        for sample_index in range(sample_count + 1):
            fraction = sample_index / sample_count
            x = start[0] + fraction * (end[0] - start[0])
            y = start[1] + fraction * (end[1] - start[1])
            for primitive in obstacles:
                lower_x = primitive.center_m[0] - primitive.half_extent_m[0]
                upper_x = primitive.center_m[0] + primitive.half_extent_m[0]
                lower_y = primitive.center_m[1] - primitive.half_extent_m[1]
                upper_y = primitive.center_m[1] + primitive.half_extent_m[1]
                dx = max(lower_x - x, 0.0, x - upper_x)
                dy = max(lower_y - y, 0.0, y - upper_y)
                minimum = min(minimum, math.hypot(dx, dy))
    return minimum


def _custom_attribute(prim: Any, name: str, value_type: Any, value: object) -> None:
    attribute = prim.GetAttribute(name)
    if not attribute:
        attribute = prim.CreateAttribute(name, value_type, custom=True)
    attribute.Set(value)


def _world_pose(stage: Any, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"required prim does not exist: {prim_path}")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    position = np.asarray(matrix.Transform(Gf.Vec3d()), dtype=np.float64)
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    orientation = np.asarray(
        (
            quaternion.GetReal(),
            imaginary[0],
            imaginary[1],
            imaginary[2],
        ),
        dtype=np.float64,
    )
    return position, orientation


def _cube(
    stage: Any,
    path: str,
    *,
    translate: Sequence[float],
    half_scale: Sequence[float],
    color: Sequence[float],
    collision: bool = True,
) -> Any:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, translate)))
    xform.AddScaleOp().Set(Gf.Vec3d(*map(float, half_scale)))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    return cube.GetPrim()


def _cylinder(
    stage: Any,
    path: str,
    *,
    translate: Sequence[float],
    half_extent: Sequence[float],
    axis: str,
    color: Sequence[float],
    collision: bool = True,
) -> Any:
    from pxr import Gf, UsdGeom, UsdPhysics

    half_x, half_y, half_z = map(float, half_extent)
    dimensions = {
        "X": (max(half_y, half_z), 2.0 * half_x),
        "Y": (max(half_x, half_z), 2.0 * half_y),
        "Z": (max(half_x, half_y), 2.0 * half_z),
    }
    try:
        radius, height = dimensions[axis]
    except KeyError as error:
        raise ValueError(f"unsupported cylinder axis: {axis}") from error
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(axis)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    UsdGeom.Xformable(cylinder).AddTranslateOp().Set(
        Gf.Vec3d(*map(float, translate))
    )
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim()).CreateCollisionEnabledAttr(True)
    return cylinder.GetPrim()


def _author_remote_decon_facility(
    stage: Any,
) -> FacilityLayoutManifest:
    """Author four connected rooms while preserving the remote decon geometry."""

    from pxr import Gf, Sdf, UsdGeom

    layout = decommissioning_facility_layout()
    root_path = layout.root_path
    for stale_path in (
        "/World/ComplexFacility",
        "/World/MazeFacility",
        layout.secondary_shield_path,
        root_path,
    ):
        if stage.GetPrimAtPath(stale_path).IsValid():
            stage.RemovePrim(stale_path)
    original_east_wall = "/World/Environment/WallEast"
    if stage.GetPrimAtPath(original_east_wall).IsValid():
        stage.RemovePrim(original_east_wall)
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    root.SetDisplayName("Four-room decommissioning validation facility")
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "decommissioning_facility"),
        ("rad:facility:layoutId", Sdf.ValueTypeNames.String, layout.layout_id),
        ("rad:facility:deterministic", Sdf.ValueTypeNames.Bool, True),
        ("rad:facility:roomCount", Sdf.ValueTypeNames.Int, len(layout.rooms)),
        ("rad:facility:corridorCount", Sdf.ValueTypeNames.Int, len(layout.corridors)),
        (
            "rad:facility:equipmentCount",
            Sdf.ValueTypeNames.Int,
            len(layout.equipment_ids),
        ),
        (
            "rad:facility:obstacleCount",
            Sdf.ValueTypeNames.Int,
            len(layout.equipment_ids),
        ),
        (
            "rad:facility:reservedRouteIds",
            Sdf.ValueTypeNames.StringArray,
            [route.route_id for route in layout.reserved_routes],
        ),
        (
            "rad:facility:deconSurfaceGeometry",
            Sdf.ValueTypeNames.String,
            "same_visible_irregular_collision_activity_mesh",
        ),
    ):
        _custom_attribute(root, name, value_type, value)

    UsdGeom.Scope.Define(stage, f"{root_path}/Rooms")
    for room in layout.rooms:
        marker = UsdGeom.Xform.Define(
            stage, f"{root_path}/Rooms/{room.room_id}"
        ).GetPrim()
        marker.SetDisplayName(room.display_name)
        for name, value_type, value in (
            ("rad:facility:roomId", Sdf.ValueTypeNames.String, room.room_id),
            (
                "rad:facility:centerM",
                Sdf.ValueTypeNames.Double3,
                Gf.Vec3d(*room.center_m),
            ),
            (
                "rad:facility:halfExtentM",
                Sdf.ValueTypeNames.Double3,
                Gf.Vec3d(*room.half_extent_m),
            ),
        ):
            _custom_attribute(marker, name, value_type, value)

    UsdGeom.Scope.Define(stage, f"{root_path}/Corridors")
    for corridor in layout.corridors:
        marker = UsdGeom.Xform.Define(
            stage, f"{root_path}/Corridors/{corridor.corridor_id}"
        ).GetPrim()
        marker.SetDisplayName(corridor.corridor_id.replace("_", " ").title())
        for name, value_type, value in (
            (
                "rad:facility:corridorId",
                Sdf.ValueTypeNames.String,
                corridor.corridor_id,
            ),
            (
                "rad:facility:fromRoomId",
                Sdf.ValueTypeNames.String,
                corridor.from_room_id,
            ),
            (
                "rad:facility:toRoomId",
                Sdf.ValueTypeNames.String,
                corridor.to_room_id,
            ),
            (
                "rad:facility:centerM",
                Sdf.ValueTypeNames.Double3,
                Gf.Vec3d(*corridor.center_m),
            ),
            (
                "rad:facility:halfExtentM",
                Sdf.ValueTypeNames.Double3,
                Gf.Vec3d(*corridor.half_extent_m),
            ),
        ):
            _custom_attribute(marker, name, value_type, value)

    equipment_roots: dict[str, Any] = {}
    for primitive in layout.primitives:
        if primitive.equipment_id is not None:
            equipment_root = equipment_roots.get(primitive.equipment_id)
            if equipment_root is None:
                equipment_root = UsdGeom.Xform.Define(
                    stage,
                    f"{root_path}/Equipment/{primitive.equipment_id}",
                ).GetPrim()
                equipment_roots[primitive.equipment_id] = equipment_root
                equipment_root.SetDisplayName(
                    primitive.display_name
                    or primitive.equipment_id.replace("_", " ").title()
                )
                for name, value_type, value in (
                    ("rad:role", Sdf.ValueTypeNames.String, "facility_equipment"),
                    (
                        "rad:facility:equipmentId",
                        Sdf.ValueTypeNames.String,
                        primitive.equipment_id,
                    ),
                    (
                        "rad:facility:equipmentType",
                        Sdf.ValueTypeNames.String,
                        primitive.equipment_type or "decommissioning_obstacle",
                    ),
                    (
                        "rad:facility:roomId",
                        Sdf.ValueTypeNames.String,
                        primitive.room_id or "",
                    ),
                    ("rad:facility:fixed", Sdf.ValueTypeNames.Bool, True),
                ):
                    _custom_attribute(equipment_root, name, value_type, value)

        path = f"{root_path}/{primitive.path}"
        if primitive.shape == "cube":
            authored = _cube(
                stage,
                path,
                translate=primitive.center_m,
                half_scale=primitive.half_extent_m,
                color=primitive.color,
                collision=primitive.collision,
            )
        elif primitive.shape == "cylinder":
            authored = _cylinder(
                stage,
                path,
                translate=primitive.center_m,
                half_extent=primitive.half_extent_m,
                axis=primitive.axis,
                color=primitive.color,
                collision=primitive.collision,
            )
        else:
            raise ValueError(f"unsupported facility primitive shape: {primitive.shape}")
        _custom_attribute(
            authored,
            "rad:facility:category",
            Sdf.ValueTypeNames.String,
            primitive.category,
        )
        _custom_attribute(
            authored,
            "rad:material:id",
            Sdf.ValueTypeNames.String,
            "steel" if primitive.category == "equipment" else "concrete",
        )
        if primitive.room_id is not None:
            _custom_attribute(
                authored,
                "rad:facility:roomId",
                Sdf.ValueTypeNames.String,
                primitive.room_id,
            )
        if primitive.corridor_id is not None:
            _custom_attribute(
                authored,
                "rad:facility:corridorId",
                Sdf.ValueTypeNames.String,
                primitive.corridor_id,
            )

    # Non-colliding stripes expose all three corridor centerlines in the GUI.
    stripe_specs = (
        (
            "Transfer",
            ((6.45, -1.50), (7.20, -1.50), (7.95, -1.50), (8.70, -1.50), (9.25, -1.50)),
            (0.18, 0.025),
        ),
        (
            "Service",
            ((11.20, 3.78), (11.20, 4.25), (11.20, 4.72)),
            (0.025, 0.14),
        ),
        (
            "Shield",
            ((14.85, 8.0), (15.45, 8.0), (16.05, 8.0), (16.65, 8.0)),
            (0.14, 0.025),
        ),
    )
    for group, points, half_xy in stripe_specs:
        for index, (x_position, y_position) in enumerate(points, start=1):
            stripe = _cube(
                stage,
                f"{root_path}/{group}CorridorStripe{index}",
                translate=(x_position, y_position, 0.006),
                half_scale=(*half_xy, 0.006),
                color=(0.90, 0.66, 0.05),
                collision=False,
            )
            _custom_attribute(
                stripe,
                "rad:facility:category",
                Sdf.ValueTypeNames.String,
                "route_marking",
            )

    remote_station = UsdGeom.Xform.Define(
        stage, "/World/DetectorStations/RemoteDeconRoom"
    )
    remote_station.ClearXformOpOrder()
    remote_station.AddTranslateOp().Set(Gf.Vec3d(11.10, 1.65, 0.80))
    station_prim = remote_station.GetPrim()
    station_prim.SetDisplayName("Remote decontamination room")
    _custom_attribute(station_prim, "rad:role", Sdf.ValueTypeNames.String, "detector_station")
    _custom_attribute(station_prim, "rad:detector:id", Sdf.ValueTypeNames.String, "gamma-counter")
    _custom_attribute(
        station_prim,
        "rad:facility:roomId",
        Sdf.ValueTypeNames.String,
        "remote_decon_room",
    )
    return layout


def _author_staging_lead_shield(
    stage: Any,
    layout: FacilityLayoutManifest,
    *,
    grasp_offset_m: tuple[float, float, float],
) -> None:
    """Author a second movable physical shield without touching the primary."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    path = layout.secondary_shield_path
    shield = UsdGeom.Xform.Define(stage, path)
    shield.AddTranslateOp().Set(Gf.Vec3d(*layout.secondary_shield_position_m))
    prim = shield.GetPrim()
    prim.SetDisplayName("Staged lead service shield 02")
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(2.2)
    # The service panel is explicitly ballasted at its wheeled base.  Without
    # the authored low centre of mass, PhysX derives it mostly from the tall
    # plate volume and the panel can topple after a correct gripper release.
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.10))
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "shield"),
        ("rad:material:id", Sdf.ValueTypeNames.String, "lead"),
        ("rad:material:mode", Sdf.ValueTypeNames.String, "solid"),
        ("rad:shield:movable", Sdf.ValueTypeNames.Bool, True),
        ("rad:shield:resourceUnits", Sdf.ValueTypeNames.Int, 1),
        ("rad:shield:staged", Sdf.ValueTypeNames.Bool, True),
        ("rad:shield:inventoryId", Sdf.ValueTypeNames.String, "staging-shield-02"),
        ("rad:manipulation:movable", Sdf.ValueTypeNames.Bool, True),
        ("rad:manipulation:removable", Sdf.ValueTypeNames.Bool, False),
        (
            "rad:manipulation:graspFrame",
            Sdf.ValueTypeNames.String,
            "ShieldGraspFrame",
        ),
        ("rad:facility:roomId", Sdf.ValueTypeNames.String, "shield_staging_room"),
    ):
        _custom_attribute(prim, name, value_type, value)
    _cube(
        stage,
        path + "/Base",
        translate=(0.0, 0.0, 0.035),
        half_scale=(0.18, 0.30, 0.035),
        color=(0.11, 0.12, 0.13),
    )
    plate = _cube(
        stage,
        path + "/Plate",
        translate=(0.0, 0.0, 0.55),
        half_scale=(0.025, 0.22, 0.35),
        color=(0.20, 0.22, 0.24),
    )
    _custom_attribute(plate, "rad:material:id", Sdf.ValueTypeNames.String, "lead")
    _custom_attribute(plate, "rad:material:mode", Sdf.ValueTypeNames.String, "solid")
    handle_inner_x = -0.025
    handle_outer_x = grasp_offset_m[0] - 0.04
    _cube(
        stage,
        path + "/Handle",
        translate=(
            0.5 * (handle_inner_x + handle_outer_x),
            grasp_offset_m[1],
            grasp_offset_m[2],
        ),
        half_scale=(
            0.5 * abs(handle_inner_x - handle_outer_x),
            0.015,
            0.018,
        ),
        color=(0.92, 0.56, 0.08),
    )
    grasp = UsdGeom.Xform.Define(stage, path + "/ShieldGraspFrame")
    grasp.AddTranslateOp().Set(Gf.Vec3d(*grasp_offset_m))


def add_real_robot_references(
    stage: Any,
    *,
    assets_root: str | None = None,
    config: RealRobotAssetConfig | None = None,
) -> dict[str, str]:
    """Replace placeholder boxes with NVIDIA's official robot USD references."""

    from isaacsim.storage.native import get_assets_root_path
    from pxr import UsdGeom

    cfg = config or RealRobotAssetConfig()
    root = assets_root or get_assets_root_path()
    if not root:
        raise RuntimeError("Isaac Sim assets root is unavailable")
    for prim_path in (cfg.countermeasure_root, cfg.measurement_root):
        if stage.GetPrimAtPath(prim_path).IsValid():
            stage.RemovePrim(prim_path)
    countermeasure = UsdGeom.Xform.Define(stage, cfg.countermeasure_root).GetPrim()
    measurement = UsdGeom.Xform.Define(stage, cfg.measurement_root).GetPrim()
    countermeasure.GetReferences().AddReference(root + cfg.countermeasure_asset)
    measurement.GetReferences().AddReference(root + cfg.measurement_asset)
    return {
        "assets_root": root,
        "countermeasure_usd": root + cfg.countermeasure_asset,
        "measurement_usd": root + cfg.measurement_asset,
    }


def author_real_robot_task_scene(
    stage: Any,
    activity_map_path: str | Path,
    *,
    config: RealRobotAssetConfig | None = None,
) -> None:
    """Attach detector/tool hardware and author physically plausible task props."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    cfg = config or RealRobotAssetConfig()
    required = (
        cfg.countermeasure_root,
        cfg.measurement_articulation,
        cfg.panda_hand_path,
    )
    missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"real robot references are not composed: {missing}")

    # The legacy floor patch is replaced by the reachable workbench below.  It
    # remains in the stage for provenance, but cannot participate in treatment
    # or radiation after the high-fidelity task scene is composed.
    legacy_surface = stage.GetPrimAtPath("/World/ContaminatedFloor")
    legacy_collision = legacy_surface.GetAttribute("physics:collisionEnabled")
    if legacy_collision:
        legacy_collision.Set(False)
    for attribute_name in ("rad:source:enabled", "rad:decon:enabled"):
        attribute = legacy_surface.GetAttribute(attribute_name)
        if attribute:
            attribute.Set(False)

    countermeasure = stage.GetPrimAtPath(cfg.countermeasure_root)
    measurement = stage.GetPrimAtPath(cfg.measurement_root)
    _custom_attribute(countermeasure, "rad:role", Sdf.ValueTypeNames.String, "countermeasure_robot")
    _custom_attribute(measurement, "rad:role", Sdf.ValueTypeNames.String, "measurement_robot")
    for prim, model, dofs, controller in (
        (
            countermeasure,
            "Clearpath Ridgeback + Franka Emika Panda",
            12,
            "holonomic articulation + Lula IK + finger joints",
        ),
        (
            measurement,
            "NVIDIA Nova Carter",
            7,
            "differential wheel articulation",
        ),
    ):
        _custom_attribute(prim, "rad:robot:model", Sdf.ValueTypeNames.String, model)
        _custom_attribute(prim, "rad:robot:dofCount", Sdf.ValueTypeNames.Int, dofs)
        _custom_attribute(
            prim,
            "rad:robot:controller",
            Sdf.ValueTypeNames.String,
            controller,
        )
        _custom_attribute(
            prim,
            "rad:robot:geometryFidelity",
            Sdf.ValueTypeNames.String,
            "manufacturer_asset",
        )

    # The Ridgeback asset represents its holonomic base with x/y/yaw joints.
    # Increase their physical drives before the timeline starts so the base can
    # overcome contact friction on the concrete validation floor.
    base_drives = (
        (
            cfg.countermeasure_root + "/world/dummy_base_prismatic_x_joint",
            "linear",
            8.0e4,
            1.2e4,
            2.5e5,
        ),
        (
            cfg.countermeasure_root + "/dummy_base_x/dummy_base_prismatic_y_joint",
            "linear",
            8.0e4,
            1.2e4,
            2.5e5,
        ),
        (
            cfg.countermeasure_root + "/dummy_base_y/dummy_base_revolute_z_joint",
            "angular",
            5.0e4,
            8.0e3,
            1.0e5,
        ),
    )
    for joint_path, drive_name, stiffness, damping, maximum_force in base_drives:
        joint = stage.GetPrimAtPath(joint_path)
        if not joint or not joint.IsValid():
            raise RuntimeError(f"Ridgeback base joint is missing: {joint_path}")
        drive = UsdPhysics.DriveAPI.Apply(joint, drive_name)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(maximum_force)

    detector = UsdGeom.Xform.Define(stage, cfg.detector_path)
    detector.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.72))
    detector_prim = detector.GetPrim()
    _custom_attribute(detector_prim, "rad:role", Sdf.ValueTypeNames.String, "detector")
    _custom_attribute(
        detector_prim,
        "rad:detector:id",
        Sdf.ValueTypeNames.String,
        "gamma-counter",
    )
    detector_body = UsdGeom.Cylinder.Define(stage, cfg.detector_path + "/GammaDetector")
    detector_body.CreateAxisAttr("Z")
    detector_body.CreateRadiusAttr(0.11)
    detector_body.CreateHeightAttr(0.28)
    detector_body.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.68, 0.08)])

    tool_root_path = cfg.decon_tool_path.rsplit("/", 1)[0]
    UsdGeom.Xform.Define(stage, tool_root_path)
    contact = UsdGeom.Xform.Define(stage, cfg.decon_tool_path)
    contact.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.22))
    contact_prim = contact.GetPrim()
    _custom_attribute(contact_prim, "rad:role", Sdf.ValueTypeNames.String, "decon_tool")
    _custom_attribute(
        contact_prim,
        "rad:decon:toolAxis",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    _cube(
        stage,
        tool_root_path + "/ToolShaft",
        translate=(0.0, 0.0, 0.11),
        half_scale=(0.025, 0.025, 0.11),
        color=(0.08, 0.24, 0.34),
        collision=False,
    )
    _cube(
        stage,
        cfg.decon_tool_path + "/Pad",
        translate=(0.0, 0.0, 0.0),
        half_scale=(0.095, 0.075, 0.012),
        color=(0.05, 0.78, 0.78),
        collision=False,
    )

    for path in (
        "/World/DeconWorkbench",
        cfg.decon_surface_path,
        cfg.shield_path,
    ):
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
    from radcounter.core.surface_decontamination import irregular_deposition_field

    surface_x, surface_y, surface_z = cfg.decon_workbench_center_m
    cells_y, cells_z = 48, 28
    source_width_m, source_height_m = 0.72, 0.56
    field = irregular_deposition_field(cells_y, cells_z).reshape(cells_z, cells_y)
    active_cells = np.argwhere(field > 0.0)
    cell_width_m = source_width_m / cells_y
    cell_height_m = source_height_m / cells_z
    points: list[Gf.Vec3f] = []
    indices: list[int] = []
    colors: list[Gf.Vec3f] = []
    maximum_activity = max(float(field.max()), 1e-12)
    wall_color = np.asarray((0.20, 0.27, 0.32), dtype=np.float64)
    for row, column in active_cells:
        y_center = surface_y + (
            (float(column) + 0.5) * cell_width_m - source_width_m * 0.5
        )
        z_center = surface_z + (
            (float(row) + 0.5) * cell_height_m - source_height_m * 0.5
        )
        half_y = cell_width_m * 0.515
        half_z = cell_height_m * 0.515
        start = len(points)
        points.extend(
            (
                Gf.Vec3f(surface_x, y_center - half_y, z_center - half_z),
                Gf.Vec3f(surface_x, y_center + half_y, z_center - half_z),
                Gf.Vec3f(surface_x, y_center + half_y, z_center + half_z),
                Gf.Vec3f(surface_x, y_center - half_y, z_center + half_z),
            )
        )
        # Clockwise winding from the room gives both triangles a -X normal,
        # matching a tool that approaches the east wall from inside the room.
        indices.extend(
            (start, start + 2, start + 1, start, start + 3, start + 2)
        )
        fraction = float(field[row, column]) / maximum_activity
        hot_overlay = np.asarray(
            (0.44 + 0.42 * fraction, 0.055 + 0.08 * fraction, 0.018),
            dtype=np.float64,
        )
        color = 0.28 * hot_overlay + 0.72 * wall_color
        colors.extend((Gf.Vec3f(*color), Gf.Vec3f(*color)))
    surface = UsdGeom.Mesh.Define(stage, cfg.decon_surface_path)
    surface.CreatePointsAttr(points)
    surface.CreateFaceVertexCountsAttr([3] * (2 * len(active_cells)))
    surface.CreateFaceVertexIndicesAttr(indices)
    surface.CreateSubdivisionSchemeAttr("none")
    display_color = surface.CreateDisplayColorAttr(colors)
    display_color.SetMetadata("interpolation", UsdGeom.Tokens.uniform)
    UsdPhysics.CollisionAPI.Apply(surface.GetPrim()).CreateCollisionEnabledAttr(True)
    surface_prim = surface.GetPrim()
    activity_path = Path(activity_map_path).resolve()
    digest = hashlib.sha256(activity_path.read_bytes()).hexdigest()
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "contaminated_surface"),
        ("rad:source:type", Sdf.ValueTypeNames.String, "surface"),
        ("rad:source:isotopeId", Sdf.ValueTypeNames.String, "Cs-137"),
        ("rad:source:activityMapUri", Sdf.ValueTypeNames.String, str(activity_path)),
        ("rad:source:activityMapSha256", Sdf.ValueTypeNames.String, digest),
        ("rad:source:hiddenFromEstimator", Sdf.ValueTypeNames.Bool, False),
        ("rad:source:enabled", Sdf.ValueTypeNames.Bool, True),
        ("rad:decon:enabled", Sdf.ValueTypeNames.Bool, True),
        ("rad:decon:activityMapUri", Sdf.ValueTypeNames.String, str(activity_path)),
        ("rad:decon:activityMapSha256", Sdf.ValueTypeNames.String, digest),
        ("rad:decon:efficiencyMean", Sdf.ValueTypeNames.Double, 0.86),
        ("rad:decon:efficiencyStd", Sdf.ValueTypeNames.Double, 0.08),
        ("rad:decon:minToolDwellS", Sdf.ValueTypeNames.Double, 0.4),
        ("rad:source:irregularMask", Sdf.ValueTypeNames.Bool, True),
        ("rad:source:candidateCellCount", Sdf.ValueTypeNames.Int, cells_y * cells_z),
        ("rad:source:activeCellCount", Sdf.ValueTypeNames.Int, len(active_cells)),
        ("rad:source:activeFaceCount", Sdf.ValueTypeNames.Int, 2 * len(active_cells)),
        (
            "rad:source:depositionModel",
            Sdf.ValueTypeNames.String,
            "gaussian_lobes_correlated_roughness_holes_satellite_droplets",
        ),
        ("rad:decon:rasterRows", Sdf.ValueTypeNames.Int, 6),
    ):
        _custom_attribute(surface_prim, name, value_type, value)
    surface_prim.SetDisplayName("Cs-137 wall-mounted planar contamination source")
    _custom_attribute(
        surface_prim,
        "rad:source:geometry",
        Sdf.ValueTypeNames.String,
        "irregular_masked_triangle_activity_map",
    )
    _custom_attribute(
        surface_prim,
        "rad:decon:surfaceOrientation",
        Sdf.ValueTypeNames.String,
        "vertical_x",
    )
    facility_layout = _author_remote_decon_facility(stage)

    shield = UsdGeom.Xform.Define(stage, cfg.shield_path)
    shield.AddTranslateOp().Set(Gf.Vec3d(*cfg.shield_initial_position_m))
    shield_prim = shield.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(shield_prim).CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(shield_prim)
    mass.CreateMassAttr(2.2)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.10))
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "shield"),
        ("rad:material:id", Sdf.ValueTypeNames.String, "lead"),
        ("rad:material:mode", Sdf.ValueTypeNames.String, "solid"),
        ("rad:shield:movable", Sdf.ValueTypeNames.Bool, True),
        ("rad:shield:resourceUnits", Sdf.ValueTypeNames.Int, 1),
        ("rad:manipulation:movable", Sdf.ValueTypeNames.Bool, True),
        ("rad:manipulation:removable", Sdf.ValueTypeNames.Bool, False),
        ("rad:manipulation:graspFrame", Sdf.ValueTypeNames.String, "ShieldGraspFrame"),
    ):
        _custom_attribute(shield_prim, name, value_type, value)
    _cube(
        stage,
        cfg.shield_path + "/Base",
        translate=(0.0, 0.0, 0.035),
        half_scale=(0.18, 0.30, 0.035),
        color=(0.11, 0.12, 0.13),
    )
    plate = _cube(
        stage,
        cfg.shield_path + "/Plate",
        translate=(0.0, 0.0, 0.55),
        half_scale=(0.025, 0.22, 0.35),
        color=(0.20, 0.22, 0.24),
    )
    _custom_attribute(plate, "rad:material:id", Sdf.ValueTypeNames.String, "lead")
    _custom_attribute(plate, "rad:material:mode", Sdf.ValueTypeNames.String, "solid")
    # Extend the service handle from the west face of the plate far enough for
    # the Franka wrist to descend without striking the plate.  The grasp frame
    # sits near the outer end while the same rigid handle remains connected to
    # the shield body.
    handle_inner_x = -0.025
    handle_outer_x = cfg.shield_grasp_offset_m[0] - 0.04
    handle_center_x = 0.5 * (handle_inner_x + handle_outer_x)
    handle_half_length_x = 0.5 * abs(handle_inner_x - handle_outer_x)
    _cube(
        stage,
        cfg.shield_path + "/Handle",
        translate=(
            handle_center_x,
            cfg.shield_grasp_offset_m[1],
            cfg.shield_grasp_offset_m[2],
        ),
        half_scale=(handle_half_length_x, 0.015, 0.018),
        color=(0.92, 0.56, 0.08),
    )
    grasp = UsdGeom.Xform.Define(stage, cfg.shield_grasp_path)
    grasp.AddTranslateOp().Set(Gf.Vec3d(*cfg.shield_grasp_offset_m))
    _author_staging_lead_shield(
        stage,
        facility_layout,
        grasp_offset_m=cfg.shield_grasp_offset_m,
    )
    UsdGeom.Xform.Define(stage, "/World/DeconCleanTrace")

    # Give every generic manipulation prop a stand-off handle that the Franka
    # can approach from above without intersecting the prop body.  The props
    # stay inside the arm's rated payload; their large visual envelopes
    # exercise collision-aware routing.
    payloads = (
        (
            "/World/HiddenContaminatedDrum",
            "GraspFrame",
            (0.0, -0.50, 0.25),
            2.7,
        ),
        (
            "/World/MovableObstacle",
            "ObstacleGraspFrame",
            (-0.72, 0.0, 0.12),
            2.8,
        ),
    )
    for object_path, frame_name, frame_offset, payload_kg in payloads:
        object_prim = stage.GetPrimAtPath(object_path)
        if not object_prim or not object_prim.IsValid():
            raise RuntimeError(f"manipulation prop is missing: {object_path}")
        UsdPhysics.MassAPI.Apply(object_prim).CreateMassAttr(payload_kg)
        _custom_attribute(
            object_prim,
            "rad:manipulation:payloadKg",
            Sdf.ValueTypeNames.Double,
            payload_kg,
        )
        frame_path = f"{object_path}/{frame_name}"
        frame = UsdGeom.Xform.Define(stage, frame_path)
        frame.ClearXformOpOrder()
        frame.AddTranslateOp().Set(Gf.Vec3d(*frame_offset))
        _cube(
            stage,
            object_path + "/ManipulatorHandle",
            translate=frame_offset,
            half_scale=(0.10, 0.015, 0.018),
            color=(0.92, 0.56, 0.08),
            collision=False,
        )
        if object_path == "/World/HiddenContaminatedDrum":
            # Join the outer crossbar to the drum while keeping the Franka
            # wrist clear of the 0.32 m drum radius during vertical approach.
            _cube(
                stage,
                object_path + "/ManipulatorHandleStem",
                translate=(0.0, -0.41, 0.25),
                half_scale=(0.018, 0.09, 0.018),
                color=(0.92, 0.56, 0.08),
                collision=False,
            )


class RidgebackFrankaController:
    """Control every Ridgeback+Franka DOF through one Isaac articulation."""

    base_joint_names = (
        "dummy_base_prismatic_x_joint",
        "dummy_base_prismatic_y_joint",
        "dummy_base_revolute_z_joint",
    )
    arm_joint_names = tuple(f"panda_joint{index}" for index in range(1, 8))
    finger_joint_names = ("panda_finger_joint1", "panda_finger_joint2")
    downward_orientation_wxyz = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float64)

    def __init__(
        self,
        stage: Any,
        stepper: Any,
        *,
        config: RealRobotAssetConfig | None = None,
        articulation: Any | None = None,
    ) -> None:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.robot_motion.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
        )
        from isaacsim.robot_motion.motion_generation.interface_config_loader import (
            load_supported_lula_kinematics_solver_config,
        )

        self.stage = stage
        self.stepper = stepper
        self.config = config or RealRobotAssetConfig()
        self.robot = articulation or SingleArticulation(
            self.config.countermeasure_root, name="radcounter_ridgeback_franka"
        )
        if articulation is None:
            self.robot.initialize()
        self.controller = self.robot.get_articulation_controller()
        self.controller.switch_control_mode(mode="position")
        self.dof_names = tuple(self.robot.dof_names)
        expected = set(self.base_joint_names + self.arm_joint_names + self.finger_joint_names)
        missing = sorted(expected.difference(self.dof_names))
        if missing:
            raise RuntimeError(f"Ridgeback+Franka DOFs are missing: {missing}")
        self.indices = {name: self.robot.get_dof_index(name) for name in expected}
        lula_config = load_supported_lula_kinematics_solver_config("Franka")
        self._lula = LulaKinematicsSolver(**lula_config)
        self._solver = ArticulationKinematicsSolver(
            self.robot,
            self._lula,
            "right_gripper",
        )
        arm = self._arm_positions()
        self._arm_min = arm.copy()
        self._arm_max = arm.copy()
        self._grasp_joint_path = (
            self.config.panda_hand_path + "/RadCounterShieldGraspJoint"
        )
        self._handle_collision_enabled = True
        self._grasp_collision_state: dict[str, bool] = {}
        self._grasped_object_path: str | None = None
        self.last_base_target = np.zeros(3, dtype=np.float64)
        self.last_base_positions = self._positions()[
            [self.indices[name] for name in self.base_joint_names]
        ].copy()
        self.last_base_motion_steps = 0
        self.trace: list[str] = ["idle"]

    @staticmethod
    def _smoothstep(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    def _positions(self) -> np.ndarray:
        return np.asarray(self.robot.get_joint_positions(), dtype=np.float64)

    def _arm_positions(self) -> np.ndarray:
        positions = self._positions()
        return positions[[self.indices[name] for name in self.arm_joint_names]]

    def _observe_arm(self) -> None:
        arm = self._arm_positions()
        self._arm_min = np.minimum(self._arm_min, arm)
        self._arm_max = np.maximum(self._arm_max, arm)

    @property
    def arm_joint_excursion_rad(self) -> float:
        return float(np.max(self._arm_max - self._arm_min))

    def _transition(self, state: str) -> None:
        self.trace.append(state)

    def _step(self, callback: FrameCallback | None, frame: int) -> None:
        self.stepper.step(render=True)
        self._observe_arm()
        if callback is not None:
            callback(1.0 / 60.0, frame)

    def _sync_kinematics_base(self) -> None:
        position, orientation = _world_pose(self.stage, self.config.panda_base_path)
        self._lula.set_robot_base_pose(position, orientation)

    def end_effector_position(self) -> np.ndarray:
        self._sync_kinematics_base()
        position, _ = self._solver.compute_end_effector_pose(position_only=True)
        return np.asarray(position, dtype=np.float64)

    def move_base(
        self,
        target_xy_yaw: Sequence[float],
        *,
        interpolation_steps: int = 90,
        settle_steps: int = 180,
        tolerance: float = 0.025,
    ) -> bool:
        from isaacsim.core.utils.types import ArticulationAction

        target = np.asarray(target_xy_yaw, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError("target_xy_yaw must contain x, y, and yaw")
        indices = np.asarray([self.indices[name] for name in self.base_joint_names], dtype=np.int32)
        self.last_base_target = target.copy()
        start = self._positions()[indices]
        self.last_base_motion_steps = 0
        for frame in range(1, interpolation_steps + 1):
            self.last_base_motion_steps += 1
            fraction = self._smoothstep(frame / interpolation_steps)
            command = start + fraction * (target - start)
            self.controller.apply_action(
                ArticulationAction(joint_positions=command, joint_indices=indices)
            )
            self._step(None, frame)
        for frame in range(1, settle_steps + 1):
            self.last_base_motion_steps += 1
            self.controller.apply_action(
                ArticulationAction(joint_positions=target, joint_indices=indices)
            )
            self._step(None, interpolation_steps + frame)
            if np.linalg.norm(self._positions()[indices] - target) <= tolerance:
                self.last_base_positions = self._positions()[indices].copy()
                return True
        self.last_base_positions = self._positions()[indices].copy()
        return False

    def navigate_to(
        self,
        target_position_m: Sequence[float],
        target_yaw_rad: float | None = None,
    ) -> ArticulatedTaskReport:
        """Move the holonomic base through articulation targets, never teleports."""

        target = np.asarray(target_position_m, dtype=np.float64)
        if target.shape not in {(2,), (3,)}:
            raise ValueError("target_position_m must contain x/y or x/y/z")
        yaw = (
            float(self.last_base_positions[2])
            if target_yaw_rad is None
            else float(target_yaw_rad)
        )
        self._transition("navigating")
        distance = float(np.linalg.norm(target[:2] - self.last_base_positions[:2]))
        interpolation_steps = max(90, min(720, int(math.ceil(distance / 0.025))))
        success = self.move_base(
            (float(target[0]), float(target[1]), yaw),
            interpolation_steps=interpolation_steps,
        )
        state = "complete" if success else "failed"
        self._transition(state)
        return ArticulatedTaskReport(
            state,
            success,
            self.last_base_motion_steps,
            "articulated base target reached" if success else "articulated base target failed",
            phases=("navigate",),
            arm_joint_excursion_rad=self.arm_joint_excursion_rad,
        )

    def navigate_route(
        self,
        route_m: Sequence[Sequence[float]],
        final_yaw_rad: float | None = None,
    ) -> ArticulatedTaskReport:
        phases: list[str] = []
        total_steps = 0
        route = tuple(route_m)
        if not route:
            return ArticulatedTaskReport(
                "failed", False, 0, "articulated base route is empty", phases=()
            )
        for index, waypoint in enumerate(route):
            yaw = final_yaw_rad if index == len(route) - 1 else None
            result = self.navigate_to(waypoint, yaw)
            total_steps += result.steps
            phases.append(f"navigate_{index + 1}")
            if not result.success:
                return ArticulatedTaskReport(
                    "failed",
                    False,
                    total_steps,
                    f"base route failed at waypoint {index + 1}",
                    phases=tuple(phases),
                    arm_joint_excursion_rad=self.arm_joint_excursion_rad,
                )
        return ArticulatedTaskReport(
            "complete",
            True,
            total_steps,
            "articulated base route completed",
            phases=tuple(phases),
            arm_joint_excursion_rad=self.arm_joint_excursion_rad,
        )

    def check_reachability(
        self,
        target_position_m: Sequence[float],
        *,
        base_position_m: Sequence[float] | None = None,
    ) -> bool:
        target = np.asarray(target_position_m, dtype=np.float64)
        if base_position_m is None:
            base, _ = _world_pose(self.stage, self.config.panda_base_path)
        else:
            base = np.asarray(base_position_m, dtype=np.float64)
        radial = float(np.linalg.norm(target[:2] - base[:2]))
        vertical = float(target[2] - base[2])
        return radial <= 0.95 and -0.35 <= vertical <= 1.15

    def move_hand(
        self,
        target_position_m: Sequence[float],
        orientation_wxyz: Sequence[float] | None = None,
        *,
        interpolation_steps: int = 75,
        settle_steps: int = 180,
        tolerance_m: float = 0.012,
        callback: FrameCallback | None = None,
    ) -> HandMotionResult:
        from isaacsim.core.utils.types import ArticulationAction

        target = np.asarray(target_position_m, dtype=np.float64)
        orientation = np.asarray(
            self.downward_orientation_wxyz
            if orientation_wxyz is None
            else orientation_wxyz,
            dtype=np.float64,
        )
        self._sync_kinematics_base()
        action, success = self._solver.compute_inverse_kinematics(
            target,
            orientation,
            position_tolerance=tolerance_m / 2.0,
            orientation_tolerance=0.08,
        )
        if not success or action.joint_positions is None or action.joint_indices is None:
            actual = self.end_effector_position()
            return HandMotionResult(
                False,
                0,
                tuple(map(float, target)),
                tuple(map(float, actual)),
                float(np.linalg.norm(actual - target)),
            )
        indices = np.asarray(action.joint_indices, dtype=np.int32)
        target_joints = np.asarray(action.joint_positions, dtype=np.float64)
        start = self._positions()[indices]
        frame_number = 0
        for frame in range(1, interpolation_steps + 1):
            frame_number += 1
            fraction = self._smoothstep(frame / interpolation_steps)
            command = start + fraction * (target_joints - start)
            self.controller.apply_action(
                ArticulationAction(joint_positions=command, joint_indices=indices)
            )
            self._step(callback, frame_number)
        actual = self.end_effector_position()
        for _ in range(settle_steps):
            if np.linalg.norm(actual - target) <= tolerance_m:
                break
            frame_number += 1
            self.controller.apply_action(action)
            self._step(callback, frame_number)
            actual = self.end_effector_position()
        error = float(np.linalg.norm(actual - target))
        return HandMotionResult(
            error <= tolerance_m,
            frame_number,
            tuple(map(float, target)),
            tuple(map(float, actual)),
            error,
        )

    def hold(self, frames: int, callback: FrameCallback | None = None) -> None:
        for frame in range(1, frames + 1):
            self._step(callback, frame)

    def set_gripper(self, finger_position_m: float, *, steps: int = 45) -> float:
        from isaacsim.core.utils.types import ArticulationAction

        target = np.asarray((finger_position_m, finger_position_m), dtype=np.float64)
        indices = np.asarray(
            [self.indices[name] for name in self.finger_joint_names], dtype=np.int32
        )
        start = self._positions()[indices]
        for frame in range(1, steps + 1):
            fraction = self._smoothstep(frame / steps)
            command = start + fraction * (target - start)
            self.controller.apply_action(
                ArticulationAction(joint_positions=command, joint_indices=indices)
            )
            self._step(None, frame)
        return float(np.sum(self._positions()[indices]))

    def stow_arm(
        self,
        *,
        interpolation_steps: int = 75,
        settle_steps: int = 120,
        tolerance_rad: float = 0.035,
    ) -> ArticulatedTaskReport:
        """Fold the Franka into a repeatable collision-safe travel posture."""

        from isaacsim.core.utils.types import ArticulationAction

        target = np.asarray((0.0, -0.60, 0.0, -2.0, 0.0, 1.50, 0.75))
        indices = np.asarray(
            [self.indices[name] for name in self.arm_joint_names], dtype=np.int32
        )
        start = self._positions()[indices]
        self._transition("stowing_arm")
        for frame in range(1, interpolation_steps + 1):
            fraction = self._smoothstep(frame / interpolation_steps)
            command = start + fraction * (target - start)
            self.controller.apply_action(
                ArticulationAction(joint_positions=command, joint_indices=indices)
            )
            self._step(None, frame)
        steps = interpolation_steps
        for frame in range(1, settle_steps + 1):
            error = float(np.max(np.abs(self._positions()[indices] - target)))
            if error <= tolerance_rad:
                self._transition("arm_stowed")
                return ArticulatedTaskReport(
                    "complete",
                    True,
                    steps,
                    "arm folded into travel posture",
                    phases=("stow_arm",),
                    arm_joint_excursion_rad=self.arm_joint_excursion_rad,
                )
            self.controller.apply_action(
                ArticulationAction(joint_positions=target, joint_indices=indices)
            )
            self._step(None, interpolation_steps + frame)
            steps += 1
        self._transition("failed")
        return ArticulatedTaskReport(
            "failed",
            False,
            steps,
            "arm failed to reach travel posture",
            phases=("stow_arm",),
            arm_joint_excursion_rad=self.arm_joint_excursion_rad,
        )

    def set_decon_tool_visible(self, visible: bool) -> None:
        from pxr import UsdGeom

        root_path = self.config.decon_tool_path.rsplit("/", 1)[0]
        imageable = UsdGeom.Imageable(self.stage.GetPrimAtPath(root_path))
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _clean_trace_patch(self, index: int) -> None:
        tool_position, _ = _world_pose(self.stage, self.config.decon_tool_path)
        _cube(
            self.stage,
            f"/World/DeconCleanTrace/Patch{index:02d}",
            translate=(tool_position[0], tool_position[1], 0.655),
            half_scale=(0.09, 0.065, 0.002),
            color=(0.10, 0.72, 0.58),
            collision=False,
        )

    def execute_decontamination(
        self,
        decontaminator: Any,
        hand_waypoints_m: Sequence[Sequence[float]],
        *,
        dwell_frames: int = 32,
        orientation_wxyz: Sequence[float] | None = None,
        approach_offset_m: Sequence[float] = (0.0, 0.0, 0.16),
        retreat_offset_m: Sequence[float] = (0.0, 0.0, 0.18),
        author_trace_patches: bool = True,
        approach_tolerance_m: float = 0.025,
        waypoint_tolerance_m: float = 0.025,
        required_coverage_fraction: float = 0.0,
        max_waypoint_speed_m_s: float | None = None,
    ) -> DecontaminationMotionReport:
        if not hand_waypoints_m:
            raise ValueError("at least one decontamination waypoint is required")
        self.set_decon_tool_visible(True)
        activity_before = float(np.sum(decontaminator.activity_bq))
        first = np.asarray(hand_waypoints_m[0], dtype=np.float64)
        approach = first + np.asarray(approach_offset_m, dtype=np.float64)
        approach_result = self.move_hand(
            approach,
            orientation_wxyz,
            tolerance_m=approach_tolerance_m,
        )
        if not approach_result.success:
            return DecontaminationMotionReport(
                False,
                0,
                0,
                0.0,
                activity_before,
                activity_before,
                0.0,
                (),
                0.0,
                (approach_result.position_error_m,),
                0.0,
            )
        accepted = 0
        rejected = 0
        removed = 0.0
        triangles: set[int] = set()
        errors: list[float] = []
        simulation_step = 0
        tool_positions: list[np.ndarray] = []

        def treatment_tick(dt_s: float, _: int) -> None:
            nonlocal accepted, rejected, removed, simulation_step
            simulation_step += 1
            result = decontaminator.tick(dt_s, simulation_step)
            accepted += result.accepted_contacts
            rejected += result.rejected_contacts
            removed += result.removed_activity_bq
            triangles.update(result.treated_triangle_indices)
            position, _ = _world_pose(self.stage, self.config.decon_tool_path)
            tool_positions.append(position)

        for index, waypoint in enumerate(hand_waypoints_m):
            contact_before = accepted
            interpolation_steps = 75
            if max_waypoint_speed_m_s is not None:
                if max_waypoint_speed_m_s <= 0.0:
                    raise ValueError("max_waypoint_speed_m_s must be positive")
                distance_m = float(
                    np.linalg.norm(
                        np.asarray(waypoint, dtype=np.float64)
                        - self.end_effector_position()
                    )
                )
                interpolation_steps = max(
                    interpolation_steps,
                    int(math.ceil(distance_m * 60.0 / max_waypoint_speed_m_s)),
                )
            motion = self.move_hand(
                waypoint,
                orientation_wxyz,
                interpolation_steps=interpolation_steps,
                tolerance_m=waypoint_tolerance_m,
                callback=treatment_tick,
            )
            errors.append(motion.position_error_m)
            if not motion.success:
                decontaminator.flush()
                return DecontaminationMotionReport(
                    False,
                    accepted,
                    rejected,
                    removed,
                    activity_before,
                    float(np.sum(decontaminator.activity_bq)),
                    max(0.0, removed / activity_before) if activity_before > 0.0 else 0.0,
                    tuple(sorted(triangles)),
                    len(triangles) / max(len(decontaminator.triangle_indices), 1),
                    tuple(errors),
                    0.0,
                )
            self.hold(dwell_frames, treatment_tick)
            if author_trace_patches and accepted > contact_before:
                self._clean_trace_patch(index)
        retreat = np.asarray(hand_waypoints_m[-1], dtype=np.float64) + np.asarray(
            retreat_offset_m, dtype=np.float64
        )
        self.move_hand(retreat, orientation_wxyz)
        decontaminator.flush()
        path_length = 0.0
        for first_position, second_position in zip(
            tool_positions, tool_positions[1:], strict=False
        ):
            path_length += float(np.linalg.norm(second_position - first_position))
        activity_after = float(np.sum(decontaminator.activity_bq))
        removed_fraction = (
            max(0.0, (activity_before - activity_after) / activity_before)
            if activity_before > 0.0
            else 0.0
        )
        coverage_fraction = len(triangles) / max(
            len(decontaminator.triangle_indices), 1
        )
        return DecontaminationMotionReport(
            accepted > 0
            and removed > 0.0
            and activity_after < activity_before
            and coverage_fraction >= required_coverage_fraction,
            accepted,
            rejected,
            removed,
            activity_before,
            activity_after,
            removed_fraction,
            tuple(sorted(triangles)),
            coverage_fraction,
            tuple(errors),
            path_length,
        )

    def execute_surface_decontamination(
        self,
        decontaminator: Any,
        duration_s: float,
    ) -> DecontaminationMotionReport:
        """Raster a reachable surface with Lula IK and live contact feedback."""

        from pxr import Usd, UsdGeom

        surface = self.stage.GetPrimAtPath(decontaminator.surface_path)
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        ).ComputeWorldBound(surface).ComputeAlignedRange()
        lower = np.asarray(bounds.GetMin(), dtype=np.float64)
        upper = np.asarray(bounds.GetMax(), dtype=np.float64)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise RuntimeError("decontamination surface has invalid world bounds")
        orientation_name = str(
            surface.GetAttribute("rad:decon:surfaceOrientation").Get() or "horizontal"
        )
        orientation_wxyz: tuple[float, float, float, float] | None = None
        approach_offset = (0.0, 0.0, 0.16)
        retreat_offset = (0.0, 0.0, 0.18)
        author_trace_patches = True
        approach_tolerance_m = 0.025
        waypoint_tolerance_m = 0.025
        required_coverage_fraction = 0.0
        max_waypoint_speed_m_s: float | None = None
        if orientation_name == "vertical_x":
            # Rotate the tool's local +Z treatment axis toward world +X so the
            # pad presses against the far room's east wall.
            half_sqrt = math.sqrt(0.5)
            orientation_wxyz = (half_sqrt, 0.0, half_sqrt, 0.0)
            hand_x = float(lower[0] - 0.139)
            # Match the six-lane boustrophedon scan in the August 6 high-wall
            # reference.  Continuous motion between endpoints treats the real
            # irregular collision faces; it does not raster a hidden rectangle.
            footprint = np.asarray(
                decontaminator.config.footprint_points_local_m,
                dtype=np.float64,
            )
            # With the tool rotated toward +X, local Y spans wall Y and local
            # X spans wall Z.  Keep the pad center one projected half-extent
            # inside the surface: the physical pad still covers the boundary,
            # while Lula is not asked to reach an unnecessarily extreme
            # center pose at the final high-wall corner.
            y_margin = min(
                float(np.max(np.abs(footprint[:, 1]))),
                max(0.0, (upper[1] - lower[1]) * 0.25),
            )
            z_margin = min(
                float(np.max(np.abs(footprint[:, 0]))),
                max(0.0, (upper[2] - lower[2]) * 0.25),
            )
            y_limits = (float(lower[1] + y_margin), float(upper[1] - y_margin))
            z_rows = np.linspace(lower[2] + z_margin, upper[2] - z_margin, 6)
            waypoints = tuple(
                (hand_x, y_position, float(z_position))
                for row, z_position in enumerate(z_rows)
                for y_position in (
                    y_limits if row % 2 == 0 else y_limits[::-1]
                )
            )
            approach_offset = (-0.08, 0.0, 0.0)
            retreat_offset = (-0.12, 0.0, 0.0)
            author_trace_patches = False
            approach_tolerance_m = 0.055
            waypoint_tolerance_m = 0.055
            required_coverage_fraction = 0.35
            max_waypoint_speed_m_s = 0.20
        else:
            margin = np.minimum(
                (upper - lower) * 0.18, np.asarray((0.07, 0.07, 0.0))
            )
            x_values = np.linspace(lower[0] + margin[0], upper[0] - margin[0], 3)
            y_values = (lower[1] + margin[1], upper[1] - margin[1])
            # The proven contact hand height is 0.139 m above a horizontal
            # surface with the downward Lula orientation.
            hand_z = float(upper[2] + 0.139)
            waypoints = tuple(
                (float(x), float(y), hand_z)
                for row, y in enumerate(y_values)
                for x in (x_values if row % 2 == 0 else x_values[::-1])
            )
        dwell_frames = max(6, int(round(max(duration_s, 0.1) * 60.0 / len(waypoints))))
        self._transition("decontaminating")
        report = self.execute_decontamination(
            decontaminator,
            waypoints,
            dwell_frames=dwell_frames,
            orientation_wxyz=orientation_wxyz,
            approach_offset_m=approach_offset,
            retreat_offset_m=retreat_offset,
            author_trace_patches=author_trace_patches,
            approach_tolerance_m=approach_tolerance_m,
            waypoint_tolerance_m=waypoint_tolerance_m,
            required_coverage_fraction=required_coverage_fraction,
            max_waypoint_speed_m_s=max_waypoint_speed_m_s,
        )
        self._transition("complete" if report.success else "failed")
        return report

    @staticmethod
    def _grasp_frame_name(target: Any) -> str:
        attribute = target.GetAttribute("rad:manipulation:graspFrame")
        return str(attribute.Get() or "") if attribute else ""

    def _attach_object(self, object_path: str) -> float:
        """Latch a verified hand/object contact after visible finger closure."""

        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        target = self.stage.GetPrimAtPath(object_path)
        if not target or not target.IsValid():
            raise RuntimeError(f"grasp target does not exist: {object_path}")
        movable = target.GetAttribute("rad:manipulation:movable")
        if not movable or not bool(movable.Get()):
            raise RuntimeError(f"grasp target is not movable: {object_path}")
        payload = target.GetAttribute("rad:manipulation:payloadKg")
        if payload and float(payload.Get() or 0.0) > 3.0:
            raise RuntimeError(f"grasp target exceeds the Franka payload: {payload.Get()} kg")
        frame_name = self._grasp_frame_name(target)
        if not frame_name:
            raise RuntimeError(f"grasp target has no grasp frame: {object_path}")
        grasp_path = f"{object_path.rstrip('/')}/{frame_name}"
        grasp_prim = self.stage.GetPrimAtPath(grasp_path)
        if not grasp_prim or not grasp_prim.IsValid():
            raise RuntimeError(f"grasp frame does not exist: {grasp_path}")
        hand_position = self.end_effector_position()
        grasp_position, _ = _world_pose(self.stage, grasp_path)
        distance = float(np.linalg.norm(hand_position - grasp_position))
        if distance > 0.055:
            raise RuntimeError(f"grasp frame is {distance:.3f} m from the gripper")

        hand = self.stage.GetPrimAtPath(self.config.panda_hand_path)
        cache = UsdGeom.XformCache()
        hand_world = cache.GetLocalToWorldTransform(hand)
        target_world = cache.GetLocalToWorldTransform(target)
        grasp_world = cache.GetLocalToWorldTransform(grasp_prim)
        anchor_world = grasp_world.Transform(Gf.Vec3d())
        hand_anchor = hand_world.GetInverse().Transform(anchor_world)
        target_anchor = target_world.GetInverse().Transform(anchor_world)
        local_rotation = hand_world.ExtractRotationQuat().GetInverse() * (
            target_world.ExtractRotationQuat()
        )
        imaginary = local_rotation.GetImaginary()
        if self.stage.GetPrimAtPath(self._grasp_joint_path).IsValid():
            self.stage.RemovePrim(self._grasp_joint_path)
        joint = UsdPhysics.FixedJoint.Define(self.stage, self._grasp_joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.config.panda_hand_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(object_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*hand_anchor))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*target_anchor))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(float(local_rotation.GetReal()), Gf.Vec3f(*map(float, imaginary)))
        )
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

        self._grasp_collision_state.clear()
        for descendant in Usd.PrimRange(target):
            if descendant.GetName() not in {"Handle", "ManipulatorHandle"}:
                continue
            collision = descendant.GetAttribute("physics:collisionEnabled")
            if collision and collision.HasAuthoredValueOpinion():
                path = str(descendant.GetPath())
                self._grasp_collision_state[path] = bool(collision.Get())
                collision.Set(False)
        self._grasped_object_path = object_path
        self._step(None, 1)
        return distance

    def _restore_grasp_collisions(self) -> None:
        """Restore handle collisions after the open fingers have retracted."""

        for path, enabled in self._grasp_collision_state.items():
            collision = self.stage.GetPrimAtPath(path).GetAttribute(
                "physics:collisionEnabled"
            )
            if collision:
                collision.Set(enabled)
        self._grasp_collision_state.clear()

    def _release_object(self, *, restore_collisions: bool = True) -> None:
        if self._grasped_object_path is None:
            raise RuntimeError("no object is grasped")
        # A fixed joint can retain the last interpolation velocity on the
        # payload even after the hand has reached its final target.  Clear
        # that residual motion before detaching so a tall service shield is
        # released from rest instead of receiving a repeatable tipping
        # impulse.  Gravity and contacts remain active after the joint is
        # removed; this is not a pose teleport or a kinematic placement.
        target = self.stage.GetPrimAtPath(self._grasped_object_path)
        from pxr import Gf

        for attribute_name in ("physics:velocity", "physics:angularVelocity"):
            attribute = target.GetAttribute(attribute_name)
            if attribute:
                attribute.Set(Gf.Vec3f(0.0))
        self.stage.RemovePrim(self._grasp_joint_path)
        if restore_collisions:
            self._restore_grasp_collisions()
        self._grasped_object_path = None
        self._step(None, 1)

    def execute_pick_and_place(
        self,
        object_path: str,
        pickup_base_position_m: Sequence[float],
        placement_base_position_m: Sequence[float],
        pickup_tool_position_m: Sequence[float] | None = None,
        placement_tool_position_m: Sequence[float] | None = None,
        tool_orientation_xyzw: Sequence[float] | None = None,
        before_release: Callable[[], None] | None = None,
        after_release: Callable[[], None] | None = None,
        pickup_base_yaw_rad: float | None = None,
        placement_base_yaw_rad: float | None = None,
        pickup_base_route_m: Sequence[Sequence[float]] | None = None,
        placement_base_route_m: Sequence[Sequence[float]] | None = None,
        target_root_position_m: Sequence[float] | None = None,
        placement_settle_tolerance_m: float = 0.15,
    ) -> ArticulatedTaskReport:
        """Execute a complete base/arm/finger pick-and-place with pose audit."""

        del pickup_tool_position_m, placement_tool_position_m, tool_orientation_xyzw
        self.set_decon_tool_visible(False)
        phases: list[str] = []
        total_steps = 0

        stow = self.stow_arm()
        total_steps += stow.steps
        phases.extend(stow.phases)
        if not stow.success:
            return ArticulatedTaskReport(
                "failed",
                False,
                total_steps,
                stow.message,
                object_path,
                tuple(phases),
                arm_joint_excursion_rad=self.arm_joint_excursion_rad,
            )

        pickup_route = (
            (pickup_base_position_m,)
            if pickup_base_route_m is None or not tuple(pickup_base_route_m)
            else tuple(pickup_base_route_m)
        )
        pickup = self.navigate_route(pickup_route, pickup_base_yaw_rad)
        total_steps += pickup.steps
        phases.extend(pickup.phases)
        if not pickup.success:
            return ArticulatedTaskReport(
                "failed", False, total_steps, "pickup base motion failed", object_path,
                tuple(phases), arm_joint_excursion_rad=self.arm_joint_excursion_rad
            )

        target = self.stage.GetPrimAtPath(object_path)
        frame_name = self._grasp_frame_name(target)
        grasp_path = f"{object_path.rstrip('/')}/{frame_name}"
        object_initial, _ = _world_pose(self.stage, object_path)
        grasp_position, _ = _world_pose(self.stage, grasp_path)
        grasp_offset = grasp_position - object_initial
        open_aperture = self.set_gripper(0.035)
        phases.append("open_gripper")
        approach = grasp_position + np.asarray((0.0, 0.0, 0.14))
        for phase, target_position in (("approach", approach), ("grasp_pose", grasp_position)):
            self._transition(phase)
            motion = self.move_hand(target_position)
            total_steps += motion.steps
            phases.append(phase)
            if not motion.success:
                return ArticulatedTaskReport(
                    "failed",
                    False,
                    total_steps,
                    (
                        f"{phase} IK motion failed: target={motion.target_position_m}, "
                        f"actual={motion.final_position_m}, "
                        f"error={motion.position_error_m:.4f} m"
                    ),
                    object_path,
                    tuple(phases), finger_aperture_open_m=open_aperture,
                    arm_joint_excursion_rad=self.arm_joint_excursion_rad,
                )
        # The authored service handle is 30 mm thick.  Command a 32 mm total
        # aperture so the physical fingers establish contact without trying to
        # crush the handle into an 8 mm gap and launching the rigid payload.
        closed_aperture = self.set_gripper(0.016)
        phases.append("close_gripper")
        self._transition("grasping")
        grasp_distance = self._attach_object(object_path)
        phases.append("attach_at_hand")
        pickup_base = np.asarray(pickup_base_position_m, dtype=np.float64)
        retract_xy = pickup_base[:2] - grasp_position[:2]
        retract_norm = float(np.linalg.norm(retract_xy))
        if retract_norm <= 1.0e-9:
            self._release_object()
            return ArticulatedTaskReport(
                "failed",
                False,
                total_steps,
                "pickup base and grasp frame have no horizontal separation",
                object_path,
                tuple(phases),
                grasp_distance,
                finger_aperture_open_m=open_aperture,
                finger_aperture_closed_m=closed_aperture,
                arm_joint_excursion_rad=self.arm_joint_excursion_rad,
            )
        retract_world = np.asarray(
            (
                0.14 * retract_xy[0] / retract_norm,
                0.14 * retract_xy[1] / retract_norm,
                0.22,
            )
        )
        lift_targets = (
            ("lift_clearance", grasp_position + np.asarray((0.0, 0.0, 0.08))),
            # Retract toward the actual pickup base while raising the load.
            # The sign changes for opposite-side grasps; a fixed world-X
            # offset would extend the loaded arm farther at yaw pi.
            ("lift_transport", grasp_position + retract_world),
        )
        for lift_phase, lift_target in lift_targets:
            lift = self.move_hand(lift_target, tolerance_m=0.025)
            total_steps += lift.steps
            phases.append(lift_phase)
            if not lift.success:
                self._release_object()
                return ArticulatedTaskReport(
                    "failed",
                    False,
                    total_steps,
                    f"loaded {lift_phase} IK motion failed",
                    object_path,
                    tuple(phases),
                    grasp_distance,
                    finger_aperture_open_m=open_aperture,
                    finger_aperture_closed_m=closed_aperture,
                    arm_joint_excursion_rad=self.arm_joint_excursion_rad,
                )

        placement_route = (
            (placement_base_position_m,)
            if placement_base_route_m is None or not tuple(placement_base_route_m)
            else tuple(placement_base_route_m)
        )
        carried = self.navigate_route(placement_route, placement_base_yaw_rad)
        total_steps += carried.steps
        phases.extend(f"carry_{phase}" for phase in carried.phases)
        if not carried.success:
            self._release_object()
            return ArticulatedTaskReport(
                "failed", False, total_steps, "loaded base route failed", object_path,
                tuple(phases), grasp_distance, finger_aperture_open_m=open_aperture,
                finger_aperture_closed_m=closed_aperture,
                arm_joint_excursion_rad=self.arm_joint_excursion_rad,
            )
        if target_root_position_m is None:
            destination_root = np.asarray(placement_base_position_m, dtype=np.float64).copy()
            destination_root[:2] += object_initial[:2] - np.asarray(
                pickup_base_position_m, dtype=np.float64
            )[:2]
            destination_root[2] = object_initial[2]
        else:
            destination_root = np.asarray(target_root_position_m, dtype=np.float64)
        # move_hand() commands a fixed world-space tool orientation. Turning
        # the mobile base only changes the side from which the arm approaches;
        # the attached object returns to its original world orientation during
        # the placement IK motion, so its grasp offset must not follow base yaw.
        destination_grasp = destination_root + grasp_offset
        preplace = destination_grasp + np.asarray((0.0, 0.0, 0.16))
        placement_targets = (
            (preplace, 0.025),
            (destination_grasp + np.asarray((0.0, 0.0, 0.08)), 0.025),
            # Finish 5 mm above the authored support plane.  The former
            # 40 mm release height made a correctly routed tall panel free-
            # fall onto one edge and topple during the settle audit.
            (destination_grasp + np.asarray((0.0, 0.0, 0.005)), 0.012),
        )
        for index, (target_position, tolerance_m) in enumerate(
            placement_targets, start=1
        ):
            motion = self.move_hand(target_position, tolerance_m=tolerance_m)
            total_steps += motion.steps
            phases.append(f"place_{index}")
            if not motion.success:
                self._release_object()
                return ArticulatedTaskReport(
                    "failed",
                    False,
                    total_steps,
                    (
                        "loaded placement IK motion failed: "
                        f"target={motion.target_position_m}, "
                        f"actual={motion.final_position_m}, "
                        f"error={motion.position_error_m:.4f} m"
                    ),
                    object_path, tuple(phases), grasp_distance,
                    finger_aperture_open_m=open_aperture,
                    finger_aperture_closed_m=closed_aperture,
                    arm_joint_excursion_rad=self.arm_joint_excursion_rad,
                )
        if before_release is not None:
            before_release()
        # Let the closed-loop arm and attached payload come fully to rest at
        # the low release pose before opening the fingers.
        self.hold(30)
        phases.append("stabilize_before_release")
        release_position, release_orientation = _world_pose(self.stage, object_path)
        self.set_gripper(0.035)
        self._transition("releasing")
        # Keep the service-handle collision disabled for the few frames in
        # which the open fingers retract. Re-enabling it while the gripper is
        # still co-located with the handle can apply a separation impulse and
        # topple an otherwise stable shield after an accurate placement.
        self._release_object(restore_collisions=False)
        phases.extend(("open_gripper", "release", "retract_after_release", "settle"))
        self.move_hand(preplace)
        self._restore_grasp_collisions()
        self.hold(90)
        if after_release is not None:
            after_release()
        final_position, final_orientation = _world_pose(self.stage, object_path)
        placement_error = float(np.linalg.norm(final_position - destination_root))
        success = placement_error <= placement_settle_tolerance_m
        state = "complete" if success else "failed"
        self._transition(state)
        return ArticulatedTaskReport(
            state,
            success,
            total_steps,
            "articulated pick-and-place completed"
            if success
            else (
                "object did not settle at the commanded pose: "
                f"error={placement_error:.4f} m, "
                f"tolerance={placement_settle_tolerance_m:.4f} m"
            ),
            object_path,
            tuple(phases),
            grasp_distance,
            placement_error,
            open_aperture,
            closed_aperture,
            self.arm_joint_excursion_rad,
            tuple(map(float, destination_root)),
            tuple(map(float, release_position)),
            tuple(map(float, final_position)),
            tuple(map(float, release_orientation)),
            tuple(map(float, final_orientation)),
        )

    def remove_to_disposal_zone(
        self,
        object_path: str,
        disposal_zone_path: str,
    ) -> ArticulatedTaskReport:
        """Secure a physically delivered object and deactivate its source."""

        from pxr import Sdf, UsdGeom

        target = self.stage.GetPrimAtPath(object_path)
        zone = self.stage.GetPrimAtPath(disposal_zone_path)
        cache = UsdGeom.BBoxCache(
            0.0,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        zone_range = cache.ComputeWorldBound(zone).ComputeAlignedRange()
        object_range = cache.ComputeWorldBound(target).ComputeAlignedRange()
        zone_lower = np.asarray(zone_range.GetMin(), dtype=np.float64)
        zone_upper = np.asarray(zone_range.GetMax(), dtype=np.float64)
        object_lower = np.asarray(object_range.GetMin(), dtype=np.float64)
        object_upper = np.asarray(object_range.GetMax(), dtype=np.float64)
        horizontal_margin_m = 0.03
        contained_xy = bool(
            np.all(object_lower[:2] >= zone_lower[:2] - horizontal_margin_m)
            and np.all(object_upper[:2] <= zone_upper[:2] + horizontal_margin_m)
        )
        supported = bool(-0.08 <= object_lower[2] - zone_upper[2] <= 0.18)
        if not contained_xy or not supported:
            return ArticulatedTaskReport(
                "failed",
                False,
                0,
                (
                    "object is not contained by the disposal zone: "
                    f"contained_xy={contained_xy}, supported={supported}"
                ),
                object_path,
            )
        source_enabled = target.GetAttribute("rad:source:enabled")
        if source_enabled:
            source_enabled.Set(False)
        disposed = target.GetAttribute("rad:disposal:disposed")
        if not disposed:
            disposed = target.CreateAttribute(
                "rad:disposal:disposed", Sdf.ValueTypeNames.Bool, custom=True
            )
        disposed.Set(True)
        for name in ("rad:manipulation:movable", "rad:manipulation:removable"):
            attribute = target.GetAttribute(name)
            if attribute:
                attribute.Set(False)
        self._transition("secured_in_disposal")
        return ArticulatedTaskReport(
            "complete",
            True,
            0,
            "object secured and source disabled in disposal zone",
            object_path,
            ("secure_disposal",),
            arm_joint_excursion_rad=self.arm_joint_excursion_rad,
        )

    def _attach_shield(self) -> float:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        hand_position = self.end_effector_position()
        grasp_position, _ = _world_pose(self.stage, self.config.shield_grasp_path)
        distance = float(np.linalg.norm(hand_position - grasp_position))
        if distance > 0.055:
            raise RuntimeError(f"shield grasp frame is {distance:.3f} m from the gripper")
        hand = self.stage.GetPrimAtPath(self.config.panda_hand_path)
        shield = self.stage.GetPrimAtPath(self.config.shield_path)
        cache = UsdGeom.XformCache()
        hand_world = cache.GetLocalToWorldTransform(hand)
        shield_world = cache.GetLocalToWorldTransform(shield)
        grasp_world = cache.GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(self.config.shield_grasp_path)
        )
        anchor_world = grasp_world.Transform(Gf.Vec3d())
        hand_anchor = hand_world.GetInverse().Transform(anchor_world)
        shield_anchor = shield_world.GetInverse().Transform(anchor_world)
        hand_rotation = hand_world.ExtractRotationQuat()
        shield_rotation = shield_world.ExtractRotationQuat()
        local_rotation = hand_rotation.GetInverse() * shield_rotation
        imaginary = local_rotation.GetImaginary()
        joint = UsdPhysics.FixedJoint.Define(self.stage, self._grasp_joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.config.panda_hand_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self.config.shield_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*hand_anchor))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*shield_anchor))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(local_rotation.GetReal()),
                Gf.Vec3f(*map(float, imaginary)),
            )
        )
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        handle = self.stage.GetPrimAtPath(self.config.shield_path + "/Handle")
        collision = handle.GetAttribute("physics:collisionEnabled")
        self._handle_collision_enabled = bool(collision.Get())
        collision.Set(False)
        self._step(None, 1)
        return distance

    def _release_shield(self) -> None:
        self.stage.RemovePrim(self._grasp_joint_path)
        handle = self.stage.GetPrimAtPath(self.config.shield_path + "/Handle")
        handle.GetAttribute("physics:collisionEnabled").Set(self._handle_collision_enabled)
        self._step(None, 1)

    def execute_shield_pick_and_place(
        self,
        *,
        pickup_base_xy_yaw: Sequence[float],
        placement_base_route_xy_yaw: Sequence[Sequence[float]],
        destination_root_position_m: Sequence[float],
        transport_hand_offset_from_grasp_m: Sequence[float] | None = None,
        placement_staging_offsets_m: Sequence[Sequence[float]] = (),
    ) -> ShieldMotionReport:
        self.set_decon_tool_visible(False)
        initial, _ = _world_pose(self.stage, self.config.shield_path)
        if not self.move_base(pickup_base_xy_yaw):
            return ShieldMotionReport(
                False, "pickup_base", math.inf, 0.0, 0.0, 0.0, math.inf,
                tuple(map(float, initial)), tuple(map(float, initial)),
                (float(np.linalg.norm(self.last_base_positions - self.last_base_target)),),
                (),
                (),
            )
        open_aperture = self.set_gripper(0.035)
        grasp_position, _ = _world_pose(self.stage, self.config.shield_grasp_path)
        approach = grasp_position + np.asarray((0.0, 0.0, 0.16))
        approach_motion = self.move_hand(approach)
        grasp_motion = self.move_hand(grasp_position) if approach_motion.success else None
        if not approach_motion.success or grasp_motion is None or not grasp_motion.success:
            return ShieldMotionReport(
                False, "pickup_hand", math.inf, open_aperture, 0.0, 0.0, math.inf,
                tuple(map(float, initial)), tuple(map(float, initial)), ()
                , (), ()
            )
        # Match the 30 mm handle thickness before creating the verified fixed
        # joint; an 8 mm commanded gap causes a large PhysX separation impulse.
        closed_aperture = self.set_gripper(0.016)
        grasp_distance = self._attach_shield()
        lift_target = grasp_position + np.asarray((0.0, 0.0, 0.24))
        if not self.move_hand(lift_target).success:
            self._release_shield()
            return ShieldMotionReport(
                False, "lift", grasp_distance, open_aperture, closed_aperture, 0.0, math.inf,
                tuple(map(float, initial)), tuple(map(float, initial)), ()
                , (), ()
            )
        if transport_hand_offset_from_grasp_m is not None:
            transport_offset = np.asarray(
                transport_hand_offset_from_grasp_m,
                dtype=np.float64,
            )
            if transport_offset.shape != (3,):
                raise ValueError("transport hand offset must contain x/y/z")
            transport_target = grasp_position + transport_offset
            transport = self.move_hand(
                transport_target,
                interpolation_steps=105,
                tolerance_m=0.025,
            )
            if not transport.success:
                self._release_shield()
                final, _ = _world_pose(self.stage, self.config.shield_path)
                return ShieldMotionReport(
                    False,
                    "transport_posture",
                    grasp_distance,
                    open_aperture,
                    closed_aperture,
                    float(final[2] - initial[2]),
                    math.inf,
                    tuple(map(float, initial)),
                    tuple(map(float, final)),
                    (),
                    (transport.steps,),
                    (transport.position_error_m,),
                )
        lifted, _ = _world_pose(self.stage, self.config.shield_path)
        base_errors: list[float] = []
        for waypoint in placement_base_route_xy_yaw:
            if not self.move_base(waypoint):
                base_error = float(
                    np.linalg.norm(self.last_base_positions - self.last_base_target)
                )
                base_errors.append(base_error)
                self._release_shield()
                final, _ = _world_pose(self.stage, self.config.shield_path)
                return ShieldMotionReport(
                    False,
                    f"placement_base_{len(base_errors)}",
                    grasp_distance,
                    open_aperture,
                    closed_aperture,
                    float(lifted[2] - initial[2]),
                    math.inf,
                    tuple(map(float, initial)),
                    tuple(map(float, final)),
                    tuple(base_errors),
                    (),
                    (),
                )
            base_errors.append(
                float(np.linalg.norm(self.last_base_positions - self.last_base_target))
            )
        destination_root = np.asarray(destination_root_position_m, dtype=np.float64)
        release_root = destination_root + np.asarray((0.0, 0.0, 0.06))
        destination_grasp = release_root + np.asarray(
            self.config.shield_grasp_offset_m, dtype=np.float64
        )
        preplace = destination_grasp + np.asarray((0.0, 0.0, 0.12))
        loaded_tolerance_m = 0.025
        placement_motions = []
        for offset_m in placement_staging_offsets_m:
            offset = np.asarray(offset_m, dtype=np.float64)
            if offset.shape != (3,):
                raise ValueError("placement staging offsets must contain x/y/z")
            placement_motions.append(
                self.move_hand(
                    destination_grasp + offset,
                    interpolation_steps=105,
                    tolerance_m=loaded_tolerance_m,
                )
            )
            if not placement_motions[-1].success:
                break
        if not placement_motions or placement_motions[-1].success:
            placement_motions.append(
                self.move_hand(
                    preplace,
                    interpolation_steps=105,
                    tolerance_m=loaded_tolerance_m,
                )
            )
        if placement_motions[-1].success:
            for height_offset in (0.08, 0.04, 0.0):
                placement_motions.append(
                    self.move_hand(
                        destination_grasp + np.asarray((0.0, 0.0, height_offset)),
                        interpolation_steps=105,
                        tolerance_m=loaded_tolerance_m,
                    )
                )
                if not placement_motions[-1].success:
                    break
        if not all(motion.success for motion in placement_motions):
            self._release_shield()
            final, _ = _world_pose(self.stage, self.config.shield_path)
            return ShieldMotionReport(
                False,
                "placement_hand",
                grasp_distance,
                open_aperture,
                closed_aperture,
                float(lifted[2] - initial[2]),
                math.inf,
                tuple(map(float, initial)),
                tuple(map(float, final)),
                tuple(base_errors),
                tuple(motion.steps for motion in placement_motions),
                tuple(motion.position_error_m for motion in placement_motions),
            )
        self.set_gripper(0.035)
        self._release_shield()
        self.hold(90)
        final, _ = _world_pose(self.stage, self.config.shield_path)
        placement_error = float(np.linalg.norm(final - destination_root))
        self.move_hand(preplace)
        return ShieldMotionReport(
            placement_error <= 0.12,
            None if placement_error <= 0.12 else "placement_settle",
            grasp_distance,
            open_aperture,
            closed_aperture,
            float(lifted[2] - initial[2]),
            placement_error,
            tuple(map(float, initial)),
            tuple(map(float, final)),
            tuple(base_errors),
            tuple(motion.steps for motion in placement_motions),
            tuple(motion.position_error_m for motion in placement_motions),
        )


class NovaCarterController:
    """Differential-wheel controller for the official Nova Carter articulation."""

    wheel_names = ("joint_wheel_left", "joint_wheel_right")

    def __init__(
        self,
        stage: Any,
        stepper: Any,
        *,
        config: RealRobotAssetConfig | None = None,
        wheel_radius_m: float = 0.14,
        wheel_base_m: float = 0.413,
        articulation: Any | None = None,
        progress_callback: NavigationProgressCallback | None = None,
    ) -> None:
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot

        self.stage = stage
        self.stepper = stepper
        self.config = config or RealRobotAssetConfig()
        self.wheel_radius_m = wheel_radius_m
        self.wheel_base_m = wheel_base_m
        self.progress_callback = progress_callback
        self.robot = articulation or WheeledRobot(
            prim_path=self.config.measurement_articulation,
            name="radcounter_nova_carter",
            wheel_dof_names=list(self.wheel_names),
        )
        if articulation is None:
            self.robot.initialize()
        self.robot.get_articulation_controller().switch_control_mode(mode="velocity")
        missing = [name for name in self.wheel_names if name not in self.robot.dof_names]
        if missing:
            raise RuntimeError(f"Nova Carter wheel DOFs are missing: {missing}")
        self.trace: list[str] = ["idle"]
        self.home_position_m: tuple[float, float, float] | None = None
        self._home_return_route_m: tuple[tuple[float, float, float], ...] = ()

    @staticmethod
    def _yaw(orientation_wxyz: np.ndarray) -> float:
        w, x, y, z = orientation_wxyz
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrap(value: float) -> float:
        return math.atan2(math.sin(value), math.cos(value))

    def set_initial_pose(
        self,
        position_m: Sequence[float],
        orientation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.robot.set_world_pose(
            position=np.asarray(position_m, dtype=np.float64),
            orientation=np.asarray(orientation_wxyz, dtype=np.float64),
        )
        self.robot.set_linear_velocity(np.zeros(3, dtype=np.float64))
        self.robot.set_angular_velocity(np.zeros(3, dtype=np.float64))
        self.stop()
        for _ in range(20):
            self.stepper.step(render=True)
        home = np.asarray(position_m, dtype=np.float64)
        self.home_position_m = tuple(map(float, home[:3]))
        self._home_return_route_m = ()

    def return_home(self) -> MeasurementMotionReport:
        """Drive back to the explicitly recorded initial position."""

        if self.home_position_m is None:
            raise RuntimeError("measurement robot starting position is not recorded")
        self.trace.append("returning_home")
        if self._home_return_route_m:
            return self.navigate_route(self._home_return_route_m)
        return self.navigate_to(self.home_position_m)

    def _command(self, linear_m_s: float, angular_rad_s: float) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        left = (
            linear_m_s - angular_rad_s * self.wheel_base_m / 2.0
        ) / self.wheel_radius_m
        right = (
            linear_m_s + angular_rad_s * self.wheel_base_m / 2.0
        ) / self.wheel_radius_m
        self.robot.apply_wheel_actions(
            ArticulationAction(joint_velocities=np.asarray((left, right), dtype=np.float64))
        )

    def stop(self) -> None:
        self._command(0.0, 0.0)

    def navigate_to(
        self,
        target_xy_m: Sequence[float],
        *,
        tolerance_m: float = 0.12,
        maximum_steps: int | None = None,
    ) -> MeasurementMotionReport:
        target = np.asarray(target_xy_m, dtype=np.float64)
        initial, _ = _world_pose(self.stage, self.config.measurement_articulation)
        if maximum_steps is None:
            initial_distance_m = float(np.linalg.norm(target[:2] - initial[:2]))
            # Budget for 0.25 m/s average progress plus turns and settling.
            # The previous fixed 1,200-frame cap could only cover about 11 m
            # and stopped Nova Carter midway through the connecting corridor.
            maximum_steps = max(
                1200,
                int(math.ceil(initial_distance_m / 0.25 * 60.0)) + 300,
            )
        if maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        self.trace.append("navigating")
        success = False
        steps = 0
        for step in range(1, maximum_steps + 1):
            steps = step
            position, orientation = _world_pose(
                self.stage, self.config.measurement_articulation
            )
            error = target[:2] - position[:2]
            distance = float(np.linalg.norm(error))
            if self.progress_callback is not None and (step == 1 or step % 15 == 0):
                self.progress_callback(
                    step,
                    tuple(map(float, position)),
                    tuple(map(float, target[:2])),
                    distance,
                )
            if distance <= tolerance_m:
                success = True
                break
            desired_yaw = math.atan2(error[1], error[0])
            yaw_error = self._wrap(desired_yaw - self._yaw(orientation))
            angular = float(np.clip(2.2 * yaw_error, -1.2, 1.2))
            linear = min(0.55, 0.9 * distance)
            if abs(yaw_error) > 0.45:
                linear = 0.0
            self._command(linear, angular)
            self.stepper.step(render=True)
        self.stop()
        for _ in range(20):
            self.stepper.step(render=True)
        final, _ = _world_pose(self.stage, self.config.measurement_articulation)
        displacement = float(np.linalg.norm(final[:2] - initial[:2]))
        if self.progress_callback is not None:
            self.progress_callback(
                steps,
                tuple(map(float, final)),
                tuple(map(float, target[:2])),
                float(np.linalg.norm(target[:2] - final[:2])),
            )
        self.trace.append("complete" if success else "failed")
        return MeasurementMotionReport(
            success,
            steps,
            displacement,
            tuple(map(float, initial)),
            tuple(map(float, final)),
        )

    def navigate_route(
        self,
        waypoints_m: Sequence[Sequence[float]],
    ) -> MeasurementMotionReport:
        """Drive a collision-planned polyline and report the aggregate motion."""

        waypoints = tuple(waypoints_m)
        initial, _ = _world_pose(self.stage, self.config.measurement_articulation)
        if not waypoints:
            return MeasurementMotionReport(
                False,
                0,
                0.0,
                tuple(map(float, initial)),
                tuple(map(float, initial)),
            )
        if self.home_position_m is not None and np.linalg.norm(
            initial[:2] - np.asarray(self.home_position_m[:2], dtype=np.float64)
        ) <= 0.20:
            outbound_nodes = (initial, *(np.asarray(item, dtype=np.float64) for item in waypoints))
            self._home_return_route_m = tuple(
                tuple(map(float, node[:3])) for node in reversed(outbound_nodes[:-1])
            )
        total_steps = 0
        for waypoint in waypoints:
            report = self.navigate_to(waypoint)
            total_steps += report.steps
            if not report.success:
                final, _ = _world_pose(
                    self.stage, self.config.measurement_articulation
                )
                return MeasurementMotionReport(
                    False,
                    total_steps,
                    float(np.linalg.norm(final[:2] - initial[:2])),
                    tuple(map(float, initial)),
                    tuple(map(float, final)),
                )
        final, _ = _world_pose(self.stage, self.config.measurement_articulation)
        return MeasurementMotionReport(
            True,
            total_steps,
            float(np.linalg.norm(final[:2] - initial[:2])),
            tuple(map(float, initial)),
            tuple(map(float, final)),
        )


__all__ = [
    "ArticulatedTaskReport",
    "DecontaminationMotionReport",
    "FacilityCorridorSpec",
    "FacilityLayoutManifest",
    "FacilityPrimitiveSpec",
    "FacilityRoomSpec",
    "FacilityRouteSpec",
    "HandMotionResult",
    "MeasurementMotionReport",
    "NovaCarterController",
    "RealRobotAssetConfig",
    "RidgebackFrankaController",
    "ShieldMotionReport",
    "add_real_robot_references",
    "author_real_robot_task_scene",
    "create_decontamination_activity_map",
    "decommissioning_facility_layout",
    "enable_real_robot_extensions",
    "facility_route_clearance_m",
]
