import asyncio

import numpy as np

from radcounter.core.actions import (
    ActionStatus,
    ActionType,
    CountermeasureAction,
    CountermeasureExecutionContext,
    DecontaminationExecutor,
    DeterministicRobotController,
    DisposalZone,
    MoveObjectExecutor,
    RemoveObjectExecutor,
    ResourceState,
    ShieldPlacementExecutor,
)
from radcounter.core.models import PointSourceState, SurfaceSourceState, TruthState


def _pose(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = [x, y, z]
    return pose


def _context() -> CountermeasureExecutionContext:
    surface = SurfaceSourceState(
        "floor",
        "/World/Floor",
        np.array([0, 1]),
        np.array([100.0, 100.0]),
        "synthetic",
    )
    waste = PointSourceState("waste", np.array([10.0, 0.0, 0.0]), 0.0, "synthetic")
    attached = PointSourceState(
        "box-source",
        np.array([0.0, 0.0, 0.0]),
        50.0,
        "synthetic",
        attached_prim_path="/World/Box",
    )
    truth = TruthState(
        point_sources={"waste": waste, "box-source": attached},
        surface_sources={"floor": surface},
    )
    return CountermeasureExecutionContext(
        truth,
        ResourceState({"tool_s": 100.0, "shield_units": 2.0, "robot_s": 100.0}),
        np.random.default_rng(5),
        object_poses_world={"/World/Box": _pose(0.0)},
        movable_objects={"/World/Box"},
        removable_objects={"/World/Box"},
        shield_poses_world={"/World/Shield": _pose(-1.0)},
        disposal_zones={
            "outside": DisposalZone(
                "outside", np.array([10.0, 0.0, 0.0]), 1.0, "deactivate_outside"
            )
        },
    )


def test_decon_updates_target_activity_only_and_transfers_to_waste() -> None:
    context = _context()
    action = CountermeasureAction(
        "decon-1",
        ActionType.DECONTAMINATE,
        "robot",
        target_region={
            "source_id": "floor",
            "triangle_indices": [0],
            "exposure_s": [float(np.log(2.0))],
        },
        parameters={
            "rate_constant_s_inv": 1.0,
            "removed_activity_mode": "transfer_to_waste",
            "waste_source_id": "waste",
        },
        predicted_duration_s=1.0,
        resource_cost={"tool_s": 1.0},
    )
    result = asyncio.run(
        DecontaminationExecutor().execute(action, DeterministicRobotController("robot"), context)
    )
    assert result.status == ActionStatus.COMPLETED
    assert np.allclose(
        context.truth_state.surface_sources["floor"].activity_bq_per_triangle, [50, 100]
    )
    assert np.isclose(context.truth_state.point_sources["waste"].activity_bq, 50.0)
    assert context.truth_state.revision.source_activity_revision == 1
    assert context.truth_state.revision.geometry_revision == 0
    assert result.public_view().truth_details is None


def test_repeated_decon_accumulates_exposure() -> None:
    context = _context()
    action = CountermeasureAction(
        "decon-repeat",
        ActionType.DECONTAMINATE,
        "robot",
        target_region={
            "source_id": "floor",
            "triangle_indices": [0],
            "exposure_s": [float(np.log(2.0))],
        },
        parameters={"rate_constant_s_inv": 1.0, "removed_activity_mode": "discard"},
    )
    executor = DecontaminationExecutor()
    robot = DeterministicRobotController("robot")
    asyncio.run(executor.execute(action, robot, context))
    asyncio.run(executor.execute(action, robot, context))
    assert np.isclose(
        context.truth_state.surface_sources["floor"].activity_bq_per_triangle[0], 25.0
    )
    assert np.isclose(context.decon_exposure_s_by_source["floor"][0], 2.0 * np.log(2.0))


def test_invalid_decon_mode_rejects_without_mutation() -> None:
    context = _context()
    before_activity = context.truth_state.surface_sources["floor"].activity_bq_per_triangle.copy()
    action = CountermeasureAction(
        "decon-invalid",
        ActionType.DECONTAMINATE,
        "robot",
        target_region={
            "source_id": "floor",
            "triangle_indices": [0],
            "exposure_s": [1.0],
        },
        parameters={"removed_activity_mode": "invalid"},
    )
    result = asyncio.run(
        DecontaminationExecutor().execute(action, DeterministicRobotController("robot"), context)
    )
    assert result.status == ActionStatus.REJECTED
    assert np.array_equal(
        context.truth_state.surface_sources["floor"].activity_bq_per_triangle,
        before_activity,
    )
    assert "floor" not in context.decon_exposure_s_by_source
    assert context.truth_state.revision.source_activity_revision == 0


def test_shield_placement_uses_actual_pose_and_bumps_geometry() -> None:
    context = _context()
    target = _pose(2.0)
    action = CountermeasureAction(
        "shield-1",
        ActionType.PLACE_SHIELD,
        "robot",
        target_prim_path="/World/Shield",
        target_pose_world=target,
        parameters={"translation_error_std_m": 0.01},
        predicted_duration_s=2.0,
        resource_cost={"shield_units": 1.0},
    )
    result = asyncio.run(
        ShieldPlacementExecutor().execute(
            action,
            DeterministicRobotController("robot", graspable_prims={"/World/Shield"}),
            context,
        )
    )
    assert result.status == ActionStatus.COMPLETED
    assert context.truth_state.revision.geometry_revision == 1
    assert not np.array_equal(context.shield_poses_world["/World/Shield"], target)
    assert context.resources.available["shield_units"] == 1.0


def test_move_object_updates_attached_source_pose_and_revisions() -> None:
    context = _context()
    action = CountermeasureAction(
        "move-1",
        ActionType.MOVE_OBJECT,
        "robot",
        target_prim_path="/World/Box",
        target_pose_world=_pose(3.0),
    )
    result = asyncio.run(
        MoveObjectExecutor().execute(
            action,
            DeterministicRobotController("robot", graspable_prims={"/World/Box"}),
            context,
        )
    )
    assert result.status == ActionStatus.COMPLETED
    assert np.allclose(context.truth_state.point_sources["box-source"].position_world_m, [3, 0, 0])
    assert context.truth_state.revision.source_pose_revision == 1
    assert context.truth_state.revision.geometry_revision == 1


def test_remove_rejects_pose_outside_disposal_zone() -> None:
    context = _context()
    action = CountermeasureAction(
        "remove-bad",
        ActionType.REMOVE_OBJECT,
        "robot",
        target_prim_path="/World/Box",
        target_pose_world=_pose(3.0),
        parameters={"disposal_zone_id": "outside"},
    )
    result = asyncio.run(
        RemoveObjectExecutor().execute(
            action,
            DeterministicRobotController("robot", graspable_prims={"/World/Box"}),
            context,
        )
    )
    assert result.status == ActionStatus.REJECTED
    assert context.truth_state.point_sources["box-source"].enabled


def test_remove_deactivates_only_after_disposal_validation() -> None:
    context = _context()
    action = CountermeasureAction(
        "remove-good",
        ActionType.REMOVE_OBJECT,
        "robot",
        target_prim_path="/World/Box",
        target_pose_world=_pose(10.0),
        parameters={"disposal_zone_id": "outside"},
    )
    result = asyncio.run(
        RemoveObjectExecutor().execute(
            action,
            DeterministicRobotController("robot", graspable_prims={"/World/Box"}),
            context,
        )
    )
    assert result.status == ActionStatus.COMPLETED
    assert not context.truth_state.point_sources["box-source"].enabled
    assert context.truth_state.revision.source_activity_revision == 1
