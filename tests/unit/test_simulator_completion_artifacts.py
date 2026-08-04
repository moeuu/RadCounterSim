from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_native_backend_declares_instances_and_packet_intersection() -> None:
    source = (ROOT / "native/src/embree_backend.cpp").read_text(encoding="utf-8")
    assert "RTC_GEOMETRY_TYPE_INSTANCE" in source
    assert "rtcSetGeometryInstancedScene" in source
    assert "rtcSetGeometryTransform" in source
    assert "rtcIntersect8" in source


def test_isaac_workflow_and_physics_callback_adapters_exist() -> None:
    workflow = (
        ROOT / "source/extensions/radcounter.isaac/radcounter/isaac/workflow/services.py"
    ).read_text(encoding="utf-8")
    candidates = (
        ROOT / "source/extensions/radcounter.isaac/radcounter/isaac/planning/scene_candidates.py"
    ).read_text(encoding="utf-8")
    loop = (
        ROOT / "source/extensions/radcounter.isaac/radcounter/isaac/runtime/physics_loop.py"
    ).read_text(encoding="utf-8")
    assert "class IsaacWorkflowServices" in workflow
    assert "class PublicMeasurement" in workflow
    assert (
        "expected_rate_cps"
        not in workflow.split("class PublicMeasurement", 1)[1].split("class WorkflowResidual", 1)[0]
    )
    assert "class IsaacActionCandidateGenerator" in candidates
    for token in (
        "mobile_path_available",
        "manipulator_reachable",
        "collision_free_placement",
        "placement_stable",
        "disposal_capacity_available",
    ):
        assert token in candidates
    assert "subscribe_physics_step_events" in loop


def test_ros_joint_trajectory_is_a_gateway_execution_path() -> None:
    action = (ROOT / "ros2_ws/src/radcounter_msgs/action/ExecuteRobotTask.action").read_text(
        encoding="utf-8"
    )
    clients = (
        ROOT / "ros2_ws/src/radcounter_robot_gateway/radcounter_robot_gateway/motion_clients.py"
    ).read_text(encoding="utf-8")
    gateway = (
        ROOT / "ros2_ws/src/radcounter_robot_gateway/radcounter_robot_gateway/gateway_node.py"
    ).read_text(encoding="utf-8")
    assert "bool use_joint_trajectory" in action
    assert "class JointTrajectoryClient" in clients
    assert "FollowJointTrajectory" in clients
    assert "self._trajectory.execute" in gateway


def test_dashboard_accepts_estimate_residual_and_plan_views() -> None:
    dashboard = (
        ROOT / "source/extensions/radcounter.isaac/radcounter/isaac/ui/dashboard.py"
    ).read_text(encoding="utf-8")
    assert "def set_workflow_view" in dashboard
    assert "SOURCE AUTHORING" in dashboard
    assert "ESTIMATE" in dashboard
    assert "RESIDUAL" in dashboard
    assert "PLAN" in dashboard
