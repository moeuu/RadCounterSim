import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

import radcounter

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
if str(EXTENSION) not in sys.path:
    sys.path.insert(0, str(EXTENSION))
extension_namespace = str(EXTENSION / "radcounter")
if extension_namespace not in radcounter.__path__:
    radcounter.__path__.append(extension_namespace)


def test_real_robot_assets_and_motion_gate_are_declared() -> None:
    root = Path(__file__).resolve().parents[2]
    module = root / "source/extensions/radcounter.isaac/radcounter/isaac/robot/real_robots.py"
    runner = root / "scripts/run_real_robot_validation.py"
    gate = root / "tests/isaac/real_robot_gate.py"
    object_gate = root / "tests/isaac/articulated_object_gate.py"
    assert module.is_file()
    assert runner.is_file()
    assert gate.is_file()
    assert object_gate.is_file()
    source = module.read_text(encoding="utf-8")
    assert "RidgebackFranka/ridgeback_franka.usd" in source
    assert "NVIDIA/NovaCarter/nova_carter.usd" in source
    assert "ArticulationAction" in source
    assert "ContactDrivenDecontaminator" not in source
    assert "FixedJoint.Define" in source


def test_robot_configs_select_real_isaac_controllers() -> None:
    root = Path(__file__).resolve().parents[2]
    countermeasure = yaml.safe_load(
        (root / "configs/robots/countermeasure_robot.yaml").read_text(encoding="utf-8")
    )
    measurement = yaml.safe_load(
        (root / "configs/robots/measurement_robot.yaml").read_text(encoding="utf-8")
    )
    assert countermeasure["reference_model_id"] == "clearpath-ridgeback-franka"
    assert countermeasure["controller"] == "ridgeback_franka_lula_ik"
    assert countermeasure["arm_dofs"] == 7
    assert countermeasure["geometry_fidelity"] == "manufacturer_asset"
    assert measurement["reference_model_id"] == "nvidia-nova-carter"
    assert measurement["controller"] == "nova_carter_differential_wheels"
    assert measurement["wheel_dof_names"] == ["joint_wheel_left", "joint_wheel_right"]


def test_gui_validation_uses_articulated_motion_without_tool_teleport() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/run_gui_validation.py").read_text(encoding="utf-8")
    assert "RidgebackFrankaController" in source
    assert "NovaCarterController" in source
    assert "create_decontamination_activity_map" in source
    assert "_position_contact_tool" not in source
    assert "IsaacPhysicsRobotController" not in source

    controller = (
        root / "source/extensions/radcounter.isaac/radcounter/isaac/robot/real_robots.py"
    ).read_text(encoding="utf-8")
    assert "_author_remote_decon_facility" in controller
    assert "CorridorNorthWall" in controller
    assert "DeconRoomEastWall" in controller
    assert "Cs-137 wall-mounted planar contamination source" in controller
    assert "irregular_deposition_field" in controller
    assert "gaussian_lobes_correlated_roughness_holes_satellite_droplets" in controller
    assert "surface.CreateFaceVertexCountsAttr([3] * (2 * len(active_cells)))" in controller
    assert "z_rows = np.linspace" in controller
    assert "float(np.max(np.abs(footprint[:, 1])))" in controller
    assert "float(np.max(np.abs(footprint[:, 0])))" in controller
    assert "required_coverage_fraction = 0.35" in controller
    assert "max_waypoint_speed_m_s = 0.20" in controller
    assert "compute_inverse_kinematics" in controller
    assert "def execute_pick_and_place" in controller
    assert "SetTargets([Sdf.Path(self.config.panda_hand_path)])" in controller
    assert "def execute_surface_decontamination" in controller
    assert "progress_callback" in controller
    assert "def return_home" in controller
    assert "def navigate_route" in controller
    assert "initial_distance_m / 0.25 * 60.0" in controller
    assert "destination_grasp = destination_root + grasp_offset" in controller
    assert "yaw_delta = placement_yaw - pickup_yaw" not in controller
    assert controller.count("self.set_gripper(0.016)") == 2
    assert "shield_grasp_offset_m: tuple[float, float, float] = (-0.24, 0.0, 0.59)" in controller
    assert "handle_outer_x = cfg.shield_grasp_offset_m[0] - 0.04" in controller
    assert "(0.0, -0.50, 0.25)" in controller
    assert 'object_path + "/ManipulatorHandleStem"' in controller
    assert '"lift_clearance"' in controller
    assert '"lift_transport"' in controller
    assert "retract_xy = pickup_base[:2] - grasp_position[:2]" in controller
    assert "grasp_position + retract_world" in controller
    assert "self._release_object(restore_collisions=False)" in controller
    release_index = controller.index("self._release_object(restore_collisions=False)")
    retract_index = controller.index("self.move_hand(preplace)", release_index)
    restore_index = controller.index("self._restore_grasp_collisions()", retract_index)
    assert release_index < retract_index < restore_index

    rules = (root / "docs/decontamination-authoring-rules.md").read_text(encoding="utf-8")
    assert "Gaussian lobes plus correlated sinusoidal roughness" in rules
    assert "continuous boustrophedon/serpentine raster" in rules
    assert "Sample the full physical pad footprint densely enough" in rules
    assert "ray count is spatial sampling density, not elapsed time" in rules
    assert "np.linspace(-0.085, 0.085, 9)" in source

    contact_model = (
        root / "source/extensions/radcounter.isaac/radcounter/isaac/robot/decontamination.py"
    ).read_text(encoding="utf-8")
    assert "effective_contact_exposure_s" in contact_model
    assert "def _update_surface_visuals" in contact_model
    assert "opacity_attr.Set([0.0 if value < 0.10" in contact_model
    assert "count / footprint_count" not in contact_model
    assert "implementation, regression tests" in (root / "AGENTS.md").read_text(encoding="utf-8")

    workflow = (
        root / "source/extensions/radcounter.isaac/radcounter/isaac/workflow/services.py"
    ).read_text(encoding="utf-8")
    assert '"motion_audit": _motion_audit(report)' in workflow
    assert "collateral_motion_audit" in workflow
    assert '"pickup_base_route_m"' in (
        root / "source/extensions/radcounter.isaac/radcounter/isaac/planning/scene_candidates.py"
    ).read_text(encoding="utf-8")
    assert '"base_route_m"' in (
        root / "source/extensions/radcounter.isaac/radcounter/isaac/planning/scene_candidates.py"
    ).read_text(encoding="utf-8")
    assert "carried_object_path" in (
        root / "source/extensions/radcounter.isaac/radcounter/isaac/planning/scene_candidates.py"
    ).read_text(encoding="utf-8")
    assert "live_candidates = generator.generate_measurement_actions(belief)" in source
    assert "generator.probe.invalidate_collision_cache()" in source
    assert "object_end_effector_offset_m=(0.90, 0.0, 0.0)" in source
    planner_source = (
        root / "source/extensions/radcounter.isaac/radcounter/isaac/planning/scene_candidates.py"
    ).read_text(encoding="utf-8")
    assert "removal_pickup_yaw = math.pi" in planner_source
    assert "removal_grasp = removal_root + grasp_from_root" in planner_source
    assert "removal_grasp_offset" not in planner_source
    assert "create_decontamination_activity_map(activity_path)" in (
        root / "scripts/run_real_robot_validation.py"
    ).read_text(encoding="utf-8")


def test_complex_decommissioning_facility_manifest_is_deterministic_and_clear() -> None:
    from radcounter.isaac.robot.real_robots import (
        RealRobotAssetConfig,
        decommissioning_facility_layout,
        facility_route_clearance_m,
    )

    layout = decommissioning_facility_layout()
    assert layout == decommissioning_facility_layout()
    assert layout.layout_id == "four_room_decommissioning_cell_v1"

    room_ids = {room.room_id for room in layout.rooms}
    assert room_ids == {
        "original_cell",
        "remote_decon_room",
        "reactor_service_room",
        "shield_staging_room",
    }
    assert len(layout.corridors) >= 3
    connected = {"original_cell"}
    while True:
        expanded = (
            connected
            | {
                corridor.to_room_id
                for corridor in layout.corridors
                if corridor.from_room_id in connected
            }
            | {
                corridor.from_room_id
                for corridor in layout.corridors
                if corridor.to_room_id in connected
            }
        )
        if expanded == connected:
            break
        connected = expanded
    assert connected == room_ids

    assert len(layout.equipment_ids) >= 10
    assert len(layout.equipment_ids) == len(set(layout.equipment_ids))
    assert {primitive.shape for primitive in layout.primitives} >= {"cube", "cylinder"}
    assert all(
        primitive.room_id in room_ids
        for primitive in layout.primitives
        if primitive.equipment_id is not None
    )

    route_ids = {route.route_id for route in layout.reserved_routes}
    assert route_ids == {
        "primary_decon_route",
        "staging_room_service_route",
        "primary_shield_25_service_route",
        "primary_shield_65_service_route",
    }
    for route_id in route_ids:
        assert facility_route_clearance_m(layout, route_id) >= 0.65
    primary_route = next(
        route for route in layout.reserved_routes if route.route_id == "primary_decon_route"
    )
    config = RealRobotAssetConfig()
    assert np.allclose(
        primary_route.waypoints_m[-1][:2],
        (config.decon_workbench_center_m[0] - 0.90, config.decon_workbench_center_m[1]),
    )

    staging_room = next(room for room in layout.rooms if room.room_id == "shield_staging_room")
    shield_position = np.asarray(layout.secondary_shield_position_m)
    room_lower = np.asarray(staging_room.center_m) - np.asarray(staging_room.half_extent_m)
    room_upper = np.asarray(staging_room.center_m) + np.asarray(staging_room.half_extent_m)
    assert np.all(shield_position[:2] > room_lower[:2])
    assert np.all(shield_position[:2] < room_upper[:2])
    assert layout.secondary_shield_path == "/World/StagingLeadShield"
    assert layout.secondary_shield_path != config.shield_path

    primitive_paths = {primitive.path for primitive in layout.primitives}
    assert {
        "ShieldServiceAlcoveFloor",
        "ShieldServiceAlcoveWestWall",
        "DeconRoomWestWallSouthJamb",
        "DeconRoomWestWallNorth",
    } <= primitive_paths


def test_primary_shield_25_and_65_routes_use_the_west_handle_service_side() -> None:
    from radcounter.isaac.planning.scene_candidates import (
        IsaacActionCandidateGenerator,
    )
    from radcounter.isaac.robot.real_robots import (
        RealRobotAssetConfig,
        decommissioning_facility_layout,
        facility_route_clearance_m,
    )

    layout = decommissioning_facility_layout()
    route_by_id = {route.route_id: route for route in layout.reserved_routes}
    generator = object.__new__(IsaacActionCandidateGenerator)
    generator.config = SimpleNamespace(end_effector_offset_m=(0.72, 0.0, 0.0))
    shield_config = RealRobotAssetConfig()
    public_source = np.asarray((14.39, 0.5700749954, 0.45))
    protected = np.asarray((0.0, 0.0, 0.45))
    root_from_center = np.asarray((0.05, 0.0, -0.45))

    for percentage, fraction in ((25, 0.25), (65, 0.65)):
        target_root = public_source + fraction * (protected - public_source)
        target_root += root_from_center
        target_grasp = target_root + np.asarray(shield_config.shield_grasp_offset_m)
        placement_base = generator._base_for_end_effector(
            target_grasp,
            robot_z=0.28,
            yaw_rad=0.0,
        )
        route_id = f"primary_shield_{percentage}_service_route"
        route_endpoint = np.asarray(route_by_id[route_id].waypoints_m[-1])

        assert np.allclose(route_endpoint, placement_base, atol=1.0e-9)
        assert np.isclose(target_root[0] - placement_base[0], 0.96)
        assert np.isclose(target_root[1], placement_base[1])
        assert placement_base[0] < target_grasp[0] < target_root[0]
        assert facility_route_clearance_m(layout, route_id) >= 0.65


def test_complex_facility_authoring_declares_auditable_metadata() -> None:
    source = (
        ROOT / "source/extensions/radcounter.isaac/radcounter/isaac/robot/real_robots.py"
    ).read_text(encoding="utf-8")

    for attribute_name in (
        "rad:facility:layoutId",
        "rad:facility:roomCount",
        "rad:facility:corridorCount",
        "rad:facility:equipmentCount",
        "rad:facility:obstacleCount",
        "rad:facility:reservedRouteIds",
        "rad:facility:equipmentType",
    ):
        assert attribute_name in source
    assert "same_visible_irregular_collision_activity_mesh" in source
    assert "def _author_staging_lead_shield" in source
    assert source.count("CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.10))") >= 2
    assert source.count("half_scale=(0.18, 0.30, 0.035)") >= 2
    assert '"rad:shield:inventoryId"' in source
    assert '"staging-shield-02"' in source
    assert "if cfg.include_validation_facility else None" in source
    assert '"/World/RemoteDeconFacility"' in source
    assert '"/World/DetectorStations/RemoteDeconRoom"' in source
    assert "if legacy_surface.IsValid():" in source
    assert "Optional relocation/disposal" in source

    planner_source = (
        ROOT / "source/extensions/radcounter.isaac/radcounter/isaac/planning/scene_candidates.py"
    ).read_text(encoding="utf-8")
    assert "for placement_yaw in (0.0,):" in planner_source
    assert "overlaps the chassis and payload at release" in planner_source


def test_carried_shield_route_avoids_drum_swept_volume() -> None:
    from radcounter.isaac.planning.scene_candidates import IsaacSceneFeasibilityProbe

    probe = object.__new__(IsaacSceneFeasibilityProbe)
    probe.config = SimpleNamespace(
        countermeasure_robot_path="/World/CountermeasureRobot",
        measurement_robot_path="/World/MeasurementRobot",
        mobile_clearance_m=0.42,
    )
    probe._collision_bounds = lambda: (  # type: ignore[method-assign]
        (
            "/World/HiddenContaminatedDrum/Drum",
            np.asarray((2.08, -0.32, 0.05)),
            np.asarray((2.72, 0.32, 1.05)),
        ),
    )
    probe.bounds = lambda path: (  # type: ignore[method-assign]
        np.asarray((4.42, 2.0, 0.0)),
        np.asarray((5.18, 3.6, 1.8)),
    )
    pickup_base = np.asarray((3.6, 2.8, 0.015))
    placement_base = np.asarray((-0.7667, -0.325, 0.015))

    base_only = probe.plan_mobile_route(
        pickup_base,
        placement_base,
        excluded_paths=("/World/LeadShield",),
    )
    carrying = probe.plan_mobile_route(
        pickup_base,
        placement_base,
        excluded_paths=("/World/LeadShield",),
        carried_object_path="/World/LeadShield",
        carried_base_position_m=pickup_base,
    )

    assert base_only is not None and len(base_only) == 1
    assert carrying is not None and len(carrying) >= 2
    assert not np.allclose(carrying[0], placement_base)


def test_manipulation_base_endpoint_cannot_overlap_unrelated_prop() -> None:
    from radcounter.isaac.planning.scene_candidates import IsaacSceneFeasibilityProbe

    probe = object.__new__(IsaacSceneFeasibilityProbe)
    probe.config = SimpleNamespace(
        countermeasure_robot_path="/World/CountermeasureRobot",
        measurement_robot_path="/World/MeasurementRobot",
        mobile_clearance_m=0.42,
    )
    probe._collision_bounds = lambda: (  # type: ignore[method-assign]
        (
            "/World/HiddenContaminatedDrum/Drum",
            np.asarray((2.08, -0.32, 0.05)),
            np.asarray((2.72, 0.32, 1.05)),
        ),
    )

    assert not probe.mobile_base_pose_available(
        np.asarray((2.215, -0.26, 0.015)),
        excluded_paths=("/World/LeadShield",),
    )
    assert probe.mobile_base_pose_available(
        np.asarray((3.895, -0.26, 0.015)),
        excluded_paths=("/World/LeadShield",),
    )


def test_curated_cad_spawn_can_ignore_only_the_compound_environment_aabb() -> None:
    from radcounter.isaac.planning.scene_candidates import IsaacSceneFeasibilityProbe

    probe = object.__new__(IsaacSceneFeasibilityProbe)
    probe.config = SimpleNamespace(
        countermeasure_robot_path="/World/CountermeasureRobot",
        measurement_robot_path="/World/MeasurementRobot",
        mobile_clearance_m=0.55,
        ignored_collision_paths=("/World/Environment",),
    )
    probe._collision_bounds = lambda: (  # type: ignore[method-assign]
        (
            "/World/Environment/Building/Mesh",
            np.asarray((-25.0, -25.0, -2.0)),
            np.asarray((2.2, 25.0, 58.0)),
        ),
    )
    probe.bounds = lambda path: None  # type: ignore[method-assign]

    start = np.asarray((0.0, -1.0, 0.0))
    target = np.asarray((1.2, -0.55, 0.0))
    route = probe.plan_mobile_route(
        start,
        target,
        moving_robot_path="/World/CountermeasureRobot",
    )

    assert route is not None
    assert np.allclose(route[-1], target)
    assert probe.mobile_base_pose_available(target)


def test_measurement_route_treats_countermeasure_robot_as_obstacle() -> None:
    from radcounter.isaac.planning.scene_candidates import IsaacSceneFeasibilityProbe

    probe = object.__new__(IsaacSceneFeasibilityProbe)
    probe.config = SimpleNamespace(
        countermeasure_robot_path="/World/CountermeasureRobot",
        measurement_robot_path="/World/MeasurementRobot",
        mobile_clearance_m=0.55,
    )
    probe._collision_bounds = lambda: ()  # type: ignore[method-assign]
    probe.bounds = lambda path: (  # type: ignore[method-assign]
        (
            np.asarray((0.65, -1.45, 0.0)),
            np.asarray((1.75, -0.55, 1.1)),
        )
        if path == "/World/CountermeasureRobot"
        else None
    )
    start = np.asarray((-3.2, 2.4, 0.0))
    target = np.asarray((3.2, -2.4, 0.8))

    route = probe.plan_mobile_route(
        start,
        target,
        moving_robot_path="/World/MeasurementRobot",
    )

    assert route is not None and len(route) >= 2
    assert not np.allclose(route[0], target)


def test_pickup_route_does_not_cross_target_object() -> None:
    from radcounter.isaac.planning.scene_candidates import IsaacSceneFeasibilityProbe

    probe = object.__new__(IsaacSceneFeasibilityProbe)
    probe.config = SimpleNamespace(
        countermeasure_robot_path="/World/CountermeasureRobot",
        measurement_robot_path="/World/MeasurementRobot",
        mobile_clearance_m=0.55,
    )
    probe._collision_bounds = lambda: (  # type: ignore[method-assign]
        (
            "/World/HiddenContaminatedDrum/Drum",
            np.asarray((2.08, -0.32, 0.05)),
            np.asarray((2.72, 0.32, 1.05)),
        ),
    )
    probe.bounds = lambda path: None  # type: ignore[method-assign]
    start = np.asarray((3.8, 0.0, 0.28))
    pickup_base = np.asarray((1.5, -0.38, 0.28))

    route = probe.plan_mobile_route(
        start,
        pickup_base,
        moving_robot_path="/World/CountermeasureRobot",
    )

    assert route is not None and len(route) >= 2
    assert not np.allclose(route[0], pickup_base)


def test_opposite_side_removal_base_preserves_object_standoff() -> None:
    from radcounter.isaac.planning.scene_candidates import (
        IsaacActionCandidateGenerator,
    )

    generator = object.__new__(IsaacActionCandidateGenerator)
    generator.config = SimpleNamespace(end_effector_offset_m=(0.90, 0.0, 0.0))
    grasp = np.asarray((2.4, -0.5, 0.8))

    base = generator._base_for_end_effector(
        grasp,
        robot_z=0.28,
        yaw_rad=np.pi,
        offset_m=(0.90, 0.0, 0.0),
    )

    assert np.allclose(base, np.asarray((3.3, -0.5, 0.28)))
    assert np.isclose(np.linalg.norm(base[:2] - grasp[:2]), 0.90)
