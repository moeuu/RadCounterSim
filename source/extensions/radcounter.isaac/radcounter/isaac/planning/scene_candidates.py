"""Generate planner candidates from public belief and the live Isaac scene."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from radcounter.core.models.actions import ActionType, CountermeasureAction
from radcounter.core.models.state import BeliefState
from radcounter.core.planning.models import ActionCandidate, ActionMetrics, FeasibilityFacts


@dataclass(frozen=True, slots=True)
class SceneCandidateConfig:
    countermeasure_robot_path: str = "/World/CountermeasureRobot"
    measurement_robot_path: str = "/World/MeasurementRobot"
    countermeasure_pose_path: str | None = None
    measurement_pose_path: str | None = None
    task_detector_path: str = "/World/DetectorStations/Protected"
    disposal_zone_path: str = "/World/DisposalZone"
    end_effector_offset_m: tuple[float, float, float] = (1.2, 0.0, 0.0)
    decon_end_effector_offset_m: tuple[float, float, float] | None = None
    object_end_effector_offset_m: tuple[float, float, float] | None = None
    mobile_clearance_m: float = 0.28
    manipulator_workspace_m: float = 1.55
    manipulator_vertical_range_m: tuple[float, float] = (-0.35, 1.15)
    # Include a near-source slot inside the remote room as well as the
    # original-cell slots.  In a multi-room facility the straight line to the
    # protected station crosses the west wall around 35%, so 25% is the first
    # physically reachable host-owned placement slot for a staged panel.
    shield_line_fractions: tuple[float, ...] = (0.25, 0.35, 0.5, 0.65)
    measurement_duration_s: float = 2.0
    shield_duration_s: float = 45.0
    decon_duration_s: float = 20.0
    object_duration_s: float = 35.0
    object_parking_offsets_m: tuple[tuple[float, float], ...] = (
        (1.6, 0.0),
        (1.6, -1.8),
        (3.4, 0.0),
        (3.4, -1.8),
    )
    dose_proxy_to_sv_h: float = 1.0e-12
    # A compound CAD building may expose one collision mesh whose AABB covers
    # all of its traversable rooms.  Curated spawn anchors can opt that root
    # out of the coarse AABB planner while PhysX still enforces its triangles.
    ignored_collision_paths: tuple[str, ...] = ()


class IsaacSceneFeasibilityProbe:
    """Evaluate path, IK, collision, grasp, support, and disposal constraints."""

    def __init__(
        self,
        stage: Any,
        config: SceneCandidateConfig,
        controller: Any | None = None,
    ) -> None:
        self.stage = stage
        self.config = config
        self.controller = controller
        self._collision_bounds_cache: tuple[tuple[str, np.ndarray, np.ndarray], ...] | None = None

    @staticmethod
    def _attribute(prim: Any, name: str, default: object = None) -> object:
        attribute = prim.GetAttribute(name)
        if not attribute or not attribute.HasAuthoredValueOpinion():
            return default
        value = attribute.Get()
        return default if value is None else value

    def world_position(self, prim_or_path: Any) -> np.ndarray:
        from pxr import Gf, UsdGeom

        prim = (
            self.stage.GetPrimAtPath(prim_or_path)
            if isinstance(prim_or_path, str)
            else prim_or_path
        )
        if not prim or not prim.IsValid():
            raise ValueError(f"USD prim does not exist: {prim_or_path}")
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        return np.asarray(matrix.Transform(Gf.Vec3d()), dtype=np.float64)

    def bounds(self, prim_or_path: Any) -> tuple[np.ndarray, np.ndarray] | None:
        from pxr import Usd, UsdGeom

        prim = (
            self.stage.GetPrimAtPath(prim_or_path)
            if isinstance(prim_or_path, str)
            else prim_or_path
        )
        if not prim or not prim.IsValid() or not UsdGeom.Imageable(prim):
            return None
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
        maximum = np.asarray(aligned.GetMax(), dtype=np.float64)
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            return None
        return minimum, maximum

    def _collision_bounds(self) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
        from pxr import Usd, UsdGeom, UsdPhysics

        if self._collision_bounds_cache is not None:
            return self._collision_bounds_cache

        result: list[tuple[str, np.ndarray, np.ndarray]] = []
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lower = np.asarray(aligned.GetMin(), dtype=np.float64)
            upper = np.asarray(aligned.GetMax(), dtype=np.float64)
            if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
                result.append((str(prim.GetPath()), lower, upper))
        self._collision_bounds_cache = tuple(result)
        return self._collision_bounds_cache

    def invalidate_collision_cache(self) -> None:
        """Refresh collision geometry before a new live-scene planning pass."""

        self._collision_bounds_cache = None

    @staticmethod
    def _is_descendant(path: str, parent: str) -> bool:
        return path == parent or path.startswith(parent.rstrip("/") + "/")

    def _navigation_obstacles(
        self,
        *,
        excluded_paths: tuple[str, ...] = (),
        moving_robot_path: str | None = None,
        carried_object_path: str | None = None,
        carried_base_position_m: np.ndarray | None = None,
    ) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
        ignored_robots = (
            (self.config.countermeasure_robot_path, self.config.measurement_robot_path)
            if moving_robot_path is None
            else (moving_robot_path,)
        )
        ignored = (
            *ignored_robots,
            *excluded_paths,
            *getattr(self.config, "ignored_collision_paths", ()),
        )
        clearance = self.config.mobile_clearance_m
        obstacles: list[tuple[str, np.ndarray, np.ndarray]] = []
        for path, lower, upper in self._collision_bounds():
            if any(self._is_descendant(path, item) for item in ignored):
                continue
            if upper[2] <= 0.08:
                continue
            expanded_lower = lower - np.asarray((clearance, clearance, 0.05))
            expanded_upper = upper + np.asarray((clearance, clearance, 0.05))
            obstacles.append((path, expanded_lower, expanded_upper))
        # Some referenced manufacturer robot assets expose their PhysX shapes
        # through schemas that are not returned by the generic
        # UsdPhysics.CollisionAPI traversal above.  Always include the other
        # articulated robot's composed bounds so one robot can never plan
        # straight through the other.
        for robot_path in (
            self.config.countermeasure_robot_path,
            self.config.measurement_robot_path,
        ):
            if any(self._is_descendant(robot_path, item) for item in ignored):
                continue
            robot_bounds = self.bounds(robot_path)
            if robot_bounds is None:
                continue
            robot_lower, robot_upper = robot_bounds
            expanded_lower = robot_lower - np.asarray((clearance, clearance, 0.05))
            expanded_upper = robot_upper + np.asarray((clearance, clearance, 0.05))
            obstacles.append((f"{robot_path}:articulated_bounds", expanded_lower, expanded_upper))
        if carried_object_path is not None:
            if carried_base_position_m is None:
                raise ValueError("carried_base_position_m is required with carried_object_path")
            carried_bounds = self.bounds(carried_object_path)
            if carried_bounds is None:
                raise ValueError(f"carried object has no bounds: {carried_object_path}")
            base = np.asarray(carried_base_position_m, dtype=np.float64)
            relative_lower = carried_bounds[0] - base
            relative_upper = carried_bounds[1] - base
            swept_margin = np.asarray((0.12, 0.12, 0.05), dtype=np.float64)
            for path, lower, upper in self._collision_bounds():
                if any(self._is_descendant(path, item) for item in ignored):
                    continue
                if upper[2] <= 0.08:
                    continue
                # Configuration-space obstacle for the carried object's full
                # AABB. A base position inside this box would make the carried
                # payload overlap the scene obstacle even when the base itself
                # remains clear.
                forbidden_lower = lower - relative_upper - swept_margin
                forbidden_upper = upper - relative_lower + swept_margin
                obstacles.append(
                    (f"{path}:carried:{carried_object_path}", forbidden_lower, forbidden_upper)
                )
        return tuple(obstacles)

    @staticmethod
    def _segment_available(
        start: np.ndarray,
        target: np.ndarray,
        obstacles: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    ) -> bool:
        distance = float(np.linalg.norm(target[:2] - start[:2]))
        sample_count = max(2, int(math.ceil(distance / 0.12)) + 1)
        samples = np.linspace(start, target, sample_count)
        samples[:, 2] = np.maximum(samples[:, 2], 0.32)
        for _, expanded_lower, expanded_upper in obstacles:
            inside = np.all((samples >= expanded_lower) & (samples <= expanded_upper), axis=1)
            # A manipulation base pose is intentionally close to the serviced
            # object and may sit inside its *inflated* clearance box without
            # touching its real collision geometry.  Permit a monotonic exit
            # from, or entry into, that endpoint margin; never permit crossing
            # through the obstacle in the middle of a route.
            first_outside = 0
            while first_outside < len(inside) and inside[first_outside]:
                first_outside += 1
            last_outside = len(inside) - 1
            while last_outside >= 0 and inside[last_outside]:
                last_outside -= 1
            if first_outside == len(inside):
                return False
            middle = inside[first_outside : last_outside + 1]
            if bool(np.any(middle)):
                return False
        return True

    def plan_mobile_route(
        self,
        start_m: np.ndarray,
        target_m: np.ndarray,
        *,
        excluded_paths: tuple[str, ...] = (),
        moving_robot_path: str | None = None,
        carried_object_path: str | None = None,
        carried_base_position_m: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...] | None:
        start = np.asarray(start_m, dtype=np.float64)
        target = np.asarray(target_m, dtype=np.float64)
        obstacles = self._navigation_obstacles(
            excluded_paths=excluded_paths,
            moving_robot_path=moving_robot_path,
            carried_object_path=carried_object_path,
            carried_base_position_m=carried_base_position_m,
        )
        if self._segment_available(start, target, obstacles):
            return (target.copy(),)

        navigation_z = max(float(start[2]), float(target[2]), 0.32)
        corner_padding = 0.04
        points = [start.copy(), target.copy()]
        for _, lower, upper in obstacles:
            for x in (lower[0] - corner_padding, upper[0] + corner_padding):
                for y in (lower[1] - corner_padding, upper[1] + corner_padding):
                    candidate = np.asarray((x, y, navigation_z), dtype=np.float64)
                    if any(
                        np.all(candidate >= other_lower) and np.all(candidate <= other_upper)
                        for _, other_lower, other_upper in obstacles
                    ):
                        continue
                    if not any(
                        np.linalg.norm(candidate[:2] - point[:2]) < 1.0e-6 for point in points
                    ):
                        points.append(candidate)

        count = len(points)
        distances = np.full(count, np.inf, dtype=np.float64)
        previous = np.full(count, -1, dtype=np.int64)
        visited = np.zeros(count, dtype=bool)
        distances[0] = 0.0
        for _ in range(count):
            candidates = np.flatnonzero(~visited)
            if candidates.size == 0:
                break
            current = int(candidates[np.argmin(distances[candidates])])
            if not np.isfinite(distances[current]) or current == 1:
                break
            visited[current] = True
            for neighbor in candidates:
                neighbor_index = int(neighbor)
                if neighbor_index == current or not self._segment_available(
                    points[current], points[neighbor_index], obstacles
                ):
                    continue
                tentative = distances[current] + float(
                    np.linalg.norm(points[neighbor_index][:2] - points[current][:2])
                )
                if tentative < distances[neighbor_index]:
                    distances[neighbor_index] = tentative
                    previous[neighbor_index] = current

        if not np.isfinite(distances[1]):
            return None
        route_indices = [1]
        while route_indices[-1] != 0:
            parent = int(previous[route_indices[-1]])
            if parent < 0:
                return None
            route_indices.append(parent)
        route_indices.reverse()
        return tuple(points[index].copy() for index in route_indices[1:])

    def mobile_path_available(
        self,
        start_m: np.ndarray,
        target_m: np.ndarray,
        *,
        excluded_paths: tuple[str, ...] = (),
        moving_robot_path: str | None = None,
        carried_object_path: str | None = None,
        carried_base_position_m: np.ndarray | None = None,
    ) -> bool:
        return (
            self.plan_mobile_route(
                start_m,
                target_m,
                excluded_paths=excluded_paths,
                moving_robot_path=moving_robot_path,
                carried_object_path=carried_object_path,
                carried_base_position_m=carried_base_position_m,
            )
            is not None
        )

    def mobile_base_pose_available(
        self,
        position_m: np.ndarray,
        *,
        excluded_paths: tuple[str, ...] = (),
    ) -> bool:
        """Check that a manipulation base pose itself clears scene geometry.

        Route segments may deliberately enter an inflated endpoint margin so a
        manipulator can service the target object.  That exception must not
        apply to unrelated obstacles at the final base pose: doing so can put
        the chassis directly through a nearby prop even though the route is
        reported as available.
        """

        position = np.asarray(position_m, dtype=np.float64)
        ignored = (
            self.config.countermeasure_robot_path,
            self.config.measurement_robot_path,
            *excluded_paths,
            *getattr(self.config, "ignored_collision_paths", ()),
        )
        clearance = float(self.config.mobile_clearance_m)
        for path, lower, upper in self._collision_bounds():
            if any(self._is_descendant(path, item) for item in ignored):
                continue
            if upper[2] <= 0.08:
                continue
            inside_xy = bool(
                lower[0] - clearance <= position[0] <= upper[0] + clearance
                and lower[1] - clearance <= position[1] <= upper[1] + clearance
            )
            if inside_xy:
                return False
        return True

    def manipulator_reachable(
        self, target_m: np.ndarray, planned_base_m: np.ndarray | None = None
    ) -> bool:
        target = np.asarray(target_m, dtype=np.float64)
        robot = self.world_position(
            self.config.countermeasure_pose_path or self.config.countermeasure_robot_path
        )
        planned_base = robot if planned_base_m is None else np.asarray(planned_base_m)
        if self.controller is not None and hasattr(self.controller, "check_reachability"):
            try:
                return bool(
                    self.controller.check_reachability(target, base_position_m=planned_base)
                )
            except TypeError:
                if np.linalg.norm(planned_base - robot) <= 0.05:
                    return bool(self.controller.check_reachability(target))
            except NotImplementedError:
                pass
        radial = float(np.linalg.norm(target[:2] - planned_base[:2]))
        vertical = float(target[2] - planned_base[2])
        minimum_z, maximum_z = self.config.manipulator_vertical_range_m
        return radial <= self.config.manipulator_workspace_m and minimum_z <= vertical <= maximum_z

    def grasp_frame_available(self, object_path: str | None) -> bool:
        if not object_path:
            return True
        prim = self.stage.GetPrimAtPath(object_path)
        if not prim or not prim.IsValid():
            return False
        frame_name = str(self._attribute(prim, "rad:manipulation:graspFrame", ""))
        if not frame_name:
            return False
        frame = self.stage.GetPrimAtPath(f"{object_path.rstrip('/')}/{frame_name}")
        return bool(frame and frame.IsValid())

    def collision_free_placement(self, object_path: str | None, target_m: np.ndarray) -> bool:
        if not object_path:
            return True
        current_bounds = self.bounds(object_path)
        if current_bounds is None:
            return False
        current_center = (current_bounds[0] + current_bounds[1]) * 0.5
        half_extent = (current_bounds[1] - current_bounds[0]) * 0.5
        candidate_center = np.asarray(target_m, dtype=np.float64)
        candidate_lower = candidate_center - half_extent
        candidate_upper = candidate_center + half_extent
        ignored = (
            object_path,
            self.config.countermeasure_robot_path,
            self.config.measurement_robot_path,
        )
        for path, lower, upper in self._collision_bounds():
            if any(self._is_descendant(path, item) for item in ignored):
                continue
            if upper[2] <= candidate_lower[2] + 0.035:
                continue
            prim = self.stage.GetPrimAtPath(path)
            movable_neighbor = False
            while prim and prim.IsValid() and str(prim.GetPath()) != "/":
                movable = prim.GetAttribute("rad:manipulation:movable")
                if movable and movable.HasAuthoredValueOpinion() and bool(movable.Get()):
                    movable_neighbor = True
                    break
                prim = prim.GetParent()
            # A payload can be geometrically separate from another prop while
            # leaving too little room for the gripper and final arm descent.
            # Preserve a manipulation envelope around movable neighbors.
            manipulation_margin = 0.22 if movable_neighbor else 0.0
            margin = np.asarray(
                (manipulation_margin, manipulation_margin, 0.0),
                dtype=np.float64,
            )
            overlap = np.all(candidate_lower - margin < upper - 0.01) and np.all(
                candidate_upper + margin > lower + 0.01
            )
            if bool(overlap):
                return False
        return bool(np.all(np.isfinite(current_center)))

    def placement_stable(self, object_path: str | None, target_m: np.ndarray) -> bool:
        if not object_path:
            return True
        object_bounds = self.bounds(object_path)
        if object_bounds is None:
            return False
        half_height = float((object_bounds[1][2] - object_bounds[0][2]) * 0.5)
        bottom = float(target_m[2] - half_height)
        for path, lower, upper in self._collision_bounds():
            if self._is_descendant(path, object_path):
                continue
            supports_xy = bool(
                lower[0] <= target_m[0] <= upper[0] and lower[1] <= target_m[1] <= upper[1]
            )
            if supports_xy and -0.08 <= bottom - upper[2] <= 0.15:
                return True
        return False

    def disposal_capacity_available(self, object_path: str | None) -> bool:
        zone = self.stage.GetPrimAtPath(self.config.disposal_zone_path)
        if not zone or not zone.IsValid() or not object_path:
            return False
        target = self.stage.GetPrimAtPath(object_path)
        target_class = str(self._attribute(target, "rad:manipulation:disposalClass", ""))
        accepted = str(self._attribute(zone, "rad:disposal:acceptsClass", ""))
        return bool(target_class and target_class == accepted)

    def facts(
        self,
        *,
        object_path: str | None,
        target_m: np.ndarray,
        pickup_base_m: np.ndarray,
        placement_base_m: np.ndarray,
        requires_grasp: bool,
        pickup_manipulator_target_m: np.ndarray | None = None,
        placement_manipulator_target_m: np.ndarray | None = None,
        requires_disposal: bool = False,
    ) -> FeasibilityFacts:
        robot_position = self.world_position(
            self.config.countermeasure_pose_path or self.config.countermeasure_robot_path
        )
        path_to_pickup = self.mobile_path_available(
            robot_position,
            pickup_base_m,
            moving_robot_path=self.config.countermeasure_robot_path,
        )
        path_to_placement = self.mobile_path_available(
            pickup_base_m,
            placement_base_m,
            excluded_paths=(() if object_path is None else (object_path,)),
            moving_robot_path=self.config.countermeasure_robot_path,
            carried_object_path=object_path if requires_grasp else None,
            carried_base_position_m=pickup_base_m if requires_grasp else None,
        )
        pickup_pose_available = self.mobile_base_pose_available(
            pickup_base_m,
            excluded_paths=(() if object_path is None else (object_path,)),
        )
        placement_pose_available = self.mobile_base_pose_available(
            placement_base_m,
            excluded_paths=(() if object_path is None else (object_path,)),
        )
        pickup_target = (
            target_m if pickup_manipulator_target_m is None else pickup_manipulator_target_m
        )
        placement_target = (
            target_m if placement_manipulator_target_m is None else placement_manipulator_target_m
        )
        return FeasibilityFacts(
            mobile_path_available=(
                path_to_pickup
                and path_to_placement
                and pickup_pose_available
                and placement_pose_available
            ),
            manipulator_reachable=(
                True
                if not requires_grasp
                else self.manipulator_reachable(pickup_target, pickup_base_m)
                and self.manipulator_reachable(placement_target, placement_base_m)
            ),
            collision_free=self.collision_free_placement(object_path, target_m),
            grasp_frame_available=(
                True if not requires_grasp else self.grasp_frame_available(object_path)
            ),
            placement_stable=self.placement_stable(object_path, target_m),
            disposal_capacity_available=(
                True if not requires_disposal else self.disposal_capacity_available(object_path)
            ),
            robot_available=bool(
                self.stage.GetPrimAtPath(self.config.countermeasure_robot_path).IsValid()
            ),
        )


class IsaacActionCandidateGenerator:
    """Create truth-free planner inputs from USD geometry and a BeliefState."""

    def __init__(
        self,
        stage: Any,
        radiation_simulation: Any,
        *,
        controller: Any | None = None,
        config: SceneCandidateConfig | None = None,
    ) -> None:
        self.stage = stage
        self.simulation = radiation_simulation
        self.config = config or SceneCandidateConfig()
        self.probe = IsaacSceneFeasibilityProbe(stage, self.config, controller)
        self._latest: dict[str, ActionCandidate] = {}

    @staticmethod
    def _attribute(prim: Any, name: str, default: object = None) -> object:
        attribute = prim.GetAttribute(name)
        if not attribute or not attribute.HasAuthoredValueOpinion():
            return default
        value = attribute.Get()
        return default if value is None else value

    @staticmethod
    def _pose(position_m: np.ndarray, *, yaw_rad: float = 0.0) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        cosine = math.cos(yaw_rad)
        sine = math.sin(yaw_rad)
        pose[:2, :2] = np.asarray(((cosine, -sine), (sine, cosine)))
        pose[:3, 3] = np.asarray(position_m, dtype=np.float64)
        return pose

    @staticmethod
    def _safe_name(path: str) -> str:
        return path.strip("/").replace("/", "-").lower()

    def _center(self, prim_or_path: Any) -> np.ndarray:
        bounds = self.probe.bounds(prim_or_path)
        if bounds is not None:
            return (bounds[0] + bounds[1]) * 0.5
        return self.probe.world_position(prim_or_path)

    def _robot_position(self, robot_path: str) -> np.ndarray:
        pose_path = robot_path
        if robot_path == self.config.countermeasure_robot_path:
            pose_path = self.config.countermeasure_pose_path or robot_path
        elif robot_path == self.config.measurement_robot_path:
            pose_path = self.config.measurement_pose_path or robot_path
        return self.probe.world_position(pose_path)

    def _public_source_samples(self, belief: BeliefState) -> tuple[np.ndarray, np.ndarray]:
        strengths_by_path: dict[str, float] = {}
        for basis_id, strength in zip(belief.basis_ids, belief.source_strength_bq, strict=True):
            basis_path = basis_id.split("#", 1)[0]
            strengths_by_path[basis_path] = strengths_by_path.get(basis_path, 0.0) + float(strength)
        positions: list[np.ndarray] = []
        strengths: list[np.ndarray] = []
        visible_sources = [
            source for source in self.simulation.sources if not source.hidden_from_estimator
        ]
        for source in visible_sources:
            value = strengths_by_path.get(source.prim_path)
            if value is None:
                continue
            positions.append(np.asarray(source.positions_m, dtype=np.float64))
            strengths.append(np.full(len(source.positions_m), value / len(source.positions_m)))
        if not positions and visible_sources and float(np.sum(belief.source_strength_bq)) > 0:
            all_positions = np.concatenate(
                [np.asarray(source.positions_m, dtype=np.float64) for source in visible_sources]
            )
            positions.append(all_positions)
            strengths.append(
                np.full(
                    len(all_positions),
                    float(np.sum(belief.source_strength_bq)) / len(all_positions),
                )
            )
        if not positions:
            return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
        return np.concatenate(positions), np.concatenate(strengths)

    def belief_rate_proxy(self, position_m: np.ndarray, belief: BeliefState) -> float:
        origins, strengths = self._public_source_samples(belief)
        if not len(origins):
            return 0.0
        targets = np.repeat(
            np.asarray(position_m, dtype=np.float64).reshape(1, 3), len(origins), axis=0
        )
        distances = np.maximum(
            np.linalg.norm(targets - origins, axis=1),
            self.simulation.configuration.minimum_distance_m,
        )
        transmission = self.simulation.transport.transmission(
            origins, targets, np.asarray([661.657], dtype=np.float64)
        )[:, 0]
        return float(np.sum(strengths * transmission / (4.0 * math.pi * distances**2)))

    def _task_position(self) -> np.ndarray:
        prim = self.stage.GetPrimAtPath(self.config.task_detector_path)
        if prim and prim.IsValid():
            return self.probe.world_position(prim)
        if self.simulation.detectors:
            return np.asarray(self.simulation.detectors[0].position_m, dtype=np.float64)
        raise RuntimeError("the scene has no task detector")

    def _facts(
        self,
        *,
        object_path: str | None,
        target_m: np.ndarray,
        pickup_base_m: np.ndarray,
        placement_base_m: np.ndarray,
        requires_grasp: bool,
        pickup_manipulator_target_m: np.ndarray | None = None,
        placement_manipulator_target_m: np.ndarray | None = None,
        requires_disposal: bool = False,
    ) -> FeasibilityFacts:
        return self.probe.facts(
            object_path=object_path,
            target_m=target_m,
            pickup_base_m=pickup_base_m,
            placement_base_m=placement_base_m,
            requires_grasp=requires_grasp,
            pickup_manipulator_target_m=pickup_manipulator_target_m,
            placement_manipulator_target_m=placement_manipulator_target_m,
            requires_disposal=requires_disposal,
        )

    def _candidate(
        self,
        action: CountermeasureAction,
        belief: BeliefState,
        facts: FeasibilityFacts,
        *,
        target_m: np.ndarray,
        remaining_fraction: float,
        information_fraction: float = 0.0,
        tags: frozenset[str] = frozenset(),
    ) -> ActionCandidate:
        baseline = self.belief_rate_proxy(self._task_position(), belief)
        expected = baseline * float(np.clip(remaining_fraction, 0.0, 1.0))
        uncertainty = float(np.sqrt(max(np.trace(belief.covariance), 0.0)))
        failed_facts = sum(
            not value
            for value in (
                facts.mobile_path_available,
                facts.manipulator_reachable,
                facts.collision_free,
                facts.grasp_frame_available,
                facts.placement_stable,
                facts.disposal_capacity_available,
                facts.robot_available,
            )
        )
        robot_position = self._robot_position(action.robot_id)
        distance = float(np.linalg.norm(np.asarray(target_m) - robot_position))
        dose_rate = expected * self.config.dose_proxy_to_sv_h
        metrics = ActionMetrics(
            expected_task_path_dose_sv=dose_rate * action.predicted_duration_s / 3600.0,
            expected_peak_dose_rate_sv_h=dose_rate,
            residual_source_uncertainty=uncertainty * max(remaining_fraction, 0.05),
            action_time_s=action.predicted_duration_s,
            resource_cost=float(sum(action.resource_cost.values())),
            robot_execution_risk=failed_facts / 7.0,
            expected_information_gain=uncertainty * information_fraction,
            distance_to_target_m=distance,
        )
        candidate = ActionCandidate(action, metrics, facts, tags)
        self._latest[action.action_id] = candidate
        return candidate

    def _base_for_end_effector(
        self,
        target_m: np.ndarray,
        *,
        robot_z: float,
        yaw_rad: float = 0.0,
        offset_m: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        offset = np.asarray(
            self.config.end_effector_offset_m if offset_m is None else offset_m,
            dtype=np.float64,
        )
        cosine = math.cos(yaw_rad)
        sine = math.sin(yaw_rad)
        rotated_offset = np.asarray(
            (
                cosine * offset[0] - sine * offset[1],
                sine * offset[0] + cosine * offset[1],
                offset[2],
            ),
            dtype=np.float64,
        )
        result = np.asarray(target_m, dtype=np.float64) - rotated_offset
        result[2] = robot_z
        return result

    def _route(
        self,
        start_m: np.ndarray,
        target_m: np.ndarray,
        object_path: str,
        *,
        carrying: bool = False,
    ) -> list[list[float]]:
        route = self.probe.plan_mobile_route(
            start_m,
            target_m,
            excluded_paths=((object_path,) if carrying else ()),
            moving_robot_path=self.config.countermeasure_robot_path,
            carried_object_path=object_path if carrying else None,
            carried_base_position_m=start_m if carrying else None,
        )
        return [] if route is None else [waypoint.tolist() for waypoint in route]

    def _measurement_candidates(self, belief: BeliefState) -> list[ActionCandidate]:
        robot = self._robot_position(self.config.measurement_robot_path)
        candidates: list[ActionCandidate] = []
        for prim in self.stage.Traverse():
            if self._attribute(prim, "rad:role", "") != "detector_station":
                continue
            target = self.probe.world_position(prim)
            route = self.probe.plan_mobile_route(
                robot,
                target,
                moving_robot_path=self.config.measurement_robot_path,
            )
            facts = FeasibilityFacts(mobile_path_available=route is not None)
            action = CountermeasureAction(
                action_id=f"measure-{self._safe_name(str(prim.GetPath()))}",
                action_type=ActionType.MEASURE,
                robot_id=self.config.measurement_robot_path,
                target_prim_path=str(prim.GetPath()),
                target_pose_world=self._pose(target),
                parameters={
                    "detector_path": str(prim.GetPath()),
                    "base_route_m": (
                        [] if route is None else [waypoint.tolist() for waypoint in route]
                    ),
                },
                predicted_duration_s=self.config.measurement_duration_s,
            )
            candidates.append(
                self._candidate(
                    action,
                    belief,
                    facts,
                    target_m=target,
                    remaining_fraction=1.0,
                    information_fraction=0.25,
                    tags=frozenset({"measurement", "scene_derived"}),
                )
            )
        return candidates

    def _decon_candidates(self, belief: BeliefState) -> list[ActionCandidate]:
        robot = self._robot_position(self.config.countermeasure_robot_path)
        candidates: list[ActionCandidate] = []
        for prim in self.stage.Traverse():
            if not bool(self._attribute(prim, "rad:decon:enabled", False)):
                continue
            target = self._center(prim)
            base = self._base_for_end_effector(
                target,
                robot_z=float(robot[2]),
                offset_m=self.config.decon_end_effector_offset_m,
            )
            efficiency = float(self._attribute(prim, "rad:decon:efficiencyMean", 0.8))
            facts = self._facts(
                object_path=None,
                target_m=target,
                pickup_base_m=base,
                placement_base_m=base,
                requires_grasp=False,
            )
            route = self.probe.plan_mobile_route(
                robot,
                base,
                moving_robot_path=self.config.countermeasure_robot_path,
            )
            action = CountermeasureAction(
                action_id=f"decon-{self._safe_name(str(prim.GetPath()))}",
                action_type=ActionType.DECONTAMINATE,
                robot_id=self.config.countermeasure_robot_path,
                target_prim_path=str(prim.GetPath()),
                target_region={"surface_path": str(prim.GetPath())},
                target_pose_world=self._pose(target),
                parameters={
                    "surface_path": str(prim.GetPath()),
                    "decon_profile": str(
                        self._attribute(
                            prim,
                            "rad:decon:profile",
                            "full_coverage_serpentine_raster",
                        )
                    ),
                    "raster_rows": int(self._attribute(prim, "rad:decon:rasterRows", 6)),
                    "duration_s": self.config.decon_duration_s,
                    "pickup_base_position_m": base.tolist(),
                    "placement_base_position_m": base.tolist(),
                    "pickup_base_route_m": (
                        [] if route is None else [waypoint.tolist() for waypoint in route]
                    ),
                    "pickup_base_yaw_rad": 0.0,
                    "decon_media": self.config.decon_duration_s,
                },
                predicted_duration_s=self.config.decon_duration_s,
            )
            candidates.append(
                self._candidate(
                    action,
                    belief,
                    facts,
                    target_m=target,
                    remaining_fraction=max(0.0, 1.0 - efficiency),
                    tags=frozenset({"decontamination", "scene_derived"}),
                )
            )
        return candidates

    def _shield_reduction(self, shield: Any) -> float:
        material_id = str(self._attribute(shield, "rad:material:id", "lead"))
        energy, attenuation = self.simulation.configuration.materials[material_id]
        coefficient = float(np.interp(661.657, energy, attenuation))
        bounds = self.probe.bounds(shield)
        thickness = 0.05 if bounds is None else float(np.min(bounds[1] - bounds[0]))
        return float(math.exp(-coefficient * max(thickness, 1.0e-4)))

    def _shield_candidates(self, belief: BeliefState) -> list[ActionCandidate]:
        source_positions, source_strengths = self._public_source_samples(belief)
        if not len(source_positions):
            return []
        source = source_positions[int(np.argmax(source_strengths))]
        protected = self._task_position()
        robot = self._robot_position(self.config.countermeasure_robot_path)
        candidates: list[ActionCandidate] = []
        for shield in self.stage.Traverse():
            if self._attribute(shield, "rad:role", "") != "shield":
                continue
            if not bool(self._attribute(shield, "rad:shield:movable", False)):
                continue
            shield_path = str(shield.GetPath())
            deployed = bool(self._attribute(shield, "rad:shield:deployed", False))
            deployment_state = "deployed" if deployed else "available"
            pickup = self._center(shield)
            shield_root = self.probe.world_position(shield)
            root_from_center = shield_root - pickup
            frame_name = str(self._attribute(shield, "rad:manipulation:graspFrame", ""))
            grasp_position = self.probe.world_position(f"{shield_path.rstrip('/')}/{frame_name}")
            grasp_from_root = grasp_position - shield_root
            pickup_base = self._base_for_end_effector(grasp_position, robot_z=float(robot[2]))
            for fraction in self.config.shield_line_fractions:
                target = source + fraction * (protected - source)
                target[2] = pickup[2]
                target_root = target + root_from_center
                placement_options: list[tuple[float, np.ndarray, np.ndarray, FeasibilityFacts]] = []
                # This panel has one physical service handle on its west face.
                # The controller preserves the payload's world orientation, so
                # a pi-yaw fallback would put the Ridgeback east of the plate
                # while its arm still reaches through to the west handle. That
                # overlaps the chassis and payload at release. Keep the base on
                # the authored handle side until orientation-aware payload
                # rotation or a verified second grasp frame is implemented.
                for placement_yaw in (0.0,):
                    target_grasp = target_root + grasp_from_root
                    placement_base = self._base_for_end_effector(
                        target_grasp,
                        robot_z=float(robot[2]),
                        yaw_rad=placement_yaw,
                    )
                    option_facts = self._facts(
                        object_path=shield_path,
                        target_m=target,
                        pickup_base_m=pickup_base,
                        placement_base_m=placement_base,
                        requires_grasp=True,
                        pickup_manipulator_target_m=grasp_position,
                        placement_manipulator_target_m=target_grasp,
                    )
                    placement_options.append(
                        (placement_yaw, target_grasp, placement_base, option_facts)
                    )
                placement_yaw, target_grasp, placement_base, facts = next(
                    (
                        option
                        for option in placement_options
                        if option[3].mobile_path_available and option[3].manipulator_reachable
                    ),
                    placement_options[0],
                )
                action = CountermeasureAction(
                    action_id=(
                        f"shield-{self._safe_name(shield_path)}-{int(round(fraction * 100)):02d}"
                    ),
                    # The identifier intentionally remains stable across this
                    # state transition.  A confirmed multi-step workflow can
                    # therefore re-resolve the same placement option against
                    # the live scene and receive MOVE_SHIELD after deployment.
                    action_type=(ActionType.MOVE_SHIELD if deployed else ActionType.PLACE_SHIELD),
                    robot_id=self.config.countermeasure_robot_path,
                    target_prim_path=shield_path,
                    target_pose_world=self._pose(target_root),
                    parameters={
                        "object_path": shield_path,
                        "pickup_base_position_m": pickup_base.tolist(),
                        "placement_base_position_m": placement_base.tolist(),
                        "pickup_base_route_m": self._route(robot, pickup_base, shield_path),
                        "placement_base_route_m": self._route(
                            pickup_base,
                            placement_base,
                            shield_path,
                            carrying=True,
                        ),
                        "pickup_base_yaw_rad": 0.0,
                        "placement_base_yaw_rad": placement_yaw,
                        "shield_type": str(self._attribute(shield, "rad:material:id", "default")),
                        "shield_units": (
                            0
                            if deployed
                            else int(self._attribute(shield, "rad:shield:resourceUnits", 1))
                        ),
                        "placement_fraction": float(fraction),
                        "deployment_state": deployment_state,
                    },
                    predicted_duration_s=self.config.shield_duration_s,
                )
                candidates.append(
                    self._candidate(
                        action,
                        belief,
                        facts,
                        target_m=target,
                        remaining_fraction=self._shield_reduction(shield),
                        tags=frozenset({"shield", "scene_derived"}),
                    )
                )
        return candidates

    def _object_candidates(self, belief: BeliefState) -> list[ActionCandidate]:
        robot = self._robot_position(self.config.countermeasure_robot_path)
        zone = self.stage.GetPrimAtPath(self.config.disposal_zone_path)
        if not zone or not zone.IsValid():
            return []
        zone_position = self._center(zone)
        parking_paths = [
            str(prim.GetPath())
            for prim in self.stage.Traverse()
            if self._attribute(prim, "rad:manipulation:graspFrame", None) is not None
            and self._attribute(prim, "rad:role", "") != "shield"
        ]
        candidates: list[ActionCandidate] = []
        for prim in self.stage.Traverse():
            if not bool(self._attribute(prim, "rad:manipulation:movable", False)):
                continue
            if self._attribute(prim, "rad:role", "") == "shield":
                continue
            object_path = str(prim.GetPath())
            pickup = self._center(prim)
            object_root = self.probe.world_position(prim)
            root_from_center = object_root - pickup
            frame_name = str(self._attribute(prim, "rad:manipulation:graspFrame", ""))
            grasp_position = self.probe.world_position(f"{object_path.rstrip('/')}/{frame_name}")
            grasp_from_root = grasp_position - object_root
            pickup_base = self._base_for_end_effector(
                grasp_position,
                robot_z=float(robot[2]),
                offset_m=self.config.object_end_effector_offset_m,
            )
            parking_slot = parking_paths.index(object_path)
            configured_offsets = self.config.object_parking_offsets_m
            offset_index = parking_slot % len(configured_offsets)
            offset_ring = parking_slot // len(configured_offsets)
            offset_xy = np.asarray(configured_offsets[offset_index], dtype=np.float64)
            offset_xy[0] += 1.8 * offset_ring
            parking = zone_position + np.asarray((offset_xy[0], offset_xy[1], 0.0))
            parking[2] = pickup[2]
            parking_root = parking + root_from_center
            placement_base = self._base_for_end_effector(
                parking_root + grasp_from_root,
                robot_z=float(robot[2]),
                offset_m=self.config.object_end_effector_offset_m,
            )
            facts = self._facts(
                object_path=object_path,
                target_m=parking,
                pickup_base_m=pickup_base,
                placement_base_m=placement_base,
                requires_grasp=True,
                pickup_manipulator_target_m=grasp_position,
                placement_manipulator_target_m=parking_root + grasp_from_root,
            )
            action = CountermeasureAction(
                action_id=f"move-{self._safe_name(object_path)}",
                action_type=ActionType.MOVE_OBJECT,
                robot_id=self.config.countermeasure_robot_path,
                target_prim_path=object_path,
                target_pose_world=self._pose(parking_root),
                parameters={
                    "object_path": object_path,
                    "pickup_base_position_m": pickup_base.tolist(),
                    "placement_base_position_m": placement_base.tolist(),
                    "pickup_base_route_m": self._route(robot, pickup_base, object_path),
                    "placement_base_route_m": self._route(
                        pickup_base,
                        placement_base,
                        object_path,
                        carrying=True,
                    ),
                    "pickup_base_yaw_rad": 0.0,
                    "placement_base_yaw_rad": 0.0,
                },
                predicted_duration_s=self.config.object_duration_s,
            )
            candidates.append(
                self._candidate(
                    action,
                    belief,
                    facts,
                    target_m=parking,
                    remaining_fraction=(0.5 if object_path in belief.basis_ids else 1.0),
                    tags=frozenset({"object_move", "scene_derived"}),
                )
            )
            if not bool(self._attribute(prim, "rad:manipulation:removable", False)):
                continue
            # A removable object may first have been parked next to the
            # disposal zone.  Re-approaching that grasp from yaw zero places
            # the chassis between the payload and the zone and leaves the
            # elevated pre-grasp outside Franka's collision-free workspace.
            # Approach from the opposite side while preserving the validated
            # 0.90 m object stand-off.
            removal_pickup_yaw = math.pi
            removal_pickup_base = self._base_for_end_effector(
                grasp_position,
                robot_z=float(robot[2]),
                yaw_rad=removal_pickup_yaw,
                offset_m=self.config.object_end_effector_offset_m,
            )
            removal_target = zone_position.copy()
            removal_target[2] = pickup[2]
            removal_root = removal_target + root_from_center
            removal_yaw = math.pi
            # The controller keeps the payload's world orientation fixed
            # while the base turns.  Preserve the authored grasp offset too;
            # rotating it here would send the carried drum one metre north
            # before asking the arm to move it back at the disposal zone.
            removal_grasp = removal_root + grasp_from_root
            removal_base = self._base_for_end_effector(
                removal_grasp,
                robot_z=float(robot[2]),
                yaw_rad=removal_yaw,
                offset_m=self.config.object_end_effector_offset_m,
            )
            removal_facts = self._facts(
                object_path=object_path,
                target_m=removal_target,
                pickup_base_m=removal_pickup_base,
                placement_base_m=removal_base,
                requires_grasp=True,
                pickup_manipulator_target_m=grasp_position,
                placement_manipulator_target_m=removal_grasp,
                requires_disposal=True,
            )
            remove = CountermeasureAction(
                action_id=f"remove-{self._safe_name(object_path)}",
                action_type=ActionType.REMOVE_OBJECT,
                robot_id=self.config.countermeasure_robot_path,
                target_prim_path=object_path,
                target_pose_world=self._pose(removal_root, yaw_rad=removal_yaw),
                parameters={
                    "object_path": object_path,
                    "disposal_zone_path": self.config.disposal_zone_path,
                    "pickup_base_position_m": removal_pickup_base.tolist(),
                    "placement_base_position_m": removal_base.tolist(),
                    "pickup_base_route_m": self._route(robot, removal_pickup_base, object_path),
                    "placement_base_route_m": self._route(
                        removal_pickup_base,
                        removal_base,
                        object_path,
                        carrying=True,
                    ),
                    "pickup_base_yaw_rad": removal_pickup_yaw,
                    "placement_base_yaw_rad": removal_yaw,
                    # Removal ends inside a bounded disposal zone.  Allow the
                    # released rigid body to settle naturally, then apply the
                    # stricter zone-containment check before disabling it.
                    "placement_settle_tolerance_m": 0.25,
                },
                predicted_duration_s=self.config.object_duration_s,
            )
            candidates.append(
                self._candidate(
                    remove,
                    belief,
                    removal_facts,
                    target_m=removal_target,
                    remaining_fraction=(0.0 if object_path in belief.basis_ids else 1.0),
                    tags=frozenset({"object_remove", "scene_derived"}),
                )
            )
        return candidates

    def generate_all(
        self, belief: BeliefState, diagnosis: object | None = None
    ) -> tuple[ActionCandidate, ...]:
        del diagnosis
        self.probe.invalidate_collision_cache()
        self._latest.clear()
        candidates = [
            *self._measurement_candidates(belief),
            *self._decon_candidates(belief),
            *self._shield_candidates(belief),
            *self._object_candidates(belief),
        ]
        return tuple(candidates)

    def generate_measurement_actions(self, belief: BeliefState) -> tuple[ActionCandidate, ...]:
        self.probe.invalidate_collision_cache()
        return tuple(self._measurement_candidates(belief))

    def generate_decon_actions(self, belief: BeliefState) -> tuple[ActionCandidate, ...]:
        self.probe.invalidate_collision_cache()
        return tuple(self._decon_candidates(belief))

    def generate_shield_actions(self, belief: BeliefState) -> tuple[ActionCandidate, ...]:
        self.probe.invalidate_collision_cache()
        return tuple(self._shield_candidates(belief))

    def generate_move_remove_actions(self, belief: BeliefState) -> tuple[ActionCandidate, ...]:
        self.probe.invalidate_collision_cache()
        return tuple(self._object_candidates(belief))

    def preview(self, action: CountermeasureAction, belief: BeliefState) -> dict[str, object]:
        candidate = self._latest.get(action.action_id)
        if candidate is None:
            raise KeyError(f"action was not generated from the current scene: {action.action_id}")
        baseline = self.belief_rate_proxy(self._task_position(), belief)
        peak = candidate.metrics.expected_peak_dose_rate_sv_h
        predicted_rate = peak / self.config.dose_proxy_to_sv_h
        return {
            "action_id": action.action_id,
            "detector_path": self.config.task_detector_path,
            "baseline_rate_proxy_cps": baseline,
            "predicted_rate_proxy_cps": predicted_rate,
            "belief_revision": vars(belief.revision),
            "truth_accessed": False,
        }

    def evaluate_task(self, belief: BeliefState) -> tuple[float, float]:
        rate = self.belief_rate_proxy(self._task_position(), belief)
        peak_sv_h = rate * self.config.dose_proxy_to_sv_h
        return peak_sv_h / 3600.0, peak_sv_h
