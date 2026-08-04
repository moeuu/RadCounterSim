"""Import and control arbitrary USD, URDF, Xacro, and MJCF robots."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np

from radcounter.core.robots import (
    JointCommand,
    JointControlMode,
    RobotCapabilities,
    RobotFleetConfig,
    RobotFormat,
    RobotImportConfig,
    RobotType,
    SpatialVelocityCommand,
    TwistCommand,
    interpolate_gripper,
    resolve_joint_groups,
    spatial_to_joint_velocity,
    twist_to_joint_velocity,
)


class GenericRobotImporter:
    def __init__(self, cache_directory: str | Path = ".cache/robots") -> None:
        self.cache_directory = Path(cache_directory).expanduser().resolve()
        self.cache_directory.mkdir(parents=True, exist_ok=True)

    def import_robot(self, stage, config: RobotImportConfig) -> str:
        from pxr import Gf, Sdf, UsdGeom

        source = self._resolve(config.uri, config.package_paths)
        usd_path = self._convert(source, config)
        prim_path = config.prim_path or f"/World/Robots/{_identifier(config.id)}"
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(str(usd_path))
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*config.translation_m))
        rotation = config.rotation_rpy_deg
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        prim.CreateAttribute("rad:robot:id", Sdf.ValueTypeNames.String).Set(config.id)
        prim.CreateAttribute("rad:robot:type", Sdf.ValueTypeNames.String).Set(
            config.robot_type.value
        )
        prim.CreateAttribute("rad:stream:focus", Sdf.ValueTypeNames.Bool).Set(True)
        return self._find_articulation_root(stage, prim_path)

    def _convert(self, source: Path, config: RobotImportConfig) -> Path:
        robot_format = config.resolved_format
        if robot_format is RobotFormat.USD:
            return source
        if robot_format is RobotFormat.XACRO:
            source = self._expand_xacro(source, config.package_paths)
            robot_format = RobotFormat.URDF
        digest = hashlib.sha256()
        digest.update(source.read_bytes())
        digest.update(config.model_dump_json().encode("utf-8"))
        output = self.cache_directory / digest.hexdigest() / f"{config.id}.usd"
        if output.exists():
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        if robot_format is RobotFormat.URDF:
            from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

            importer_config = URDFImporterConfig(
                urdf_path=str(source),
                usd_path=str(output),
                merge_fixed_joints=config.merge_fixed_joints,
                merge_mesh=config.merge_meshes,
                collision_from_visuals=config.collision_from_visuals,
                collision_type=config.collision_type,
                allow_self_collision=config.allow_self_collision,
                ros_package_paths=[
                    {"path": str(Path(path).expanduser().resolve())}
                    for path in config.package_paths
                ],
                robot_type=_isaac_robot_type(config),
                fix_base=config.fixed_base,
            )
            imported = URDFImporter(importer_config).import_urdf()
        elif robot_format is RobotFormat.MJCF:
            from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

            importer_config = MJCFImporterConfig(
                mjcf_path=str(source),
                usd_path=str(output),
                import_scene=True,
                merge_mesh=config.merge_meshes,
                collision_from_visuals=config.collision_from_visuals,
                collision_type=config.collision_type,
                allow_self_collision=config.allow_self_collision,
                robot_type=_isaac_robot_type(config),
                fix_base=config.fixed_base,
            )
            imported = MJCFImporter(importer_config).import_mjcf()
        else:
            raise ValueError(f"Unsupported robot format: {robot_format.value}")
        return Path(imported or output).expanduser().resolve()

    def _expand_xacro(self, source: Path, package_paths: tuple[str, ...]) -> Path:
        executable = shutil.which("xacro")
        if executable is None:
            raise RuntimeError("Xacro import requires the ROS xacro executable")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        target = self.cache_directory / "xacro" / digest / f"{source.stem}.urdf"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            environment = None
            if package_paths:
                import os

                environment = dict(os.environ)
                existing = environment.get("AMENT_PREFIX_PATH", "")
                environment["AMENT_PREFIX_PATH"] = ":".join((*package_paths, existing))
            with target.open("w", encoding="utf-8") as stream:
                subprocess.run(
                    [executable, str(source)],
                    check=True,
                    stdout=stream,
                    env=environment,
                )
        return target

    def _resolve(self, uri: str, package_paths: tuple[str, ...]) -> Path:
        if uri.startswith("package://"):
            relative = uri.removeprefix("package://")
            package, _, rest = relative.partition("/")
            for root in package_paths:
                root_path = Path(root).expanduser().resolve()
                candidates = (root_path / package / rest, root_path / "share" / package / rest)
                for candidate in candidates:
                    if candidate.exists():
                        return candidate.resolve()
            raise FileNotFoundError(f"ROS package URI not found: {uri}")
        parsed = urlparse(uri)
        if parsed.scheme in {"http", "https"}:
            suffix = Path(parsed.path).suffix
            target = (
                self.cache_directory
                / "downloads"
                / (hashlib.sha256(uri.encode("utf-8")).hexdigest() + suffix)
            )
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                urlretrieve(uri, target)
            return target
        path = Path(parsed.path if parsed.scheme == "file" else uri).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _find_articulation_root(stage, root_path: str) -> str:
        from pxr import Usd, UsdPhysics

        root = stage.GetPrimAtPath(root_path)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return str(prim.GetPath())
        raise RuntimeError(f"Imported robot at {root_path} has no UsdPhysics.ArticulationRootAPI")


class GenericArticulationController:
    """Thin named-DOF facade over Isaac Sim 6 experimental Articulation."""

    def __init__(self, articulation_path: str, config: RobotImportConfig) -> None:
        self.articulation_path = articulation_path
        self.config = config
        self._articulation = None
        self._dof_names: tuple[str, ...] = ()
        self._groups: dict[str, tuple[str, ...]] = {}
        self._ik_solver = None

    def bind(self) -> RobotCapabilities:
        from isaacsim.core.experimental.prims import Articulation

        self._articulation = Articulation(self.articulation_path)
        self._articulation.initialize_cpp_data_view()
        self._dof_names = tuple(str(name) for name in self._articulation.dof_names)
        self._groups, unresolved = resolve_joint_groups(
            self._dof_names,
            self.config.joint_groups,
        )
        if self.config.lula_kinematics is not None:
            from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

            lula = self.config.lula_kinematics
            self._ik_solver = LulaKinematicsSolver(
                lula.robot_description_path,
                lula.urdf_path,
            )
        return RobotCapabilities(
            dof_names=self._dof_names,
            groups=self._groups,
            has_base_controller=self.config.base_controller is not None,
            grippers=tuple(item.name for item in self.config.grippers),
            has_inverse_kinematics=self.config.lula_kinematics is not None,
            unresolved_groups=unresolved,
            supports_spatial_velocity=(
                self.config.robot_type in {RobotType.AERIAL, RobotType.FLOATING}
                or (
                    self.config.base_controller is not None
                    and self.config.base_controller.type.lower() in {"aerial", "floating"}
                )
            ),
        )

    def command_joint(self, command: JointCommand) -> None:
        articulation = self._require_bound()
        indices = self._indices(command.names)
        values = np.asarray([command.values], dtype=np.float32)
        mode = JointControlMode(command.mode)
        if mode is JointControlMode.POSITION:
            articulation.set_dof_position_targets(values, dof_indices=indices)
        elif mode is JointControlMode.VELOCITY:
            articulation.set_dof_velocity_targets(values, dof_indices=indices)
        else:
            articulation.set_dof_efforts(values, dof_indices=indices)

    def command_group(self, group_name: str, values: tuple[float, ...]) -> None:
        group = next(item for item in self.config.joint_groups if item.name == group_name)
        names = self._groups.get(group_name, ())
        if len(names) != len(values):
            raise ValueError(f"Group {group_name!r} expects {len(names)} values")
        self.command_joint(JointCommand(names, values, group.mode.value))

    def command_twist(self, command: TwistCommand) -> None:
        articulation = self._require_bound()
        base = self.config.base_controller
        if base is None:
            raise RuntimeError("Robot has no base_controller")
        values = twist_to_joint_velocity(command, base)
        if base.type.lower() == "floating" and not base.joint_names:
            linear = np.asarray(
                [[command.linear_x_m_s, command.linear_y_m_s, 0.0]],
                dtype=np.float32,
            )
            angular = np.asarray([[0.0, 0.0, command.angular_z_rad_s]], dtype=np.float32)
            articulation.set_velocities(linear, angular)
            return
        articulation.set_dof_velocity_targets(
            np.asarray([values], dtype=np.float32),
            dof_indices=self._indices(base.joint_names),
        )

    def command_spatial_velocity(self, command: SpatialVelocityCommand) -> None:
        articulation = self._require_bound()
        base = self.config.base_controller
        if base is None:
            raise RuntimeError("Robot has no base_controller")
        values = spatial_to_joint_velocity(command, base)
        if not base.joint_names:
            articulation.set_velocities(
                np.asarray([values[:3]], dtype=np.float32),
                np.asarray([values[3:]], dtype=np.float32),
            )
            return
        articulation.set_dof_velocity_targets(
            np.asarray([values], dtype=np.float32),
            dof_indices=self._indices(base.joint_names),
        )

    def command_gripper(self, name: str, fraction_closed: float) -> None:
        config = next(item for item in self.config.grippers if item.name == name)
        values = interpolate_gripper(config, fraction_closed)
        self.command_joint(
            JointCommand(
                names=config.joint_names,
                values=tuple(float(value) for value in values),
                mode=JointControlMode.POSITION.value,
            )
        )

    def command_cartesian(
        self,
        position_m: tuple[float, float, float],
        orientation_wxyz: tuple[float, float, float, float] | None = None,
    ) -> bool:
        if self._ik_solver is None or self.config.lula_kinematics is None:
            raise RuntimeError("Robot has no configured Lula kinematics solver")
        orientation = (
            None if orientation_wxyz is None else np.asarray(orientation_wxyz, dtype=np.float64)
        )
        positions, success = self._ik_solver.compute_inverse_kinematics(
            self.config.lula_kinematics.end_effector_frame,
            np.asarray(position_m, dtype=np.float64),
            orientation,
        )
        if success:
            names = tuple(str(name) for name in self._ik_solver.get_joint_names())
            self.command_joint(
                JointCommand(
                    names=names,
                    values=tuple(float(value) for value in positions),
                    mode=JointControlMode.POSITION.value,
                )
            )
        return bool(success)

    def joint_state(self) -> dict[str, tuple[float, float]]:
        articulation = self._require_bound()
        positions = np.asarray(articulation.get_dof_positions()).reshape(-1)
        velocities = np.asarray(articulation.get_dof_velocities()).reshape(-1)
        return {
            name: (float(positions[index]), float(velocities[index]))
            for index, name in enumerate(self._dof_names)
        }

    def _indices(self, names: tuple[str, ...]) -> list[int]:
        missing = [name for name in names if name not in self._dof_names]
        if missing:
            raise KeyError(f"Unknown DOFs: {', '.join(missing)}")
        return [self._dof_names.index(name) for name in names]

    def _require_bound(self):
        if self._articulation is None:
            raise RuntimeError("Controller must be bound after the stage starts simulating")
        return self._articulation


class RobotFleetManager:
    def __init__(
        self,
        stage,
        fleet: RobotFleetConfig,
        cache_directory: str | Path = ".cache/robots",
    ) -> None:
        self.stage = stage
        self.fleet = fleet
        self.importer = GenericRobotImporter(cache_directory)
        self.controllers: dict[str, GenericArticulationController] = {}

    def import_all(self) -> Mapping[str, str]:
        paths = {}
        for config in self.fleet.robots:
            path = self.importer.import_robot(self.stage, config)
            paths[config.id] = path
            self.controllers[config.id] = GenericArticulationController(path, config)
        return paths

    def bind_all(self) -> Mapping[str, RobotCapabilities]:
        return {robot_id: controller.bind() for robot_id, controller in self.controllers.items()}

    def controller(self, robot_id: str) -> GenericArticulationController:
        return self.controllers[robot_id]


def _identifier(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in value)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"Robot_{cleaned}"


def _isaac_robot_type(config: RobotImportConfig) -> str:
    values = {
        "default": "Default",
        "end_effector": "End Effector",
        "manipulator": "Manipulator",
        "humanoid": "Humanoid",
    }
    return values.get(config.robot_type.value, "Default")
