"""Configuration-driven RTX and custom sensors mounted on arbitrary robots."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from radcounter.core.robots import (
    RobotFleetConfig,
    RobotImportConfig,
    RobotSensorBackendRegistry,
    RobotSensorType,
    SensorMountConfig,
)


def _identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return result if result and not result[0].isdigit() else f"sensor_{result}"


def _quaternion_wxyz(rotation_rpy_deg: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = (math.radians(value) * 0.5 for value in rotation_rpy_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        ],
        dtype=np.float64,
    )


@dataclass
class MountedIsaacSensor:
    robot_id: str
    config: SensorMountConfig
    prim_path: str
    runtime: object | None = None
    drawing_point_cloud: bool = False

    def read(self, channel: str | None = None) -> object:
        if self.runtime is None:
            raise RuntimeError(
                f"Sensor {self.robot_id}/{self.config.id} is pose-only; "
                "configure a backend to produce measurements"
            )
        if hasattr(self.runtime, "get_data"):
            selected = channel or self._default_channel()
            return self.runtime.get_data(selected)
        if hasattr(self.runtime, "read"):
            return self.runtime.read(channel)
        raise TypeError(f"Sensor runtime at {self.prim_path} has no read API")

    def close(self) -> None:
        if self.drawing_point_cloud and hasattr(self.runtime, "detach_writer"):
            self.runtime.detach_writer("draw-point-cloud")
        if hasattr(self.runtime, "close"):
            self.runtime.close()

    def _default_channel(self) -> str:
        if self.config.lidar is not None:
            return self.config.lidar.annotators[0]
        if self.config.camera is not None:
            return self.config.camera.annotators[0]
        return ""


class IsaacRobotSensorRigManager:
    """Mount and read sensors for every robot in a heterogeneous fleet."""

    def __init__(
        self,
        stage,
        backend_registry: RobotSensorBackendRegistry | None = None,
        load_entry_point_plugins: bool = True,
    ) -> None:
        self.stage = stage
        self.backends = backend_registry or RobotSensorBackendRegistry()
        if load_entry_point_plugins:
            self.backends.load_entry_points()
        self._mounted: dict[tuple[str, str], MountedIsaacSensor] = {}
        self._rtx_enabled = False

    def mount_fleet(self, fleet: RobotFleetConfig) -> Mapping[tuple[str, str], MountedIsaacSensor]:
        for robot in fleet.robots:
            self.mount_robot(robot)
        return dict(self._mounted)

    def mount_robot(
        self,
        robot: RobotImportConfig,
        robot_root_path: str | None = None,
    ) -> tuple[MountedIsaacSensor, ...]:
        root = robot_root_path or robot.prim_path or f"/World/Robots/{_identifier(robot.id)}"
        mounted = []
        for config in robot.sensor_mounts:
            if not config.enabled:
                continue
            key = (robot.id, config.id)
            if key in self._mounted:
                raise ValueError(f"Sensor {robot.id}/{config.id} is already mounted")
            item = self._mount(robot.id, root, config)
            self._mounted[key] = item
            mounted.append(item)
        return tuple(mounted)

    def sensor(self, robot_id: str, sensor_id: str) -> MountedIsaacSensor:
        try:
            return self._mounted[(robot_id, sensor_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown mounted sensor: {robot_id}/{sensor_id}") from exc

    def read(self, robot_id: str, sensor_id: str, channel: str | None = None) -> object:
        return self.sensor(robot_id, sensor_id).read(channel)

    def world_pose(
        self,
        robot_id: str,
        sensor_id: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        from pxr import Usd, UsdGeom

        item = self.sensor(robot_id, sensor_id)
        prim = self.stage.GetPrimAtPath(item.prim_path)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform.ExtractTranslation()
        quaternion = transform.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        return (
            tuple(float(value) for value in translation),
            (
                float(quaternion.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ),
        )

    def close(self) -> None:
        for item in self._mounted.values():
            item.close()
        self._mounted.clear()

    def _mount(
        self,
        robot_id: str,
        robot_root_path: str,
        config: SensorMountConfig,
    ) -> MountedIsaacSensor:
        parent = self._parent_path(robot_root_path, config.parent_link)
        if not self.stage.GetPrimAtPath(parent).IsValid():
            raise ValueError(
                f"Parent link {parent!r} for sensor {robot_id}/{config.id} does not exist"
            )
        container = f"{parent}/RadCounterSensors"
        self.stage.DefinePrim(container, "Xform")
        path = f"{container}/{_identifier(config.id)}"

        if config.sensor_type is RobotSensorType.LIDAR:
            runtime, drawing = self._mount_lidar(path, config)
        elif config.sensor_type is RobotSensorType.CAMERA:
            runtime, drawing = self._mount_camera(path, config)
        else:
            self._author_xform(path, config)
            runtime, drawing = self._mount_configured_backend(path, config), False

        self._author_metadata(path, robot_id, config)
        return MountedIsaacSensor(robot_id, config, path, runtime, drawing)

    def _mount_lidar(self, path: str, mount: SensorMountConfig) -> tuple[object, bool]:
        self._ensure_rtx_extension()
        from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor

        config = mount.lidar
        assert config is not None
        create_args: dict[str, object] = {
            "config": config.model,
            "translations": np.asarray([mount.pose.translation_m], dtype=np.float64),
            "orientations": _quaternion_wxyz(mount.pose.rotation_rpy_deg),
        }
        if config.variant is not None:
            create_args["variant"] = config.variant
        authored = Lidar.create(path, **create_args)
        runtime = LidarSensor(authored, annotators=list(config.annotators))
        if config.draw_point_cloud:
            runtime.attach_writer("draw-point-cloud")
        return runtime, config.draw_point_cloud

    def _mount_camera(self, path: str, mount: SensorMountConfig) -> tuple[object, bool]:
        self._ensure_rtx_extension()
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from pxr import UsdGeom

        config = mount.camera
        assert config is not None
        authored = RtxCamera(
            path=path,
            translations=np.asarray([mount.pose.translation_m], dtype=np.float64),
            orientations=_quaternion_wxyz(mount.pose.rotation_rpy_deg),
            tick_rate=mount.update_rate_hz,
        )
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
        camera.GetFocalLengthAttr().Set(config.focal_length_mm)
        camera.GetHorizontalApertureAttr().Set(config.horizontal_aperture_mm)
        if config.vertical_aperture_mm is not None:
            camera.GetVerticalApertureAttr().Set(config.vertical_aperture_mm)
        camera.GetClippingRangeAttr().Set(config.clipping_range_m)
        runtime = CameraSensor(
            authored,
            resolution=(config.height_px, config.width_px),
            annotators=list(config.annotators),
        )
        return runtime, False

    def _mount_configured_backend(self, path: str, mount: SensorMountConfig) -> object | None:
        backend = None
        options: dict[str, object] = {}
        if mount.custom is not None:
            backend = mount.custom.backend
            options = mount.custom.options
        elif mount.radiation is not None:
            backend = mount.radiation.backend
        if backend is None:
            return None
        return self.backends.create(
            backend,
            stage=self.stage,
            prim_path=path,
            mount=mount,
            options=options,
            pose_provider=lambda: self._world_pose_for_path(path),
        )

    def _author_xform(self, path: str, mount: SensorMountConfig) -> None:
        from pxr import Gf, UsdGeom

        xform = UsdGeom.Xform.Define(self.stage, path)
        xformable = UsdGeom.Xformable(xform.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*mount.pose.translation_m))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*mount.pose.rotation_rpy_deg))

    def _author_metadata(
        self,
        path: str,
        robot_id: str,
        mount: SensorMountConfig,
    ) -> None:
        from pxr import Sdf

        prim = self.stage.GetPrimAtPath(path)
        values = {
            "rad:sensor:id": mount.id,
            "rad:sensor:robotId": robot_id,
            "rad:sensor:type": mount.sensor_type.value,
            "rad:sensor:frameId": mount.effective_frame_id,
            "rad:sensor:updateRateHz": mount.update_rate_hz,
            "rad:sensor:rosEnabled": mount.ros.enabled,
            "rad:sensor:rosTopics": json.dumps(mount.ros.topics, sort_keys=True),
        }
        for key, value in values.items():
            if isinstance(value, bool):
                value_type = Sdf.ValueTypeNames.Bool
            elif isinstance(value, float):
                value_type = Sdf.ValueTypeNames.Double
            else:
                value_type = Sdf.ValueTypeNames.String
            prim.CreateAttribute(key, value_type).Set(value)
        for key, value in mount.metadata.items():
            prim.CreateAttribute(
                f"rad:sensor:meta:{_identifier(key)}",
                Sdf.ValueTypeNames.String,
            ).Set(str(value))

    def _ensure_rtx_extension(self) -> None:
        if self._rtx_enabled:
            return
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.sensors.experimental.rtx", True)
        self._rtx_enabled = True

    def _world_pose_for_path(
        self,
        path: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        from pxr import Usd, UsdGeom

        prim = self.stage.GetPrimAtPath(path)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform.ExtractTranslation()
        quaternion = transform.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        return (
            tuple(float(value) for value in translation),
            (
                float(quaternion.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ),
        )

    @staticmethod
    def _parent_path(robot_root_path: str, parent_link: str) -> str:
        if not parent_link:
            return robot_root_path.rstrip("/")
        if parent_link.startswith("/"):
            return parent_link.rstrip("/")
        return f"{robot_root_path.rstrip('/')}/{parent_link.strip('/')}"
