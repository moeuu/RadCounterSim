"""Portable robot sensor-rig descriptions and custom backend boundary."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RobotSensorType(StrEnum):
    LIDAR = "lidar"
    CAMERA = "camera"
    RADIATION = "radiation"
    IMU = "imu"
    GNSS = "gnss"
    CUSTOM = "custom"


class LidarDimension(StrEnum):
    PLANAR = "2d"
    VOLUMETRIC = "3d"


class SensorPoseConfig(BaseModel):
    """Sensor-to-parent-link transform using ROS-style xyz and fixed-axis rpy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


class RosSensorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    namespace: str = ""
    frame_id: str | None = None
    publish_tf: bool = True
    qos_profile: str = "sensor_data"
    topics: dict[str, str] = Field(default_factory=dict)


class LidarSensorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: LidarDimension = LidarDimension.VOLUMETRIC
    model: str = "Example_Rotary"
    variant: str | None = None
    annotators: tuple[str, ...] = ("generic-model-output",)
    draw_point_cloud: bool = False
    minimum_range_m: float | None = None
    maximum_range_m: float | None = None
    horizontal_fov_deg: float | None = None
    vertical_fov_deg: float | None = None
    horizontal_resolution_deg: float | None = None
    vertical_channels: int | None = None
    range_noise_std_m: float = 0.0


class CameraSensorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width_px: int = Field(default=1280, ge=1)
    height_px: int = Field(default=720, ge=1)
    annotators: tuple[str, ...] = ("rgb",)
    focal_length_mm: float = Field(default=18.0, gt=0.0)
    horizontal_aperture_mm: float = Field(default=20.955, gt=0.0)
    vertical_aperture_mm: float | None = Field(default=None, gt=0.0)
    clipping_range_m: tuple[float, float] = (0.05, 1000.0)
    lens_model: str = "pinhole"
    distortion_coefficients: tuple[float, ...] = ()

    @model_validator(mode="after")
    def validate_clip_range(self) -> CameraSensorConfig:
        near, far = self.clipping_range_m
        if near <= 0.0 or far <= near:
            raise ValueError("Camera clipping range must satisfy 0 < near < far")
        if not self.annotators:
            raise ValueError("Camera requires at least one annotator")
        return self


class RadiationSensorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detector_model_id: str
    detector_config_uri: str | None = None
    integration_time_s: float = Field(default=1.0, gt=0.0)
    backend: str | None = None


class CustomSensorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str
    options: dict[str, object] = Field(default_factory=dict)


class SensorMountConfig(BaseModel):
    """One configurable sensor mounted to a robot link."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    id: str
    sensor_type: RobotSensorType = Field(alias="type")
    parent_link: str = ""
    frame_id: str | None = None
    enabled: bool = True
    update_rate_hz: float = Field(default=10.0, gt=0.0)
    pose: SensorPoseConfig = Field(default_factory=SensorPoseConfig)
    ros: RosSensorConfig = Field(default_factory=RosSensorConfig)
    lidar: LidarSensorConfig | None = None
    camera: CameraSensorConfig | None = None
    radiation: RadiationSensorConfig | None = None
    custom: CustomSensorConfig | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> SensorMountConfig:
        required = {
            RobotSensorType.LIDAR: self.lidar,
            RobotSensorType.CAMERA: self.camera,
            RobotSensorType.RADIATION: self.radiation,
            RobotSensorType.CUSTOM: self.custom,
        }
        if self.sensor_type in required and required[self.sensor_type] is None:
            raise ValueError(f"Sensor {self.id!r} requires a {self.sensor_type.value} block")
        return self

    @property
    def effective_frame_id(self) -> str:
        return self.frame_id or self.ros.frame_id or self.id


class SensorRigConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    sensors: tuple[SensorMountConfig, ...]

    def model_post_init(self, __context: object) -> None:
        ids = [sensor.id for sensor in self.sensors]
        if len(ids) != len(set(ids)):
            raise ValueError("Sensor IDs must be unique within a rig")


class RobotSensorRuntime(Protocol):
    def read(self, channel: str | None = None) -> object: ...
    def close(self) -> None: ...


RobotSensorBackendFactory = Callable[..., RobotSensorRuntime]


class RobotSensorBackendRegistry:
    """Factories for vendor SDKs, physical sensors, and project-specific sensors."""

    ENTRY_POINT_GROUP = "radcounter.robot_sensors"

    def __init__(self) -> None:
        self._factories: dict[str, RobotSensorBackendFactory] = {}

    def register(self, name: str, factory: RobotSensorBackendFactory) -> None:
        if name in self._factories:
            raise ValueError(f"Sensor backend {name!r} is already registered")
        self._factories[name] = factory

    def load_entry_points(self) -> None:
        for item in entry_points(group=self.ENTRY_POINT_GROUP):
            self.register(item.name, item.load())

    def create(self, name: str, **context: object) -> RobotSensorRuntime:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"Unknown robot sensor backend: {name}") from exc
        return factory(**context)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def load_sensor_rig(path: str | Path) -> SensorRigConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return SensorRigConfig.model_validate(payload)
