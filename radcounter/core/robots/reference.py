"""Traceable catalog of real robots represented by bundled simulator models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .models import RobotGeometryFidelity, RobotReferenceConfig


@dataclass(frozen=True, slots=True)
class SensorHardpoint:
    id: str
    parent_link: str
    translation_m: tuple[float, float, float]
    rotation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    intended_sensor: str = ""


@dataclass(frozen=True, slots=True)
class RealRobotReference:
    id: str
    manufacturer: str
    model: str
    roles: tuple[str, ...]
    source_urls: tuple[str, ...]
    dimensions_m: tuple[float, float, float]
    mass_kg: float | None
    locomotion: str
    mechanisms: tuple[str, ...]
    maximum_speed_m_s: float | None = None
    nominal_endurance_s: float | None = None
    sensor_hardpoints: tuple[SensorHardpoint, ...] = ()

    def config(
        self,
        fidelity: RobotGeometryFidelity = RobotGeometryFidelity.REFERENCE_PROCEDURAL,
    ) -> RobotReferenceConfig:
        return RobotReferenceConfig(
            model_id=self.id,
            manufacturer=self.manufacturer,
            model=self.model,
            source_urls=self.source_urls,
            geometry_fidelity=fidelity,
            dimensions_m=self.dimensions_m,
            mass_kg=self.mass_kg,
        )


PACKBOT_FUKUSHIMA = RealRobotReference(
    id="irobot-packbot-fukushima",
    manufacturer="iRobot (PackBot product line now Teledyne FLIR)",
    model="PackBot, Fukushima Daiichi deployment configuration",
    roles=("radiation survey", "CBRN reconnaissance", "inspection", "communications relay"),
    source_urls=(
        "https://www.tepco.co.jp/en/nu/fukushima-np/f1-roadmap/images/11042801a-e.pdf",
        "https://photo.tepco.co.jp/en/date/2011/201104-e/110417-01e.html",
        "https://defense.flir.com/defense-products/packbot-525/",
    ),
    dimensions_m=(0.70, 0.53, 0.18),
    mass_kg=35.0,
    locomotion="dual crawler tracks with independently articulated front flippers",
    mechanisms=(
        "stair-capable tracked mobility",
        "remote pan/tilt inspection camera",
        "modular CBRN and radiation payload interfaces",
    ),
    maximum_speed_m_s=9.3 / 3.6,
    nominal_endurance_s=4.0 * 3600.0,
    sensor_hardpoints=(
        SensorHardpoint("front_camera", "camera_link", (0.31, 0.0, 0.27), intended_sensor="RGB inspection camera"),
        SensorHardpoint("survey_lidar", "lidar_link", (0.02, 0.0, 0.39), intended_sensor="3-D LiDAR"),
        SensorHardpoint("radiation", "radiation_link", (0.08, 0.18, 0.40), intended_sensor="gamma dose-rate or spectroscopic detector"),
    ),
)

ELIOS3_RAD = RealRobotReference(
    id="flyability-elios3-rad",
    manufacturer="Flyability / Mirion Technologies",
    model="Elios 3 RAD",
    roles=("indoor radiation survey", "3-D LiDAR mapping", "confined-space inspection"),
    source_urls=(
        "https://www.flyability.com/news/flyability-launches-a-radiation-survey-meter-payload-for-indoor-inspection-drone-elios-3-in-partnership-with-mirion-technologies",
        "https://knowledge.flyability.com/aircraft/elios-3/elios-3-payload/radiation-sensor",
        "https://www.flyability.com/hubfs/Elios-3/eBooks/Surveying-Payload/E3-Surveying-Payload_Brochure_web.pdf",
    ),
    dimensions_m=(0.48, 0.48, 0.38),
    mass_kg=None,
    locomotion="collision-tolerant caged quadrotor",
    mechanisms=(
        "protective fixed cage",
        "Ouster LiDAR with onboard SLAM",
        "three optical cameras",
        "Mirion RDS-32 radiation payload",
    ),
    nominal_endurance_s=7.5 * 60.0,
    sensor_hardpoints=(
        SensorHardpoint("front_camera", "camera_link", (0.18, 0.0, -0.01), intended_sensor="inspection RGB/thermal camera group"),
        SensorHardpoint("lidar", "lidar_link", (-0.02, 0.0, -0.10), intended_sensor="Ouster OS0 LiDAR"),
        SensorHardpoint("radiation", "radiation_link", (-0.12, 0.0, -0.12), intended_sensor="Mirion RDS-32 WR"),
    ),
)

MHI_MEISTER = RealRobotReference(
    id="mhi-meister",
    manufacturer="Mitsubishi Heavy Industries",
    model="MEISTeR",
    roles=("shield handling", "obstacle removal", "core sampling", "dry decontamination", "suction"),
    source_urls=(
        "https://www.mhi.com/business/products-services/energy-environment/nuclear-power-generation/robot-mechatronics/meister",
        "https://www.mhi.com/news/1402201775.html",
    ),
    dimensions_m=(1.25, 0.70, 1.30),
    mass_kg=440.0,
    locomotion="four independently terrain-following crawler modules",
    mechanisms=(
        "dual seven-axis manipulators, 15 kg payload each",
        "interchangeable gripper, cutter, drill, suction and abrasive-blast tools",
        "automatic upper-body center-of-gravity compensation",
    ),
    maximum_speed_m_s=2.0 / 3.6,
    nominal_endurance_s=2.0 * 3600.0,
    sensor_hardpoints=(
        SensorHardpoint("head_camera", "head_camera_link", (0.30, 0.0, 1.17), intended_sensor="stereo inspection camera"),
        SensorHardpoint("dose_meter", "base_link", (0.12, 0.28, 0.66), intended_sensor="ion chamber"),
        SensorHardpoint("left_tool", "left_tool0", (0.0, 0.0, 0.0), intended_sensor="tool camera / contact sensor"),
        SensorHardpoint("right_tool", "right_tool0", (0.0, 0.0, 0.0), intended_sensor="tool camera / force sensor"),
    ),
)

HITACHI_AROUNDER = RealRobotReference(
    id="hitachi-ge-arounder",
    manufacturer="Hitachi-GE Nuclear Energy / IRID",
    model="Arounder low-section high-pressure-water decontamination machine",
    roles=("high-pressure water decontamination", "coating removal", "surface scarification"),
    source_urls=(
        "https://www.hitachi-hgne.co.jp/news/2013/20130308.html",
        "https://irid.or.jp/_pdf/20150714_5.pdf",
    ),
    dimensions_m=(1.529, 0.600, 1.255),
    mass_kg=840.0,
    locomotion="crawler with foldable four-degree-of-freedom treatment arm",
    mechanisms=(
        "25-200 MPa reciprocating high-pressure nozzle",
        "sealed brush-skirt treatment head",
        "simultaneous contaminated-water and debris recovery",
        "remote pump, recovery tank, hose reel and corner rollers",
    ),
    sensor_hardpoints=(
        SensorHardpoint("navigation_camera", "base_link", (0.42, 0.0, 0.82), intended_sensor="radiation-resistant navigation camera"),
        SensorHardpoint("head_camera", "treatment_head", (0.08, 0.0, 0.18), intended_sensor="treatment-head camera"),
    ),
)

REAL_ROBOT_REFERENCES = MappingProxyType(
    {
        item.id: item
        for item in (PACKBOT_FUKUSHIMA, ELIOS3_RAD, MHI_MEISTER, HITACHI_AROUNDER)
    }
)


def get_real_robot_reference(model_id: str) -> RealRobotReference:
    try:
        return REAL_ROBOT_REFERENCES[model_id]
    except KeyError as exc:
        choices = ", ".join(sorted(REAL_ROBOT_REFERENCES))
        raise KeyError(f"Unknown real robot reference {model_id!r}; choose one of: {choices}") from exc


def reference_config(model_id: str) -> RobotReferenceConfig:
    return get_real_robot_reference(model_id).config()
