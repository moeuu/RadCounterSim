"""Composable environment, robot, and detector selections for RadCounterSim."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from radcounter.core.environment import EnvironmentImportConfig, load_environment_descriptor
from radcounter.core.robots import RobotFleetConfig, load_robot_fleet
from radcounter.core.sensors import DetectorDescriptor
from radcounter.core.sensors.catalog import popular_detector_catalog
from radcounter.core.sensors.plugins import load_detector_descriptor

DEFAULT_CATALOG_RELATIVE_PATH = Path("configs/system/catalog.yaml")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvironmentSpawnAnchor(_FrozenModel):
    """A curated, environment-relative pose that keeps catalog robots operable."""

    kind: Literal["ground_robot", "aerial_robot", "work_surface", "camera"]
    translation_m: tuple[float, float, float]
    yaw_deg: float = 0.0
    description: str = ""


class EnvironmentCatalogEntry(_FrozenModel):
    display_name: str = Field(min_length=1)
    descriptor_uri: str = Field(min_length=1)
    description: str = ""
    setup_hint: str | None = None
    preparation_scripts: tuple[str, ...] = ()
    spawn_anchors: dict[str, EnvironmentSpawnAnchor] = Field(default_factory=dict)


class ReferenceRobotPlacement(_FrozenModel):
    robot_id: str = Field(min_length=1)
    reference_model_id: str = Field(min_length=1)
    prim_path: str = Field(pattern=r"^/World(?:/.*)?$")
    spawn_anchor: str | None = None
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0


class RobotSetCatalogEntry(_FrozenModel):
    display_name: str = Field(min_length=1)
    kind: Literal["decommissioning", "reference", "fleet", "none"]
    description: str = ""
    fleet_uri: str | None = None
    reference_robots: tuple[ReferenceRobotPlacement, ...] = ()

    @model_validator(mode="after")
    def validate_payload(self) -> RobotSetCatalogEntry:
        if self.kind == "fleet" and not self.fleet_uri:
            raise ValueError("fleet robot sets require fleet_uri")
        if self.kind != "fleet" and self.fleet_uri is not None:
            raise ValueError("fleet_uri is valid only for fleet robot sets")
        if self.kind == "reference" and not self.reference_robots:
            raise ValueError("reference robot sets require reference_robots")
        if self.kind != "reference" and self.reference_robots:
            raise ValueError("reference_robots is valid only for reference robot sets")
        ids = [robot.robot_id for robot in self.reference_robots]
        if len(ids) != len(set(ids)):
            raise ValueError("reference robot IDs must be unique within a robot set")
        return self


class DetectorPlacement(_FrozenModel):
    detector_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    descriptor_uri: str | None = None
    parent_robot_id: str | None = None
    parent_sensor_link: str | None = None
    parent_prim_path: str | None = None
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def validate_parent(self) -> DetectorPlacement:
        robot_parent = self.parent_robot_id is not None
        prim_parent = self.parent_prim_path is not None
        if robot_parent == prim_parent:
            raise ValueError(
                "a detector requires exactly one of parent_robot_id or parent_prim_path"
            )
        if self.parent_sensor_link is not None and not robot_parent:
            raise ValueError("parent_sensor_link requires parent_robot_id")
        if self.parent_prim_path is not None and not self.parent_prim_path.startswith("/World"):
            raise ValueError("parent_prim_path must be rooted below /World")
        return self


class DetectorSetCatalogEntry(_FrozenModel):
    display_name: str = Field(min_length=1)
    description: str = ""
    detectors: tuple[DetectorPlacement, ...]

    def model_post_init(self, __context: object) -> None:
        ids = [detector.detector_id for detector in self.detectors]
        if len(ids) != len(set(ids)):
            raise ValueError("detector IDs must be unique within a detector set")


class SystemProfileConfig(_FrozenModel):
    display_name: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    robot_set: str = Field(min_length=1)
    detector_set: str = Field(min_length=1)
    runtime_config_uri: str = Field(min_length=1)
    application_mode: Literal["decommissioning", "configurable"] = "configurable"
    description: str = ""


class SystemCatalog(_FrozenModel):
    schema_version: Literal[1]
    default_profile: str = Field(min_length=1)
    environments: dict[str, EnvironmentCatalogEntry]
    robot_sets: dict[str, RobotSetCatalogEntry]
    detector_sets: dict[str, DetectorSetCatalogEntry]
    profiles: dict[str, SystemProfileConfig]

    @model_validator(mode="after")
    def validate_references(self) -> SystemCatalog:
        if self.default_profile not in self.profiles:
            raise ValueError(f"default profile is missing: {self.default_profile}")
        for profile_id, profile in self.profiles.items():
            missing = []
            if profile.environment not in self.environments:
                missing.append(f"environment={profile.environment}")
            if profile.robot_set not in self.robot_sets:
                missing.append(f"robot_set={profile.robot_set}")
            if profile.detector_set not in self.detector_sets:
                missing.append(f"detector_set={profile.detector_set}")
            if missing:
                raise ValueError(f"profile {profile_id!r} has missing references: {missing}")
        return self


@dataclass(frozen=True)
class ResolvedDetector:
    placement: DetectorPlacement
    descriptor: DetectorDescriptor


@dataclass(frozen=True)
class ResolvedSystemSelection:
    catalog_path: Path
    profile_id: str
    profile: SystemProfileConfig
    environment_id: str
    environment_entry: EnvironmentCatalogEntry
    environment_descriptor_path: Path
    environment_config: EnvironmentImportConfig
    robot_set_id: str
    robot_set: RobotSetCatalogEntry
    robot_fleet_path: Path | None
    robot_fleet: RobotFleetConfig | None
    detector_set_id: str
    detector_set: DetectorSetCatalogEntry
    detectors: tuple[ResolvedDetector, ...]
    runtime_config_path: Path
    has_overrides: bool = False

    @property
    def configurable(self) -> bool:
        return self.profile.application_mode == "configurable" or self.has_overrides

    @property
    def environment_source_path(self) -> Path | None:
        parsed = urlparse(self.environment_config.uri)
        if parsed.scheme in {"http", "https"}:
            return None
        raw = Path(parsed.path if parsed.scheme == "file" else self.environment_config.uri)
        if raw.is_absolute():
            return raw.expanduser().resolve()
        return (self.environment_descriptor_path.parent / raw).expanduser().resolve()

    @property
    def environment_ready(self) -> bool:
        source = self.environment_source_path
        return source is None or source.is_file()

    @property
    def environment_preparation_scripts(self) -> tuple[Path, ...]:
        return tuple(
            _catalog_path(self.catalog_path, uri)
            for uri in self.environment_entry.preparation_scripts
        )

    def spawn_anchor(self, anchor_id: str) -> EnvironmentSpawnAnchor:
        """Resolve a named pose from the selected environment's operator layout."""

        try:
            return self.environment_entry.spawn_anchors[anchor_id]
        except KeyError as error:
            raise ValueError(
                f"environment {self.environment_id!r} has no spawn anchor {anchor_id!r}"
            ) from error

    def as_dict(self) -> dict[str, object]:
        source = self.environment_source_path
        return {
            "profile": self.profile_id,
            "display_name": self.profile.display_name,
            "application_mode": (
                "configurable" if self.configurable else self.profile.application_mode
            ),
            "environment": {
                "id": self.environment_id,
                "display_name": self.environment_entry.display_name,
                "descriptor": str(self.environment_descriptor_path),
                "source": str(source) if source is not None else self.environment_config.uri,
                "ready": self.environment_ready,
                "setup_hint": self.environment_entry.setup_hint,
                "preparation_scripts": [str(path) for path in self.environment_preparation_scripts],
                "spawn_anchors": {
                    anchor_id: anchor.model_dump(mode="json")
                    for anchor_id, anchor in self.environment_entry.spawn_anchors.items()
                },
            },
            "robot_set": {
                "id": self.robot_set_id,
                "display_name": self.robot_set.display_name,
                "kind": self.robot_set.kind,
                "fleet": str(self.robot_fleet_path) if self.robot_fleet_path else None,
                "robots": [
                    {
                        "robot_id": robot.robot_id,
                        "reference_model_id": robot.reference_model_id,
                        "prim_path": robot.prim_path,
                        "spawn_anchor": robot.spawn_anchor,
                    }
                    for robot in self.robot_set.reference_robots
                ],
            },
            "detector_set": {
                "id": self.detector_set_id,
                "display_name": self.detector_set.display_name,
                "detectors": [
                    {
                        "detector_id": item.placement.detector_id,
                        "model_id": item.descriptor.model_id,
                        "display_name": item.descriptor.display_name,
                        "parent_robot_id": item.placement.parent_robot_id,
                        "parent_prim_path": item.placement.parent_prim_path,
                    }
                    for item in self.detectors
                ],
            },
            "runtime_config": str(self.runtime_config_path),
        }


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_catalog_path() -> Path:
    override = os.environ.get("RADCOUNTER_SYSTEM_CATALOG")
    if override:
        return Path(override).expanduser().resolve()
    return repository_root() / DEFAULT_CATALOG_RELATIVE_PATH


def default_selection_path() -> Path:
    override = os.environ.get("RADCOUNTER_SELECTION_FILE")
    if override:
        return Path(override).expanduser().resolve()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "radcountersim/system-selection.json"


def load_system_catalog(path: str | Path | None = None) -> tuple[Path, SystemCatalog]:
    catalog_path = Path(path or default_catalog_path()).expanduser().resolve()
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("system catalog root must be a mapping")
    return catalog_path, SystemCatalog.model_validate(payload)


def _catalog_path(catalog_path: Path, uri: str) -> Path:
    candidate = Path(uri).expanduser()
    if not candidate.is_absolute():
        candidate = catalog_path.parent / candidate
    return candidate.resolve()


def _rebase_path(value: str, base: Path) -> str:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "file"}:
        return value
    candidate = Path(value).expanduser()
    return str(candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve())


def _resolved_fleet(path: Path) -> RobotFleetConfig:
    fleet = load_robot_fleet(path)
    robots = []
    for robot in fleet.robots:
        update: dict[str, object] = {
            "uri": _rebase_path(robot.uri, path.parent),
            "package_paths": tuple(_rebase_path(item, path.parent) for item in robot.package_paths),
        }
        if robot.lula_kinematics is not None:
            update["lula_kinematics"] = robot.lula_kinematics.model_copy(
                update={
                    "robot_description_path": _rebase_path(
                        robot.lula_kinematics.robot_description_path, path.parent
                    ),
                    "urdf_path": _rebase_path(robot.lula_kinematics.urdf_path, path.parent),
                }
            )
        robots.append(robot.model_copy(update=update))
    return fleet.model_copy(update={"robots": tuple(robots)})


def resolve_system_selection(
    *,
    catalog_path: str | Path | None = None,
    profile_id: str | None = None,
    environment_id: str | None = None,
    robot_set_id: str | None = None,
    detector_set_id: str | None = None,
) -> ResolvedSystemSelection:
    resolved_catalog_path, catalog = load_system_catalog(catalog_path)
    selected_profile_id = profile_id or catalog.default_profile
    try:
        profile = catalog.profiles[selected_profile_id]
    except KeyError as error:
        raise KeyError(f"unknown system profile: {selected_profile_id}") from error
    selected_environment = environment_id or profile.environment
    selected_robots = robot_set_id or profile.robot_set
    selected_detectors = detector_set_id or profile.detector_set
    try:
        environment_entry = catalog.environments[selected_environment]
        robot_set = catalog.robot_sets[selected_robots]
        detector_set = catalog.detector_sets[selected_detectors]
    except KeyError as error:
        raise KeyError(f"unknown system component: {error.args[0]}") from error

    descriptor_path = _catalog_path(resolved_catalog_path, environment_entry.descriptor_uri)
    environment_config = load_environment_descriptor(descriptor_path)
    fleet_path = None
    fleet = None
    if robot_set.fleet_uri:
        fleet_path = _catalog_path(resolved_catalog_path, robot_set.fleet_uri)
        fleet = _resolved_fleet(fleet_path)

    builtins = popular_detector_catalog()
    detectors = []
    for placement in detector_set.detectors:
        if placement.descriptor_uri:
            descriptor = load_detector_descriptor(
                _catalog_path(resolved_catalog_path, placement.descriptor_uri)
            )
        else:
            try:
                descriptor = builtins[placement.model_id]
            except KeyError as error:
                raise KeyError(
                    f"unknown built-in detector model {placement.model_id!r}; "
                    "set descriptor_uri for a custom detector"
                ) from error
        if descriptor.model_id != placement.model_id:
            raise ValueError(
                f"detector {placement.detector_id!r} requested model {placement.model_id!r} "
                f"but its descriptor defines {descriptor.model_id!r}"
            )
        detectors.append(ResolvedDetector(placement, descriptor))

    available_robot_ids = {
        "reference": {robot.robot_id for robot in robot_set.reference_robots},
        "fleet": {robot.id for robot in fleet.robots} if fleet is not None else set(),
        "decommissioning": {"countermeasure", "measurement"},
        "none": set(),
    }[robot_set.kind]
    missing_spawn_anchors = sorted(
        {
            robot.spawn_anchor
            for robot in robot_set.reference_robots
            if robot.spawn_anchor is not None
            and robot.spawn_anchor not in environment_entry.spawn_anchors
        }
    )
    if missing_spawn_anchors:
        raise ValueError(
            f"environment {selected_environment!r} does not provide spawn anchors required by "
            f"robot set {selected_robots!r}: {missing_spawn_anchors}"
        )
    missing_detector_robots = sorted(
        {
            item.placement.parent_robot_id
            for item in detectors
            if item.placement.parent_robot_id is not None
            and item.placement.parent_robot_id not in available_robot_ids
        }
    )
    if missing_detector_robots:
        raise ValueError(
            f"detector set {selected_detectors!r} requires robots not present in "
            f"{selected_robots!r}: {missing_detector_robots}"
        )

    runtime_path = _catalog_path(resolved_catalog_path, profile.runtime_config_uri)
    if not runtime_path.is_file():
        raise FileNotFoundError(f"runtime configuration is missing: {runtime_path}")
    return ResolvedSystemSelection(
        catalog_path=resolved_catalog_path,
        profile_id=selected_profile_id,
        profile=profile,
        environment_id=selected_environment,
        environment_entry=environment_entry,
        environment_descriptor_path=descriptor_path,
        environment_config=environment_config,
        robot_set_id=selected_robots,
        robot_set=robot_set,
        robot_fleet_path=fleet_path,
        robot_fleet=fleet,
        detector_set_id=selected_detectors,
        detector_set=detector_set,
        detectors=tuple(detectors),
        runtime_config_path=runtime_path,
        has_overrides=(
            (environment_id is not None and environment_id != profile.environment)
            or (robot_set_id is not None and robot_set_id != profile.robot_set)
            or (detector_set_id is not None and detector_set_id != profile.detector_set)
        ),
    )


def save_active_selection(
    selection_path: str | Path,
    *,
    catalog_path: str | Path,
    profile_id: str,
    environment_id: str | None = None,
    robot_set_id: str | None = None,
    detector_set_id: str | None = None,
) -> Path:
    target = Path(selection_path).expanduser().resolve()
    payload = {
        "schema_version": 1,
        "catalog_path": str(Path(catalog_path).expanduser().resolve()),
        "profile_id": profile_id,
        "environment_id": environment_id,
        "robot_set_id": robot_set_id,
        "detector_set_id": detector_set_id,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_active_selection(
    selection_path: str | Path | None = None,
) -> ResolvedSystemSelection:
    path = Path(selection_path or default_selection_path()).expanduser().resolve()
    if not path.is_file():
        return resolve_system_selection(
            catalog_path=os.environ.get("RADCOUNTER_SYSTEM_CATALOG"),
            profile_id=os.environ.get("RADCOUNTER_SYSTEM_PROFILE"),
            environment_id=os.environ.get("RADCOUNTER_ENVIRONMENT"),
            robot_set_id=os.environ.get("RADCOUNTER_ROBOT_SET"),
            detector_set_id=os.environ.get("RADCOUNTER_DETECTOR_SET"),
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported active selection schema: {path}")
    return resolve_system_selection(
        catalog_path=os.environ.get("RADCOUNTER_SYSTEM_CATALOG") or payload["catalog_path"],
        profile_id=os.environ.get("RADCOUNTER_SYSTEM_PROFILE") or payload["profile_id"],
        environment_id=os.environ.get("RADCOUNTER_ENVIRONMENT") or payload.get("environment_id"),
        robot_set_id=os.environ.get("RADCOUNTER_ROBOT_SET") or payload.get("robot_set_id"),
        detector_set_id=os.environ.get("RADCOUNTER_DETECTOR_SET") or payload.get("detector_set_id"),
    )
