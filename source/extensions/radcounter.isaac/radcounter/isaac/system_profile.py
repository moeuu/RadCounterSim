"""Compose a selected environment, robot set, and detector set in Isaac Sim."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from radcounter.core.environment import EnvironmentImportPipeline
from radcounter.core.system_profiles import ResolvedSystemSelection

from .robot import (
    RealRobotAssetConfig,
    RobotFleetManager,
    add_real_robot_references,
    enable_real_robot_extensions,
    spawn_reference_robot,
)
from .usd import EnvironmentUsdWriter


@dataclass(frozen=True)
class ComposedSystem:
    stage_path: Path
    environment_manifest_path: Path
    runtime_config_path: Path
    robot_paths: dict[str, str]
    detector_paths: dict[str, str]
    disabled_detector_paths: tuple[str, ...]
    rebased_asset_paths: dict[str, str]
    profile_manifest: dict[str, object]
    fleet_manager: RobotFleetManager | None = None


@dataclass(frozen=True)
class _RobotHandle:
    root_path: str
    sensor_links: dict[str, str]


def prepare_environment_stage(selection: ResolvedSystemSelection) -> tuple[Path, Path]:
    if not selection.environment_ready:
        raise FileNotFoundError(
            f"selected environment is not ready: {selection.environment_source_path}; "
            f"{selection.environment_entry.setup_hint or 'provide the configured source file'}"
        )
    result = EnvironmentImportPipeline().import_environment(
        selection.environment_config,
        base_directory=selection.environment_descriptor_path.parent,
    )
    stage_path = EnvironmentUsdWriter().write(
        result.scene,
        result.output_directory / "environment.usda",
    )
    return stage_path, result.manifest_path


def compose_selected_system(
    stage: Any,
    selection: ResolvedSystemSelection,
    *,
    stage_path: Path,
) -> ComposedSystem:
    disabled_detector_paths = _disable_existing_detectors(stage)
    handles, fleet_manager = _compose_robots(stage, selection)
    detector_paths = _compose_detectors(stage, selection, handles)
    rebased_assets = _rebase_referenced_asset_paths(stage)
    runtime_path = _write_runtime_config(selection, stage_path)
    stage.GetRootLayer().Save()
    manifest = {
        **selection.as_dict(),
        "stage": str(stage_path),
        "runtime_config": str(runtime_path),
        "robot_paths": {key: value.root_path for key, value in handles.items()},
        "detector_paths": detector_paths,
        "disabled_detector_paths": list(disabled_detector_paths),
        "rebased_asset_paths": rebased_assets,
    }
    return ComposedSystem(
        stage_path=stage_path,
        environment_manifest_path=stage_path.parent / "manifest.json",
        runtime_config_path=runtime_path,
        robot_paths={key: value.root_path for key, value in handles.items()},
        detector_paths=detector_paths,
        disabled_detector_paths=disabled_detector_paths,
        rebased_asset_paths=rebased_assets,
        profile_manifest=manifest,
        fleet_manager=fleet_manager,
    )


def _disable_existing_detectors(stage: Any) -> tuple[str, ...]:
    """Deactivate authored detectors so a selected detector set is replacement, not additive."""

    from pxr import Sdf

    disabled = []
    for prim in stage.Traverse():
        role = prim.GetAttribute("rad:role")
        detector_id = prim.GetAttribute("rad:detector:id")
        if not role or not detector_id or not detector_id.Get():
            continue
        if str(role.Get()) not in {"detector", "detector_station"}:
            continue
        role.Set("inactive_detector")
        prim.CreateAttribute(
            "rad:detector:disabledByProfile", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(True)
        disabled.append(str(prim.GetPath()))
    return tuple(disabled)


def _rebase_referenced_asset_paths(stage: Any) -> dict[str, str]:
    """Make relative RadCounter sidecars stable in a generated composition layer."""

    rebased: dict[str, str] = {}
    root_layer_path = Path(stage.GetRootLayer().realPath).resolve()
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if not name.startswith("rad:") or not name.lower().endswith("uri"):
                continue
            value = attribute.Get()
            if not isinstance(value, str) or not value:
                continue
            parsed = urlparse(value)
            candidate = Path(parsed.path if parsed.scheme == "file" else value).expanduser()
            if parsed.scheme in {"http", "https"} or candidate.is_absolute():
                continue
            resolved = None
            for specification in attribute.GetPropertyStack():
                layer_path = Path(specification.layer.realPath or "")
                if not layer_path.is_file() or layer_path.resolve() == root_layer_path:
                    continue
                layer_candidate = (layer_path.parent / candidate).resolve()
                if layer_candidate.is_file():
                    resolved = layer_candidate
                    break
            if resolved is None:
                continue
            attribute.Set(str(resolved))
            rebased[f"{prim.GetPath()}.{name}"] = str(resolved)
    return rebased


def _compose_robots(
    stage: Any,
    selection: ResolvedSystemSelection,
) -> tuple[dict[str, _RobotHandle], RobotFleetManager | None]:
    kind = selection.robot_set.kind
    handles: dict[str, _RobotHandle] = {}
    if kind == "none":
        return handles, None
    if kind == "reference":
        from pxr import Gf, Sdf

        for placement in selection.robot_set.reference_robots:
            spawned = spawn_reference_robot(
                stage,
                placement.reference_model_id,
                placement.prim_path,
            )
            anchor = (
                None
                if placement.spawn_anchor is None
                else selection.spawn_anchor(placement.spawn_anchor)
            )
            translation = placement.translation_m if anchor is None else anchor.translation_m
            yaw_deg = placement.yaw_deg if anchor is None else anchor.yaw_deg
            spawned.translation_op.Set(Gf.Vec3d(*translation))
            spawned.yaw_op.Set(float(yaw_deg))
            if anchor is not None:
                prim = stage.GetPrimAtPath(spawned.prim_path)
                prim.CreateAttribute(
                    "rad:spawn:anchorId", Sdf.ValueTypeNames.String, custom=True
                ).Set(placement.spawn_anchor)
                prim.CreateAttribute(
                    "rad:spawn:environmentId", Sdf.ValueTypeNames.String, custom=True
                ).Set(selection.environment_id)
            handles[placement.robot_id] = _RobotHandle(
                spawned.prim_path,
                dict(spawned.sensor_links),
            )
        return handles, None
    if kind == "fleet":
        if selection.robot_fleet is None:
            raise RuntimeError("resolved fleet robot set has no fleet configuration")
        manager = RobotFleetManager(stage, selection.robot_fleet)
        imported = manager.import_all()
        for robot in selection.robot_fleet.robots:
            root = robot.prim_path or imported[robot.id]
            handles[robot.id] = _RobotHandle(root, {})
        return handles, manager
    if kind == "decommissioning":
        from pxr import Gf, Sdf, UsdGeom

        enable_real_robot_extensions()
        config = RealRobotAssetConfig()
        add_real_robot_references(stage, config=config)
        for path, anchor_id in (
            (config.countermeasure_root, "ground-primary"),
            (config.measurement_root, "ground-secondary"),
        ):
            anchor = selection.spawn_anchor(anchor_id)
            prim = stage.GetPrimAtPath(path)
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(*anchor.translation_m))
            xformable.AddRotateZOp().Set(float(anchor.yaw_deg))
            prim.CreateAttribute("rad:spawn:anchorId", Sdf.ValueTypeNames.String, custom=True).Set(
                anchor_id
            )
            prim.CreateAttribute(
                "rad:spawn:environmentId", Sdf.ValueTypeNames.String, custom=True
            ).Set(selection.environment_id)
        handles["countermeasure"] = _RobotHandle(config.countermeasure_root, {})
        handles["measurement"] = _RobotHandle(
            config.measurement_articulation,
            {"radiation": config.detector_path},
        )
        return handles, None
    raise AssertionError(kind)


def _find_robot_link(stage: Any, handle: _RobotHandle, link_name: str) -> str:
    explicit = handle.sensor_links.get(link_name)
    if explicit:
        return explicit
    root = stage.GetPrimAtPath(handle.root_path)
    if not root.IsValid():
        raise ValueError(f"robot root does not exist: {handle.root_path}")
    candidates = {link_name, f"{link_name}_link"}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(handle.root_path.rstrip("/") + "/"):
            continue
        if prim.GetName() in candidates:
            return path
    raise ValueError(f"robot {handle.root_path} has no sensor link {link_name!r}")


def _compose_detectors(
    stage: Any,
    selection: ResolvedSystemSelection,
    handles: dict[str, _RobotHandle],
) -> dict[str, str]:
    from pxr import Gf, Sdf, UsdGeom

    paths: dict[str, str] = {}
    for resolved in selection.detectors:
        placement = resolved.placement
        if placement.parent_robot_id is not None:
            try:
                handle = handles[placement.parent_robot_id]
            except KeyError as error:
                raise ValueError(
                    f"detector {placement.detector_id!r} requires robot "
                    f"{placement.parent_robot_id!r}, which is not selected"
                ) from error
            parent_path = (
                _find_robot_link(stage, handle, placement.parent_sensor_link)
                if placement.parent_sensor_link
                else handle.root_path
            )
        else:
            parent_path = str(placement.parent_prim_path)
        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            raise ValueError(
                f"detector {placement.detector_id!r} parent does not exist: {parent_path}"
            )
        safe_id = "".join(
            character if character.isalnum() else "_" for character in placement.detector_id
        )
        path = f"{parent_path.rstrip('/')}/RadCounterDetectors/{safe_id}"
        xform = UsdGeom.Xform.Define(stage, path)
        xformable = UsdGeom.Xformable(xform.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*placement.translation_m))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*placement.rotation_rpy_deg))
        prim = xform.GetPrim()
        prim.CreateAttribute("rad:role", Sdf.ValueTypeNames.String, custom=True).Set("detector")
        prim.CreateAttribute("rad:detector:id", Sdf.ValueTypeNames.String, custom=True).Set(
            placement.detector_id
        )
        prim.CreateAttribute("rad:detector:modelId", Sdf.ValueTypeNames.String, custom=True).Set(
            resolved.descriptor.model_id
        )
        body = UsdGeom.Cylinder.Define(stage, path + "/Body")
        body.CreateAxisAttr("X")
        body.CreateRadiusAttr(0.045)
        body.CreateHeightAttr(0.12)
        body.CreateDisplayColorAttr([Gf.Vec3f(0.94, 0.67, 0.08)])
        paths[placement.detector_id] = path
    return paths


def _write_runtime_config(selection: ResolvedSystemSelection, stage_path: Path) -> Path:
    payload = json.loads(selection.runtime_config_path.read_text(encoding="utf-8"))
    detectors = {}
    for resolved in selection.detectors:
        descriptor = resolved.descriptor
        response = next(
            (
                item
                for item in descriptor.particle_responses
                if item.radiation_type.value == "gamma"
            ),
            descriptor.particle_responses[0],
        )
        response_energy = np.asarray(
            response.effective_area_m2.energies_kev, dtype=np.float64
        )
        effective_area = np.asarray(
            response.effective_area_m2.values,
            dtype=np.float64,
        )
        if descriptor.energy_bin_edges_kev:
            bin_edges = np.asarray(descriptor.energy_bin_edges_kev, dtype=np.float64)
        else:
            bin_edges = np.asarray((0.0, float(response_energy[-1]) * 1.001), dtype=np.float64)
        response_matrix = np.zeros((len(response_energy), len(bin_edges) - 1), dtype=np.float64)
        bin_indices = np.searchsorted(bin_edges, response_energy, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        response_matrix[np.arange(len(response_energy)), bin_indices] = effective_area
        background = np.full(
            len(bin_edges) - 1,
            descriptor.background_cps / (len(bin_edges) - 1),
            dtype=np.float64,
        )
        detectors[resolved.placement.detector_id] = {
            "energy_bin_edges_keV": bin_edges.tolist(),
            "response_energy_keV": response_energy.tolist(),
            "effective_area_m2_per_bin": response_matrix.tolist(),
            "background_cps_per_bin": background.tolist(),
            "dead_time_s": descriptor.dead_time_s,
        }
    payload["scene"] = str(stage_path)
    payload["detectors"] = detectors
    payload["selected_system"] = {
        "profile": selection.profile_id,
        "environment": selection.environment_id,
        "robot_set": selection.robot_set_id,
        "detector_set": selection.detector_set_id,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    environment_cache = Path(selection.environment_config.cache_directory).expanduser().resolve()
    target = environment_cache.parent / "system_profiles" / digest
    target.mkdir(parents=True, exist_ok=True)
    path = target / "runtime.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
    return path
