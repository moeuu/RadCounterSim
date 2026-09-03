"""USD authoring, scene discovery, and change tracking for radiation metadata."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from radcounter.core.models.state import RevisionState
from radcounter.core.scene import (
    DecontaminationSurfaceDescriptor,
    ManipulationDescriptor,
    MaterialDescriptor,
    MaterialMode,
    RadiationRole,
    RadiationSceneSnapshot,
    ShieldDescriptor,
    SourceDescriptor,
    SourceType,
    SurfaceActivityMap,
    UsdRadiationAttributes,
    classify_stage_changes,
)


class UsdRadiationRegistryUnavailable(RuntimeError):
    """Raised when registry operations are requested outside a USD runtime."""


def _pxr_modules() -> tuple[Any, Any, Any, Any]:
    try:
        from pxr import Sdf, Tf, Usd, UsdGeom  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise UsdRadiationRegistryUnavailable(
            "USD radiation registry requires the Isaac Sim pxr runtime"
        ) from error
    return Sdf, Tf, Usd, UsdGeom


def _attribute_value(prim: Any, name: str, default: Any = None) -> Any:
    attribute = prim.GetAttribute(name)
    if not attribute.IsValid() or not attribute.HasAuthoredValue():
        return default
    value = attribute.Get()
    if hasattr(value, "path"):
        return str(value.path)
    return value


def _stage_base_directory(stage: Any) -> Path:
    root_layer = stage.GetRootLayer()
    identifier = str(root_layer.realPath or root_layer.identifier)
    if not identifier or identifier.startswith("anon:"):
        return Path.cwd()
    return Path(identifier).expanduser().resolve().parent


class UsdMetadataAuthor:
    """Author canonical custom attributes without requiring a custom USD schema."""

    def __init__(self) -> None:
        self._sdf, _, _, self._usd_geom = _pxr_modules()

    def _set(self, prim: Any, name: str, type_name: Any, value: Any) -> None:
        attribute = prim.GetAttribute(name)
        if not attribute.IsValid():
            attribute = prim.CreateAttribute(name, type_name, custom=True)
        attribute.Set(value)

    def author_source(self, prim: Any, source: SourceDescriptor) -> None:
        value_types = self._sdf.ValueTypeNames
        self._set(prim, UsdRadiationAttributes.ROLE, value_types.String, RadiationRole.SOURCE.value)
        self._set(
            prim, UsdRadiationAttributes.SOURCE_TYPE, value_types.String, source.source_type.value
        )
        self._set(
            prim, UsdRadiationAttributes.SOURCE_ISOTOPE_ID, value_types.String, source.isotope_id
        )
        self._set(
            prim, UsdRadiationAttributes.SOURCE_ACTIVITY_BQ, value_types.Double, source.activity_bq
        )
        self._set(
            prim,
            UsdRadiationAttributes.SOURCE_SURFACE_ACTIVITY_BQ_M2,
            value_types.Double,
            source.surface_activity_bq_m2,
        )
        self._set(
            prim,
            UsdRadiationAttributes.SOURCE_HIDDEN_FROM_ESTIMATOR,
            value_types.Bool,
            source.hidden_from_estimator,
        )
        self._set(
            prim,
            UsdRadiationAttributes.SOURCE_MOVABLE_WITH_PRIM,
            value_types.Bool,
            source.movable_with_prim,
        )
        self._set(
            prim, UsdRadiationAttributes.SOURCE_ENABLED, value_types.Bool, source.enabled
        )
        if source.activity_map_uri is not None:
            self._set(
                prim,
                UsdRadiationAttributes.SOURCE_ACTIVITY_MAP_URI,
                value_types.Asset,
                self._sdf.AssetPath(source.activity_map_uri),
            )
        if source.activity_map_sha256 is not None:
            self._set(
                prim,
                UsdRadiationAttributes.SOURCE_ACTIVITY_MAP_SHA256,
                value_types.String,
                source.activity_map_sha256,
            )

    def author_material(self, prim: Any, material: MaterialDescriptor) -> None:
        value_types = self._sdf.ValueTypeNames
        self._set(
            prim, UsdRadiationAttributes.MATERIAL_ID, value_types.String, material.material_id
        )
        self._set(
            prim, UsdRadiationAttributes.MATERIAL_MODE, value_types.String, material.mode.value
        )
        if material.thickness_m is not None:
            self._set(
                prim,
                UsdRadiationAttributes.MATERIAL_THICKNESS_M,
                value_types.Double,
                material.thickness_m,
            )
        if material.attenuation_uri is not None:
            self._set(
                prim,
                UsdRadiationAttributes.MATERIAL_ATTENUATION_URI,
                value_types.Asset,
                self._sdf.AssetPath(material.attenuation_uri),
            )

    def author_shield(self, prim: Any, shield: ShieldDescriptor) -> None:
        value_types = self._sdf.ValueTypeNames
        self._set(prim, UsdRadiationAttributes.ROLE, value_types.String, RadiationRole.SHIELD.value)
        self._set(
            prim, UsdRadiationAttributes.MATERIAL_ID, value_types.String, shield.material_id
        )
        if not prim.IsA(self._usd_geom.Gprim):
            self._set(
                prim,
                UsdRadiationAttributes.MATERIAL_CONTAINER_ONLY,
                value_types.Bool,
                True,
            )
        self._set(
            prim, UsdRadiationAttributes.SHIELD_MOVABLE, value_types.Bool, shield.movable
        )
        self._set(
            prim,
            UsdRadiationAttributes.SHIELD_RESOURCE_UNITS,
            value_types.Int,
            shield.resource_units,
        )

    def author_decontamination_surface(
        self, prim: Any, surface: DecontaminationSurfaceDescriptor
    ) -> None:
        value_types = self._sdf.ValueTypeNames
        self._set(
            prim,
            UsdRadiationAttributes.ROLE,
            value_types.String,
            RadiationRole.CONTAMINATED_SURFACE.value,
        )
        self._set(
            prim, UsdRadiationAttributes.DECON_ENABLED, value_types.Bool, surface.enabled
        )
        self._set(
            prim,
            UsdRadiationAttributes.DECON_SUBSTRATE_MATERIAL_ID,
            value_types.String,
            surface.substrate_material_id,
        )
        self._set(
            prim,
            UsdRadiationAttributes.DECON_TREATMENT_MODEL_URI,
            value_types.Asset,
            self._sdf.AssetPath(surface.treatment_model_uri),
        )
        self._set(
            prim,
            UsdRadiationAttributes.DECON_TREATMENT_MODEL_SHA256,
            value_types.String,
            surface.treatment_model_sha256,
        )
        self._set(
            prim,
            UsdRadiationAttributes.DECON_MIN_TOOL_DWELL_S,
            value_types.Double,
            surface.min_tool_dwell_s,
        )
        if surface.activity_map_uri is not None:
            self._set(
                prim,
                UsdRadiationAttributes.DECON_ACTIVITY_MAP_URI,
                value_types.Asset,
                self._sdf.AssetPath(surface.activity_map_uri),
            )
        if surface.activity_map_sha256 is not None:
            self._set(
                prim,
                UsdRadiationAttributes.DECON_ACTIVITY_MAP_SHA256,
                value_types.String,
                surface.activity_map_sha256,
            )

    def author_manipulation(self, prim: Any, manipulation: ManipulationDescriptor) -> None:
        value_types = self._sdf.ValueTypeNames
        self._set(
            prim,
            UsdRadiationAttributes.MANIPULATION_MOVABLE,
            value_types.Bool,
            manipulation.movable,
        )
        self._set(
            prim,
            UsdRadiationAttributes.MANIPULATION_REMOVABLE,
            value_types.Bool,
            manipulation.removable,
        )
        if manipulation.grasp_frame is not None:
            self._set(
                prim,
                UsdRadiationAttributes.MANIPULATION_GRASP_FRAME,
                value_types.String,
                manipulation.grasp_frame,
            )
        if manipulation.disposal_class is not None:
            self._set(
                prim,
                UsdRadiationAttributes.MANIPULATION_DISPOSAL_CLASS,
                value_types.String,
                manipulation.disposal_class,
            )


class UsdRadiationRegistry:
    """Discover every radiation-relevant prim and validate referenced sidecars."""

    def __init__(
        self,
        stage: Any,
        *,
        base_directory: str | Path | None = None,
        strict_activity_maps: bool = True,
    ) -> None:
        self.stage = stage
        self.base_directory = (
            _stage_base_directory(stage)
            if base_directory is None
            else Path(base_directory).expanduser().resolve()
        )
        self.strict_activity_maps = strict_activity_maps
        self.snapshot = RadiationSceneSnapshot()

    @staticmethod
    def _world_transform(prim: Any, xform_cache: Any) -> tuple[float, ...]:
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))

    def _load_activity_map(
        self, uri: str | None, expected_sha256: str | None
    ) -> SurfaceActivityMap | None:
        if uri is None:
            return None
        try:
            return SurfaceActivityMap.load(
                uri,
                base_directory=self.base_directory,
                expected_sha256=expected_sha256,
            )
        except (FileNotFoundError, ValueError):
            if self.strict_activity_maps:
                raise
            return None

    def refresh(self) -> RadiationSceneSnapshot:
        """Rebuild the registry from the current composed USD stage."""

        _, _, usd, usd_geom = _pxr_modules()
        xform_cache = usd_geom.XformCache(usd.TimeCode.Default())
        sources: dict[str, SourceDescriptor] = {}
        materials: dict[str, MaterialDescriptor] = {}
        shields: dict[str, ShieldDescriptor] = {}
        decon_surfaces: dict[str, DecontaminationSurfaceDescriptor] = {}
        manipulable: dict[str, ManipulationDescriptor] = {}
        activity_maps: dict[str, SurfaceActivityMap] = {}
        geometry_paths: set[str] = set()
        detector_paths: set[str] = set()
        for prim in self.stage.Traverse():
            if not prim.IsValid() or not prim.IsActive():
                continue
            path = str(prim.GetPath())
            role_value = str(_attribute_value(prim, UsdRadiationAttributes.ROLE, ""))
            source_type_value = _attribute_value(prim, UsdRadiationAttributes.SOURCE_TYPE)
            world_transform = self._world_transform(prim, xform_cache)
            if role_value == RadiationRole.DETECTOR.value:
                detector_paths.add(path)
            if source_type_value is not None or role_value == RadiationRole.SOURCE.value:
                source_type = SourceType(str(source_type_value or SourceType.POINT.value))
                source = SourceDescriptor(
                    prim_path=path,
                    source_type=source_type,
                    isotope_id=str(
                        _attribute_value(prim, UsdRadiationAttributes.SOURCE_ISOTOPE_ID, "")
                    ),
                    activity_bq=float(
                        _attribute_value(prim, UsdRadiationAttributes.SOURCE_ACTIVITY_BQ, 0.0)
                    ),
                    surface_activity_bq_m2=float(
                        _attribute_value(
                            prim,
                            UsdRadiationAttributes.SOURCE_SURFACE_ACTIVITY_BQ_M2,
                            0.0,
                        )
                    ),
                    activity_map_uri=_attribute_value(
                        prim, UsdRadiationAttributes.SOURCE_ACTIVITY_MAP_URI
                    ),
                    activity_map_sha256=_attribute_value(
                        prim, UsdRadiationAttributes.SOURCE_ACTIVITY_MAP_SHA256
                    ),
                    hidden_from_estimator=bool(
                        _attribute_value(
                            prim,
                            UsdRadiationAttributes.SOURCE_HIDDEN_FROM_ESTIMATOR,
                            False,
                        )
                    ),
                    movable_with_prim=bool(
                        _attribute_value(
                            prim, UsdRadiationAttributes.SOURCE_MOVABLE_WITH_PRIM, False
                        )
                    ),
                    enabled=bool(
                        _attribute_value(prim, UsdRadiationAttributes.SOURCE_ENABLED, True)
                    ),
                    world_transform=world_transform,
                )
                sources[path] = source
                sidecar = self._load_activity_map(
                    source.activity_map_uri, source.activity_map_sha256
                )
                if sidecar is not None:
                    activity_maps[path] = sidecar
            material_id = _attribute_value(prim, UsdRadiationAttributes.MATERIAL_ID)
            if material_id is not None:
                mode = MaterialMode(
                    str(
                        _attribute_value(
                            prim,
                            UsdRadiationAttributes.MATERIAL_MODE,
                            MaterialMode.SOLID.value,
                        )
                    )
                )
                thickness_value = _attribute_value(
                    prim, UsdRadiationAttributes.MATERIAL_THICKNESS_M
                )
                materials[path] = MaterialDescriptor(
                    path,
                    str(material_id),
                    mode,
                    None if thickness_value is None else float(thickness_value),
                    _attribute_value(prim, UsdRadiationAttributes.MATERIAL_ATTENUATION_URI),
                )
                geometry_paths.add(path)
            if role_value == RadiationRole.SHIELD.value:
                if material_id is None:
                    raise ValueError(f"shield {path} has no radiation material ID")
                shields[path] = ShieldDescriptor(
                    path,
                    str(material_id),
                    bool(_attribute_value(prim, UsdRadiationAttributes.SHIELD_MOVABLE, False)),
                    int(
                        _attribute_value(
                            prim, UsdRadiationAttributes.SHIELD_RESOURCE_UNITS, 1
                        )
                    ),
                    world_transform,
                )
                geometry_paths.add(path)
            decon_enabled_value = _attribute_value(prim, UsdRadiationAttributes.DECON_ENABLED)
            if (
                decon_enabled_value is not None
                or role_value == RadiationRole.CONTAMINATED_SURFACE.value
            ):
                map_uri = _attribute_value(
                    prim,
                    UsdRadiationAttributes.DECON_ACTIVITY_MAP_URI,
                    _attribute_value(prim, UsdRadiationAttributes.SOURCE_ACTIVITY_MAP_URI),
                )
                map_sha = _attribute_value(
                    prim,
                    UsdRadiationAttributes.DECON_ACTIVITY_MAP_SHA256,
                    _attribute_value(prim, UsdRadiationAttributes.SOURCE_ACTIVITY_MAP_SHA256),
                )
                surface = DecontaminationSurfaceDescriptor(
                    path,
                    bool(True if decon_enabled_value is None else decon_enabled_value),
                    map_uri,
                    map_sha,
                    str(
                        _attribute_value(
                            prim, UsdRadiationAttributes.DECON_SUBSTRATE_MATERIAL_ID, ""
                        )
                    ),
                    str(
                        _attribute_value(
                            prim, UsdRadiationAttributes.DECON_TREATMENT_MODEL_URI, ""
                        )
                    ),
                    str(
                        _attribute_value(
                            prim, UsdRadiationAttributes.DECON_TREATMENT_MODEL_SHA256, ""
                        )
                    ),
                    float(
                        _attribute_value(
                            prim, UsdRadiationAttributes.DECON_MIN_TOOL_DWELL_S, 0.0
                        )
                    ),
                )
                decon_surfaces[path] = surface
                if path not in activity_maps:
                    sidecar = self._load_activity_map(map_uri, map_sha)
                    if sidecar is not None:
                        activity_maps[path] = sidecar
                geometry_paths.add(path)
            movable_value = _attribute_value(
                prim, UsdRadiationAttributes.MANIPULATION_MOVABLE
            )
            removable_value = _attribute_value(
                prim, UsdRadiationAttributes.MANIPULATION_REMOVABLE
            )
            if movable_value is not None or removable_value is not None:
                manipulable[path] = ManipulationDescriptor(
                    path,
                    bool(movable_value),
                    bool(removable_value),
                    _attribute_value(prim, UsdRadiationAttributes.MANIPULATION_GRASP_FRAME),
                    _attribute_value(
                        prim, UsdRadiationAttributes.MANIPULATION_DISPOSAL_CLASS
                    ),
                )
                geometry_paths.add(path)
        self.snapshot = RadiationSceneSnapshot(
            sources=sources,
            materials=materials,
            shields=shields,
            decontamination_surfaces=decon_surfaces,
            manipulable_prims=manipulable,
            activity_maps=activity_maps,
            geometry_prim_paths=frozenset(geometry_paths),
            detector_prim_paths=frozenset(detector_paths),
        )
        return self.snapshot


class UsdStageRevisionTracker:
    """Convert USD ``ObjectsChanged`` notices into radiation revisions."""

    def __init__(
        self,
        stage: Any,
        registry: UsdRadiationRegistry,
        revision: RevisionState | None = None,
        on_change: Callable[[RevisionState], None] | None = None,
    ) -> None:
        self.stage = stage
        self.registry = registry
        self.revision = revision if revision is not None else RevisionState()
        self.on_change = on_change
        self._listener: Any | None = None

    def start(self) -> None:
        if self._listener is not None:
            return
        _, tf, usd, _ = _pxr_modules()
        self._listener = tf.Notice.Register(
            usd.Notice.ObjectsChanged, self._on_objects_changed, self.stage
        )

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.Revoke()
            self._listener = None

    def _on_objects_changed(self, notice: Any, _sender: Any) -> None:
        snapshot = self.registry.snapshot
        changed_paths = tuple(str(path) for path in notice.GetChangedInfoOnlyPaths())
        resynced_paths = tuple(str(path) for path in notice.GetResyncedPaths())
        delta = classify_stage_changes(
            changed_paths,
            resynced_paths,
            source_prim_paths=frozenset(snapshot.sources),
            geometry_prim_paths=snapshot.geometry_prim_paths,
            detector_prim_paths=snapshot.detector_prim_paths,
        )
        delta.apply(self.revision)
        if delta.registry_refresh:
            self.registry.refresh()
        if self.on_change is not None:
            self.on_change(self.revision.copy())
