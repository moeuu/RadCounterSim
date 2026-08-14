"""Metric visual-twin ingestion with texture-preserving USD conversion."""

from __future__ import annotations

import fnmatch
import hashlib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from radcounter.core.environment import load_environment_manifest
from radcounter.core.rendering import DigitalTwinAssetConfig


@dataclass(frozen=True)
class IngestionReport:
    source_uri: str
    mounted_uri: str
    prim_path: str
    source_meters_per_unit: float
    applied_scale: float
    instanceable_prim_count: int
    normalized_manifest: bool


def _local_path(uri: str, base_directory: str) -> Path | None:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        return None
    value = Path(parsed.path if parsed.scheme == "file" else uri).expanduser()
    return (Path(base_directory) / value).resolve() if not value.is_absolute() else value.resolve()


class DigitalTwinIngestor:
    """Mount native USD, converted DCC assets, or a normalized mesh manifest."""

    _USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
    _CONVERTIBLE_SUFFIXES = {".obj", ".fbx", ".gltf", ".glb", ".dae", ".stl", ".ply"}

    def __init__(self, stage, *, base_directory: str = ".") -> None:
        self.stage = stage
        self.base_directory = base_directory

    async def mount(self, config: DigitalTwinAssetConfig) -> IngestionReport:
        if config.visual_uri is None:
            assert config.physics_manifest_uri is not None
            return self._mount_manifest(config.physics_manifest_uri, config)
        local = _local_path(config.visual_uri, self.base_directory)
        suffix = Path(urllib.parse.urlparse(config.visual_uri).path).suffix.lower()
        if suffix == ".json" and local is not None:
            return self._mount_manifest(str(local), config)
        if suffix not in self._USD_SUFFIXES:
            if local is None or suffix not in self._CONVERTIBLE_SUFFIXES:
                raise ValueError(
                    "The visual twin must be USD, a DCC mesh supported by Omni asset converter, "
                    "or a RadCounterSim environment manifest. CAD/BIM/LiDAR sources should first "
                    "be normalized with radcounter-import-environment."
                )
            mounted_uri = str(await self._convert_to_usd(local, config))
        else:
            mounted_uri = str(local) if local is not None else config.visual_uri
        source_mpu, source_up = self._inspect_usd(mounted_uri)
        source_mpu = config.source_meters_per_unit or source_mpu
        source_up = config.source_up_axis if config.source_up_axis != "auto" else source_up
        target_mpu = self._stage_meters_per_unit()
        applied_scale = source_mpu / target_mpu * config.scale
        root = self._reference_asset(mounted_uri, config, applied_scale, source_up)
        count = self._mark_instanceable(root, config.instanceable_patterns)
        return IngestionReport(
            source_uri=config.visual_uri,
            mounted_uri=mounted_uri,
            prim_path=config.prim_path,
            source_meters_per_unit=source_mpu,
            applied_scale=applied_scale,
            instanceable_prim_count=count,
            normalized_manifest=False,
        )

    async def _convert_to_usd(self, source: Path, config: DigitalTwinAssetConfig) -> Path:
        import omni.kit.asset_converter

        digest_state = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest_state.update(block)
        digest = digest_state.hexdigest()[:20]
        cache = Path(config.cache_directory).expanduser()
        if not cache.is_absolute():
            cache = Path(self.base_directory) / cache
        output = cache.resolve() / f"{source.stem}-{digest}.usdc"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file():
            return output
        context = omni.kit.asset_converter.AssetConverterContext()
        for name, value in {
            "ignore_materials": False,
            "ignore_animations": False,
            "merge_all_meshes": False,
            "use_meter_as_world_unit": False,
            "embed_textures": False,
        }.items():
            if hasattr(context, name):
                setattr(context, name, value)
        converter = omni.kit.asset_converter.get_instance()
        task = converter.create_converter_task(str(source), str(output), None, context)
        if not await task.wait_until_finished():
            error = task.get_error_message() if hasattr(task, "get_error_message") else "unknown"
            raise RuntimeError(f"Omni asset conversion failed for {source}: {error}")
        return output

    def _mount_manifest(
        self,
        manifest_uri: str,
        config: DigitalTwinAssetConfig,
    ) -> IngestionReport:
        from pxr import Gf, Sdf, UsdGeom

        local = _local_path(manifest_uri, self.base_directory)
        if local is None:
            raise ValueError("normalized environment manifests must be local files")
        result = load_environment_manifest(local)
        root = UsdGeom.Xform.Define(self.stage, config.prim_path)
        xformable = UsdGeom.Xformable(root.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*config.translation_m))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*config.rotation_rpy_deg))
        xformable.AddScaleOp().Set(Gf.Vec3f(config.scale))
        for mesh in result.scene.meshes:
            if not mesh.visual_enabled:
                continue
            path = f"{config.prim_path}/Meshes/{mesh.mesh_id.replace('-', '_')}"
            authored = UsdGeom.Mesh.Define(self.stage, path)
            authored.CreatePointsAttr([Gf.Vec3f(*point) for point in mesh.vertices_m])
            authored.CreateFaceVertexCountsAttr([3] * len(mesh.triangles))
            authored.CreateFaceVertexIndicesAttr(mesh.triangles.reshape(-1).tolist())
            authored.CreateSubdivisionSchemeAttr("none")
            prim = authored.GetPrim()
            prim.CreateAttribute("rad:material:id", Sdf.ValueTypeNames.String).Set(mesh.material_id)
            prim.CreateAttribute("rad:visual:normalizedFallback", Sdf.ValueTypeNames.Bool).Set(True)
            UsdGeom.Imageable(prim).CreatePurposeAttr("render")
        return IngestionReport(
            source_uri=manifest_uri,
            mounted_uri=str(local),
            prim_path=config.prim_path,
            source_meters_per_unit=1.0,
            applied_scale=config.scale,
            instanceable_prim_count=0,
            normalized_manifest=True,
        )

    def _reference_asset(
        self,
        uri: str,
        config: DigitalTwinAssetConfig,
        scale: float,
        source_up_axis: str,
    ):
        from pxr import Gf, Sdf, UsdGeom

        root = UsdGeom.Xform.Define(self.stage, config.prim_path)
        prim = root.GetPrim()
        prim.GetReferences().AddReference(uri)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*config.translation_m))
        rotation = list(config.rotation_rpy_deg)
        if source_up_axis.upper() == "Y":
            rotation[0] += 90.0
        elif source_up_axis.upper() == "X":
            rotation[1] -= 90.0
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        xformable.AddScaleOp().Set(Gf.Vec3f(scale))
        prim.CreateAttribute("rad:digitalTwin:sourceUri", Sdf.ValueTypeNames.String).Set(uri)
        prim.CreateAttribute("rad:digitalTwin:metricScale", Sdf.ValueTypeNames.Double).Set(scale)
        return prim

    def _mark_instanceable(self, root, patterns: tuple[str, ...]) -> int:
        if not patterns:
            return 0
        count = 0
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(str(root.GetPath())):
                continue
            if not any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns):
                continue
            if prim.HasAuthoredReferences() or prim.HasAuthoredPayloads():
                prim.SetInstanceable(True)
                count += 1
        return count

    def _inspect_usd(self, uri: str) -> tuple[float, str]:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(uri, load=Usd.Stage.LoadNone)
        if stage is None:
            return 1.0, "Z"
        return float(UsdGeom.GetStageMetersPerUnit(stage)), str(UsdGeom.GetStageUpAxis(stage))

    def _stage_meters_per_unit(self) -> float:
        from pxr import UsdGeom

        value = float(UsdGeom.GetStageMetersPerUnit(self.stage))
        return value if value > 0.0 else 1.0
