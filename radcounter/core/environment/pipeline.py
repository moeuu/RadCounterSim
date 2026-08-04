"""Format-neutral environment import, resolution, normalization, and caching."""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.metadata
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from radcounter.core.environment.geometry import apply_transform, normalization_matrix
from radcounter.core.environment.importers import (
    EnvironmentImporter,
    EnvironmentImporterRegistry,
    EnvironmentImportError,
    ImportContext,
    run_external_converter,
)
from radcounter.core.environment.models import (
    EnvironmentFormat,
    EnvironmentImportConfig,
    EnvironmentImportResult,
    EnvironmentMesh,
    EnvironmentScene,
    RawEnvironmentScene,
)

_SUFFIX_FORMATS = {
    ".usd": EnvironmentFormat.USD,
    ".usda": EnvironmentFormat.USD,
    ".usdc": EnvironmentFormat.USD,
    ".usdz": EnvironmentFormat.USD,
    ".gltf": EnvironmentFormat.GLTF,
    ".glb": EnvironmentFormat.GLB,
    ".obj": EnvironmentFormat.OBJ,
    ".stl": EnvironmentFormat.STL,
    ".ply": EnvironmentFormat.PLY,
    ".dae": EnvironmentFormat.DAE,
    ".3mf": EnvironmentFormat.THREE_MF,
    ".off": EnvironmentFormat.OFF,
    ".fbx": EnvironmentFormat.FBX,
    ".sdf": EnvironmentFormat.SDF,
    ".world": EnvironmentFormat.SDF,
    ".urdf": EnvironmentFormat.URDF,
    ".xacro": EnvironmentFormat.XACRO,
    ".step": EnvironmentFormat.STEP,
    ".stp": EnvironmentFormat.STEP,
    ".iges": EnvironmentFormat.IGES,
    ".igs": EnvironmentFormat.IGES,
    ".brep": EnvironmentFormat.BREP,
    ".pcd": EnvironmentFormat.PCD,
    ".xyz": EnvironmentFormat.XYZ,
    ".pts": EnvironmentFormat.XYZ,
    ".ifc": EnvironmentFormat.IFC,
    ".e57": EnvironmentFormat.E57,
    ".las": EnvironmentFormat.LAS,
    ".laz": EnvironmentFormat.LAZ,
    ".bt": EnvironmentFormat.OCTOMAP,
    ".ot": EnvironmentFormat.OCTOMAP,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EnvironmentAssetResolver:
    """Resolve local, ROS package, Gazebo model, and HTTPS environment URIs."""

    def __init__(self, config: EnvironmentImportConfig, base_directory: Path) -> None:
        self.config = config
        self.base_directory = base_directory.resolve()
        self.cache_root = Path(config.cache_directory).expanduser().resolve()

    @staticmethod
    def _search_values(name: str) -> tuple[Path, ...]:
        return tuple(
            Path(value).expanduser()
            for value in os.environ.get(name, "").split(os.pathsep)
            if value
        )

    def resolve(self, uri: str, relative_to: Path | None = None) -> Path:
        expanded = os.path.expandvars(os.path.expanduser(uri))
        parsed = urllib.parse.urlparse(expanded)
        if parsed.scheme in {"http", "https"}:
            return self._download(expanded)
        if parsed.scheme == "file":
            candidate = Path(urllib.request.url2pathname(parsed.path))
        elif expanded.startswith("model://"):
            candidate = self._resolve_model(expanded.removeprefix("model://"))
        elif expanded.startswith("package://"):
            candidate = self._resolve_package(expanded.removeprefix("package://"))
        else:
            candidate = Path(expanded)
            if not candidate.is_absolute():
                candidate = (relative_to or self.base_directory) / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if candidate.is_dir() and (candidate / "model.sdf").is_file():
            return candidate / "model.sdf"
        return candidate

    def _resolve_model(self, relative: str) -> Path:
        roots = tuple(Path(path).expanduser() for path in self.config.model_search_paths)
        roots += self._search_values("GZ_SIM_RESOURCE_PATH")
        roots += self._search_values("GAZEBO_MODEL_PATH")
        for root in roots:
            candidate = root / relative
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Gazebo model URI was not found: model://{relative}")

    def _resolve_package(self, relative: str) -> Path:
        package, separator, remainder = relative.partition("/")
        if not package or not separator:
            raise FileNotFoundError(f"invalid ROS package URI: package://{relative}")
        roots = tuple(Path(path).expanduser() for path in self.config.package_search_paths)
        roots += self._search_values("ROS_PACKAGE_PATH")
        for prefix in self._search_values("AMENT_PREFIX_PATH"):
            roots += (prefix / "share",)
        for root in roots:
            candidates = [root / package / remainder]
            if root.name == package:
                candidates.append(root / remainder)
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(f"ROS package URI was not found: package://{relative}")

    def _download(self, uri: str) -> Path:
        url_digest = hashlib.sha256(uri.encode()).hexdigest()
        suffix = Path(urllib.parse.urlparse(uri).path).suffix
        destination = self.cache_root / "downloads" / f"{url_digest}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            temporary = destination.with_suffix(destination.suffix + ".part")
            with (
                urllib.request.urlopen(uri, timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                while block := response.read(1024 * 1024):
                    stream.write(block)
            temporary.replace(destination)
        return destination


def detect_environment_format(path: Path, requested: EnvironmentFormat) -> EnvironmentFormat:
    if requested != EnvironmentFormat.AUTO:
        return requested
    detected = _SUFFIX_FORMATS.get(path.suffix.lower())
    if detected is not None:
        return detected
    if path.suffix.lower() == ".xml":
        import xml.etree.ElementTree as ET

        root = ET.parse(path).getroot().tag
        if root == "robot":
            return EnvironmentFormat.URDF
        if root in {"sdf", "world", "model"}:
            return EnvironmentFormat.SDF
    raise EnvironmentImportError(
        f"cannot detect the environment format from {path.name}; set environment.format"
    )


class EnvironmentImportPipeline:
    """Import any registered source into one portable normalized scene."""

    def __init__(self, registry: EnvironmentImporterRegistry | None = None) -> None:
        self.registry = registry or EnvironmentImporterRegistry.with_builtins()
        self._load_plugins()

    def _load_plugins(self) -> None:
        for entry_point in importlib.metadata.entry_points(
            group="radcounter.environment_importers"
        ):
            loaded = entry_point.load()
            importer = loaded() if callable(loaded) and not hasattr(loaded, "formats") else loaded
            self.registry.register(importer)

    def register_importer(self, importer: EnvironmentImporter, *, replace: bool = False) -> None:
        self.registry.register(importer, replace=replace)

    def import_environment(
        self,
        config: EnvironmentImportConfig,
        *,
        base_directory: str | Path = ".",
    ) -> EnvironmentImportResult:
        base = Path(base_directory).expanduser().resolve()
        resolver = EnvironmentAssetResolver(config, base)
        source = resolver.resolve(config.uri, base)
        if not source.is_file():
            raise EnvironmentImportError(f"environment source is not a file: {source}")
        source_digest = file_sha256(source)
        if config.expected_sha256 is not None and source_digest != config.expected_sha256:
            raise EnvironmentImportError(
                "environment SHA256 mismatch: "
                f"expected {config.expected_sha256}, got {source_digest}"
            )
        environment_format = detect_environment_format(source, config.format)
        cache_key = self._cache_key(source_digest, config)
        output_directory = Path(config.cache_directory).expanduser().resolve() / cache_key
        output_directory.mkdir(parents=True, exist_ok=True)
        if environment_format == EnvironmentFormat.USD:
            scene = EnvironmentScene(
                environment_id=config.environment_id,
                source_uri=str(source),
                source_sha256=source_digest,
                source_format=environment_format,
                dependencies=((str(source), source_digest),),
                native_stage_uri=str(source),
                import_config=config,
            )
        else:
            context = ImportContext(
                config=config,
                resolve_uri=lambda uri, relative: resolver.resolve(uri, relative),
            )
            try:
                raw_scene = self.registry.get(environment_format).load(source, context)
            except EnvironmentImportError:
                if config.external_converter is None:
                    raise
                converted, converted_format = run_external_converter(
                    source, config, output_directory
                )
                raw_scene = self.registry.get(converted_format).load(converted, context)
                context.dependencies.add(converted)
            scene = self._normalize(
                raw_scene,
                source=source,
                source_digest=source_digest,
                environment_format=environment_format,
                config=config,
            )
        normalized_path = self._write_cache(scene, config, output_directory)
        return EnvironmentImportResult(
            scene=scene,
            cache_key=cache_key,
            output_directory=output_directory,
            manifest_path=output_directory / "manifest.json",
            normalized_mesh_path=normalized_path,
        )

    @staticmethod
    def _cache_key(source_digest: str, config: EnvironmentImportConfig) -> str:
        digest = hashlib.sha256()
        digest.update(source_digest.encode())
        digest.update(config.model_dump_json(exclude={"expected_sha256"}).encode())
        return digest.hexdigest()[:24]

    @staticmethod
    def _material(name: str, config: EnvironmentImportConfig) -> str:
        for rule in config.material_rules:
            if fnmatch.fnmatchcase(name, rule.pattern):
                return rule.material_id
        return config.default_material_id

    def _normalize(
        self,
        raw_scene: RawEnvironmentScene,
        *,
        source: Path,
        source_digest: str,
        environment_format: EnvironmentFormat,
        config: EnvironmentImportConfig,
    ) -> EnvironmentScene:
        transform = normalization_matrix(config, raw_scene.coordinate_defaults)
        reverse_winding = np.linalg.det(transform[:3, :3]) < 0.0
        meshes: list[EnvironmentMesh] = []
        triangle_count = 0
        for index, raw in enumerate(raw_scene.meshes):
            triangles = raw.triangles[:, ::-1] if reverse_winding else raw.triangles
            triangle_count += len(triangles)
            if triangle_count > config.max_triangles and not config.streaming.enabled:
                raise EnvironmentImportError(
                    f"environment exceeds max_triangles={config.max_triangles}; "
                    "decimate the source or raise the explicit limit"
                )
            material_name = raw.source_material or raw.name
            meshes.append(
                EnvironmentMesh(
                    mesh_id=f"mesh-{index:06d}",
                    name=raw.name,
                    vertices_m=apply_transform(raw.vertices, transform),
                    triangles=triangles,
                    material_id=self._material(material_name, config),
                    visual_enabled=raw.visual_enabled,
                    collision_enabled=config.collision.enabled and raw.collision_enabled,
                    radiation_enabled=raw.radiation_enabled,
                    display_color_rgb=raw.display_color_rgb,
                )
            )
        dependencies = tuple(
            (str(path), file_sha256(path))
            for path in sorted(set(raw_scene.dependencies) | {source})
            if path.is_file()
        )
        return EnvironmentScene(
            environment_id=config.environment_id,
            source_uri=str(source),
            source_sha256=source_digest,
            source_format=environment_format,
            meshes=tuple(meshes),
            dependencies=dependencies,
            warnings=raw_scene.warnings,
            import_config=config,
        )

    @staticmethod
    def _write_cache(
        scene: EnvironmentScene,
        config: EnvironmentImportConfig,
        output_directory: Path,
    ) -> Path | None:
        arrays: dict[str, np.ndarray] = {}
        mesh_records: list[dict[str, object]] = []
        for index, mesh in enumerate(scene.meshes):
            vertex_key = f"vertices_{index:06d}"
            triangle_key = f"triangles_{index:06d}"
            arrays[vertex_key] = mesh.vertices_m
            arrays[triangle_key] = mesh.triangles
            mesh_records.append(
                {
                    "mesh_id": mesh.mesh_id,
                    "name": mesh.name,
                    "vertex_array": vertex_key,
                    "triangle_array": triangle_key,
                    "material_id": mesh.material_id,
                    "visual_enabled": mesh.visual_enabled,
                    "collision_enabled": mesh.collision_enabled,
                    "radiation_enabled": mesh.radiation_enabled,
                    "display_color_rgb": mesh.display_color_rgb,
                    "bounds_m": mesh.bounds_m,
                }
            )
        normalized_path: Path | None = None
        if arrays:
            normalized_path = output_directory / "normalized_meshes.npz"
            with tempfile.NamedTemporaryFile(
                dir=output_directory, suffix=".npz", delete=False
            ) as stream:
                temporary_path = Path(stream.name)
            np.savez_compressed(temporary_path, **arrays)
            temporary_path.replace(normalized_path)
        manifest = {
            "schema_version": "1.0",
            "environment_id": scene.environment_id,
            "source_uri": scene.source_uri,
            "source_sha256": scene.source_sha256,
            "source_format": scene.source_format.value,
            "native_stage_uri": scene.native_stage_uri,
            "normalized_mesh_path": normalized_path.name if normalized_path else None,
            "vertex_count": scene.vertex_count,
            "triangle_count": scene.triangle_count,
            "bounds_m": scene.bounds_m,
            "dependencies": [dict(path=path, sha256=digest) for path, digest in scene.dependencies],
            "warnings": scene.warnings,
            "meshes": mesh_records,
            "resolved_import_config": config.model_dump(mode="json"),
        }
        manifest_path = output_directory / "manifest.json"
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_manifest.replace(manifest_path)
        return normalized_path


def load_environment_manifest(path: str | Path) -> EnvironmentImportResult:
    """Load a previously normalized cache without requiring its source importer."""

    manifest_path = Path(path).expanduser().resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1.0" or "meshes" not in raw:
        raise ValueError(f"not a RadCounterSim environment manifest: {manifest_path}")
    config = EnvironmentImportConfig.model_validate(raw["resolved_import_config"])
    normalized_name = raw.get("normalized_mesh_path")
    arrays = None
    if normalized_name:
        arrays = np.load(manifest_path.parent / normalized_name, allow_pickle=False)
    meshes: list[EnvironmentMesh] = []
    try:
        for record in raw["meshes"]:
            if arrays is None:
                raise ValueError("environment manifest has meshes but no normalized array file")
            meshes.append(
                EnvironmentMesh(
                    mesh_id=record["mesh_id"],
                    name=record["name"],
                    vertices_m=arrays[record["vertex_array"]],
                    triangles=arrays[record["triangle_array"]],
                    material_id=record["material_id"],
                    visual_enabled=record["visual_enabled"],
                    collision_enabled=record["collision_enabled"],
                    radiation_enabled=record["radiation_enabled"],
                    display_color_rgb=(
                        tuple(record["display_color_rgb"])
                        if record.get("display_color_rgb") is not None
                        else None
                    ),
                )
            )
    finally:
        if arrays is not None:
            arrays.close()
    scene = EnvironmentScene(
        environment_id=raw["environment_id"],
        source_uri=raw["source_uri"],
        source_sha256=raw["source_sha256"],
        source_format=EnvironmentFormat(raw["source_format"]),
        meshes=tuple(meshes),
        dependencies=tuple((item["path"], item["sha256"]) for item in raw.get("dependencies", [])),
        warnings=tuple(raw.get("warnings", [])),
        native_stage_uri=raw.get("native_stage_uri"),
        import_config=config,
    )
    normalized_path = manifest_path.parent / normalized_name if normalized_name else None
    return EnvironmentImportResult(
        scene=scene,
        cache_key=manifest_path.parent.name,
        output_directory=manifest_path.parent,
        manifest_path=manifest_path,
        normalized_mesh_path=normalized_path,
    )
