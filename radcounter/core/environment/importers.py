"""Built-in mesh, robotics-world, CAD, and point-cloud importers."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from radcounter.core.environment.geometry import apply_transform, pose_matrix, voxel_surface_mesh
from radcounter.core.environment.models import (
    AxisDirection,
    CollisionGeometrySource,
    CoordinateDefaults,
    EnvironmentFormat,
    EnvironmentImportConfig,
    RawEnvironmentMesh,
    RawEnvironmentScene,
)


class EnvironmentImportError(RuntimeError):
    """Raised when an environment cannot be converted without silent fallback."""


class EnvironmentDependencyError(EnvironmentImportError):
    """Raised when an explicitly requested format adapter is unavailable."""


@dataclass
class ImportContext:
    config: EnvironmentImportConfig
    resolve_uri: Callable[[str, Path], Path]
    dependencies: set[Path] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


class EnvironmentImporter(Protocol):
    formats: frozenset[EnvironmentFormat]

    def load(self, source: Path, context: ImportContext) -> RawEnvironmentScene: ...


def _material_color(geometry: object) -> tuple[str | None, tuple[float, float, float] | None]:
    visual = getattr(geometry, "visual", None)
    material = getattr(visual, "material", None)
    name = getattr(material, "name", None)
    diffuse = getattr(material, "diffuse", None)
    if diffuse is None:
        return str(name) if name else None, None
    values = np.asarray(diffuse, dtype=np.float64).reshape(-1)
    if len(values) < 3:
        return str(name) if name else None, None
    scale = 255.0 if values[:3].max(initial=0.0) > 1.0 else 1.0
    return str(name) if name else None, tuple(np.clip(values[:3] / scale, 0.0, 1.0))


def _trimesh_scene(
    source: Path,
    *,
    name_prefix: str = "",
    parent_transform: np.ndarray | None = None,
    visual_enabled: bool = True,
    collision_enabled: bool = True,
    radiation_enabled: bool = True,
    point_cloud_voxel_size: float = 0.1,
    max_point_cloud_voxels: int = 500_000,
) -> tuple[RawEnvironmentMesh, ...]:
    try:
        import trimesh
    except ModuleNotFoundError as error:
        raise EnvironmentDependencyError(
            "standard environment import requires the uv-managed trimesh dependency"
        ) from error
    try:
        scene = trimesh.load_scene(source, process=False)
    except Exception as error:
        raise EnvironmentImportError(
            f"trimesh could not read {source.suffix or 'this format'}: {error}. "
            "Configure external_converter for formats such as proprietary FBX."
        ) from error
    parent = np.eye(4) if parent_transform is None else parent_transform
    meshes: list[RawEnvironmentMesh] = []
    for node_name in sorted(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        vertices_value = getattr(geometry, "vertices", None)
        faces_value = getattr(geometry, "faces", None)
        if vertices_value is None:
            continue
        if faces_value is None or len(faces_value) == 0:
            vertices, triangles = voxel_surface_mesh(
                np.asarray(vertices_value),
                voxel_size=point_cloud_voxel_size,
                max_voxels=max_point_cloud_voxels,
            )
        else:
            vertices = np.asarray(vertices_value, dtype=np.float64)
            faces = np.asarray(faces_value, dtype=np.int64)
            if faces.ndim != 2:
                continue
            if faces.shape[1] == 3:
                triangles = faces
            else:
                triangles = np.vstack(
                    [
                        np.column_stack(
                            (
                                np.full(len(face) - 2, face[0]),
                                face[1:-1],
                                face[2:],
                            )
                        )
                        for face in faces
                    ]
                )
        vertices = apply_transform(vertices, parent @ np.asarray(transform, dtype=np.float64))
        material, color = _material_color(geometry)
        mesh_name = f"{name_prefix}{node_name or geometry_name}"
        meshes.append(
            RawEnvironmentMesh(
                mesh_name,
                vertices,
                triangles,
                source_material=material or str(geometry_name),
                visual_enabled=visual_enabled,
                collision_enabled=collision_enabled,
                radiation_enabled=radiation_enabled,
                display_color_rgb=color,
            )
        )
    if not meshes:
        raise EnvironmentImportError(
            f"environment contains no triangle or point geometry: {source}"
        )
    return tuple(meshes)


class TrimeshEnvironmentImporter:
    formats = frozenset(
        {
            EnvironmentFormat.GLTF,
            EnvironmentFormat.GLB,
            EnvironmentFormat.OBJ,
            EnvironmentFormat.STL,
            EnvironmentFormat.PLY,
            EnvironmentFormat.DAE,
            EnvironmentFormat.THREE_MF,
            EnvironmentFormat.OFF,
            EnvironmentFormat.FBX,
        }
    )

    def load(self, source: Path, context: ImportContext) -> RawEnvironmentScene:
        context.dependencies.add(source)
        defaults = CoordinateDefaults()
        if source.suffix.lower() in {".gltf", ".glb", ".fbx"}:
            defaults = CoordinateDefaults(
                up_axis=AxisDirection.POS_Y,
                forward_axis=AxisDirection.NEG_Z,
            )
        return RawEnvironmentScene(
            _trimesh_scene(
                source,
                point_cloud_voxel_size=_raw_point_cloud_voxel_size(context.config),
                max_point_cloud_voxels=context.config.max_point_cloud_voxels,
            ),
            coordinate_defaults=defaults,
            dependencies=tuple(sorted(context.dependencies)),
            warnings=tuple(context.warnings),
        )


def _parse_floats(text: str | None, count: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    values = tuple(float(value) for value in text.replace(",", " ").split())
    if len(values) != count:
        raise EnvironmentImportError(f"expected {count} numeric values, received {len(values)}")
    return values


def _raw_point_cloud_voxel_size(config: EnvironmentImportConfig) -> float:
    unit_scale = (
        1.0
        if config.coordinate_system.units.value == "auto"
        else config.coordinate_system.units.metres
    )
    return config.point_cloud_voxel_size_m / (unit_scale * config.scale)


def _origin(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4)
    xyz = _parse_floats(element.get("xyz"), 3, (0.0, 0.0, 0.0))
    rpy = _parse_floats(element.get("rpy"), 3, (0.0, 0.0, 0.0))
    return pose_matrix(xyz, rpy)


def _sdf_pose(element: ET.Element | None) -> np.ndarray:
    values = _parse_floats(element.text if element is not None else None, 6, (0.0,) * 6)
    return pose_matrix(values[:3], values[3:])


def _primitive_mesh(geometry: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ModuleNotFoundError as error:
        raise EnvironmentDependencyError("primitive geometry requires trimesh") from error
    box = geometry.find("box")
    if box is not None:
        size_text = box.get("size") or (box.findtext("size") if box is not None else None)
        size = _parse_floats(size_text, 3, (1.0, 1.0, 1.0))
        mesh = trimesh.creation.box(extents=size)
    else:
        cylinder = geometry.find("cylinder")
        sphere = geometry.find("sphere")
        plane = geometry.find("plane")
        if cylinder is not None:
            radius = float(cylinder.get("radius") or cylinder.findtext("radius") or 0.5)
            length = float(
                cylinder.get("length")
                or cylinder.get("height")
                or cylinder.findtext("length")
                or 1.0
            )
            mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=32)
        elif sphere is not None:
            radius = float(sphere.get("radius") or sphere.findtext("radius") or 0.5)
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        elif plane is not None:
            size = _parse_floats(plane.findtext("size"), 2, (10.0, 10.0))
            mesh = trimesh.creation.box(extents=(size[0], size[1], 0.01))
        else:
            raise EnvironmentImportError("unsupported URDF/SDF primitive geometry")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _robot_geometry(
    geometry: ET.Element,
    transform: np.ndarray,
    name: str,
    context: ImportContext,
    reference_directory: Path,
    *,
    visual: bool,
    collision: bool,
    radiation: bool,
) -> tuple[RawEnvironmentMesh, ...]:
    mesh_element = geometry.find("mesh")
    if mesh_element is not None:
        uri = (
            mesh_element.get("filename") or mesh_element.get("uri") or mesh_element.findtext("uri")
        )
        if not uri:
            raise EnvironmentImportError(f"mesh URI is missing for {name}")
        source = context.resolve_uri(uri, reference_directory)
        context.dependencies.add(source)
        scale = _parse_floats(
            mesh_element.get("scale") or mesh_element.findtext("scale"),
            3,
            (1.0, 1.0, 1.0),
        )
        mesh_transform = transform.copy()
        mesh_transform[:3, :3] = mesh_transform[:3, :3] @ np.diag(scale)
        return _trimesh_scene(
            source,
            name_prefix=f"{name}/",
            parent_transform=mesh_transform,
            visual_enabled=visual,
            collision_enabled=collision,
            radiation_enabled=radiation,
        )
    vertices, triangles = _primitive_mesh(geometry)
    return (
        RawEnvironmentMesh(
            name,
            apply_transform(vertices, transform),
            triangles,
            visual_enabled=visual,
            collision_enabled=collision,
            radiation_enabled=radiation,
        ),
    )


def _selection(
    source: CollisionGeometrySource,
    *,
    has_collision: bool,
    is_visual: bool,
    show_collision: bool,
) -> tuple[bool, bool, bool]:
    if source == CollisionGeometrySource.VISUAL:
        selected = is_visual
    elif source == CollisionGeometrySource.COLLISION:
        selected = not is_visual
    elif source == CollisionGeometrySource.BOTH:
        selected = True
    else:
        selected = (not is_visual) if has_collision else is_visual
    visible = is_visual or (selected and show_collision)
    return visible, selected, selected


class RoboticsEnvironmentImporter:
    formats = frozenset({EnvironmentFormat.SDF, EnvironmentFormat.URDF, EnvironmentFormat.XACRO})

    def load(self, source: Path, context: ImportContext) -> RawEnvironmentScene:
        context.dependencies.add(source)
        if source.suffix.lower() == ".xacro":
            source = self._expand_xacro(source, context)
        root = ET.parse(source).getroot()
        if root.tag == "robot":
            meshes = self._load_urdf(root, source, context)
        elif root.tag in {"sdf", "world", "model"}:
            meshes = self._load_sdf_root(root, source, context)
        else:
            raise EnvironmentImportError(f"unsupported robotics XML root element: {root.tag}")
        return RawEnvironmentScene(
            tuple(meshes),
            coordinate_defaults=CoordinateDefaults(),
            dependencies=tuple(sorted(context.dependencies)),
            warnings=tuple(context.warnings),
        )

    def _expand_xacro(self, source: Path, context: ImportContext) -> Path:
        executable = shutil.which("xacro")
        if executable is None:
            raise EnvironmentDependencyError(
                "Xacro input requires the ROS xacro executable or an external converter"
            )
        output = Path(tempfile.mkdtemp(prefix="radcounter-xacro-")) / "expanded.urdf"
        completed = subprocess.run(
            [executable, str(source), "-o", str(output)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise EnvironmentImportError(f"xacro failed: {completed.stderr.strip()}")
        context.dependencies.add(output)
        return output

    def _load_urdf(
        self,
        root: ET.Element,
        source: Path,
        context: ImportContext,
    ) -> list[RawEnvironmentMesh]:
        links = {element.get("name", "link"): element for element in root.findall("link")}
        parent_joint: dict[str, tuple[str, np.ndarray]] = {}
        children: dict[str, list[str]] = {}
        for joint in root.findall("joint"):
            parent_element = joint.find("parent")
            child_element = joint.find("child")
            if parent_element is None or child_element is None:
                continue
            parent = parent_element.get("link", "")
            child = child_element.get("link", "")
            parent_joint[child] = (parent, _origin(joint.find("origin")))
            children.setdefault(parent, []).append(child)
        roots = sorted(set(links) - set(parent_joint)) or sorted(links)[:1]
        world: dict[str, np.ndarray] = {}

        def visit(link_name: str, transform: np.ndarray) -> None:
            world[link_name] = transform
            for child in children.get(link_name, []):
                visit(child, transform @ parent_joint[child][1])

        for root_name in roots:
            visit(root_name, np.eye(4))
        meshes: list[RawEnvironmentMesh] = []
        selection = context.config.collision.geometry_source
        for link_name, link in links.items():
            collision_elements = link.findall("collision")
            has_collision = bool(collision_elements)
            for kind, elements in (
                ("visual", link.findall("visual")),
                ("collision", collision_elements),
            ):
                is_visual = kind == "visual"
                visible, collision, radiation = _selection(
                    selection,
                    has_collision=has_collision,
                    is_visual=is_visual,
                    show_collision=context.config.collision.show_collision_geometry,
                )
                if is_visual is False and selection == CollisionGeometrySource.VISUAL:
                    continue
                geometry_items = elements
                for index, element in enumerate(geometry_items):
                    geometry = element.find("geometry")
                    if geometry is None:
                        continue
                    transform = world.get(link_name, np.eye(4)) @ _origin(element.find("origin"))
                    meshes.extend(
                        _robot_geometry(
                            geometry,
                            transform,
                            f"{link_name}/{kind}-{index}",
                            context,
                            source.parent,
                            visual=visible,
                            collision=collision,
                            radiation=radiation,
                        )
                    )
        if not meshes:
            raise EnvironmentImportError(f"URDF contains no usable geometry: {source}")
        return meshes

    def _load_sdf_root(
        self,
        root: ET.Element,
        source: Path,
        context: ImportContext,
    ) -> list[RawEnvironmentMesh]:
        containers = [root]
        if root.tag == "sdf":
            containers = list(root.findall("world")) + list(root.findall("model"))
        meshes: list[RawEnvironmentMesh] = []
        for container in containers:
            if container.tag == "model":
                meshes.extend(self._load_sdf_model(container, np.eye(4), source, context))
            else:
                for model in container.findall("model"):
                    meshes.extend(self._load_sdf_model(model, np.eye(4), source, context))
                for include in container.findall("include"):
                    meshes.extend(self._load_sdf_include(include, np.eye(4), source, context))
        if not meshes:
            raise EnvironmentImportError(f"SDF world contains no usable geometry: {source}")
        return meshes

    def _load_sdf_include(
        self,
        include: ET.Element,
        parent_transform: np.ndarray,
        source: Path,
        context: ImportContext,
    ) -> list[RawEnvironmentMesh]:
        uri = include.findtext("uri")
        if not uri:
            return []
        included = context.resolve_uri(uri, source.parent)
        if included.is_dir():
            included = included / "model.sdf"
        context.dependencies.add(included)
        include_root = ET.parse(included).getroot()
        transform = parent_transform @ _sdf_pose(include.find("pose"))
        models = include_root.findall("model") if include_root.tag == "sdf" else [include_root]
        meshes: list[RawEnvironmentMesh] = []
        for model in models:
            meshes.extend(self._load_sdf_model(model, transform, included, context))
        return meshes

    def _load_sdf_model(
        self,
        model: ET.Element,
        parent_transform: np.ndarray,
        source: Path,
        context: ImportContext,
    ) -> list[RawEnvironmentMesh]:
        model_name = model.get("name", "model")
        model_transform = parent_transform @ _sdf_pose(model.find("pose"))
        meshes: list[RawEnvironmentMesh] = []
        selection = context.config.collision.geometry_source
        for link in model.findall("link"):
            link_name = link.get("name", "link")
            link_transform = model_transform @ _sdf_pose(link.find("pose"))
            collisions = link.findall("collision")
            has_collision = bool(collisions)
            for kind, elements in (("visual", link.findall("visual")), ("collision", collisions)):
                is_visual = kind == "visual"
                visible, collision, radiation = _selection(
                    selection,
                    has_collision=has_collision,
                    is_visual=is_visual,
                    show_collision=context.config.collision.show_collision_geometry,
                )
                if not is_visual and selection == CollisionGeometrySource.VISUAL:
                    continue
                for index, element in enumerate(elements):
                    geometry = element.find("geometry")
                    if geometry is None:
                        continue
                    transform = link_transform @ _sdf_pose(element.find("pose"))
                    meshes.extend(
                        _robot_geometry(
                            geometry,
                            transform,
                            f"{model_name}/{link_name}/{kind}-{index}",
                            context,
                            source.parent,
                            visual=visible,
                            collision=collision,
                            radiation=radiation,
                        )
                    )
        for nested in model.findall("model"):
            meshes.extend(self._load_sdf_model(nested, model_transform, source, context))
        for include in model.findall("include"):
            meshes.extend(self._load_sdf_include(include, model_transform, source, context))
        return meshes


class CadEnvironmentImporter:
    formats = frozenset({EnvironmentFormat.STEP, EnvironmentFormat.IGES, EnvironmentFormat.BREP})

    def load(self, source: Path, context: ImportContext) -> RawEnvironmentScene:
        try:
            import gmsh
        except (ModuleNotFoundError, OSError) as error:
            raise EnvironmentDependencyError(
                "STEP/IGES/BREP import requires `uv sync --group cad` and the Gmsh runtime"
            ) from error
        context.dependencies.add(source)
        initialized = False
        try:
            gmsh.initialize()
            initialized = True
            gmsh.option.setNumber("General.Terminal", 0)
            try:
                gmsh.option.setString("Geometry.OCCTargetUnit", "M")
            except Exception:
                context.warnings.append("Gmsh did not accept OCCTargetUnit; set units explicitly")
            gmsh.model.add("radcounter_environment")
            gmsh.model.occ.importShapes(str(source))
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", context.config.cad_mesh_size_m)
            gmsh.option.setNumber("Mesh.MeshSizeMin", context.config.cad_mesh_size_m / 4.0)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.model.mesh.generate(2)
            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            node_tags_array = np.asarray(node_tags, dtype=np.int64)
            vertices = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
            order = np.argsort(node_tags_array)
            sorted_tags = node_tags_array[order]
            triangles: list[np.ndarray] = []
            element_types, _, element_nodes = gmsh.model.mesh.getElements(dim=2)
            for element_type, flat_nodes in zip(element_types, element_nodes, strict=True):
                properties = gmsh.model.mesh.getElementProperties(element_type)
                nodes_per_element = int(properties[3])
                primary_nodes = int(properties[5])
                tags = np.asarray(flat_nodes, dtype=np.int64).reshape(-1, nodes_per_element)
                tags = tags[:, :primary_nodes]
                locations = np.searchsorted(sorted_tags, tags)
                if np.any(locations >= len(sorted_tags)) or not np.array_equal(
                    sorted_tags[locations], tags
                ):
                    raise EnvironmentImportError("Gmsh returned an unresolved CAD node tag")
                indices = order[locations]
                if primary_nodes == 3:
                    triangles.append(indices)
                elif primary_nodes == 4:
                    triangles.extend((indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]))
                elif primary_nodes > 3:
                    triangles.extend(
                        np.column_stack(
                            (
                                np.full(primary_nodes - 2, row[0]),
                                row[1:-1],
                                row[2:],
                            )
                        )
                        for row in indices
                    )
            if not triangles:
                raise EnvironmentImportError(f"CAD model generated no surface triangles: {source}")
            mesh = RawEnvironmentMesh(source.stem, vertices, np.vstack(triangles))
            return RawEnvironmentScene(
                (mesh,),
                coordinate_defaults=CoordinateDefaults(),
                dependencies=tuple(sorted(context.dependencies)),
                warnings=tuple(context.warnings),
            )
        except EnvironmentImportError:
            raise
        except Exception as error:
            raise EnvironmentImportError(f"Gmsh CAD tessellation failed: {error}") from error
        finally:
            if initialized:
                gmsh.finalize()


def load_pcd_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load PCD coordinates and optional packed RGB without a PCL dependency.

    PCL may emit repeated ``_`` padding fields and reserve bytes after the
    declared point records.  NumPy structured dtypes reject duplicate names,
    and consuming the entire payload interprets the reserve as extra points.
    Field names are therefore made unique and binary input is bounded by the
    POINTS/WIDTH/HEIGHT declaration.
    """

    with path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise EnvironmentImportError("PCD header is incomplete")
            decoded = line.decode("ascii", errors="strict").strip()
            header_lines.append(decoded)
            if decoded.upper().startswith("DATA "):
                break
        header: dict[str, list[str]] = {}
        for line in header_lines:
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            header[key.upper()] = values
        fields = header.get("FIELDS") or header.get("FIELD")
        if not fields or not {"x", "y", "z"}.issubset(fields):
            raise EnvironmentImportError("PCD requires x, y, z fields")
        data_mode = header["DATA"][0].lower()
        if data_mode == "ascii":
            data = np.loadtxt(stream, dtype=np.float64, ndmin=2)
            points = data[:, [fields.index("x"), fields.index("y"), fields.index("z")]]
            colors = _pcd_ascii_colors(data, fields)
            return points, colors
        if data_mode == "binary_compressed":
            raise EnvironmentImportError(
                "binary_compressed PCD requires an external converter to PLY or binary PCD"
            )
        if data_mode != "binary":
            raise EnvironmentImportError(f"unsupported PCD DATA mode: {data_mode}")
        sizes = [int(value) for value in header.get("SIZE", ["4"] * len(fields))]
        types = header.get("TYPE", ["F"] * len(fields))
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        type_map = {
            ("F", 4): "<f4",
            ("F", 8): "<f8",
            ("I", 1): "<i1",
            ("I", 2): "<i2",
            ("I", 4): "<i4",
            ("U", 1): "<u1",
            ("U", 2): "<u2",
            ("U", 4): "<u4",
        }
        dtype_fields: list[tuple[str, object]] = []
        safe_names: list[str] = []
        occurrences: dict[str, int] = {}
        for name, size, kind, count in zip(fields, sizes, types, counts, strict=True):
            dtype = type_map.get((kind.upper(), size))
            if dtype is None:
                raise EnvironmentImportError(f"unsupported PCD field type: {kind}{size}")
            occurrence = occurrences.get(name, 0)
            occurrences[name] = occurrence + 1
            safe_name = name if occurrence == 0 else f"{name}_{occurrence}"
            safe_names.append(safe_name)
            dtype_fields.append((safe_name, dtype if count == 1 else (dtype, (count,))))
        record_dtype = np.dtype(dtype_fields)
        declared_points = int(
            (header.get("POINTS") or [
                str(int(header.get("WIDTH", ["0"])[0]) * int(header.get("HEIGHT", ["1"])[0]))
            ])[0]
        )
        expected_bytes = declared_points * record_dtype.itemsize
        payload = stream.read()
        if len(payload) < expected_bytes:
            raise EnvironmentImportError(
                f"PCD binary payload is truncated: expected {expected_bytes} bytes, "
                f"received {len(payload)}"
            )
        records = np.frombuffer(payload[:expected_bytes], dtype=record_dtype)
        first_names = {
            original: safe
            for original, safe in zip(fields, safe_names, strict=True)
            if original not in fields[: fields.index(original)]
        }
        points = np.column_stack(
            tuple(records[first_names[axis]] for axis in ("x", "y", "z"))
        ).astype(np.float64)
        colors = _pcd_binary_colors(records, first_names)
        return points, colors


def _pcd_ascii_colors(data: np.ndarray, fields: list[str]) -> np.ndarray | None:
    if all(channel in fields for channel in ("r", "g", "b")):
        colors = data[:, [fields.index("r"), fields.index("g"), fields.index("b")]]
        return np.clip(colors / 255.0, 0.0, 1.0)
    packed_name = "rgb" if "rgb" in fields else "rgba" if "rgba" in fields else None
    if packed_name is None:
        return None
    packed_float = np.asarray(data[:, fields.index(packed_name)], dtype="<f4")
    return _unpack_pcd_rgb(packed_float.view("<u4"))


def _pcd_binary_colors(
    records: np.ndarray,
    first_names: dict[str, str],
) -> np.ndarray | None:
    if all(channel in first_names for channel in ("r", "g", "b")):
        colors = np.column_stack(
            tuple(records[first_names[channel]] for channel in ("r", "g", "b"))
        ).astype(np.float64)
        maximum = float(np.max(colors)) if colors.size else 0.0
        return np.clip(colors / 255.0 if maximum > 1.0 else colors, 0.0, 1.0)
    packed_name = "rgb" if "rgb" in first_names else "rgba" if "rgba" in first_names else None
    if packed_name is None:
        return None
    packed_values = np.asarray(records[first_names[packed_name]])
    if np.issubdtype(packed_values.dtype, np.floating):
        packed = packed_values.astype("<f4", copy=False).view("<u4")
    else:
        packed = packed_values.astype("<u4", copy=False)
    return _unpack_pcd_rgb(packed)


def _unpack_pcd_rgb(packed: np.ndarray) -> np.ndarray:
    return np.column_stack(
        ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)
    ).astype(np.float64) / 255.0


def _load_pcd(path: Path) -> np.ndarray:
    return load_pcd_xyz_rgb(path)[0]


class PointCloudEnvironmentImporter:
    formats = frozenset({EnvironmentFormat.PCD, EnvironmentFormat.XYZ})

    def load(self, source: Path, context: ImportContext) -> RawEnvironmentScene:
        context.dependencies.add(source)
        points = (
            _load_pcd(source)
            if source.suffix.lower() == ".pcd"
            else np.loadtxt(source, ndmin=2)[:, :3]
        )
        vertices, triangles = voxel_surface_mesh(
            points,
            voxel_size=_raw_point_cloud_voxel_size(context.config),
            max_voxels=context.config.max_point_cloud_voxels,
        )
        return RawEnvironmentScene(
            (RawEnvironmentMesh(source.stem, vertices, triangles),),
            dependencies=tuple(sorted(context.dependencies)),
            warnings=tuple(context.warnings),
        )


class EnvironmentImporterRegistry:
    """Extensible format registry; third-party packages may register loaders."""

    def __init__(self) -> None:
        self._importers: dict[EnvironmentFormat, EnvironmentImporter] = {}

    def register(self, importer: EnvironmentImporter, *, replace: bool = False) -> None:
        for environment_format in importer.formats:
            if environment_format in self._importers and not replace:
                raise ValueError(f"an importer is already registered for {environment_format}")
            self._importers[environment_format] = importer

    def get(self, environment_format: EnvironmentFormat) -> EnvironmentImporter:
        try:
            return self._importers[environment_format]
        except KeyError as error:
            raise EnvironmentImportError(
                f"no importer is registered for {environment_format.value}; "
                "configure external_converter or register a radcounter environment importer"
            ) from error

    @classmethod
    def with_builtins(cls) -> EnvironmentImporterRegistry:
        registry = cls()
        for importer in (
            TrimeshEnvironmentImporter(),
            RoboticsEnvironmentImporter(),
            CadEnvironmentImporter(),
            PointCloudEnvironmentImporter(),
        ):
            registry.register(importer)
        return registry


def run_external_converter(
    source: Path,
    config: EnvironmentImportConfig,
    output_directory: Path,
) -> tuple[Path, EnvironmentFormat]:
    converter = config.external_converter
    if converter is None:
        raise EnvironmentImportError("no external converter is configured")
    suffix = "." + converter.output_format.value
    output = output_directory / f"converted{suffix}"
    command = tuple(
        argument.replace("{input}", str(source)).replace("{output}", str(output))
        for argument in converter.command
    )
    executable = shutil.which(command[0]) if not os.path.isabs(command[0]) else command[0]
    if executable is None:
        raise EnvironmentDependencyError(
            f"external converter executable is unavailable: {command[0]}"
        )
    completed = subprocess.run(
        (executable, *command[1:]),
        check=False,
        capture_output=True,
        text=True,
        timeout=converter.timeout_s,
    )
    if completed.returncode != 0 or not output.is_file():
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise EnvironmentImportError(
            f"external converter failed ({completed.returncode}): {detail}"
        )
    return output, converter.output_format
