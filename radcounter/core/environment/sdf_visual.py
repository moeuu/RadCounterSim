"""Read visual meshes and PBR metadata from a Gazebo SDF model.

The core package deliberately does not depend on Gazebo or Isaac Sim.  This
module turns an SDF model into a small, renderer-neutral description that an
Isaac extension (or another renderer) can consume without losing visual poses,
mesh scale, or the metal/roughness texture set.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SdfPose:
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SdfPbrTextures:
    base_color: Path | None = None
    normal: Path | None = None
    roughness: Path | None = None
    metallic: Path | None = None
    emissive: Path | None = None


@dataclass(frozen=True)
class SdfVisualAsset:
    link_name: str
    visual_name: str
    link_pose: SdfPose
    visual_pose: SdfPose
    mesh_path: Path
    mesh_scale: tuple[float, float, float]
    textures: SdfPbrTextures


@dataclass(frozen=True)
class SdfVisualModel:
    name: str
    pose: SdfPose
    source_path: Path
    visuals: tuple[SdfVisualAsset, ...]


def _parse_vector(text: str | None, size: int, default: float) -> tuple[float, ...]:
    if not text or not text.strip():
        return (default,) * size
    values = tuple(float(value) for value in text.split())
    if len(values) != size:
        raise ValueError(f"expected {size} values, got {len(values)}: {text!r}")
    return values


def _parse_pose(element: ET.Element | None) -> SdfPose:
    values = _parse_vector(element.text if element is not None else None, 6, 0.0)
    return SdfPose(
        translation_m=(values[0], values[1], values[2]),
        rotation_rpy_rad=(values[3], values[4], values[5]),
    )


def _resolve_uri(uri: str | None, model_root: Path, model_name: str) -> Path | None:
    if not uri or not uri.strip():
        return None
    value = uri.strip()
    if value.startswith("file://"):
        return Path(value[7:]).expanduser().resolve()
    if value.startswith("model://"):
        relative = value[len("model://") :]
        prefix = f"{model_name}/"
        if relative.startswith(prefix):
            relative = relative[len(prefix) :]
        else:
            parts = relative.split("/", 1)
            relative = parts[1] if len(parts) == 2 else parts[0]
        return (model_root / relative).resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (model_root / path).resolve()


def load_sdf_visual_model(path: str | Path) -> SdfVisualModel:
    """Load mesh visuals from an SDF file and validate local asset references."""

    source_path = Path(path).expanduser().resolve()
    root = ET.parse(source_path).getroot()
    model = root if root.tag == "model" else root.find("model")
    if model is None:
        raise ValueError(f"SDF does not contain a model: {source_path}")

    model_name = model.get("name") or source_path.parent.name
    model_root = source_path.parent
    visuals: list[SdfVisualAsset] = []

    for link in model.findall("link"):
        link_name = link.get("name") or f"link_{len(visuals)}"
        link_pose = _parse_pose(link.find("pose"))
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            mesh_path = _resolve_uri(mesh.findtext("uri"), model_root, model_name)
            if mesh_path is None:
                continue
            if not mesh_path.is_file():
                raise FileNotFoundError(f"SDF visual mesh is missing: {mesh_path}")

            metal = visual.find("material/pbr/metal")

            def texture(tag: str, material: ET.Element | None = metal) -> Path | None:
                resolved = _resolve_uri(
                    material.findtext(tag) if material is not None else None,
                    model_root,
                    model_name,
                )
                if resolved is not None and not resolved.is_file():
                    raise FileNotFoundError(f"SDF texture is missing: {resolved}")
                return resolved

            scale = _parse_vector(mesh.findtext("scale"), 3, 1.0)
            visuals.append(
                SdfVisualAsset(
                    link_name=link_name,
                    visual_name=visual.get("name") or f"visual_{len(visuals)}",
                    link_pose=link_pose,
                    visual_pose=_parse_pose(visual.find("pose")),
                    mesh_path=mesh_path,
                    mesh_scale=(scale[0], scale[1], scale[2]),
                    textures=SdfPbrTextures(
                        base_color=texture("albedo_map"),
                        normal=texture("normal_map"),
                        roughness=texture("roughness_map"),
                        metallic=texture("metalness_map"),
                        emissive=texture("emissive_map"),
                    ),
                )
            )

    if not visuals:
        raise ValueError(f"SDF does not contain mesh visuals: {source_path}")
    return SdfVisualModel(
        name=model_name,
        pose=_parse_pose(model.find("pose")),
        source_path=source_path,
        visuals=tuple(visuals),
    )
