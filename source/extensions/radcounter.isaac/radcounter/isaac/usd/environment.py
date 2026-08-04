"""Author normalized external environments as USD for Isaac, PhysX, and Embree."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from radcounter.core.environment.geometry import normalization_matrix
from radcounter.core.environment.models import (
    AxisDirection,
    CollisionApproximation,
    CoordinateDefaults,
    EnvironmentImportConfig,
    EnvironmentScene,
)


class EnvironmentUsdUnavailable(RuntimeError):
    """Raised when USD authoring is called outside an Isaac/Kit host."""


def _pxr_modules() -> tuple[Any, ...]:
    try:
        from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, Vt
    except ModuleNotFoundError as error:
        raise EnvironmentUsdUnavailable(
            "environment USD authoring requires Isaac Sim's pxr modules"
        ) from error
    return Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, Vt


def _author_environment_attributes(prim: Any, sdf: Any, scene: EnvironmentScene) -> None:
    values = {
        "rad:environment:id": (sdf.ValueTypeNames.String, scene.environment_id),
        "rad:environment:sourceUri": (sdf.ValueTypeNames.Asset, scene.source_uri),
        "rad:environment:sourceSha256": (sdf.ValueTypeNames.String, scene.source_sha256),
        "rad:environment:sourceFormat": (
            sdf.ValueTypeNames.String,
            scene.source_format.value,
        ),
    }
    for name, (value_type, value) in values.items():
        prim.CreateAttribute(name, value_type, custom=True).Set(value)


def _material_for_name(name: str, config: EnvironmentImportConfig) -> str:
    for rule in config.material_rules:
        if fnmatch.fnmatchcase(name, rule.pattern):
            return rule.material_id
    return config.default_material_id


class EnvironmentUsdWriter:
    """Create a stable generated stage while preserving the source as immutable input."""

    def write(self, scene: EnvironmentScene, output_path: str | Path) -> Path:
        gf, sdf, tf, usd, usd_geom, usd_physics, vt = _pxr_modules()
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        stage = usd.Stage.CreateNew(str(destination))
        if stage is None:
            raise RuntimeError(f"USD could not create an environment stage: {destination}")
        usd_geom.SetStageUpAxis(stage, usd_geom.Tokens.z)
        usd_geom.SetStageMetersPerUnit(stage, 1.0)
        world = usd_geom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        environment = usd_geom.Xform.Define(stage, "/World/Environment")
        _author_environment_attributes(environment.GetPrim(), sdf, scene)
        if scene.native_stage_uri is not None:
            self._reference_native(
                stage,
                environment.GetPrim(),
                scene,
                gf,
                sdf,
                tf,
                usd,
                usd_geom,
                usd_physics,
            )
        else:
            used_names: set[str] = set()
            for mesh in scene.meshes:
                name = tf.MakeValidIdentifier(mesh.name.rsplit("/", maxsplit=1)[-1])
                name = name or mesh.mesh_id.replace("-", "_")
                if name in used_names:
                    name = f"{name}_{mesh.mesh_id.replace('-', '_')}"
                used_names.add(name)
                schema = usd_geom.Mesh.Define(stage, f"/World/Environment/{name}")
                prim = schema.GetPrim()
                points = vt.Vec3fArray(
                    [gf.Vec3f(float(x), float(y), float(z)) for x, y, z in mesh.vertices_m]
                )
                schema.GetPointsAttr().Set(points)
                schema.GetFaceVertexCountsAttr().Set([3] * len(mesh.triangles))
                schema.GetFaceVertexIndicesAttr().Set(mesh.triangles.reshape(-1).tolist())
                schema.CreateSubdivisionSchemeAttr(usd_geom.Tokens.none)
                prim.CreateAttribute(
                    "rad:environment:meshId", sdf.ValueTypeNames.String, custom=True
                ).Set(mesh.mesh_id)
                if mesh.radiation_enabled:
                    prim.CreateAttribute(
                        "rad:material:id", sdf.ValueTypeNames.String, custom=True
                    ).Set(mesh.material_id)
                    prim.CreateAttribute(
                        "rad:material:mode", sdf.ValueTypeNames.String, custom=True
                    ).Set("solid")
                color = mesh.display_color_rgb or (0.52, 0.57, 0.60)
                schema.CreateDisplayColorPrimvar().Set([gf.Vec3f(*color)])
                if not mesh.visual_enabled:
                    usd_geom.Imageable(prim).MakeInvisible()
                if mesh.collision_enabled:
                    self._apply_collision(prim, scene.import_config, usd_physics)
        stage.GetRootLayer().Save()
        return destination

    @staticmethod
    def _apply_collision(
        prim: Any, config: EnvironmentImportConfig | None, usd_physics: Any
    ) -> None:
        if config is None or not config.collision.enabled:
            return
        usd_physics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
        approximation = config.collision.approximation
        if approximation != CollisionApproximation.NONE:
            token = (
                "none" if approximation == CollisionApproximation.TRIANGLE_MESH else "convexHull"
            )
            usd_physics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(token)

    def _reference_native(
        self,
        stage: Any,
        root: Any,
        scene: EnvironmentScene,
        gf: Any,
        sdf: Any,
        tf: Any,
        usd: Any,
        usd_geom: Any,
        usd_physics: Any,
    ) -> None:
        del tf
        source_path = Path(scene.native_stage_uri or "").resolve()
        source_stage = usd.Stage.Open(str(source_path))
        if source_stage is None:
            raise RuntimeError(f"USD could not open the source environment: {source_path}")
        default_prim = source_stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            root.GetReferences().AddReference(str(source_path), default_prim.GetPath())
        else:
            children = list(source_stage.GetPseudoRoot().GetChildren())
            if not children:
                raise RuntimeError(f"source USD has no root prims: {source_path}")
            for child in children:
                destination = stage.OverridePrim(root.GetPath().AppendChild(child.GetName()))
                destination.GetReferences().AddReference(str(source_path), child.GetPath())
        config = scene.import_config or EnvironmentImportConfig(uri=str(source_path))
        up_token = usd_geom.GetStageUpAxis(source_stage)
        defaults = CoordinateDefaults(
            unit_scale_m=float(usd_geom.GetStageMetersPerUnit(source_stage)),
            up_axis=(AxisDirection.POS_Y if up_token == usd_geom.Tokens.y else AxisDirection.POS_Z),
            forward_axis=AxisDirection.POS_X,
        )
        matrix = normalization_matrix(config, defaults)
        gf_matrix = gf.Matrix4d()
        for row in range(4):
            gf_matrix.SetRow(row, gf.Vec4d(*[float(value) for value in matrix[row]]))
        usd_geom.Xformable(root).AddTransformOp().Set(gf_matrix)
        for prim in stage.Traverse():
            if prim == root or not prim.IsA(usd_geom.Gprim):
                continue
            material = _material_for_name(str(prim.GetPath()), config)
            prim.CreateAttribute("rad:material:id", sdf.ValueTypeNames.String, custom=True).Set(
                material
            )
            prim.CreateAttribute("rad:material:mode", sdf.ValueTypeNames.String, custom=True).Set(
                "solid"
            )
            if prim.IsA(usd_geom.Mesh):
                self._apply_collision(prim, config, usd_physics)
