"""USD tile stages for out-of-core environments."""

from __future__ import annotations

import re
from pathlib import Path

from radcounter.core.environment.streaming import (
    EnvironmentStreamingIndex,
    load_tile_lod,
)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"Mesh_{cleaned}"


class LargeEnvironmentUsdWriter:
    """Writes one payload per tile/LOD and a lightweight root stage."""

    def write(
        self,
        index_path: str | Path,
        stage_path: str | Path,
    ) -> Path:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

        index = EnvironmentStreamingIndex.load(index_path)
        target = Path(stage_path).expanduser().resolve()
        payload_directory = target.parent / f"{target.stem}_tiles"
        payload_directory.mkdir(parents=True, exist_ok=True)
        payload_uris: dict[tuple[str, int], Path] = {}

        for tile in index.tiles:
            for lod in tile.lods:
                payload_path = payload_directory / f"{tile.tile_id}.lod{lod.level}.usda"
                payload_stage = Usd.Stage.CreateNew(str(payload_path))
                payload_stage.SetMetadata("metersPerUnit", 1.0)
                UsdGeom.SetStageUpAxis(payload_stage, UsdGeom.Tokens.z)
                root = UsdGeom.Xform.Define(payload_stage, "/EnvironmentTile")
                payload_stage.SetDefaultPrim(root.GetPrim())
                for mesh_index, mesh in enumerate(load_tile_lod(lod.uri)):
                    name = _safe_name(f"{mesh.name}_{mesh_index:03d}")
                    usd_mesh = UsdGeom.Mesh.Define(
                        payload_stage,
                        f"/EnvironmentTile/{name}",
                    )
                    usd_mesh.CreatePointsAttr(
                        Vt.Vec3fArray.FromNumpy(mesh.vertices_m.astype("float32"))
                    )
                    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.triangles)))
                    usd_mesh.CreateFaceVertexIndicesAttr(
                        Vt.IntArray.FromNumpy(mesh.triangles.astype("int32").reshape(-1))
                    )
                    usd_mesh.CreateSubdivisionSchemeAttr("none")
                    prim = usd_mesh.GetPrim()
                    material_id = str(mesh.metadata.get("material_id", "default"))
                    prim.CreateAttribute(
                        "rad:material:id",
                        Sdf.ValueTypeNames.String,
                    ).Set(material_id)
                    prim.CreateAttribute(
                        "rad:stream:tileId",
                        Sdf.ValueTypeNames.String,
                    ).Set(tile.tile_id)
                    prim.CreateAttribute(
                        "rad:stream:lod",
                        Sdf.ValueTypeNames.Int,
                    ).Set(lod.level)
                    if lod.level == 0 and bool(mesh.metadata.get("collision", True)):
                        UsdPhysics.CollisionAPI.Apply(prim)
                        prim.CreateAttribute(
                            "rad:transport:enabled",
                            Sdf.ValueTypeNames.Bool,
                        ).Set(True)
                payload_stage.GetRootLayer().Save()
                payload_uris[(tile.tile_id, lod.level)] = payload_path

        stage = Usd.Stage.CreateNew(str(target))
        stage.SetMetadata("metersPerUnit", 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        root = UsdGeom.Xform.Define(stage, "/World/Environment")
        root_prim = root.GetPrim()
        root_prim.CreateAttribute(
            "rad:stream:indexUri",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath(str(Path(index_path).expanduser().resolve())))
        root_prim.CreateAttribute(
            "rad:stream:enabled",
            Sdf.ValueTypeNames.Bool,
        ).Set(True)

        for tile in index.tiles:
            tile_prim = UsdGeom.Xform.Define(
                stage,
                f"/World/Environment/{_safe_name(tile.tile_id)}",
            ).GetPrim()
            tile_prim.CreateAttribute(
                "rad:stream:tileId",
                Sdf.ValueTypeNames.String,
            ).Set(tile.tile_id)
            tile_prim.CreateAttribute(
                "rad:stream:boundsMin",
                Sdf.ValueTypeNames.Double3,
            ).Set(Gf.Vec3d(*tile.bounds_min_m))
            tile_prim.CreateAttribute(
                "rad:stream:boundsMax",
                Sdf.ValueTypeNames.Double3,
            ).Set(Gf.Vec3d(*tile.bounds_max_m))
            for lod in tile.lods:
                relative = payload_uris[(tile.tile_id, lod.level)].relative_to(target.parent)
                tile_prim.CreateAttribute(
                    f"rad:stream:lod{lod.level}Uri",
                    Sdf.ValueTypeNames.Asset,
                ).Set(Sdf.AssetPath(relative.as_posix()))
                tile_prim.CreateAttribute(
                    f"rad:stream:lod{lod.level}Triangles",
                    Sdf.ValueTypeNames.Int64,
                ).Set(lod.triangle_count)

        stage.GetRootLayer().Save()
        return target
